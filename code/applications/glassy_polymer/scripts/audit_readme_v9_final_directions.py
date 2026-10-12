"""Fixed-final-mesh directional finite differences and finite scale orbits.

No constitutive source or frozen matrix is changed. Every forward evaluation is
cached atomically. The six directions and all amplitudes are selected before
running any perturbed history. ``--benchmark`` runs only C06/base and the first
central pair; rerunning without that flag resumes those exact computations.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time


def verify_frozen_outputs():
    """Read-only NumPy check; this branch executes before importing JAX."""
    import numpy as np
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    app = Path(__file__).resolve().parents[1]
    output = args.output or app / "results/readme_v9_final_directions"
    manifest_path = output / "fixed_object_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    signature = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    parts, completed = [], {}
    for row in manifest["row_manifest"]:
        path = output / f"{row['curve_id']}_evaluations.npz"
        if not path.exists():
            completed[row["curve_id"]] = 0
            continue
        with np.load(path) as archive:
            if str(archive["manifest_signature"].item()) != signature:
                raise RuntimeError(f"Manifest signature mismatch: {path.name}")
            values = archive["predictions"].copy()
            labels = archive["evaluation_labels"].tolist()
        completed[row["curve_id"]] = int(np.isfinite(values).all(axis=1).sum())
        parts.append(values)
    if len(parts) != 10 or any(value != 41 for value in completed.values()):
        print(json.dumps({"status": "INCOMPLETE", "completed_evaluations_by_curve": completed}, indent=2))
        return 2
    prediction = np.concatenate(parts, axis=1)
    matrix_path = app / "results/v9_creep_continuation/creep_C1_L4_full.npz"
    if hashlib.sha256(matrix_path.read_bytes()).hexdigest() != manifest["final_matrix_sha256"]:
        raise RuntimeError("Frozen final matrix SHA mismatch")
    with np.load(matrix_path) as archive:
        matrix = archive["jacobian"]
    report = json.loads((output / "directional_summary.json").read_text())
    if report["status"] != "completed" or len(report["directional_checks"]) != 18:
        raise RuntimeError("Incomplete directional report")
    sigma1 = manifest["sigma1"]
    floor = np.sqrt(np.finfo(float).eps) * sigma1
    checks = 0
    for item in report["directional_checks"]:
        key, step = item["direction"], item["step"]
        ad = matrix @ np.asarray(manifest["directions"][key])
        fd = (prediction[labels.index(f"{key}__{step:g}__plus")]
              - prediction[labels.index(f"{key}__{step:g}__minus")]) / (2 * step)
        error = np.linalg.norm(fd-ad)
        expected = {"absolute_l2_error": error, "error_over_sigma1": error/sigma1,
                    "protected_relative_error": error/max(np.linalg.norm(ad), np.linalg.norm(fd), floor),
                    "error_over_nonzero_weak_sigma21": error/manifest["nonzero_weak_sigma21"],
                    "error_over_coupled_delta2": error/manifest["coupled_delta2"]}
        for key, value in expected.items():
            if not np.isclose(item[key], value, rtol=2.e-9, atol=2.e-13):
                raise RuntimeError(f"Directional algebra mismatch: {item['direction']}/{step}/{key}")
            checks += 1
    for item in report["finite_scale_orbits"]:
        c = item["scale_factor"]
        delta = prediction[labels.index(f"finite_orbit__{c:g}")] - prediction[0]
        if not np.isclose(item["weighted_l2_output_difference"], np.linalg.norm(delta), rtol=2.e-9, atol=1.e-15):
            raise RuntimeError("Finite scale-orbit algebra mismatch")
        checks += 1
    print(json.dumps({"status": "PASS", "checks": checks, "shape": [779, 22],
                      "direction_amplitude_pairs": 18, "finite_scale_orbits": 4}, indent=2))
    return 0


if "--verify" in sys.argv:
    raise SystemExit(verify_frozen_outputs())

# Keep independent audit workers bounded on the shared machine.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")

import jax
import jax.numpy as jnp
import numpy as np

from applications.glassy_polymer.canonical import CANONICAL_REGULARIZATION, SENSITIVITY_RESOLUTION
from applications.glassy_polymer.constitutive import pmma_parameters
from applications.glassy_polymer.data import resolve_data_root
from applications.glassy_polymer.experiments import predict
from applications.glassy_polymer.objectives import path_integration_weights
from applications.glassy_polymer.scripts.audit_v9_geometry_convergence import (
    CORE_SOURCES, digest, peak_rss_mb, write_json,
)
from applications.glassy_polymer.scripts.run_joint_observability import (
    record_balancing_factors, selected_experiments,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/readme_v9_final_directions"
FROZEN_OUTPUT = OUTPUT
FINAL = ROOT / "results/v9_creep_continuation/creep_C1_L4_full.npz"
SUMMARY = ROOT / "results/v9_creep_continuation/creep_continuation_summary.json"
V9_SUMMARY = ROOT / "results/v9_geometry_convergence/geometry_convergence_summary.json"
STEPS = (1.e-3, 1.e-4, 1.e-5)
ORBIT_FACTORS = (.5, .8, 1.25, 2.)
RANDOM_SEED = 20260907
EXECUTION_SOURCE_SHA256 = digest(Path(__file__))


def portable_experiment(experiment):
    descriptor = {}
    for key, value in vars(experiment).items():
        if isinstance(value, np.ndarray):
            value = value.tolist()
        elif key == "source_path":
            value = str(value.relative_to(resolve_data_root()))
        descriptor[key] = value
    descriptor["source_sha256"] = digest(experiment.source_path)
    return descriptor


def path_neutral_repr(value):
    return re.sub(r"source_path=(?:PosixPath|WindowsPath)\([^)]*\)",
                  "source_path=<verified-relative-path-and-content-sha256>", value)


def signed_unit(vector):
    vector = np.asarray(vector, dtype=float)
    vector = vector / np.linalg.norm(vector)
    if vector[np.argmax(abs(vector))] < 0:
        vector = -vector
    return vector


def _atomic_npz(path, **values):
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez(temporary, **values)
    temporary.replace(path)


def preflight():
    continuation = json.loads(SUMMARY.read_text())
    v9 = json.loads(V9_SUMMARY.read_text())
    with np.load(FINAL) as archive:
        matrix = archive["jacobian"].copy()
        names = archive["parameter_names"].tolist()
        settings = json.loads(str(archive["resolution"].item()))
        archived_gauge = archive["analytic_gauge"].copy()
    final_record = continuation["levels"][-1]["matrices"]["fine"]
    if final_record["sha256"] != digest(FINAL) or final_record["resolution"] != settings:
        raise RuntimeError("Final-matrix metadata/hash mismatch")
    base = pmma_parameters()
    if matrix.shape != (779, 22) or names != list(base._fields):
        raise RuntimeError("Unexpected final matrix shape/column order")
    source_hashes = {name: digest(ROOT / name) for name in CORE_SOURCES}
    for name, value in source_hashes.items():
        if continuation["source_hashes"][name] != value or v9["source_hashes"][name] != value:
            raise RuntimeError(f"Frozen numerical source changed: {name}")
    source_hashes.update({name: digest(ROOT / name) for name in
                         ("integration_mesh.py", "data.py", "data_cleaning.py")})
    resolution = replace(SENSITIVITY_RESOLUTION, **settings)
    rows = continuation["row_manifest"]
    experiments = selected_experiments()
    factors = record_balancing_factors(experiments)
    legacy_path = FROZEN_OUTPUT / "legacy_fixed_object_manifest_v1.json"
    reference_path = FROZEN_OUTPUT / "fixed_object_manifest.json"
    reference = json.loads(reference_path.read_text()) if reference_path.exists() else None
    legacy = json.loads(legacy_path.read_text()) if legacy_path.exists() else reference
    legacy_rows = {row["curve_id"]: row for row in legacy["row_manifest"]} if legacy else {}
    reference_rows = {row["curve_id"]: row for row in reference["row_manifest"]} if reference else {}
    weights_by_curve = {}
    input_manifest = []
    cursor = 0
    recovery_mask = np.zeros(779, dtype=bool)
    for experiment, row in zip(experiments, rows):
        curve = experiment.curve_id
        scale = (max(float(np.max(abs(experiment.y))), .02)
                 if experiment.family in {"creep", "recovery"} else 110.)
        if (row["curve_id"] != curve or row["row_start"] != cursor
                or row["row_end_exclusive"] != cursor + len(experiment.x)
                or row["observation_scale"] != scale
                or row["sqrt_family_record_observable_mass"] != factors[curve]):
            raise RuntimeError(f"Final fixed-row definition changed: {curve}")
        weights = factors[curve] * np.sqrt(path_integration_weights(experiment.x)) / scale
        weights_by_curve[curve] = weights
        # Reconstruct the exact generating-cache signature, including its own
        # historical resolution fields, even when those fields do not affect
        # a particular non-creep history.
        if experiment.family == "creep":
            cache = ROOT / f"results/v9_creep_continuation/creep_C1_{curve}.npz"
            historical_sources = continuation["source_hashes"]
            historical_settings = settings
        else:
            cache = ROOT / f"results/v9_geometry_convergence/strain_recovery_L4_{curve}.npz"
            historical_sources = v9["source_hashes"]
            historical_settings = next(item["resolution"] for item in v9["levels"]
                                       if item["label"] == "strain_recovery_L4")
        context = {"source_hashes": historical_sources, "resolution": historical_settings,
                   "row": row, "names": names, "x": experiment.x.tolist(),
                   "y": experiment.y.tolist(), "weights": weights.tolist(),
                   "experiment": repr(experiment)}
        candidates = [repr(experiment)]
        historical_repr = legacy_rows.get(curve, {}).get("experiment")
        if isinstance(historical_repr, str):
            if path_neutral_repr(historical_repr) != path_neutral_repr(repr(experiment)):
                raise RuntimeError(f"Historical protocol/record metadata changed: {curve}")
            candidates.append(historical_repr)
        descriptor = portable_experiment(experiment)
        frozen_descriptor = reference_rows.get(curve, {}).get("experiment")
        if isinstance(frozen_descriptor, dict) and frozen_descriptor != descriptor:
            raise RuntimeError(f"Portable experiment/content binding changed: {curve}")
        signatures = []
        for candidate in candidates:
            context["experiment"] = candidate
            signatures.append(hashlib.sha256(json.dumps(context, sort_keys=True).encode()).hexdigest())
        with np.load(cache) as archive:
            signature = str(archive["signature"].item())
            if signature not in signatures:
                raise RuntimeError(f"Frozen per-history source/input signature changed: {curve}")
            if not np.array_equal(archive["jacobian"], matrix[cursor:cursor + len(weights)]):
                raise RuntimeError(f"Frozen matrix differs from curve cache: {curve}")
        input_manifest.append({**row, "x": experiment.x.tolist(), "y": experiment.y.tolist(),
                               "row_weights": weights.tolist(), "experiment": descriptor,
                               "matrix_block_cache": str(cache.relative_to(ROOT)),
                               "matrix_block_sha256": digest(cache),
                               "matrix_block_signature_verified": signature})
        if experiment.family == "recovery":
            recovery_mask[cursor:cursor + len(weights)] = True
        cursor += len(weights)
    if cursor != 779 or len(experiments) != len(rows) or len(rows) != 10:
        raise RuntimeError("Final observation registry mismatch")
    _, singular, modes = np.linalg.svd(matrix, full_matrices=False)
    gauge_raw = np.zeros(22)
    gauge_raw[names.index("gamma0")] = 1.
    for name in ("s0", "kappa1_sat", "kappa2_sat"):
        gauge_raw[names.index(name)] = float(base.m0)
    for name in ("s1", "s2"):
        gauge_raw[names.index(name)] = float(base.m_ve)
    gauge = gauge_raw / np.linalg.norm(gauge_raw)
    if not np.allclose(gauge, archived_gauge, atol=1.e-15, rtol=0.):
        raise RuntimeError("Analytic scale tangent disagrees with frozen matrix")
    rng = np.random.default_rng(RANDOM_SEED)
    directions = {
        "major_v1": signed_unit(modes[0]),
        "weak_non_gauge_v21": signed_unit(modes[-2]),
        "recovery_dominant": signed_unit(np.linalg.svd(matrix[recovery_mask], full_matrices=False)[2][0]),
        "fixed_random_1": signed_unit(rng.normal(size=22)),
        "fixed_random_2": signed_unit(rng.normal(size=22)),
        "analytic_scale_tangent": gauge,
    }
    manifest = {
        "schema_version": 2, "shape": [779, 22], "parameter_names": names,
        "nominal_parameters": {name: float(getattr(base, name)) for name in names},
        "resolution": settings, "source_hashes": source_hashes,
        "final_matrix_sha256": digest(FINAL), "continuation_summary_sha256": digest(SUMMARY),
        "row_manifest": input_manifest, "steps": list(STEPS), "orbit_factors": list(ORBIT_FACTORS),
        "random_seed": RANDOM_SEED, "directions": {key: value.tolist() for key, value in directions.items()},
        "direction_selection": "v1 and v21 of the frozen final weighted matrix; recovery-block v1; two seeded Gaussian unit vectors; analytic gauge. Signs set by largest-magnitude coefficient, except positive-gamma0 gauge.",
        "sigma1": float(singular[0]), "nonzero_weak_sigma21": float(singular[-2]),
        "coupled_delta2": continuation["levels"][-1]["comparisons"]["coupled_all_axes"]["absolute_spectral_difference"],
        "analytic_gauge_raw": gauge_raw.tolist(),
        "fixed_mesh_basis": "All meshes and interpolation rows depend only on frozen experiment controls and resolution, never perturbed material parameters.",
        "orbit_definition": "gamma0*=c; (s0,kappa1_sat,kappa2_sat)*=c**m0; (s1,s2)*=c**m_ve; other parameters unchanged. Virgin kappa=0 transforms consistently. This is a nonlinear physical-parameter orbit, exactly a straight line in the present log-ratio coordinates because m0,m_ve stay fixed; it is not a generic additive physical-parameter perturbation.",
    }
    return base, names, experiments, weights_by_curve, resolution, matrix, directions, manifest


def migrate_v1_context(path, current, experiments):
    """Rebind only absolute-path metadata after exact science/cache checks.

    Old manifest, complete original scalar cache, and old producer hashes are
    preserved. No response entry or physical parameter vector is recomputed or
    changed by this migration.
    """
    previous = json.loads(path.read_text())
    if previous == current:
        return
    if previous.get("schema_version") != 1:
        raise RuntimeError("Existing audit uses a different scientific object")
    normalized = json.loads(json.dumps(previous))
    normalized["schema_version"] = 2
    for old_row, new_row, experiment in zip(normalized["row_manifest"], current["row_manifest"], experiments):
        if path_neutral_repr(old_row["experiment"]) != path_neutral_repr(repr(experiment)):
            raise RuntimeError("Cannot migrate changed protocol metadata")
        old_row["experiment"] = new_row["experiment"]
    if normalized != current:
        raise RuntimeError("Migration attempted to change arrays, weights, mesh, columns, or directions")
    old_signature = hashlib.sha256(json.dumps(previous, sort_keys=True).encode()).hexdigest()
    new_signature = hashlib.sha256(json.dumps(current, sort_keys=True).encode()).hexdigest()
    backup = OUTPUT / "legacy_fixed_object_manifest_v1.json"
    if not backup.exists():
        backup.write_bytes(path.read_bytes())
    migrated = []
    for row in previous["row_manifest"]:
        cache = OUTPUT / f"{row['curve_id']}_evaluations.npz"
        if not cache.exists():
            continue
        with np.load(cache) as archive:
            payload = {key: archive[key].copy() for key in archive.files}
        if str(payload["manifest_signature"].item()) != old_signature:
            raise RuntimeError("V1 cache binding mismatch during migration")
        original_sha = digest(cache)
        original = OUTPUT / f"{row['curve_id']}_evaluations_context_v1.npz"
        if not original.exists():
            original.write_bytes(cache.read_bytes())
        payload["manifest_signature"] = new_signature
        payload["previous_manifest_signature"] = old_signature
        payload["previous_cache_sha256"] = original_sha
        _atomic_npz(cache, **payload)
        migrated.append({"curve_id": row["curve_id"], "original_cache_sha256": original_sha,
                         "rebound_cache_sha256": digest(cache),
                         "predictions_and_parameter_vectors_bitwise_unchanged": True})
    write_json(OUTPUT / "portable_context_migration.json", {
        "old_signature": old_signature, "new_signature": new_signature,
        "reason": "Replace workstation source_path repr by raw-root-relative path plus workbook content SHA; all other scientific fields are identical.",
        "caches": migrated})


def evaluations(base, names, directions):
    base_values = np.array([float(getattr(base, name)) for name in names])
    values = [("base", base_values)]
    for key, vector in directions.items():
        for h in STEPS:
            for sign, label in ((1., "plus"), (-1., "minus")):
                # Match replace_log_ratio_block's JAX exponential and reference multiplication.
                physical = np.asarray(jnp.asarray(base_values) * jnp.exp(jnp.asarray(sign * h * vector)))
                values.append((f"{key}__{h:g}__{label}", physical))
    for c in ORBIT_FACTORS:
        orbit = base_values.copy()
        orbit[names.index("gamma0")] *= c
        for key in ("s0", "kappa1_sat", "kappa2_sat"):
            orbit[names.index(key)] *= c ** float(base.m0)
        for key in ("s1", "s2"):
            orbit[names.index(key)] *= c ** float(base.m_ve)
        values.append((f"finite_orbit__{c:g}", orbit))
    return values


def merge_partition_outputs(sources, manifest, values, prefer_incoming):
    """Merge disjoint, fully bound evaluations; preserve every prior cache.

    An explicit scalar-pair preference can replace a duplicate batched point,
    with both arrays retained and their difference reported, never silently.
    """
    signature = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    labels = np.asarray([key for key, _ in values])
    physical = np.asarray([value for _, value in values])
    records = []
    for source in sources:
        source = source.resolve()
        source_manifest = json.loads((source / "fixed_object_manifest.json").read_text())
        if source_manifest != manifest:
            raise RuntimeError(f"Partition fixed-object manifest differs: {source}")
        for row in manifest["row_manifest"]:
            curve = row["curve_id"]
            incoming_path = source / f"{curve}_evaluations.npz"
            if not incoming_path.exists():
                continue
            with np.load(incoming_path) as archive:
                incoming = {key: archive[key].copy() for key in archive.files}
            if (str(incoming["manifest_signature"].item()) != signature
                    or not np.array_equal(incoming["evaluation_labels"], labels)
                    or not np.array_equal(incoming["physical_parameter_vectors"], physical)):
                raise RuntimeError(f"Partition parameters/labels/signature differ: {incoming_path}")
            target = OUTPUT / incoming_path.name
            if target.exists():
                with np.load(target) as archive:
                    current = {key: archive[key].copy() for key in archive.files}
                if str(current["manifest_signature"].item()) != signature:
                    raise RuntimeError("Target was not migrated to the portable context")
                backup = OUTPUT / f"{curve}_before_merge_{digest(target)[:16]}.npz"
                if not backup.exists():
                    backup.write_bytes(target.read_bytes())
            else:
                current = {"predictions": np.full_like(incoming["predictions"], np.nan),
                           "evaluation_seconds": np.full(41, np.nan),
                           "manifest_signature": signature, "evaluation_labels": labels,
                           "physical_parameter_vectors": physical,
                           "execution_source_sha256": EXECUTION_SOURCE_SHA256}
            for payload in (current, incoming):
                if "evaluation_source_sha256" not in payload:
                    hashes = np.full(41, "", dtype="U64")
                    hashes[np.isfinite(payload["predictions"]).all(axis=1)] = str(np.asarray(payload["execution_source_sha256"]).item())
                    payload["evaluation_source_sha256"] = hashes
            scalar_backup = OUTPUT / f"{curve}_scalar_benchmark_v1.npz"
            if scalar_backup.exists():
                with np.load(scalar_backup) as scalar:
                    for i in np.flatnonzero(np.isfinite(scalar["predictions"]).all(axis=1)):
                        if not np.array_equal(current["predictions"][i], scalar["predictions"][i]):
                            raise RuntimeError("Original scalar values changed before partition merge")
                        current["evaluation_source_sha256"][i] = str(scalar["execution_source_sha256"].item())
            selected, overlaps = [], []
            for i in np.flatnonzero(np.isfinite(incoming["predictions"]).all(axis=1)):
                overlap = np.isfinite(current["predictions"][i]).all()
                if overlap:
                    difference = incoming["predictions"][i] - current["predictions"][i]
                    identical = np.array_equal(incoming["predictions"][i], current["predictions"][i])
                    overlaps.append({"evaluation_index": int(i), "label": str(labels[i]),
                                     "bitwise_identical": identical,
                                     "l2_difference": float(np.linalg.norm(difference)),
                                     "linf_difference": float(np.max(abs(difference))),
                                     "incoming_preferred": bool(i in prefer_incoming)})
                    if i not in prefer_incoming:
                        if not identical:
                            raise RuntimeError("Non-identical duplicate lacks an explicit, archived merge policy")
                        continue
                current["predictions"][i] = incoming["predictions"][i]
                current["evaluation_seconds"][i] = incoming["evaluation_seconds"][i]
                current["evaluation_source_sha256"][i] = incoming["evaluation_source_sha256"][i]
                selected.append(int(i))
            current["execution_source_sha256"] = EXECUTION_SOURCE_SHA256
            _atomic_npz(target, **current)
            records.append({"curve_id": curve, "source_directory": os.path.relpath(source, OUTPUT),
                            "source_cache_sha256": digest(incoming_path), "merged_cache_sha256": digest(target),
                            "selected_evaluation_indices": selected, "duplicates": overlaps})
        for snapshot in source.glob("COMPUTATION_SOURCE_*.py"):
            destination = OUTPUT / snapshot.name
            if destination.exists() and digest(destination) != digest(snapshot):
                raise RuntimeError("Producer snapshot name/content conflict")
            if not destination.exists():
                destination.write_bytes(snapshot.read_bytes())
    write_json(OUTPUT / f"partition_merge_{time.time_ns()}.json", {
        "manifest_signature": signature, "records": records,
        "predeclared_prefer_incoming_indices": sorted(prefer_incoming),
        "policy": "Only explicitly listed duplicates may prefer incoming scalar results, to retain a same-kernel central pair; both original caches and measured differences are retained."})


def nominal_replay_comparison(manifest, observations):
    """Compare independent A1 nominal replays without generating any history."""
    reference = Path(os.environ.get(
        "REPRO_README_V9_GUARD_ROOT", ROOT / "results/readme_v9_derivative_guards")).resolve()
    definition_path = reference / "audit_definition.json"
    required = [definition_path, reference / "COMPUTATION_SOURCE.py"]
    for row in manifest["row_manifest"]:
        required.extend(reference / f"mother_final_{row['curve_id']}_{suffix}"
                        for suffix in ("observations.npz", "summary.json"))
    if not all(path.exists() for path in required):
        return {"status": "reference_unavailable", "histories_computed_for_comparison": 0,
                "interpretation": "Optional independent A1 nominal replay is not complete; no substitution for directional finite differences."}
    definition = json.loads(definition_path.read_text())
    if (definition["mother_matrix_sha256"] != manifest["final_matrix_sha256"]
            or definition["mother_resolution"] != manifest["resolution"]
            or digest(reference / "COMPUTATION_SOURCE.py") != definition["driver_sha256"]):
        raise RuntimeError("A1 nominal reference object/producer differs")
    for name, source_sha in definition["core_source_sha256"].items():
        if digest(ROOT / name) != source_sha:
            raise RuntimeError(f"A1 nominal reference numerical source changed: {name}")
    records, differences = [], []
    for row in manifest["row_manifest"]:
        curve = row["curve_id"]
        report = json.loads((reference / f"mother_final_{curve}_summary.json").read_text())
        if (report["resolution"] != manifest["resolution"]
                or report["source_sha256"] != row["experiment"]["source_sha256"]):
            raise RuntimeError(f"A1 nominal reference resolution/raw record differs: {curve}")
        with np.load(reference / f"mother_final_{curve}_observations.npz") as archive:
            if (not np.array_equal(archive["x"], row["x"])
                    or not np.array_equal(archive["measured"], row["y"])):
                raise RuntimeError(f"A1 nominal reference native observations differ: {curve}")
            weighted = archive["prediction"] * np.asarray(row["row_weights"])
        target = observations[0, row["row_start"]:row["row_end_exclusive"]]
        difference = target - weighted
        if not np.isfinite(difference).all():
            raise RuntimeError(f"Non-finite independent nominal comparison: {curve}")
        differences.append(difference)
        records.append({"curve_id": curve, "native_point_count": len(weighted),
                        "weighted_l2_difference": float(np.linalg.norm(difference)),
                        "weighted_linf_difference": float(np.max(abs(difference))),
                        "relative_l2_difference": float(np.linalg.norm(difference) /
                                                        max(np.linalg.norm(weighted), 1.e-30))})
    combined = np.concatenate(differences)
    return {"status": "completed", "histories_computed_for_comparison": 0,
            "reference_relative_to_output": os.path.relpath(reference, OUTPUT),
            "reference_files_sha256": {path.name: digest(path) for path in required},
            "curves": records,
            "joint_weighted_l2_difference": float(np.linalg.norm(combined)),
            "maximum_weighted_absolute_difference": float(np.max(abs(combined))),
            "interpretation": "A2 nominal production predictions minus independent A1 captured-history predictions, each multiplied by the identical frozen row weights; an implementation-consistency comparison, not a replacement for the 18 sampled directional checks or a continuum error bound."}


def analyze(matrix, directions, manifest, values, curve_results):
    if len(curve_results) != 10:
        return None
    labels = [label for label, _ in values]
    observations = np.concatenate([curve_results[row["curve_id"]]["predictions"]
                                   for row in manifest["row_manifest"]], axis=1)
    complete = np.all(np.isfinite(observations), axis=1)
    if not complete.all():
        return None
    sigma1 = manifest["sigma1"]
    denominator_floor = np.sqrt(np.finfo(float).eps) * sigma1
    results = []
    fd_vectors = []
    ad_vectors = []
    for key, vector in directions.items():
        ad = matrix @ vector
        for h in STEPS:
            plus = observations[labels.index(f"{key}__{h:g}__plus")]
            minus = observations[labels.index(f"{key}__{h:g}__minus")]
            fd = (plus - minus) / (2 * h)
            error = np.linalg.norm(fd - ad)
            adnorm, fdnorm = np.linalg.norm(ad), np.linalg.norm(fd)
            results.append({"direction": key, "step": h,
                            "ad_norm": float(adnorm), "fd_norm": float(fdnorm),
                            "absolute_l2_error": float(error), "absolute_linf_error": float(np.max(abs(fd-ad))),
                            "error_over_sigma1": float(error / sigma1),
                            "protected_relative_error": float(error / max(adnorm, fdnorm, denominator_floor)),
                            "error_over_nonzero_weak_sigma21": float(error / manifest["nonzero_weak_sigma21"]),
                            "error_over_coupled_delta2": float(error / manifest["coupled_delta2"]),
                            "ad_recovery_l2_norm": float(np.linalg.norm(ad[630:646])),
                            "fd_recovery_l2_error": float(np.linalg.norm((fd-ad)[630:646]))})
            fd_vectors.append(fd)
            ad_vectors.append(ad)
    base = observations[labels.index("base")]
    orbit_results = []
    for c in ORBIT_FACTORS:
        difference = observations[labels.index(f"finite_orbit__{c:g}")] - base
        orbit_results.append({"scale_factor": c, "weighted_l2_output_difference": float(np.linalg.norm(difference)),
                              "weighted_linf_output_difference": float(np.max(abs(difference))),
                              "difference_over_sigma1": float(np.linalg.norm(difference)/sigma1),
                              "difference_over_base_output_norm": float(np.linalg.norm(difference)/max(np.linalg.norm(base), 1.e-30))})
    _atomic_npz(OUTPUT / "assembled_directional_results.npz", predictions=observations,
                evaluation_labels=np.asarray(labels), physical_parameter_vectors=np.asarray([v for _, v in values]),
                fd_vectors=np.asarray(fd_vectors), ad_vectors=np.asarray(ad_vectors),
                direction_names=np.asarray([item["direction"] for item in results]),
                step_sizes=np.asarray([item["step"] for item in results]))
    return {"status": "completed", "directional_checks": results, "finite_scale_orbits": orbit_results,
            "independent_nominal_replay": nominal_replay_comparison(manifest, observations),
            "unique_parameter_points_per_curve": 41,
            "unique_curve_parameter_combinations": 410,
            "execution_counting_convention": "41 predeclared unique parameter points per path, hence 410 unique curve/point combinations. These are not total kernel/history execution counts: padded nominal points in final batches, benchmark overlaps, and the explicitly preserved scalar-pair duplicate cause additional real executions. Per-batch timings and per-point producer bindings are retained; this report does not claim a complete count of all historical/padded executions.",
            "protected_relative_denominator_floor": denominator_floor,
            "protected_relative_definition": "||FD-Jv||2 / max(||Jv||2,||FD||2,sqrt(machine_epsilon)*sigma1); gauge interpretation relies on absolute/sigma1 scales, not its near-zero observable directional derivative.",
            "acceptance_interpretation": "No fitted tolerance or post hoc magic threshold. Report all three amplitudes, error relative to the nonzero direction, weakest nongauge sensitivity, and matched two-mesh delta2. These are checks of one fixed discrete forward map, not continuum derivative/error bounds."}


def main():
    global OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--curves", nargs="+", help="Only selected curves; partial output remains explicitly incomplete")
    parser.add_argument("--postprocess-only", action="store_true")
    parser.add_argument("--batch-size", type=int, default=6,
                        help="Dynamic parameter points per vmap; one complete physical history per point")
    parser.add_argument("--batch-benchmark", action="store_true",
                        help="Run C06 first batch with overlap against completed scalar evaluations")
    parser.add_argument("--benchmark-curve", default="C06",
                        help="A single fixed-object curve for the bounded batch benchmark")
    parser.add_argument("--evaluation-indices", nargs="+", type=int,
                        help="Partition the predeclared 41 evaluations into independent output directories")
    parser.add_argument("--merge-inputs", nargs="+", type=Path,
                        help="Merge independently cached partitions with identical portable manifests")
    parser.add_argument("--prefer-incoming-indices", nargs="*", type=int, default=[],
                        help="Explicit duplicate policy, recorded with both preserved arrays and their differences")
    parser.add_argument("--output", type=Path, help="Fresh or resumable result directory")
    args = parser.parse_args()
    if args.output:
        OUTPUT = args.output.resolve()
    if not 1 <= args.batch_size <= 18:
        raise ValueError("Batch size must lie in [1,18]")
    if args.evaluation_indices and (len(set(args.evaluation_indices)) != len(args.evaluation_indices)
                                   or not all(0 <= i < 41 for i in args.evaluation_indices)):
        raise ValueError("Evaluation indices must be distinct members of 0..40")
    jax.config.update("jax_enable_x64", True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    snapshot = OUTPUT / f"COMPUTATION_SOURCE_{EXECUTION_SOURCE_SHA256}.py"
    if not snapshot.exists():
        snapshot.write_bytes(Path(__file__).read_bytes())
    base, names, experiments, weights_by_curve, resolution, matrix, directions, manifest = preflight()
    manifest_signature = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    manifest_path = OUTPUT / "fixed_object_manifest.json"
    if manifest_path.exists():
        migrate_v1_context(manifest_path, manifest, experiments)
    write_json(manifest_path, manifest)
    values = evaluations(base, names, directions)
    if args.merge_inputs:
        merge_partition_outputs(args.merge_inputs, manifest, values, set(args.prefer_incoming_indices))
    curve_results = {}
    timings = []
    start = time.monotonic()
    ordered = sorted(experiments, key=lambda experiment: experiment.curve_id != "C06")
    for experiment in ordered:
        curve = experiment.curve_id
        path = OUTPUT / f"{curve}_evaluations.npz"
        predictions = np.full((len(values), len(experiment.x)), np.nan)
        seconds = np.full(len(values), np.nan)
        producer_hashes = np.full(len(values), "", dtype="U64")
        if path.exists():
            with np.load(path) as archive:
                if str(archive["manifest_signature"].item()) != manifest_signature:
                    raise RuntimeError(f"Stale direction/input cache: {path}")
                predictions = archive["predictions"].copy()
                seconds = archive["evaluation_seconds"].copy()
                if "evaluation_source_sha256" in archive:
                    producer_hashes = archive["evaluation_source_sha256"].copy()
                else:
                    producer_hashes[np.isfinite(predictions).all(axis=1)] = str(archive["execution_source_sha256"].item())
            # The first batched benchmark deliberately preserved its three
            # scalar overlaps. Retain the actual original producer binding.
            scalar_backup = OUTPUT / f"{curve}_scalar_benchmark_v1.npz"
            if scalar_backup.exists():
                with np.load(scalar_backup) as scalar:
                    for i in np.flatnonzero(np.isfinite(scalar["predictions"]).all(axis=1)):
                        if not np.array_equal(predictions[i], scalar["predictions"][i]):
                            raise RuntimeError("Original scalar benchmark value changed")
                        producer_hashes[i] = str(scalar["execution_source_sha256"].item())
        curve_results[curve] = {"predictions": predictions}
        if args.postprocess_only or ((args.benchmark or args.batch_benchmark) and curve != args.benchmark_curve) or (args.curves and curve not in args.curves):
            continue
        weights = jnp.asarray(weights_by_curve[curve])

        def observations(physical_values):
            params = base._replace(**dict(zip(names, physical_values)))
            return weights * predict(experiment, params, CANONICAL_REGULARIZATION,
                                      checkpoint=True, resolution=resolution)

        batch_size = 1 if args.benchmark else args.batch_size
        pending = [i for i in range(len(values)) if not np.isfinite(predictions[i]).all()]
        if args.evaluation_indices is not None:
            pending = [i for i in pending if i in args.evaluation_indices]
        if args.benchmark:
            pending = [i for i in pending if i < 3]
        if args.batch_benchmark:
            pending = list(range(batch_size))
        if not pending:
            print(f"cached complete {curve}", flush=True)
            continue
        compile_start = time.monotonic()
        sample = jnp.asarray(np.stack([values[0][1]] * batch_size))
        print(f"compile {curve} batch={batch_size}", flush=True)
        write_json(OUTPUT / "execution_progress.json", {"status": "running", "stage": "compiling",
                   "curve_id": curve, "batch_size": batch_size,
                   "current_run_elapsed_seconds": time.monotonic()-start})
        # A size-one fallback must really be the original scalar program:
        # vmap(size=1) can still transform scalar conditionals into selects.
        kernel = jax.jit(observations) if batch_size == 1 else jax.jit(jax.vmap(observations))
        evaluate = kernel.lower(sample[0] if batch_size == 1 else sample).compile()
        compile_seconds = time.monotonic() - compile_start
        print(f"compiled {curve} batch={batch_size} {compile_seconds:.2f}s", flush=True)
        for offset in range(0, len(pending), batch_size):
            indices = pending[offset:offset + batch_size]
            padded = indices + [0] * (batch_size - len(indices))
            batch_parameters = np.stack([values[i][1] for i in padded])
            before = time.monotonic()
            print(f"compute {curve} batch labels={[values[i][0] for i in indices]}", flush=True)
            write_json(OUTPUT / "execution_progress.json", {"status": "running", "stage": "batch_forward",
                       "curve_id": curve, "batch_size": batch_size,
                       "evaluations": [values[i][0] for i in indices],
                       "compile_seconds": compile_seconds,
                       "current_run_elapsed_seconds": time.monotonic()-start,
                       "completed_batches_this_run": timings})
            argument = batch_parameters[0] if batch_size == 1 else batch_parameters
            outputs = np.asarray(evaluate(jnp.asarray(argument)))
            if batch_size == 1:
                outputs = outputs[None, :]
            elapsed = time.monotonic() - before
            overlap = []
            for position, index in enumerate(indices):
                if np.isfinite(predictions[index]).all():
                    overlap.append({"evaluation": values[index][0],
                                    "max_absolute_difference": float(np.max(abs(outputs[position]-predictions[index]))),
                                    "l2_difference": float(np.linalg.norm(outputs[position]-predictions[index]))})
                else:
                    predictions[index] = outputs[position]
                    producer_hashes[index] = EXECUTION_SOURCE_SHA256
                    # Shared batch wall time, not a fictitious scalar runtime.
                    seconds[index] = elapsed
            _atomic_npz(path, predictions=predictions, evaluation_seconds=seconds,
                        manifest_signature=manifest_signature,
                        evaluation_labels=np.asarray([label for label, _ in values]),
                        physical_parameter_vectors=np.asarray([value for _, value in values]),
                        evaluation_source_sha256=producer_hashes,
                        execution_source_sha256=EXECUTION_SOURCE_SHA256)
            timing = {"curve_id": curve, "evaluations": [values[i][0] for i in indices],
                      "batch_size": batch_size, "batch_runtime_seconds": elapsed,
                      "compile_seconds": compile_seconds if offset == 0 else 0.,
                      "seconds_per_parameter_point": elapsed / len(indices),
                      "scalar_overlap": overlap,
                      "finite": bool(np.isfinite(outputs).all()), "peak_rss_mb": peak_rss_mb()}
            if args.batch_benchmark:
                _atomic_npz(OUTPUT / f"{curve}_batch{batch_size}_benchmark.npz", predictions=outputs,
                            evaluation_indices=np.asarray(padded), physical_parameter_vectors=batch_parameters,
                            manifest_signature=manifest_signature)
            timings.append(timing)
            print(json.dumps(timing), flush=True)
            write_json(OUTPUT / "execution_progress.json", {"status": "running", "last_evaluation": timing,
                                                           "current_run_elapsed_seconds": time.monotonic()-start,
                                                           "current_run_timings": timings})
            if not timing["finite"]:
                raise FloatingPointError(f"Non-finite batch output preserved for {curve}")
        del evaluate
        jax.clear_caches()
        gc.collect()
    results = analyze(matrix, directions, manifest, values, curve_results)
    report = {"status": "benchmark_completed" if args.benchmark or args.batch_benchmark else "incomplete",
              "manifest_signature": manifest_signature, "manifest_sha256": digest(manifest_path),
              "execution_source_sha256": EXECUTION_SOURCE_SHA256, "elapsed_seconds_this_run": time.monotonic()-start,
              "timings_this_run": timings, "peak_rss_mb": peak_rss_mb(),
              "requested_evaluation_indices": args.evaluation_indices,
              "completed_evaluations_by_curve": {curve: int(np.isfinite(item["predictions"]).all(axis=1).sum())
                                                  for curve, item in curve_results.items()}}
    report["execution_source_snapshots"] = {digest(path): path.name
        for path in sorted(OUTPUT.glob("COMPUTATION_SOURCE_*.py"))}
    report["cache_producer_convention"] = "Each evaluation_source_sha256 binds the actual numerical producer; execution_source_sha256 identifies the latest cache/report writer and need not equal producers of preserved cached values. Scalar V1 and batch overlap evidence are archived separately."
    if results:
        report.update(results)
    write_json(OUTPUT / (f"batch{args.batch_size}_benchmark_summary.json" if args.batch_benchmark else
                         "benchmark_summary.json" if args.benchmark else "directional_summary.json"), report)
    print("FINAL", report["status"], f"elapsed={time.monotonic()-start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
