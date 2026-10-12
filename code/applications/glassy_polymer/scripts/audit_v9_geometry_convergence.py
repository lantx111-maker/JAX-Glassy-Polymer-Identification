"""Continue matched full-Jacobian mesh refinement without changing the operator.

The first sequence halves only the PMMA strain and recovery-time increments.
Its three creep blocks have exactly unchanged integration meshes and are reused.
After the declared angular gates pass, an independent creep-time refinement
recomputes only those three blocks. Every completed history is cached with its
inputs and source hashes. Two-mesh differences remain discrete comparisons.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
import gc
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from applications.glassy_polymer.calibration import replace_log_ratio_block
from applications.glassy_polymer.canonical import CANONICAL_REGULARIZATION, SENSITIVITY_RESOLUTION
from applications.glassy_polymer.constitutive import pmma_parameters
from applications.glassy_polymer.experiments import predict
from applications.glassy_polymer.objectives import path_integration_weights
from applications.glassy_polymer.scripts.run_joint_observability import (
    NAMES as CONDITIONAL_NAMES, record_balancing_factors, selected_experiments,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/v9_geometry_convergence"
V8 = ROOT / "results/v8_full_geometry_audit"
CORE_SOURCES = (
    "constitutive.py", "canonical.py", "experiments.py", "protocols.py",
    "objectives.py", "calibration.py", "scripts/run_joint_observability.py",
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def peak_rss_mb():
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(raw / (1024**2 if sys.platform == "darwin" else 1024))


def angle(matrix_a, matrix_b, dimension):
    a = np.linalg.svd(matrix_a, full_matrices=False)[2][-dimension:]
    b = np.linalg.svd(matrix_b, full_matrices=False)[2][-dimension:]
    cosines = np.linalg.svd(a @ b.T, compute_uv=False)
    return float(np.max(np.degrees(np.arccos(np.clip(cosines, -1.0, 1.0)))))


def geometry(matrix, indices, recovery_mask, gauge):
    singular = np.linalg.svd(matrix, compute_uv=False)
    joint = matrix[:, indices]
    singular_joint = np.linalg.svd(joint, compute_uv=False)
    return {
        "singular_values": singular.tolist(),
        "joint12_condition_number": float(singular_joint[0] / singular_joint[-1]),
        "gauge_derivative_relative_sigma1": float(np.linalg.norm(matrix @ gauge) / singular[0]),
        "recovery_excluded_joint12_weak3_angle_degrees": angle(joint, joint[~recovery_mask], 3),
        "recovery_excluded_joint12_algebraic_rank": int(np.linalg.matrix_rank(joint[~recovery_mask])),
    }


def compare(a, b, indices, label_a, label_b):
    singular_a = np.linalg.svd(a, compute_uv=False)
    singular_b = np.linalg.svd(b, compute_uv=False)
    delta = float(np.linalg.norm(a - b, ord=2))
    joint_a = np.linalg.svd(a[:, indices], compute_uv=False)
    joint_b = np.linalg.svd(b[:, indices], compute_uv=False)
    condition_a = joint_a[0] / joint_a[-1]
    condition_b = joint_b[0] / joint_b[-1]
    result = {
        "coarser_level": label_a, "finer_level": label_b,
        "absolute_spectral_difference": delta,
        "relative_spectral_difference": delta / singular_b[0],
        "relative_frobenius_difference": float(np.linalg.norm(a - b) / np.linalg.norm(b)),
        "full22_weak1_angle_degrees": angle(a, b, 1),
        "full22_weak2_angle_degrees": angle(a, b, 2),
        "full22_weak3_angle_degrees": angle(a, b, 3),
        "joint12_weak3_angle_degrees": angle(a[:, indices], b[:, indices], 3),
        "joint12_condition_number_relative_change": float(abs(condition_a - condition_b) / condition_b),
        "coarser_threshold_counts_half_one_two_delta": [int(sum(singular_a > factor * delta)) for factor in (.5, 1., 2.)],
        "finer_threshold_counts_half_one_two_delta": [int(sum(singular_b > factor * delta)) for factor in (.5, 1., 2.)],
    }
    result["passes_angular_gates"] = bool(result["full22_weak3_angle_degrees"] < 1.0
                                          and result["joint12_weak3_angle_degrees"] < 1.0)
    result["passes_declared_comparison"] = bool(result["passes_angular_gates"]
                                               and result["relative_frobenius_difference"] < .01
                                               and result["joint12_condition_number_relative_change"] < .05)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-new-levels", type=int, default=4)
    parser.add_argument("--wall-budget-seconds", type=float, default=1800.)
    parser.add_argument("--skip-creep-probe", action="store_true")
    parser.add_argument("--force-recompute", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.max_new_levels <= 4:
        raise ValueError("The maximum authorized strain/recovery refinement depth is four new levels")
    jax.config.update("jax_enable_x64", True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    source_hashes = {name: digest(ROOT / name) for name in CORE_SOURCES}
    source_hashes["scripts/audit_v9_geometry_convergence.py"] = digest(Path(__file__))
    v8_summary = json.loads((V8 / "full_geometry_summary.json").read_text())
    for name in CORE_SOURCES:
        if source_hashes[name] != v8_summary["source_hashes"][name]:
            raise RuntimeError(f"V8 numerical source changed: {name}")
    v8_compute_hash = v8_summary["source_hashes"]["scripts/audit_v8_full_geometry.py"]
    if v8_compute_hash not in (digest(ROOT / "scripts/audit_v8_full_geometry.py"),
                               digest(V8 / "COMPUTATION_SOURCE_V1.txt")):
        raise RuntimeError("V8 generating-script provenance cannot be verified")
    experiments = selected_experiments()
    factors = record_balancing_factors(experiments)
    names = list(pmma_parameters()._fields)
    base = pmma_parameters()
    indices = [names.index(name) for name in CONDITIONAL_NAMES]
    rows = json.loads((V8 / "row_manifest.json").read_text())
    baseline_resolution = replace(SENSITIVITY_RESOLUTION,
                                  pmma_max_true_strain_increment=1.25e-4,
                                  recovery_max_time_increment=1.5625e-2)
    if asdict(baseline_resolution) != v8_summary["meshes"]["ultrafine"]:
        raise RuntimeError("V8 finer mesh disagrees with the declared V9 baseline")
    with np.load(V8 / "matched_full_jacobians.npz") as archive:
        previous = archive["ultrafine"].copy()
        gauge = archive["analytic_gauge"].copy()
        if names != archive["parameter_names"].tolist():
            raise RuntimeError("V8 column order changed")
    recovery_mask = np.zeros(779, dtype=bool)
    cursor = 0
    weights_by_curve = {}
    for experiment, row in zip(experiments, rows):
        scale = (max(float(np.max(np.abs(experiment.y))), .02)
                 if experiment.family in {"creep", "recovery"} else 110.)
        if (row["curve_id"] != experiment.curve_id or row["row_start"] != cursor
                or row["row_end_exclusive"] != cursor + len(experiment.x)
                or row["observation_scale"] != scale
                or row["sqrt_family_record_observable_mass"] != factors[experiment.curve_id]):
            raise RuntimeError(f"V8 fixed-row geometry changed at {experiment.curve_id}")
        weights = factors[experiment.curve_id] * np.sqrt(path_integration_weights(experiment.x)) / scale
        weights_by_curve[experiment.curve_id] = weights
        context_v8 = json.dumps({
            "source_hashes": v8_summary["source_hashes"],
            "resolution": asdict(baseline_resolution), "names": names,
            "row": row, "x": experiment.x.tolist(), "y": experiment.y.tolist(),
            "weights": weights.tolist(),
        }, sort_keys=True)
        with np.load(V8 / f"ultrafine_{experiment.curve_id}.npz") as archive:
            signature = hashlib.sha256(context_v8.encode()).hexdigest()
            if str(archive["signature"].item()) != signature:
                raise RuntimeError(f"V8 cached source/input signature mismatch: {experiment.curve_id}")
            if not np.array_equal(archive["jacobian"], previous[cursor:cursor + len(weights)]):
                raise RuntimeError("V8 concatenated matrix differs from its per-history cache")
        if experiment.family == "recovery":
            recovery_mask[cursor:cursor + len(weights)] = True
        cursor += len(weights)
    if cursor != 779 or len(rows) != 10 or previous.shape != (779, 22):
        raise RuntimeError("Unexpected mother-matrix shape")
    summary = {
        "source_hashes": source_hashes,
        "v8_baseline_array_sha256": digest(V8 / "matched_full_jacobians.npz"),
        "v8_per_history_signatures_verified": True,
        "shape": [779, 22], "parameter_names": names,
        "row_manifest": rows,
        "declared_gates": {"full22_weak3_angle_degrees": 1.,
                           "joint12_weak3_angle_degrees": 1.,
                           "relative_frobenius_difference": .01,
                           "joint12_condition_number_relative_change": .05},
        "interpretation": "Two-mesh qualification of the fixed log-parameter response geometry; no continuum error bound or posterior uncertainty is inferred.",
        "baseline_level": "v8_ultrafine",
        "baseline_resolution": asdict(baseline_resolution),
        "baseline_geometry": geometry(previous, indices, recovery_mask, gauge),
        "levels": [], "comparisons": [], "history_timings": [],
        "status": "running", "creep_probe": None,
    }
    output_summary = OUTPUT / "geometry_convergence_summary.json"
    write_json(output_summary, summary)
    previous_label = "v8_ultrafine"
    passed_level = None
    previous_new_level_seconds = None
    resolution = baseline_resolution

    def calculate(label, local_resolution, inherited, recompute_creep_only=False):
        nonlocal summary
        blocks = []
        level_started = time.monotonic()
        for experiment, row in zip(experiments, rows):
            curve = experiment.curve_id
            selected = ((experiment.family == "creep") if recompute_creep_only
                        else (experiment.family != "creep"))
            history_started = time.monotonic()
            if not selected:
                block = inherited[row["row_start"]:row["row_end_exclusive"]].copy()
                mode = "reuse_exactly_unchanged_protocol_mesh"
            else:
                weights = weights_by_curve[curve]
                context = json.dumps({
                    "source_hashes": source_hashes, "resolution": asdict(local_resolution),
                    "row": row, "names": names, "x": experiment.x.tolist(),
                    "y": experiment.y.tolist(), "weights": weights.tolist(),
                    "experiment": repr(experiment),
                }, sort_keys=True)
                signature = hashlib.sha256(context.encode()).hexdigest()
                path = OUTPUT / f"{label}_{curve}.npz"
                if path.exists() and not args.force_recompute:
                    with np.load(path) as archive:
                        if str(archive["signature"].item()) != signature:
                            raise RuntimeError(f"Source/input change prevents reuse of {path}")
                        block = archive["jacobian"].copy()
                    mode = "verified_cache"
                else:
                    def observations(values):
                        params = replace_log_ratio_block(base, names, values)
                        return jnp.asarray(weights) * predict(
                            experiment, params, CANONICAL_REGULARIZATION,
                            checkpoint=True, resolution=local_resolution)
                    print(f"compute {label} {curve} peakRSS={peak_rss_mb():.1f}MB", flush=True)
                    block = np.asarray(jax.jit(jax.jacfwd(observations))(
                        jnp.zeros(22, dtype=jnp.float64)))
                    if block.shape != (len(weights), 22) or not np.isfinite(block).all():
                        raise FloatingPointError(f"Invalid Jacobian {label} {curve}")
                    temporary = path.with_name(path.stem + ".tmp.npz")
                    np.savez(temporary, jacobian=block, signature=signature,
                             computation_seconds=time.monotonic() - history_started)
                    temporary.replace(path)
                    mode = "computed"
                    jax.clear_caches()
                    gc.collect()
            seconds = time.monotonic() - history_started
            summary["history_timings"].append({
                "level": label, "curve_id": curve, "mode": mode,
                "elapsed_seconds": seconds, "peak_rss_mb": peak_rss_mb(),
            })
            summary["elapsed_seconds"] = time.monotonic() - started
            write_json(output_summary, summary)
            print(f"done {label} {curve} {mode} {seconds:.1f}s", flush=True)
            blocks.append(block)
        matrix = np.concatenate(blocks)
        np.savez(OUTPUT / f"{label}_full_jacobian.npz", jacobian=matrix,
                 parameter_names=np.asarray(names), analytic_gauge=gauge,
                 resolution=json.dumps(asdict(local_resolution), sort_keys=True))
        item = {"label": label, "resolution": asdict(local_resolution),
                "geometry": geometry(matrix, indices, recovery_mask, gauge),
                "elapsed_seconds": time.monotonic() - level_started,
                "peak_rss_mb": peak_rss_mb()}
        summary["levels"].append(item)
        write_json(output_summary, summary)
        return matrix, item["elapsed_seconds"]

    for depth in range(1, args.max_new_levels + 1):
        elapsed = time.monotonic() - started
        projected_next = (1.8 * previous_new_level_seconds
                          if previous_new_level_seconds is not None else 0.)
        if depth > 1 and elapsed + projected_next > args.wall_budget_seconds:
            summary["status"] = "paused_before_projected_wall_budget"
            summary["projected_next_level_seconds"] = projected_next
            print(f"PAUSED: elapsed={elapsed:.1f}s next estimate={projected_next:.1f}s "
                  f"budget={args.wall_budget_seconds:.1f}s", flush=True)
            break
        resolution = replace(baseline_resolution,
                             pmma_max_true_strain_increment=1.25e-4 / 2**depth,
                             recovery_max_time_increment=1.5625e-2 / 2**depth)
        label = f"strain_recovery_L{depth}"
        current, level_seconds = calculate(label, resolution, previous)
        comparison = compare(previous, current, indices, previous_label, label)
        summary["comparisons"].append(comparison)
        print(json.dumps(comparison, indent=2), flush=True)
        write_json(output_summary, summary)
        previous_new_level_seconds = level_seconds
        previous, previous_label = current, label
        if comparison["passes_declared_comparison"]:
            passed_level = label
            summary["status"] = "strain_recovery_gates_passed"
            break
    else:
        summary["status"] = "maximum_authorized_strain_recovery_depth_reached"

    summary["strain_recovery_passed_level"] = passed_level
    if passed_level is not None and not args.skip_creep_probe:
        creep_resolution = replace(resolution,
                                   pmma_max_time_increment=resolution.pmma_max_time_increment / 2.,
                                   pmma_severe_creep_max_time_increment=resolution.pmma_severe_creep_max_time_increment / 2.)
        creep_label = previous_label + "_creep_half"
        creep_matrix, _ = calculate(creep_label, creep_resolution, previous,
                                    recompute_creep_only=True)
        creep_comparison = compare(previous, creep_matrix, indices, previous_label, creep_label)
        summary["creep_probe"] = creep_comparison
        summary["status"] = ("strain_recovery_and_creep_probe_passed"
                             if creep_comparison["passes_declared_comparison"]
                             else "strain_recovery_passed_creep_probe_failed")
        print("independent_creep_probe", json.dumps(creep_comparison, indent=2), flush=True)
    summary["elapsed_seconds"] = time.monotonic() - started
    summary["peak_rss_mb"] = peak_rss_mb()
    write_json(output_summary, summary)
    with (OUTPUT / "history_timings.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary["history_timings"][0])
        writer.writeheader()
        writer.writerows(summary["history_timings"])
    print("FINAL", summary["status"], f"elapsed={summary['elapsed_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
