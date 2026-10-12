"""Compare full-parameter sensitivity matrices on matched observation rows.

Physical integration meshes vary; observation coordinates and weighting
remain fixed so that the comparison measures integration sensitivity."""

from __future__ import annotations

import csv
import argparse
from dataclasses import asdict, replace
import gc
import hashlib
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from applications.glassy_polymer.calibration import replace_log_ratio_block
from applications.glassy_polymer.canonical import (
    CANONICAL_REGULARIZATION,
    SENSITIVITY_RESOLUTION,
)
from applications.glassy_polymer.constitutive import pmma_parameters
from applications.glassy_polymer.experiments import predict
from applications.glassy_polymer.objectives import path_integration_weights
from applications.glassy_polymer.scripts.run_joint_observability import (
    NAMES as CONDITIONAL_NAMES,
    record_balancing_factors,
    selected_experiments,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "initial_geometry_meshes"
OLD_MATRIX = ROOT / "results/parameter_screening/common_22_parameter_jacobian_ad.npy"


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _angle(matrix_a, matrix_b, dimension):
    va = np.linalg.svd(matrix_a, full_matrices=False)[2][-dimension:]
    vb = np.linalg.svd(matrix_b, full_matrices=False)[2][-dimension:]
    cosines = np.linalg.svd(va @ vb.T, compute_uv=False)
    return float(np.max(np.degrees(np.arccos(np.clip(cosines, -1.0, 1.0)))))


def _mechanism_audit(matrices, names, row_manifest, summary):
    """Reuse the matched matrices; the archived recovery factor stays frozen."""
    source = ROOT / "results/conditional_geometry/common_geometry_spectra.csv"
    with source.open(newline="") as handle:
        archived = next(csv.DictReader(handle))
    factor = float(archived["recovery_discrepancy_factor"])
    indices = [names.index(name) for name in CONDITIONAL_NAMES]
    recovery = np.zeros(matrices["sensitivity"].shape[0], dtype=bool)
    for row in row_manifest:
        if row["family"] == "recovery":
            recovery[row["row_start"]:row["row_end_exclusive"]] = True
    mechanism = {
        "frozen_recovery_factor": factor,
        "recovery_factor_provenance": "results/conditional_geometry/common_geometry_spectra.csv",
        "recovery_factor_interpretation": "The same archived 0.8418 factor is applied on both meshes; the recovery RMSE is not re-estimated on the fine mesh.",
        "meshes": {},
    }
    for label, matrix in matrices.items():
        joint = matrix[:, indices]
        weighted = joint * np.where(recovery, factor, 1.0)[:, None]
        excluded = joint[~recovery]
        blocks = {"joint12": joint, "flow6": joint[:, :6], "retained6": joint[:, 6:]}
        spectral = {}
        for key, block in blocks.items():
            spectrum = np.linalg.svd(block, compute_uv=False)
            spectral[key] = {
                "sigma_max": float(spectrum[0]), "sigma_min": float(spectrum[-1]),
                "condition_number": float(spectrum[0] / spectrum[-1]),
            }
        _, singular, modes = np.linalg.svd(matrix, full_matrices=False)
        weak_modes = {}
        for index in (19, 20, 21):
            vector = modes[index]
            sign_index = int(np.argmax(abs(vector)))
            vector = vector * (1 if vector[sign_index] >= 0 else -1)
            weak_modes[str(index + 1)] = {
                "singular_value": float(singular[index]),
                "coefficients": dict(zip(names, vector.tolist())),
                "dominant_coefficients_abs_ge_0_05": {
                    name: float(value) for name, value in zip(names, vector)
                    if abs(value) >= 0.05
                },
            }
        excluded_s = np.linalg.svd(excluded, compute_uv=False)
        mechanism["meshes"][label] = {
            "block_spectra": spectral,
            "joint12_weak3_angle_recovery_factor_degrees": _angle(joint, weighted, 3),
            "joint12_weak3_angle_recovery_excluded_degrees": _angle(joint, excluded, 3),
            "joint12_algebraic_rank_without_recovery": int(np.linalg.matrix_rank(excluded)),
            "joint12_sigma_min_without_recovery": float(excluded_s[-1]),
            "recovery_rows_relative_rowspan_projection_residual": float(
                np.linalg.norm(joint[recovery] - joint[recovery] @ np.linalg.pinv(excluded) @ excluded)
                / np.linalg.norm(joint[recovery])
            ),
            "weak_full22_modes": weak_modes,
        }
    mechanism["cross_mesh_weak_subspace_angles_degrees"] = {
        str(dimension): _angle(matrices["sensitivity"], matrices["ultrafine"], dimension)
        for dimension in (1, 2, 3)
    }
    summary["mechanism_complementarity"] = mechanism
    summary["postprocessing_source_sha256"] = _digest(Path(__file__))
    (OUTPUT / "mechanism_complementarity.json").write_text(json.dumps(mechanism, indent=2) + "\n")
    (OUTPUT / "full_geometry_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(mechanism, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--postprocess-only", action="store_true",
                        help="Analyze existing matched matrices without rerunning histories.")
    parser.add_argument("--reuse-cache", action="store_true",
                        help="Reuse per-history arrays only if all input/source signatures match.")
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.postprocess_only:
        with np.load(OUTPUT / "matched_full_jacobians.npz") as stored:
            matrices = {key: stored[key].copy() for key in ("sensitivity", "ultrafine")}
            names = stored["parameter_names"].tolist()
        summary = json.loads((OUTPUT / "full_geometry_summary.json").read_text())
        row_manifest = json.loads((OUTPUT / "row_manifest.json").read_text())
        _mechanism_audit(matrices, names, row_manifest, summary)
        return
    base = pmma_parameters()
    names = list(base._fields)
    experiments = selected_experiments()
    factors = record_balancing_factors(experiments)
    levels = {
        "sensitivity": SENSITIVITY_RESOLUTION,
        "ultrafine": replace(
            SENSITIVITY_RESOLUTION,
            pmma_max_true_strain_increment=1.25e-4,
            recovery_max_time_increment=1.5625e-2,
        ),
    }
    source_hashes = {
        name: _digest(ROOT / name) for name in (
            "constitutive.py", "canonical.py", "experiments.py", "protocols.py",
            "objectives.py", "calibration.py", "scripts/run_joint_observability.py",
            "scripts/compare_initial_geometry_meshes.py",
        )
    }
    matrices = {}
    row_manifest = []
    cursor = 0
    started = time.monotonic()
    for experiment in experiments:
        scale = (max(float(np.max(np.abs(experiment.y))), 0.02)
                 if experiment.family in {"creep", "recovery"} else 110.0)
        weights = (factors[experiment.curve_id]
                   * np.sqrt(path_integration_weights(experiment.x)) / scale)
        row_manifest.append({
            "curve_id": experiment.curve_id,
            "family": experiment.family,
            "row_start": cursor,
            "row_end_exclusive": cursor + len(weights),
            "observation_scale": scale,
            "sqrt_family_record_observable_mass": factors[experiment.curve_id],
        })
        cursor += len(weights)
    if cursor != 779 or len(names) != 22:
        raise RuntimeError(f"Unexpected mother geometry: {cursor} rows, {len(names)} columns")

    for label, resolution in levels.items():
        blocks = []
        for experiment, row in zip(experiments, row_manifest):
            weights = (factors[experiment.curve_id]
                       * np.sqrt(path_integration_weights(experiment.x))
                       / row["observation_scale"])
            context = json.dumps({
                "source_hashes": source_hashes,
                "resolution": asdict(resolution), "names": names,
                "row": row, "x": experiment.x.tolist(),
                "y": experiment.y.tolist(), "weights": weights.tolist(),
            }, sort_keys=True)
            signature = hashlib.sha256(context.encode()).hexdigest()
            cache = OUTPUT / f"{label}_{experiment.curve_id}.npz"
            if args.reuse_cache and cache.exists():
                with np.load(cache) as stored:
                    if str(stored["signature"].item()) != signature:
                        raise RuntimeError(f"Stale cache: {cache}; rerun without --reuse-cache to recompute")
                    block = stored["jacobian"].copy()
                print(f"cached {label} {experiment.curve_id} {block.shape}", flush=True)
            else:
                def observations(values):
                    params = replace_log_ratio_block(base, names, values)
                    response = predict(
                        experiment, params, CANONICAL_REGULARIZATION,
                        checkpoint=True, resolution=resolution,
                    )
                    return jnp.asarray(weights) * response

                print(f"computing {label} {experiment.curve_id}", flush=True)
                block = np.asarray(jax.jit(jax.jacfwd(observations))(
                    jnp.zeros(len(names), dtype=jnp.float64)
                ))
                if not np.all(np.isfinite(block)):
                    raise FloatingPointError(f"Non-finite {label} {experiment.curve_id}")
                np.savez(cache, jacobian=block, signature=signature)
                print(f"completed {label} {experiment.curve_id} {block.shape}; "
                      f"elapsed {time.monotonic() - started:.1f} s", flush=True)
                jax.clear_caches()
                gc.collect()
            blocks.append(block)
        matrices[label] = np.concatenate(blocks, axis=0)

    coarse, fine = matrices["sensitivity"], matrices["ultrafine"]
    archived = np.load(OLD_MATRIX)
    delta = coarse - fine
    delta2 = float(np.linalg.norm(delta, ord=2))
    delta_frobenius = float(np.linalg.norm(delta))
    _, singular, vectors = np.linalg.svd(coarse, full_matrices=False)
    fine_singular = np.linalg.svd(fine, compute_uv=False)
    old_relative_difference = float(np.linalg.norm(coarse - archived) / np.linalg.norm(archived))
    if old_relative_difference > 1.0e-10:
        raise RuntimeError(f"Recomputed baseline differs from archived mother: {old_relative_difference}")
    conditional = [names.index(name) for name in CONDITIONAL_NAMES]
    gauge = np.zeros(len(names))
    gauge[names.index("gamma0")] = 1.0
    for name in ("s0", "kappa1_sat", "kappa2_sat"):
        gauge[names.index(name)] = float(base.m0)
    for name in ("s1", "s2"):
        gauge[names.index(name)] = float(base.m_ve)
    gauge /= np.linalg.norm(gauge)
    rank_rows = []
    for multiplier in (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 4.0):
        threshold = multiplier * delta2
        rank_rows.append({
            "delta2_multiplier": multiplier,
            "absolute_threshold": threshold,
            "threshold_relative_coarse_sigma1": threshold / singular[0],
            "coarse_directions_above_threshold": int(np.sum(singular > threshold)),
            "fine_directions_above_threshold": int(np.sum(fine_singular > threshold)),
        })
    with (OUTPUT / "threshold_rank_robustness.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rank_rows[0])
        writer.writeheader()
        writer.writerows(rank_rows)
    with (OUTPUT / "full_spectra.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("mode", "sensitivity_sigma", "ultrafine_sigma",
                         "absolute_change", "change_within_two_mesh_spectral_bound"))
        for i, (a, b) in enumerate(zip(singular, fine_singular), start=1):
            writer.writerow((i, a, b, abs(a - b), bool(abs(a - b) <= delta2 * (1 + 1e-12))))
    with (OUTPUT / "weak_mode_coefficients.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("mode", "singular_value", *names))
        for index in range(17, 22):
            writer.writerow((index + 1, singular[index], *vectors[index]))
    summary = {
        "description": "Matched 779-row, 22-column weighted discrete Jacobian mesh audit",
        "source_hashes": source_hashes,
        "archived_baseline_sha256": _digest(OLD_MATRIX),
        "baseline_relative_frobenius_difference_from_archive": old_relative_difference,
        "shape": list(coarse.shape), "parameter_names": names,
        "meshes": {key: asdict(value) for key, value in levels.items()},
        "refined_mesh_axes": ["pmma_max_true_strain_increment", "recovery_max_time_increment"],
        "unrefined_mesh_axes_note": "Creep time-step maxima and graded-start policies are held fixed.",
        "two_mesh_absolute_spectral_difference": delta2,
        "two_mesh_absolute_frobenius_difference": delta_frobenius,
        "two_mesh_spectral_difference_relative_coarse_sigma1": delta2 / singular[0],
        "two_mesh_spectral_difference_relative_fine_sigma1": delta2 / fine_singular[0],
        "two_mesh_relative_frobenius_difference": delta_frobenius / float(np.linalg.norm(fine)),
        "coarse_sigma1": float(singular[0]), "fine_sigma1": float(fine_singular[0]),
        "weyl_all_discrete_singular_differences_within_delta2": bool(np.max(abs(singular-fine_singular)) <= delta2 * (1 + 1e-12)),
        "interpretation": "Delta2 bounds singular-value differences between these two discrete matrices only; it does not bound the continuum error, prove all time meshes converged, or quantify experimental uncertainty.",
        "coarse_count_above_delta2": int(np.sum(singular > delta2)),
        "fine_count_above_delta2": int(np.sum(fine_singular > delta2)),
        "joint12_weak3_angle_degrees": _angle(coarse[:, conditional], fine[:, conditional], 3),
        "full22_weak3_angle_degrees": _angle(coarse, fine, 3),
        "analytic_scaling_gauge_unit_vector": dict(zip(names, gauge.tolist())),
        "coarse_gauge_derivative_relative_sigma1": float(np.linalg.norm(coarse @ gauge) / singular[0]),
        "fine_gauge_derivative_relative_sigma1": float(np.linalg.norm(fine @ gauge) / fine_singular[0]),
        "absolute_cosine_weakest_mode_with_analytic_gauge": float(abs(vectors[-1] @ gauge)),
        "elapsed_seconds": time.monotonic() - started,
    }
    np.savez(OUTPUT / "matched_full_jacobians.npz", sensitivity=coarse, ultrafine=fine,
             delta=delta, parameter_names=np.asarray(names), analytic_gauge=gauge)
    (OUTPUT / "row_manifest.json").write_text(json.dumps(row_manifest, indent=2) + "\n")
    (OUTPUT / "full_geometry_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    _mechanism_audit(matrices, names, row_manifest, summary)


if __name__ == "__main__":
    main()
