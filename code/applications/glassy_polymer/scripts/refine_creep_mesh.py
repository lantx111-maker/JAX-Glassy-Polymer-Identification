"""Refine the independent creep time mesh and compare parameter geometry.

The observation rows and non-creep discretization are fixed. Independent
creep refinement precedes the coupled time-mesh comparison."""

from __future__ import annotations

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
from applications.glassy_polymer.canonical import CANONICAL_REGULARIZATION, SENSITIVITY_RESOLUTION
from applications.glassy_polymer.constitutive import pmma_parameters
from applications.glassy_polymer.experiments import predict
from applications.glassy_polymer.objectives import path_integration_weights
from applications.glassy_polymer.scripts.refine_geometry_mesh import (
    CORE_SOURCES, angle, compare, digest, geometry, peak_rss_mb, write_json,
)
from applications.glassy_polymer.scripts.run_joint_observability import (
    NAMES as CONDITIONAL_NAMES, record_balancing_factors, selected_experiments,
)


ROOT = Path(__file__).resolve().parents[1]
REFINED = ROOT / "results/geometry_refinement"
OUTPUT = ROOT / "results/creep_refinement"


def read_matrix(path):
    with np.load(path) as archive:
        return archive["jacobian"].copy(), archive["parameter_names"].tolist(), json.loads(str(archive["resolution"].item()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-new-levels", type=int, default=3)
    parser.add_argument("--force-recompute", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.max_new_levels <= 3:
        raise ValueError("At most three additional creep refinements are authorized")
    jax.config.update("jax_enable_x64", True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    refinement_summary = json.loads((REFINED / "geometry_convergence_summary.json").read_text())
    sources = {name: digest(ROOT / name) for name in CORE_SOURCES}
    sources["scripts/refine_geometry_mesh.py"] = digest(ROOT / "scripts/refine_geometry_mesh.py")
    for name, value in sources.items():
        if refinement_summary["source_hashes"][name] != value:
            raise RuntimeError(f"REFINED numerical source changed: {name}")
    sources["scripts/refine_creep_mesh.py"] = digest(Path(__file__))
    baseline_file = REFINED / "strain_recovery_L4_creep_half_full_jacobian.npz"
    l3_file = REFINED / "strain_recovery_L3_full_jacobian.npz"
    l4_file = REFINED / "strain_recovery_L4_full_jacobian.npz"
    previous, names, baseline_settings = read_matrix(baseline_file)
    l3, l3_names, l3_settings = read_matrix(l3_file)
    l4, l4_names, l4_settings = read_matrix(l4_file)
    base = pmma_parameters()
    if names != list(base._fields) or names != l3_names or names != l4_names:
        raise RuntimeError("REFINED parameter column order changed")
    rows = refinement_summary["row_manifest"]
    experiments = selected_experiments()
    factors = record_balancing_factors(experiments)
    indices = [names.index(name) for name in CONDITIONAL_NAMES]
    recovery_mask = np.zeros(779, dtype=bool)
    creep_mask = np.zeros(779, dtype=bool)
    for experiment, row in zip(experiments, rows):
        if row["curve_id"] != experiment.curve_id or row["row_end_exclusive"] - row["row_start"] != len(experiment.x):
            raise RuntimeError("REFINED observation rows changed")
        scale = (max(float(np.max(np.abs(experiment.y))), .02)
                 if experiment.family in {"creep", "recovery"} else 110.)
        if scale != row["observation_scale"] or factors[experiment.curve_id] != row["sqrt_family_record_observable_mass"]:
            raise RuntimeError("REFINED observation weights changed")
        selected = slice(row["row_start"], row["row_end_exclusive"])
        if experiment.family == "recovery":
            recovery_mask[selected] = True
        if experiment.family == "creep":
            creep_mask[selected] = True
    if previous.shape != (779, 22) or not np.array_equal(previous[~creep_mask], l4[~creep_mask]):
        raise RuntimeError("REFINED creep-probe matrix did not preserve the L4 non-creep blocks")
    with np.load(baseline_file) as archive:
        gauge = archive["analytic_gauge"].copy()
    summary = {
        "source_hashes": sources,
        "source_array_sha256": {
            "results/geometry_refinement/" + path.name: digest(path)
            for path in (baseline_file, l3_file, l4_file)
        },
        "shape": [779, 22], "parameter_names": names, "row_manifest": rows,
        "declared_gates": refinement_summary["declared_gates"],
        "baseline_resolution": baseline_settings,
        "baseline_label": "refined_L4_creep_half",
        "inherited_creep_probe_failure": refinement_summary["creep_probe"],
        "status": "running", "levels": [], "history_timings": [],
        "interpretation": "Pure creep, same-creep strain/recovery, and coupled comparisons are separate; a pass in one is not substituted for another.",
    }
    summary_file = OUTPUT / "creep_continuation_summary.json"
    write_json(summary_file, summary)
    previous_label = "refined_L4_creep_half"
    previous_creep_sources = {row["curve_id"]: "results/geometry_refinement/strain_recovery_L4_creep_half_full_jacobian.npz"
                              for row in rows if row["family"] == "creep"}

    def export(label, matrix, settings, source_by_curve):
        path = OUTPUT / f"{label}.npz"
        np.savez(path, jacobian=matrix, parameter_names=np.asarray(names),
                 analytic_gauge=gauge, resolution=json.dumps(settings, sort_keys=True),
                 source_block_level_by_curve=json.dumps(source_by_curve, sort_keys=True))
        return {"path": "results/creep_refinement/" + path.name,
                "sha256": digest(path), "shape": [779, 22],
                "parameter_names": names, "resolution": settings,
                "source_block_level_by_curve": source_by_curve}

    for depth in range(1, args.max_new_levels + 1):
        label = f"creep_C{depth}"
        settings = dict(baseline_settings)
        settings["pmma_max_time_increment"] /= 2**depth
        settings["pmma_severe_creep_max_time_increment"] /= 2**depth
        resolution = replace(SENSITIVITY_RESOLUTION, **settings)
        current = previous.copy()
        current_sources = {}
        for experiment, row in zip(experiments, rows):
            curve = experiment.curve_id
            if experiment.family != "creep":
                current_sources[curve] = "results/geometry_refinement/strain_recovery_L4_full_jacobian.npz"
                continue
            weights = factors[curve] * np.sqrt(path_integration_weights(experiment.x)) / row["observation_scale"]
            context = json.dumps({"source_hashes": sources, "resolution": settings,
                                  "row": row, "names": names, "x": experiment.x.tolist(),
                                  "y": experiment.y.tolist(), "weights": weights.tolist(),
                                  "experiment": repr(experiment)}, sort_keys=True)
            signature = hashlib.sha256(context.encode()).hexdigest()
            cache = OUTPUT / f"{label}_{curve}.npz"
            history_started = time.monotonic()
            if cache.exists() and not args.force_recompute:
                with np.load(cache) as archive:
                    if str(archive["signature"].item()) != signature:
                        raise RuntimeError(f"Incompatible cached inputs/sources: {cache}")
                    block = archive["jacobian"].copy()
                mode = "verified_cache"
            else:
                def observations(values):
                    params = replace_log_ratio_block(base, names, values)
                    return jnp.asarray(weights) * predict(experiment, params, CANONICAL_REGULARIZATION,
                                                          checkpoint=True, resolution=resolution)
                print(f"compute {label} {curve} creep_dt={resolution.creep_time_increment('PMMA', experiment.target)}", flush=True)
                block = np.asarray(jax.jit(jax.jacfwd(observations))(jnp.zeros(22, dtype=jnp.float64)))
                if block.shape != (len(weights), 22) or not np.isfinite(block).all():
                    raise FloatingPointError(f"Invalid matrix {label} {curve}")
                temporary = cache.with_name(cache.stem + ".tmp.npz")
                np.savez(temporary, jacobian=block, signature=signature,
                         computation_seconds=time.monotonic() - history_started)
                temporary.replace(cache)
                jax.clear_caches()
                gc.collect()
                mode = "computed"
            current[row["row_start"]:row["row_end_exclusive"]] = block
            current_sources[curve] = "results/creep_refinement/" + cache.name
            seconds = time.monotonic() - history_started
            summary["history_timings"].append({"level": label, "curve_id": curve,
                                                "elapsed_seconds": seconds, "mode": mode,
                                                "peak_rss_mb": peak_rss_mb()})
            write_json(summary_file, summary)
            print(f"done {label} {curve} {seconds:.1f}s peakRSS={peak_rss_mb():.1f}MB", flush=True)
        # Strain/recovery comparison at exactly the same new creep blocks.
        same_creep_l3 = l3.copy()
        same_creep_l3[creep_mask] = current[creep_mask]
        same_settings = dict(l3_settings)
        same_settings["pmma_max_time_increment"] = settings["pmma_max_time_increment"]
        same_settings["pmma_severe_creep_max_time_increment"] = settings["pmma_severe_creep_max_time_increment"]
        same_sources = {row["curve_id"]: (current_sources[row["curve_id"]] if row["family"] == "creep"
                                          else "results/geometry_refinement/strain_recovery_L3_full_jacobian.npz")
                        for row in rows}
        # Both mesh axes halve between this composite coarse and current fine.
        coupled_coarse = l3.copy()
        coupled_coarse[creep_mask] = previous[creep_mask]
        coupled_settings = dict(same_settings)
        coupled_settings["pmma_max_time_increment"] *= 2.
        coupled_settings["pmma_severe_creep_max_time_increment"] *= 2.
        coupled_sources = {row["curve_id"]: (previous_creep_sources[row["curve_id"]] if row["family"] == "creep"
                                             else "results/geometry_refinement/strain_recovery_L3_full_jacobian.npz")
                           for row in rows}
        exported = {
            "fine": export(label + "_L4_full", current, settings, current_sources),
            "same_creep_L3": export(label + "_L3_same_creep", same_creep_l3, same_settings, same_sources),
            "coupled_coarse": export(label + "_L3_previous_creep", coupled_coarse, coupled_settings, coupled_sources),
        }
        comparisons = {
            "independent_creep": compare(previous, current, indices, previous_label, label + "_L4_full"),
            "same_creep_strain_recovery": compare(same_creep_l3, current, indices, label + "_L3_same_creep", label + "_L4_full"),
            "coupled_all_axes": compare(coupled_coarse, current, indices, label + "_L3_previous_creep", label + "_L4_full"),
        }
        item = {"label": label, "matrices": exported, "comparisons": comparisons,
                "geometry": geometry(current, indices, recovery_mask, gauge)}
        summary["levels"].append(item)
        summary["elapsed_seconds"] = time.monotonic() - started
        write_json(summary_file, summary)
        print(json.dumps(comparisons, indent=2), flush=True)
        previous, previous_label = current, label + "_L4_full"
        previous_creep_sources = {row["curve_id"]: current_sources[row["curve_id"]]
                                  for row in rows if row["family"] == "creep"}
        if comparisons["independent_creep"]["passes_declared_comparison"]:
            summary["status"] = ("all_three_comparisons_passed" if all(
                value["passes_declared_comparison"] for value in comparisons.values())
                else "creep_passed_but_other_comparison_failed")
            break
    else:
        summary["status"] = "maximum_additional_creep_depth_reached_without_creep_pass"
    summary["elapsed_seconds"] = time.monotonic() - started
    summary["peak_rss_mb"] = peak_rss_mb()
    write_json(summary_file, summary)
    print("FINAL", summary["status"], f"elapsed={summary['elapsed_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
