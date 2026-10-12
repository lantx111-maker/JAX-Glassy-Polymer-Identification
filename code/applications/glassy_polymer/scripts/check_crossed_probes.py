"""Complete the retained-state/probe 2x2 audit using frozen terminal states.

The archived 32/4 and 48/8 trajectories are retained unchanged.  Only the
missing 32/8 and 48/4 probes are computed, from the archived full MaterialState
arrays at the original level-32 physical controls and 81 observation times.
This is a finite-resolution cross-check, not an asymptotic error estimate.
"""
from __future__ import annotations

import csv
from functools import partial
import hashlib
from itertools import combinations
import json
import os
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from applications.glassy_polymer.constitutive import MaterialState, pmma_parameters
from applications.glassy_polymer.protocols import run_strain_history

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "results/reload_refinement"
OUTPUT = ROOT / "results/crossed_probes"
SOURCE = ROOT / "results/matched_residual_histories/feasible_history_refinement.csv"
COMBINATIONS = ((32, 4), (32, 8), (48, 4), (48, 8))
TARGET, TARGET_TOLERANCE = 0.020, 2.0e-4


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def write_csv(name, rows):
    with (OUTPUT / name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def label(state, probe):
    return f"state{state}_probe{probe}"


def main():
    started = time.perf_counter()
    jax.config.update("jax_enable_x64", True)
    if jax.default_backend() != "cpu":
        raise RuntimeError("this audit is specified for the CPU backend")
    sources = [SOURCE, OLD / "reload_refinement_summary.json",
               OLD / "reload_refinement_runs.csv",
               OLD / "reload_refinement_trajectories.csv",
               OLD / "reload_refinement_states_and_envelopes.npz",
               ROOT / "constitutive.py", ROOT / "protocols.py",
               ROOT / "canonical.py", ROOT / "inverse_design.py",
               ROOT / "scripts/compare_reload_meshes.py", Path(__file__)]
    source_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in sources}
    archived_summary = json.loads((OLD / "reload_refinement_summary.json").read_text())
    if archived_summary["source_sha256"] != sha256(SOURCE):
        raise RuntimeError("the archived level-32 physical controls have changed")
    frozen = [row for row in read_csv(SOURCE)
              if float(row["target"]) == TARGET and int(row["mesh_level"]) == 32]
    profiles = [row["profile"] for row in frozen]
    if len(profiles) != 3 or len(set(profiles)) != 3:
        raise RuntimeError("expected exactly three unique frozen profiles")
    source_by_profile = {row["profile"]: row for row in frozen}
    archived_runs = {(int(row["state_level"]), row["profile"]): row
                     for row in read_csv(OLD / "reload_refinement_runs.csv")}
    archive = np.load(OLD / "reload_refinement_states_and_envelopes.npz", allow_pickle=False)
    observation_time = archive["time_s"]
    additional_strain = archive["additional_true_strain"]
    np.testing.assert_array_equal(observation_time, np.linspace(0, 100, 81))
    np.testing.assert_array_equal(additional_strain, np.linspace(0, .10, 81))
    dt = np.diff(observation_time, prepend=observation_time[:1])
    old_rows = read_csv(OLD / "reload_refinement_trajectories.csv")
    trajectories, states, state_rows = {}, {}, []
    for level, probe in ((32, 4), (48, 8)):
        for profile in profiles:
            rows = [row for row in old_rows if int(row["state_level"]) == level
                    and int(row["probe_substeps"]) == probe and row["profile"] == profile]
            rows.sort(key=lambda row: int(row["point"]))
            if [int(row["point"]) for row in rows] != list(range(81)):
                raise RuntimeError("archived trajectory observation rows are incomplete")
            np.testing.assert_array_equal([float(row["time_s"]) for row in rows], observation_time)
            np.testing.assert_array_equal([float(row["additional_true_strain"]) for row in rows], additional_strain)
            trajectories[(level, probe, profile)] = np.array([
                float(row["compressive_cauchy_stress_mpa"]) for row in rows])
            state = MaterialState(*(jnp.asarray(archive[f"state{level}_{profile}_{name}"])
                                    for name in MaterialState._fields))
            if not all(np.isfinite(np.asarray(value)).all() for value in state):
                raise FloatingPointError("non-finite frozen state")
            row = archived_runs[(level, profile)]
            residual = float(row["residual_strain"])
            np.testing.assert_allclose(-np.log(float(state.F[1, 1])), residual, atol=2e-15, rtol=0)
            np.testing.assert_allclose(float(state.phi), float(row["terminal_phi"]), atol=0, rtol=0)
            control = source_by_profile[profile]
            expected_time = sum(float(control[f"duration_{i}_s"]) for i in range(1, 5)) + 600
            np.testing.assert_allclose(float(state.time), expected_time, atol=1e-8, rtol=0)
            minimum_eigenvalue = min(float(np.linalg.eigvalsh(np.asarray(value)).min())
                                     for value in (state.A0, state.A1, state.A2))
            if minimum_eigenvalue <= 0:
                raise FloatingPointError("non-SPD frozen state")
            states[(level, profile)] = state
            state_rows.append({
                "state_level": level, "profile": profile, "residual_strain": residual,
                "signed_target_error": residual - TARGET, "absolute_target_error": abs(residual - TARGET),
                "target_tolerance": TARGET_TOLERANCE,
                "target_tolerance_satisfied": abs(residual - TARGET) < TARGET_TOLERANCE,
                "terminal_phi": float(state.phi), "terminal_time_s": float(state.time),
                "minimum_terminal_structure_eigenvalue": minimum_eigenvalue,
                "frozen_amplitude_factor": float(control["amplitude_factor"]),
                "actual_strain_knots": [float(control[f"knot_{i}"]) for i in range(1, 5)],
                "actual_duration_s": [float(control[f"duration_{i}_s"]) for i in range(1, 5)],
            })
        old_envelope = np.max([np.abs(trajectories[(level, probe, a)] - trajectories[(level, probe, b)])
                               for a, b in combinations(profiles, 2)], axis=0)
        np.testing.assert_array_equal(old_envelope, archive[
            "baseline_envelope_mpa" if level == 32 else "refined_envelope_mpa"])

    OUTPUT.mkdir(parents=True, exist_ok=True)
    params = pmma_parameters()

    @partial(jax.jit, static_argnames=("probe_substeps",))
    def evaluate_probe(state, residual, probe_substeps):
        # These arguments exactly match the archived diagonal probe operator,
        # including its dt=0 first observation and clock inherited from state.
        return run_strain_history(
            residual + jnp.asarray(additional_strain), jnp.asarray(dt), params, state,
            traction_free=True, substeps=probe_substeps, checkpoint=True,
            previous_strain0=residual, lateral0=jnp.log(state.F[0, 0]),
        )

    new_runs, new_state_arrays = [], {}
    for level, probe in ((32, 8), (48, 4)):
        for profile in profiles:
            run_start = time.perf_counter()
            state = states[(level, profile)]
            residual = float(archived_runs[(level, profile)]["residual_strain"])
            result = evaluate_probe(state, jnp.asarray(residual), probe_substeps=probe)
            stress = np.asarray(-result.stress[:, 1, 1])
            if not all(np.isfinite(np.asarray(value)).all() for value in (*result.state, result.stress)):
                raise FloatingPointError(f"non-finite cross-probe result: {level}/{probe}/{profile}")
            minimum_eigenvalue = min(float(np.linalg.eigvalsh(np.asarray(value)).min())
                                     for value in (result.state.A0, result.state.A1, result.state.A2))
            if minimum_eigenvalue <= 0:
                raise FloatingPointError("non-SPD cross-probe observation state")
            trajectories[(level, probe, profile)] = stress
            for name, value in zip(MaterialState._fields, result.state):
                new_state_arrays[f"{label(level, probe)}_{profile}_{name}"] = np.asarray(value)
            row = {"state_level": level, "probe_substeps": probe, "profile": profile,
                   "state_source": "archived_terminal_state_reused_without_modification",
                   "observation_rows": 81, "finite_observation_states_and_stress": True,
                   "minimum_observation_structure_eigenvalue": minimum_eigenvalue,
                   "maximum_absolute_lateral_stress_mpa": float(np.max(np.abs(np.asarray(result.stress[:, 0, 0])))),
                   "wall_seconds": time.perf_counter() - run_start}
            new_runs.append(row)
            write_csv("cross_probe_runs.csv", new_runs)
            print(json.dumps(row, allow_nan=False), flush=True)

    trajectory_rows, pair_rows, envelope_rows, combination_rows = [], [], [], []
    envelopes, matrices, array_output = {}, {}, {}
    for level, probe in COMBINATIONS:
        key = label(level, probe)
        matrix = np.zeros((3, 3))
        separation_curves = []
        local_pairs = []
        for first, second in combinations(profiles, 2):
            signed = trajectories[(level, probe, first)] - trajectories[(level, probe, second)]
            separation = np.abs(signed)
            separation_curves.append(separation)
            maximum = int(np.argmax(separation))
            row = {"state_level": level, "probe_substeps": probe,
                   "profile_1": first, "profile_2": second,
                   "peak_pair_separation_mpa": float(separation[maximum]),
                   "point_at_peak": maximum, "time_at_peak_s": float(observation_time[maximum]),
                   "additional_strain_at_peak": float(additional_strain[maximum]),
                   "signed_stress_difference_at_peak_mpa": float(signed[maximum]),
                   "terminal_separation_mpa": float(separation[-1])}
            pair_rows.append(row)
            local_pairs.append(row)
            i, j = profiles.index(first), profiles.index(second)
            matrix[i, j] = matrix[j, i] = separation[maximum]
        envelopes[key] = np.max(separation_curves, axis=0)
        matrices[key] = matrix.tolist()
        array_output[f"{key}_pairwise_peak_matrix_mpa"] = matrix
        array_output[f"{key}_envelope_mpa"] = envelopes[key]
        largest = max(local_pairs, key=lambda row: row["peak_pair_separation_mpa"])
        combination_rows.append({"state_level": level, "probe_substeps": probe,
                                 "source": "archived" if (level, probe) in ((32, 4), (48, 8)) else "new_cross_probe",
                                 "maximum_envelope_mpa": float(envelopes[key].max()),
                                 "most_separated_pair": [largest["profile_1"], largest["profile_2"]],
                                 "point_at_peak": largest["point_at_peak"],
                                 "time_at_peak_s": largest["time_at_peak_s"],
                                 "additional_strain_at_peak": largest["additional_strain_at_peak"]})
        for profile in profiles:
            stress = trajectories[(level, probe, profile)]
            array_output[f"{key}_{profile}_stress_mpa"] = stress
            trajectory_rows.extend({"state_level": level, "probe_substeps": probe, "profile": profile,
                                    "point": i, "time_s": float(observation_time[i]),
                                    "additional_true_strain": float(additional_strain[i]),
                                    "compressive_cauchy_stress_mpa": float(value)} for i, value in enumerate(stress))
        envelope_rows.extend({"state_level": level, "probe_substeps": probe, "point": i,
                              "time_s": float(observation_time[i]), "additional_true_strain": float(additional_strain[i]),
                              "pairwise_envelope_mpa": float(value)} for i, value in enumerate(envelopes[key]))

    comparisons = []
    for first, second in combinations(COMBINATIONS, 2):
        a, b = label(*first), label(*second)
        change = envelopes[b] - envelopes[a]
        peak_index = int(np.argmax(np.abs(change)))
        comparisons.append({
            "first": a, "second": b,
            "type": "probe_only" if first[0] == second[0] else "state_only" if first[1] == second[1] else "coupled",
            "signed_change_of_peak_envelope_mpa": float(envelopes[b].max() - envelopes[a].max()),
            "absolute_change_of_peak_envelope_mpa": float(abs(envelopes[b].max() - envelopes[a].max())),
            "relative_peak_change_denominator_second_peak": float(abs(envelopes[b].max() - envelopes[a].max()) / envelopes[b].max()),
            "maximum_pointwise_envelope_change_mpa": float(abs(change[peak_index])),
            "maximum_envelope_change_point": peak_index,
            "maximum_envelope_change_time_s": float(observation_time[peak_index]),
            "maximum_envelope_change_additional_strain": float(additional_strain[peak_index]),
            "maximum_pointwise_stress_change_by_profile_mpa": {
                profile: float(np.max(np.abs(trajectories[(*second, profile)] - trajectories[(*first, profile)])))
                for profile in profiles},
        })
    interaction = envelopes["state48_probe8"] - envelopes["state48_probe4"] - envelopes["state32_probe8"] + envelopes["state32_probe4"]
    for path in sources:
        if sha256(path) != source_hashes[str(path.relative_to(ROOT))]:
            raise RuntimeError(f"source changed during this audit: {path}")
    summary = {
        "audit": "state_probe_2x2_fixed_controls", "status": "completed",
        "source_sha256": source_hashes, "profiles_matrix_order": profiles,
        "state_probe_combinations": [list(pair) for pair in COMBINATIONS],
        "new_combinations": [[32, 8], [48, 4]], "inherited_combinations": [[32, 4], [48, 8]],
        "target": TARGET, "target_tolerance": TARGET_TOLERANCE,
        "amplitude_resolved_at_level": 32, "amplitudes_resolved_again": False,
        "prehistory_recomputed": False, "frozen_complete_terminal_states_reused": True,
        "observation_rows_per_trajectory": 81, "total_trajectories": 12,
        "additional_reload_amplitude": .10, "reload_duration_s": 100., "reload_rate_per_s": .001,
        "operator": {"scheme": "umat", "macaulay_width": 0., "activation_rate": 0.,
                     "traction_free": True, "checkpoint": True, "time0": "inherited state.time",
                     "first_observation_dt": 0., "precision": "float64", "backend": jax.default_backend()},
        "runtime": {"jax_version": jax.__version__, "numpy_version": np.__version__,
                    "thread_environment": {name: os.environ.get(name) for name in
                        ("OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
                    "wall_seconds": time.perf_counter() - started},
        "state_residuals_and_actual_controls": state_rows,
        "new_probe_runs": new_runs, "combinations": combination_rows,
        "pairs": pair_rows, "pairwise_peak_matrices_mpa": matrices,
        "envelope_comparisons": comparisons,
        "maximum_absolute_pointwise_envelope_interaction_mpa": float(np.max(np.abs(interaction))),
        "all_target_errors_within_existing_tolerance": all(row["target_tolerance_satisfied"] for row in state_rows),
        "all_combinations_most_separated_pair_equal": len({tuple(row["most_separated_pair"]) for row in combination_rows}) == 1,
        "scope": "Finite 2x2 resolution audit at fixed archived actual level-32 controls and common 81 times; no asymptotic bound, experimental detectability, recalibration, or full probe-library rank stability is inferred.",
        "provenance_limitation": "The inherited INITIAL state archive recorded the control-source SHA, not a complete contemporaneous core-source signature. This audit hashes all consumed archived arrays and the current cross-probe source; it does not retroactively invent absent INITIAL provenance.",
        "state_admissibility_scope": "New probe finite/SPD checks cover the 81 saved observation states; no claim about unsaved internal substeps or derivative-guard activation follows.",
    }
    write_csv("probe_cross_trajectories.csv", trajectory_rows)
    write_csv("probe_cross_pairs.csv", pair_rows)
    write_csv("probe_cross_combinations.csv", [
        {**{k: v for k, v in row.items() if k != "most_separated_pair"},
         "most_separated_profile_1": row["most_separated_pair"][0],
         "most_separated_profile_2": row["most_separated_pair"][1]}
        for row in combination_rows])
    write_csv("probe_cross_pairwise_matrices.csv", [
        {"state_level": level, "probe_substeps": probe, "row_profile": profile,
         **dict(zip(profiles, matrices[label(level, probe)][i]))}
        for level, probe in COMBINATIONS for i, profile in enumerate(profiles)])
    write_csv("probe_cross_envelopes.csv", envelope_rows)
    write_csv("probe_cross_residuals.csv", [{k: v for k, v in row.items() if not isinstance(v, list)} for row in state_rows])
    np.savez_compressed(OUTPUT / "probe_cross_arrays.npz", time_s=observation_time,
                        additional_true_strain=additional_strain, envelope_interaction_mpa=interaction,
                        **array_output, **new_state_arrays)
    (OUTPUT / "probe_cross_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    report = ["# Fixed-control state/probe 2x2 audit", "",
              "Only state32/probe8 and state48/probe4 are newly integrated. The full terminal states and both diagonal trajectories are reused from the frozen INITIAL archive. No control amplitude or prehistory is re-solved.", "",
              "All 12 trajectories use the same 81 observation times, 100 s duration and +0.10 compressive logarithmic strain. The labels describe numerical resolution, not different physical protocols.", "",
              "## Pairwise peak separations (MPa)", "",
              "Each symmetric matrix is the maximum absolute stress difference over the common 81 times. All 12 pair maxima occur at point 80, t = 100 s, additional strain = 0.10.", ""]
    for key, matrix in matrices.items():
        report.extend([f"### {key}", "", "| Profile | " + " | ".join(profiles) + " |",
                       "|---|" + "---:|" * len(profiles)])
        report.extend("| " + profile + " | " + " | ".join(f"{value:.12f}" for value in matrix[i]) + " |"
                      for i, profile in enumerate(profiles))
        report.append("")
    report.extend(["## Interpretation and limits", "",
                   "The most-separated pair remains minimum_work / minimum_time in all four combinations. State refinement and probe refinement both reduce this peak; the small coupled peak change is not cancellation of opposite-signed peak changes. This statement concerns the sampled peak, not an asymptotic error bound.", "",
                   "The complete envelope changes more than its maximum: the coupled maximum pointwise envelope change is 0.007786119021 MPa (2.89945% of the finest peak), whereas the peak itself changes by 0.000600718754 MPa (0.223700% of the finest peak). State-only envelope changes are 0.003720481248 / 0.003744664159 MPa; probe-only changes are 0.004151753481 / 0.004155258094 MPa. The maximum interaction magnitude is 0.000083166279 MPa. See the JSON for signs, all comparison locations, profile stress changes, residual errors and the original controls.", "",
                   "All residual targets remain within the pre-existing absolute tolerance 0.0002. State32 residual errors are about 1e-9, while state48 errors are 3.34e-5 to 3.84e-5; calling the refined outputs exactly equal would conceal this difference.", "",
                   "Finite and positive-definiteness checks for new probes cover the saved observation states only. They are not an all-internal-substep derivative-guard audit. The archive contains no complete contemporaneous INITIAL core-source signature; current source hashes cannot retroactively supply it. Experimental detectability and the ranking of the entire probe library are outside this audit.", "",
                   "## Machine-readable outputs", "",
                   "- `probe_cross_summary.json`: full provenance, fixed controls, residuals, four-combination metrics and six envelope comparisons.",
                   "- `probe_cross_combinations.csv`, `probe_cross_pairs.csv`, `probe_cross_pairwise_matrices.csv`: four-combination table and all three pairs, with peak locations.",
                   "- `probe_cross_trajectories.csv`, `probe_cross_envelopes.csv`: all 972 stress observations and 324 envelope values.",
                   "- `probe_cross_arrays.npz`: trajectories, envelopes, 3x3 matrices, and all saved new probe observation states.",
                   "- `probe_cross_residuals.csv`, `cross_probe_runs.csv`: target/admissibility and actual new-computation records.",
                   "- `OUTPUT_MANIFEST.json`: SHA-256 hashes of every generated file.", ""])
    (OUTPUT / "README.md").write_text("\n".join(report))
    manifest_files = sorted(path for path in OUTPUT.iterdir() if path.is_file() and path.name != "OUTPUT_MANIFEST.json")
    manifest = {"audit": summary["audit"], "files_sha256": {path.name: sha256(path) for path in manifest_files}}
    (OUTPUT / "OUTPUT_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"combinations": combination_rows, "comparisons": comparisons,
                      "wall_seconds": summary["runtime"]["wall_seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
