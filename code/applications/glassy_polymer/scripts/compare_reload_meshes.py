"""Refine the retained state and 10% reload without re-solving design controls.

The baseline is state level 32 plus four probe substeps; the comparison is
state level 48 plus eight probe substeps. Both use identical 81 observation
times and identical additional-strain coordinates. Only the originally frozen
level-32 amplitudes are used. This is a coupled two-resolution comparison, not
an asymptotic error estimate or a re-ranking of the complete probe library.
"""

from __future__ import annotations

import csv
import hashlib
import json
from itertools import combinations
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from applications.glassy_polymer.constitutive import pmma_parameters
from applications.glassy_polymer.inverse_design import observable_retained_response
from applications.glassy_polymer.protocols import run_strain_history
from applications.glassy_polymer.scripts.match_residual_histories import encode


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "matched_residual_histories" / "feasible_history_refinement.csv"
OUTPUT = ROOT / "results" / "reload_refinement"
TARGET = 0.020
TARGET_TOLERANCE = 2.0e-4
LEVELS = ((32, 4), (48, 8))


def _write_csv(name, rows):
    with (OUTPUT / name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    jax.config.update("jax_enable_x64", True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    params = pmma_parameters()
    with SOURCE.open() as handle:
        frozen = [row for row in csv.DictReader(handle)
                  if abs(float(row["target"]) - TARGET) < 1e-12
                  and int(row["mesh_level"]) == 32]
    if len(frozen) != 3 or len({row["profile"] for row in frozen}) != 3:
        raise RuntimeError("expected three distinct frozen level-32 schedules")
    profiles = [row["profile"] for row in frozen]
    observation_time = np.linspace(0.0, 100.0, 81)
    additional_strain = np.linspace(0.0, 0.10, 81)
    observation_dt = np.diff(observation_time, prepend=observation_time[:1])
    runs, trajectories, state_arrays = [], {}, {}
    for state_level, probe_substeps in LEVELS:
        for source_row in frozen:
            started = time.perf_counter()
            profile = source_row["profile"]
            result = observable_retained_response(
                jnp.asarray(encode(source_row)), params, TARGET,
                n_per_segment=5, loading_substeps=state_level,
                recovery_steps=state_level, recovery_substeps=state_level,
                work_weight=0.0, time_weight=0.0,
            )
            state = result.recovered_state
            residual = float(result.residual_strain)
            strain = residual + additional_strain
            probe = run_strain_history(
                jnp.asarray(strain), jnp.asarray(observation_dt), params, state,
                traction_free=True, substeps=probe_substeps, checkpoint=True,
                previous_strain0=residual,
                lateral0=float(jnp.log(state.F[0, 0])),
            )
            stress = np.asarray(-probe.stress[:, 1, 1])
            if not all(np.all(np.isfinite(np.asarray(value))) for value in (*state, *probe.state, probe.stress)):
                raise FloatingPointError(f"non-finite retained state or probe: {profile}, level {state_level}")
            minimum_eigenvalue = float(min(
                np.linalg.eigvalsh(np.asarray(values)).min()
                for values in (probe.state.A0, probe.state.A1, probe.state.A2)
            ))
            if minimum_eigenvalue <= 0.0:
                raise FloatingPointError("non-SPD structure tensor during reload")
            trajectories[(state_level, profile)] = stress
            for name, values in zip(state._fields, state):
                state_arrays[f"state{state_level}_{profile}_{name}"] = np.asarray(values)
            row = {
                "profile": profile, "state_level": state_level,
                "probe_substeps": probe_substeps,
                "frozen_amplitude_factor": float(source_row["amplitude_factor"]),
                "amplitude_resolved_at_level": 32, "amplitude_resolved_again": False,
                "residual_strain": residual, "absolute_target_error": abs(residual - TARGET),
                "target_tolerance": TARGET_TOLERANCE,
                "target_tolerance_satisfied": abs(residual - TARGET) < TARGET_TOLERANCE,
                "terminal_phi": float(state.phi),
                "probe_observation_rows": len(stress),
                "minimum_probe_structure_eigenvalue": minimum_eigenvalue,
                "maximum_probe_stress_mpa": float(np.max(stress)),
                "wall_seconds": time.perf_counter() - started,
            }
            runs.append(row)
            _write_csv("reload_refinement_runs.csv", runs)
            print(json.dumps(row, allow_nan=False), flush=True)
        jax.clear_caches()

    trajectory_rows = [
        {"state_level": level, "probe_substeps": substeps, "profile": profile,
         "point": index, "time_s": float(observation_time[index]),
         "additional_true_strain": float(additional_strain[index]),
         "compressive_cauchy_stress_mpa": float(stress)}
        for level, substeps in LEVELS for profile in profiles
        for index, stress in enumerate(trajectories[(level, profile)])
    ]
    _write_csv("reload_refinement_trajectories.csv", trajectory_rows)
    pair_rows = []
    envelopes = {}
    for level, substeps in LEVELS:
        separations = []
        for first, second in combinations(profiles, 2):
            separation = np.abs(trajectories[(level, first)] - trajectories[(level, second)])
            separations.append(separation)
            maximum = int(np.argmax(separation))
            pair_rows.append({
                "state_level": level, "probe_substeps": substeps,
                "profile_1": first, "profile_2": second,
                "maximum_pair_separation_mpa": float(separation[maximum]),
                "time_at_maximum_s": float(observation_time[maximum]),
                "additional_strain_at_maximum": float(additional_strain[maximum]),
                "terminal_pair_separation_mpa": float(separation[-1]),
            })
        envelopes[level] = np.max(separations, axis=0)
    _write_csv("reload_refinement_pairs.csv", pair_rows)
    maximum_baseline = float(np.max(envelopes[32]))
    maximum_refined = float(np.max(envelopes[48]))
    profile_changes = {
        profile: float(np.max(np.abs(trajectories[(32, profile)] - trajectories[(48, profile)])))
        for profile in profiles
    }
    baseline_ranking = sorted([row for row in pair_rows if row["state_level"] == 32],
                              key=lambda row: row["maximum_pair_separation_mpa"], reverse=True)
    refined_ranking = sorted([row for row in pair_rows if row["state_level"] == 48],
                             key=lambda row: row["maximum_pair_separation_mpa"], reverse=True)
    summary = {
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "target": TARGET, "frozen_amplitude_level": 32,
        "state_and_probe_resolutions": [list(item) for item in LEVELS],
        "same_observation_rows": 81, "precision": "float64",
        "reload_amplitude": 0.10, "reload_rate_per_s": 1.0e-3,
        "baseline_maximum_pairwise_separation_mpa": maximum_baseline,
        "refined_maximum_pairwise_separation_mpa": maximum_refined,
        "absolute_change_of_maximum_separation_mpa": abs(maximum_refined - maximum_baseline),
        "relative_change_of_maximum_separation": abs(maximum_refined - maximum_baseline) / maximum_refined,
        "maximum_pointwise_envelope_change_mpa": float(np.max(np.abs(envelopes[32] - envelopes[48]))),
        "maximum_pointwise_stress_change_by_profile_mpa": profile_changes,
        "most_separated_pair_baseline": [baseline_ranking[0]["profile_1"], baseline_ranking[0]["profile_2"]],
        "most_separated_pair_refined": [refined_ranking[0]["profile_1"], refined_ranking[0]["profile_2"]],
        "maximum_target_error_baseline": max(row["absolute_target_error"] for row in runs if row["state_level"] == 32),
        "maximum_target_error_refined": max(row["absolute_target_error"] for row in runs if row["state_level"] == 48),
        "all_residual_targets_within_existing_tolerance": all(row["target_tolerance_satisfied"] for row in runs),
        "scope": "Two coupled state/probe resolutions at fixed controls. No asymptotic bound, experimental validation, or stability of the full probe-library ranking is inferred.",
        "acceptance": "The existing 2e-4 residual target tolerance is checked. Stress-separation changes are reported quantitatively without inventing a prior amplitude acceptance threshold.",
        "runs": runs, "pairs": pair_rows,
    }
    (OUTPUT / "reload_refinement_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    np.savez_compressed(
        OUTPUT / "reload_refinement_states_and_envelopes.npz",
        time_s=observation_time, additional_true_strain=additional_strain,
        baseline_envelope_mpa=envelopes[32], refined_envelope_mpa=envelopes[48],
        **state_arrays,
    )
    print(json.dumps({key: value for key, value in summary.items() if key not in {"runs", "pairs"}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
