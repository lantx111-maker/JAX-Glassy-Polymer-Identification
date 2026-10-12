"""Select an observable follow-on probe for equal-residual 0.020 histories."""

from __future__ import annotations

import csv
import json
from itertools import combinations
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from applications.glassy_polymer.constitutive import pmma_parameters
from applications.glassy_polymer.inverse_design import (
    decode_physical_history,
    observable_retained_response,
)
from applications.glassy_polymer.protocols import run_strain_history
from applications.glassy_polymer.scripts.match_residual_histories import encode


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "matched_residual_histories" / "feasible_history_refinement.csv"
OUTPUT = ROOT / "results" / "subsequent_probes"
TARGET = 0.020
LEVEL = 32

mpl.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 8.2,
    "axes.labelsize": 8.2, "xtick.labelsize": 7.2,
    "ytick.labelsize": 7.2, "legend.fontsize": 7.0,
})


def _state_coordinates(result, params):
    values = np.concatenate((
        np.asarray([
            result.terminal_phi / params.gamma0,
            result.terminal_kappa1 / params.s0,
            result.terminal_kappa2 / params.s0,
            result.terminal_b1 / params.E,
            result.terminal_b2 / params.E,
        ]),
        np.log(np.asarray(result.terminal_A0_eigenvalues)),
        np.log(np.asarray(result.terminal_A1_eigenvalues)),
        np.log(np.asarray(result.terminal_A2_eigenvalues)),
    ))
    return values


def _probe_history(name, residual):
    if name.startswith("reload_"):
        amplitude = float(name.split("_")[1].replace("pct", "")) / 100.0
        count = {0.02: 41, 0.05: 61, 0.10: 81}[amplitude]
        strain = np.linspace(residual, residual + amplitude, count)
        time = np.linspace(0.0, amplitude / 1.0e-3, count)
    elif name == "hold_120s":
        time = np.linspace(0.0, 120.0, 61)
        strain = np.full(time.shape, residual)
    elif name == "reverse_2pct":
        time = np.linspace(0.0, 20.0, 41)
        strain = np.linspace(residual, residual - 0.02, 41)
    elif name == "reload5_dwell120":
        loading_time = np.linspace(0.0, 50.0, 51)
        loading_strain = np.linspace(residual, residual + 0.05, 51)
        dwell_time = 50.0 + np.linspace(2.0, 120.0, 60)
        dwell_strain = np.full(dwell_time.shape, residual + 0.05)
        time = np.concatenate((loading_time, dwell_time))
        strain = np.concatenate((loading_strain, dwell_strain))
    else:
        raise ValueError(name)
    return time, strain


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    params = pmma_parameters()
    source_rows = [
        row for row in csv.DictReader(SOURCE.open())
        if abs(float(row["target"]) - TARGET) < 1.0e-12
        and int(row["mesh_level"]) == LEVEL
    ]
    if len(source_rows) != 3:
        raise RuntimeError("expected three level-32 observation-scale schedules")

    schedules = {}
    physical_rows = []
    for row in source_rows:
        profile = row["profile"]
        raw = encode(row)
        result = observable_retained_response(
            jnp.asarray(raw), params, TARGET, n_per_segment=5,
            loading_substeps=LEVEL, recovery_steps=LEVEL,
            recovery_substeps=LEVEL, work_weight=0.0, time_weight=0.0,
        )
        schedules[profile] = (raw, result)
        physical_rows.append({
            "profile": profile,
            "residual_strain": float(result.residual_strain),
            "phi_per_s": float(result.terminal_phi),
            "kappa1_mpa": float(result.terminal_kappa1),
            "kappa2_mpa": float(result.terminal_kappa2),
            "b1_mpa": float(result.terminal_b1),
            "b2_mpa": float(result.terminal_b2),
            **{f"A0_eigenvalue_{i + 1}": float(value)
               for i, value in enumerate(result.terminal_A0_eigenvalues)},
            **{f"A1_eigenvalue_{i + 1}": float(value)
               for i, value in enumerate(result.terminal_A1_eigenvalues)},
            **{f"A2_eigenvalue_{i + 1}": float(value)
               for i, value in enumerate(result.terminal_A2_eigenvalues)},
        })
    with (OUTPUT / "primary_target_absolute_states.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=physical_rows[0].keys())
        writer.writeheader()
        writer.writerows(physical_rows)

    profiles = [row["profile"] for row in source_rows]
    state_coordinates = {
        profile: _state_coordinates(schedules[profile][1], params)
        for profile in profiles
    }
    state_distance_rows = []
    for first, second in combinations(profiles, 2):
        state_distance_rows.append({
            "profile_1": first,
            "profile_2": second,
            "scaled_state_distance": float(np.linalg.norm(
                state_coordinates[first] - state_coordinates[second]
            )),
        })
    with (OUTPUT / "primary_target_scaled_state_distances.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=state_distance_rows[0].keys())
        writer.writeheader()
        writer.writerows(state_distance_rows)

    probe_names = (
        "reload_2pct", "reload_5pct", "reload_10pct", "hold_120s",
        "reverse_2pct", "reload5_dwell120",
    )
    trajectories = {}
    probe_rows = []
    trajectory_rows = []
    for probe in probe_names:
        local = {}
        for profile in profiles:
            result = schedules[profile][1]
            state = result.recovered_state
            residual = float(result.residual_strain)
            time, strain = _probe_history(probe, residual)
            dt = np.diff(time, prepend=time[:1])
            simulation = run_strain_history(
                jnp.asarray(strain), jnp.asarray(dt), params, state,
                traction_free=True, substeps=4, checkpoint=True,
                previous_strain0=residual,
                lateral0=float(jnp.log(state.F[0, 0])),
            )
            stress = np.asarray(-simulation.stress[:, 1, 1])
            local[profile] = stress
            for index, (time_value, strain_value, stress_value) in enumerate(
                zip(time, strain, stress)
            ):
                trajectory_rows.append({
                    "probe": probe, "profile": profile, "point": index,
                    "time_s": time_value, "compressive_true_strain": strain_value,
                    "compressive_cauchy_stress_mpa": stress_value,
                })
        pairwise = [
            np.abs(local[first] - local[second])
            for first, second in combinations(profiles, 2)
        ]
        maximum_separation = float(max(np.max(value) for value in pairwise))
        terminal_separation = float(max(value[-1] for value in pairwise))
        probe_rows.append({
            "probe": probe,
            "maximum_pairwise_stress_separation_mpa": maximum_separation,
            "terminal_pairwise_stress_separation_mpa": terminal_separation,
        })
        trajectories[probe] = local
        print(probe, maximum_separation, flush=True)

    selected_probe = max(
        probe_rows, key=lambda row: row["maximum_pairwise_stress_separation_mpa"]
    )["probe"]
    for row in probe_rows:
        row["selected_discriminator"] = row["probe"] == selected_probe
    with (OUTPUT / "follow_on_probe_ranking.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=probe_rows[0].keys())
        writer.writeheader()
        writer.writerows(probe_rows)
    with (OUTPUT / "follow_on_probe_trajectories.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=trajectory_rows[0].keys())
        writer.writeheader()
        writer.writerows(trajectory_rows)

    # Publication figure: observation-scale histories, fine check, absolute
    # terminal states, and a data-driven follow-on discriminator.
    fig, axes = plt.subplots(2, 2, figsize=(8.25, 5.9), constrained_layout=True)
    colors = ("#1768ac", "#d95f02", "#2a9d8f")
    for color, profile in zip(colors, profiles):
        raw, _ = schedules[profile]
        time, strain = decode_physical_history(jnp.asarray(raw), n_per_segment=5)
        axes[0, 0].plot(time, strain, lw=1.5, color=color,
                        label=profile.replace("minimum_", "").replace("_", " "))
    axes[0, 0].set(xlabel="loading-program time (s)", ylabel="compressive true strain")
    axes[0, 0].legend(frameon=False)

    all_refinement = list(csv.DictReader(SOURCE.open()))
    x = np.arange(len(profiles))
    for level, marker in ((24, "o"), (32, "s"), (48, "^")):
        values = [
            float(next(
                row["residual_strain"] for row in all_refinement
                if abs(float(row["target"]) - TARGET) < 1.0e-12
                and row["profile"] == profile and int(row["mesh_level"]) == level
            )) for profile in profiles
        ]
        axes[0, 1].plot(x, values, marker=marker, label=f"level {level}")
    axes[0, 1].axhline(TARGET, color="0.3", ls="--", lw=1.0)
    axes[0, 1].set_xticks(x, [profile.replace("minimum_", "").replace("_", "\n")
                              for profile in profiles])
    axes[0, 1].set_ylabel(r"$\varepsilon_{\rm res}^{600}$")
    axes[0, 1].legend(frameon=False)

    state_labels = (r"$\phi$", r"$\kappa_1$", r"$\kappa_2$", r"$b_1$", r"$b_2$")
    state_values = np.asarray([[row[key] for key in (
        "phi_per_s", "kappa1_mpa", "kappa2_mpa", "b1_mpa", "b2_mpa"
    )] for row in physical_rows], dtype=float)
    column_mean = np.mean(state_values, axis=0)
    color_values = (state_values - column_mean) / np.maximum(np.abs(column_mean), 1.0e-14)
    limit = max(np.max(np.abs(color_values)), 1.0e-12)
    image = axes[1, 0].imshow(color_values, cmap="coolwarm", vmin=-limit, vmax=limit,
                              aspect="auto")
    axes[1, 0].set_xticks(np.arange(len(state_labels)), state_labels)
    axes[1, 0].set_yticks(np.arange(len(profiles)),
                          [profile.replace("minimum_", "").replace("_", " ")
                           for profile in profiles])
    for i in range(state_values.shape[0]):
        for j in range(state_values.shape[1]):
            axes[1, 0].text(j, i, f"{state_values[i, j]:.3g}", ha="center",
                            va="center", fontsize=6.5)
    axes[1, 0].set_title("absolute terminal states (physical units)")
    fig.colorbar(image, ax=axes[1, 0], label="scaled deviation for color only")

    bars = axes[1, 1].bar(
        np.arange(len(probe_rows)),
        [row["maximum_pairwise_stress_separation_mpa"] for row in probe_rows],
        color=["#c1121f" if row["selected_discriminator"] else "#8ecae6"
               for row in probe_rows],
    )
    axes[1, 1].set_xticks(
        np.arange(len(probe_rows)),
        [row["probe"].replace("_", "\n") for row in probe_rows], rotation=25,
    )
    axes[1, 1].set_ylabel("maximum pairwise stress separation (MPa)")
    axes[1, 1].set_title(f"selected follow-on probe: {selected_probe.replace('_', ' ')}")
    axes[1, 1].bar_label(bars, fmt="%.2f", fontsize=6.3, padding=2)
    for index, axis in enumerate(axes.flat):
        axis.text(0.02, 0.98, f"({chr(97 + index)})", transform=axis.transAxes,
                  va="top", fontweight="bold")
        axis.grid(alpha=0.2)
    fig.savefig(OUTPUT / "residual_design_and_reload.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT / "residual_design_and_reload.png", dpi=360,
                bbox_inches="tight")
    plt.close(fig)

    summary = {
        "primary_target": TARGET,
        "interpretation": "observation-scale target, not an observed-range claim",
        "maximum_level32_to_level48_change": max(
            abs(float(next(row["residual_strain"] for row in all_refinement
                           if abs(float(row["target"]) - TARGET) < 1.0e-12
                           and row["profile"] == profile
                           and int(row["mesh_level"]) == 32))
                - float(next(row["residual_strain"] for row in all_refinement
                             if abs(float(row["target"]) - TARGET) < 1.0e-12
                             and row["profile"] == profile
                             and int(row["mesh_level"]) == 48)))
            for profile in profiles
        ),
        "selected_follow_on_probe": selected_probe,
        "selected_maximum_stress_separation_mpa": next(
            row["maximum_pairwise_stress_separation_mpa"] for row in probe_rows
            if row["selected_discriminator"]
        ),
        "scaled_state_distances": state_distance_rows,
    }
    (OUTPUT / "follow_on_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
