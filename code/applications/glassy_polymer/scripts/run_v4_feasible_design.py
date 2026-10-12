"""Construct target-feasible V4 histories and compare their retained states.

This is deliberately a bracketed one-dimensional amplitude solve on three
predeclared history shapes.  It does not claim a work- or time-optimal point.
"""

from __future__ import annotations

import csv
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
from applications.glassy_polymer.scripts.plot_observable_design_landscape import (
    scaled_controls,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "observable_inverse_design" / "observable_design_controls.csv"
OUTPUT = ROOT / "results" / "v4_feasible_design"
TARGETS = (0.02, 0.07)
SOLVE_LEVEL = 32
LEVELS = (24, 32, 48)

mpl.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 8.5,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
    "legend.fontsize": 7.5,
})


def logit(value):
    value = np.clip(value, 1.0e-7, 1.0 - 1.0e-7)
    return np.log(value / (1.0 - value))


def encode(row) -> np.ndarray:
    knots = np.asarray([float(row[f"knot_{index}"]) for index in range(1, 5)])
    durations = np.asarray([
        float(row[f"duration_{index}_s"]) for index in range(1, 5)
    ])
    return np.concatenate((
        logit(knots / 0.55),
        logit((durations - 20.0) / 380.0),
    ))


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    params = pmma_parameters()
    profiles = [(row["profile"], encode(row)) for row in csv.DictReader(SOURCE.open())]

    @jax.jit
    def evaluate_sensitivity_level(raw, target):
        return observable_retained_response(
            raw, params, target, n_per_segment=5,
            loading_substeps=SOLVE_LEVEL, recovery_steps=SOLVE_LEVEL,
            recovery_substeps=SOLVE_LEVEL,
            work_weight=0.0, time_weight=0.0,
        ).residual_strain

    selected = []
    for target in TARGETS:
        for profile, base_raw in profiles:
            lower, upper = 0.05, 1.20
            lower_value = float(evaluate_sensitivity_level(
                jnp.asarray(scaled_controls(base_raw, lower, 1.0)), target
            ))
            upper_value = float(evaluate_sensitivity_level(
                jnp.asarray(scaled_controls(base_raw, upper, 1.0)), target
            ))
            if not lower_value <= target <= upper_value:
                raise RuntimeError(
                    f"target {target} is not bracketed for {profile}: "
                    f"[{lower_value}, {upper_value}]"
                )
            for _ in range(24):
                midpoint = 0.5 * (lower + upper)
                raw = scaled_controls(base_raw, midpoint, 1.0)
                value = float(evaluate_sensitivity_level(jnp.asarray(raw), target))
                if value < target:
                    lower = midpoint
                else:
                    upper = midpoint
            factor = 0.5 * (lower + upper)
            selected.append((target, profile, factor,
                             scaled_controls(base_raw, factor, 1.0)))
            print(target, profile, factor, flush=True)

    rows = []
    production = {}
    for target, profile, factor, raw in selected:
        for level in LEVELS:
            result = observable_retained_response(
                jnp.asarray(raw), params, target, n_per_segment=5,
                loading_substeps=level, recovery_steps=level,
                recovery_substeps=level, work_weight=0.0, time_weight=0.0,
            )
            time, strain = decode_physical_history(jnp.asarray(raw), n_per_segment=5)
            nodes = np.asarray([0, 5, 10, 15, 20])
            row = {
                "target": target,
                "profile": profile,
                "status": "bracketed_amplitude_feasible_schedule",
                "amplitude_factor": factor,
                "mesh_level": level,
                "steps_per_loading_interval": level,
                "stress_transition_steps": level,
                "zero_stress_hold_steps": level * level,
                "residual_strain": float(result.residual_strain),
                "target_error": float(result.residual_strain) - target,
                "reload_stiffness_mpa": float(result.reload_stiffness),
                "reload_stress_at_two_percent_mpa": float(
                    result.reload_stress_at_two_percent
                ),
                "positive_axial_cauchy_work_mpa": float(result.input_work),
                "net_axial_cauchy_work_mpa": float(result.net_work),
                "absolute_axial_cauchy_work_mpa": float(result.cumulative_absolute_work),
                "peak_stress_mpa": float(result.peak_stress),
                "maximum_rate_per_s": float(result.maximum_rate),
                "loading_program_time_s": float(result.loading_program_time),
                "terminal_phi": float(result.terminal_phi),
                "terminal_kappa1": float(result.terminal_kappa1),
                "terminal_kappa2": float(result.terminal_kappa2),
                "terminal_b1": float(result.terminal_b1),
                "terminal_b2": float(result.terminal_b2),
                **{f"terminal_A0_eigenvalue_{i + 1}": float(value)
                   for i, value in enumerate(result.terminal_A0_eigenvalues)},
                **{f"terminal_A1_eigenvalue_{i + 1}": float(value)
                   for i, value in enumerate(result.terminal_A1_eigenvalues)},
                **{f"terminal_A2_eigenvalue_{i + 1}": float(value)
                   for i, value in enumerate(result.terminal_A2_eigenvalues)},
                **{f"knot_{i + 1}": float(np.asarray(strain)[nodes[i + 1]])
                   for i in range(4)},
                **{f"duration_{i + 1}_s": float(np.diff(np.asarray(time)[nodes])[i])
                   for i in range(4)},
            }
            rows.append(row)
            if level == SOLVE_LEVEL:
                production[(target, profile)] = (raw, result)
            print(target, profile, level, row["residual_strain"], flush=True)

    with (OUTPUT / "feasible_history_refinement.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # The primary figure uses the observation-scale target; 0.07 remains an
    # explicitly extrapolative secondary calculation in the exported table.
    target = 0.02
    names = [name for name, _ in profiles]
    state_labels = [
        r"$\phi$", r"$\kappa_1$", r"$\kappa_2$", r"$b_1$", r"$b_2$",
        r"$\lambda_{\min}(A_0)$", r"$\lambda_{\min}(A_1)$",
        r"$\lambda_{\min}(A_2)$", "2% reload stress",
    ]
    state = []
    for name in names:
        result = production[(target, name)][1]
        state.append([
            float(result.terminal_phi), float(result.terminal_kappa1),
            float(result.terminal_kappa2), float(result.terminal_b1),
            float(result.terminal_b2), float(jnp.min(result.terminal_A0_eigenvalues)),
            float(jnp.min(result.terminal_A1_eigenvalues)),
            float(jnp.min(result.terminal_A2_eigenvalues)),
            float(result.reload_stress_at_two_percent),
        ])
    state = np.asarray(state)
    mean = np.mean(state, axis=0)
    relative = (state - mean) / np.maximum(np.abs(mean), 1.0e-12)

    fig, axes = plt.subplots(2, 2, figsize=(8.3, 6.0), constrained_layout=True)
    colors = ("#1f77b4", "#d95f02", "#2a9d8f")
    for color, name in zip(colors, names):
        raw, _ = production[(target, name)]
        time, strain = decode_physical_history(jnp.asarray(raw), n_per_segment=5)
        axes[0, 0].plot(time, strain, color=color, lw=1.4,
                        label=name.replace("minimum_", "").replace("_", " "))
    axes[0, 0].set(xlabel="loading-program time (s)",
                   ylabel="compressive true strain")
    axes[0, 0].legend(frameon=False)
    axes[0, 0].grid(alpha=0.25)

    x = np.arange(len(names))
    for level, marker in zip(LEVELS, ("o", "s", "^")):
        values = [next(row["residual_strain"] for row in rows
                       if row["target"] == target and row["profile"] == name
                       and row["mesh_level"] == level) for name in names]
        axes[0, 1].plot(x, values, marker=marker, label=f"level {level}")
    axes[0, 1].axhline(target, color="0.35", ls="--", lw=1.0)
    axes[0, 1].set_xticks(x, [name.replace("minimum_", "").replace("_", "\n")
                             for name in names])
    axes[0, 1].set_ylabel("residual strain after transition--recovery")
    axes[0, 1].legend(frameon=False)
    axes[0, 1].grid(alpha=0.25)

    image = axes[1, 0].imshow(relative, cmap="coolwarm", vmin=-np.max(np.abs(relative)),
                              vmax=np.max(np.abs(relative)), aspect="auto")
    axes[1, 0].set_xticks(np.arange(len(state_labels)), state_labels,
                          rotation=55, ha="right")
    axes[1, 0].set_yticks(np.arange(len(names)),
                          [name.replace("minimum_", "").replace("_", " ") for name in names])
    axes[1, 0].set_title("equal residual, different terminal state")
    fig.colorbar(image, ax=axes[1, 0], label="relative deviation from profile mean")

    work = [float(production[(target, name)][1].input_work) for name in names]
    stiffness = [float(production[(target, name)][1].reload_stress_at_two_percent)
                 for name in names]
    duration = [float(production[(target, name)][1].loading_program_time) for name in names]
    scatter = axes[1, 1].scatter(work, stiffness, c=duration, s=65, cmap="viridis")
    for name, work_value, stiffness_value in zip(names, work, stiffness):
        axes[1, 1].annotate(name.replace("minimum_", "").replace("_", " "),
                            (work_value, stiffness_value), xytext=(4, 3),
                            textcoords="offset points", fontsize=7)
    axes[1, 1].set(xlabel="positive axial Cauchy-work proxy (MPa)",
                   ylabel="stress after 2% reload (MPa)")
    axes[1, 1].grid(alpha=0.25)
    fig.colorbar(scatter, ax=axes[1, 1], label="loading-program time (s)")
    for index, axis in enumerate(axes.flat):
        axis.text(0.02, 0.98, f"({chr(97 + index)})", transform=axis.transAxes,
                  va="top", fontweight="bold")
    fig.savefig(OUTPUT / "v4_feasible_history_design.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT / "v4_feasible_history_design.png", dpi=360, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
