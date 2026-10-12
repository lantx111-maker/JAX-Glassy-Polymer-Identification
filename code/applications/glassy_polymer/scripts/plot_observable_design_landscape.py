"""Compute a two-dimensional physical design slice for the balanced history."""

from __future__ import annotations

import csv
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from applications.glassy_polymer.constitutive import pmma_parameters
from applications.glassy_polymer.inverse_design import observable_retained_response


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "observable_inverse_design"
TARGET = 0.07


def sigmoid(value):
    return 1.0 / (1.0 + np.exp(-value))


def logit(value):
    value = np.clip(value, 1.0e-6, 1.0 - 1.0e-6)
    return np.log(value / (1.0 - value))


def selected_balanced_controls() -> np.ndarray:
    rows = list(csv.DictReader((RESULTS / "observable_design_multistart.csv").open()))
    candidates = [row for row in rows if row["profile"] == "balanced"]
    selected = min(candidates, key=lambda row: float(row["objective"]))
    return np.asarray([float(selected[f"raw_{index}"]) for index in range(1, 9)])


def scaled_controls(raw: np.ndarray, amplitude: float, duration: float) -> np.ndarray:
    knots = 0.55 * sigmoid(raw[:4])
    durations = 20.0 + 380.0 * sigmoid(raw[4:])
    knots = np.clip(amplitude * knots, 1.0e-5, 0.55 - 1.0e-5)
    durations = np.clip(duration * durations, 20.0 + 1.0e-4, 400.0 - 1.0e-4)
    return np.concatenate((logit(knots / 0.55), logit((durations - 20.0) / 380.0)))


def main():
    jax.config.update("jax_enable_x64", True)
    mpl.rcParams.update({"font.size": 9.0, "pdf.fonttype": 42, "ps.fonttype": 42})
    raw = selected_balanced_controls()
    amplitude_factors = np.linspace(0.72, 1.28, 17)
    duration_factors = np.linspace(0.50, 1.50, 17)
    controls = np.stack([
        scaled_controls(raw, amplitude, duration)
        for duration in duration_factors for amplitude in amplitude_factors
    ])
    params = pmma_parameters()

    @jax.jit
    @jax.vmap
    def evaluate(local_raw):
        result = observable_retained_response(
            local_raw, params, TARGET, n_per_segment=5,
            loading_substeps=24, recovery_steps=24, recovery_substeps=24,
            work_weight=1.0e-4, time_weight=2.0e-3,
        )
        return (result.residual_strain, result.input_work,
                result.peak_stress, result.maximum_rate)

    residual, work, peak_stress, maximum_rate = [
        np.asarray(value).reshape(len(duration_factors), len(amplitude_factors))
        for value in evaluate(jnp.asarray(controls))
    ]
    if not all(np.isfinite(value).all() for value in (residual, work, peak_stress, maximum_rate)):
        raise RuntimeError("non-finite design landscape")
    center_y = int(np.argmin(np.abs(duration_factors - 1.0)))
    center_x = int(np.argmin(np.abs(amplitude_factors - 1.0)))
    d_residual_d_amplitude = np.gradient(residual, amplitude_factors, axis=1)[center_y, center_x]
    d_residual_d_duration = np.gradient(residual, duration_factors, axis=0)[center_y, center_x]
    gradient_ratio = abs(d_residual_d_amplitude) / max(abs(d_residual_d_duration), 1.0e-14)
    center_values = {
        "amplitude_factor": float(amplitude_factors[center_x]),
        "duration_factor": float(duration_factors[center_y]),
        "residual_strain_after_600s": float(residual[center_y, center_x]),
        "cumulative_positive_work_mpa": float(work[center_y, center_x]),
        "peak_stress_mpa": float(peak_stress[center_y, center_x]),
        "maximum_rate_per_s": float(maximum_rate[center_y, center_x]),
    }
    controls = {
        row["profile"]: row for row in csv.DictReader(
            (RESULTS / "observable_design_controls.csv").open()
        )
    }
    if not np.allclose(
        [center_values["amplitude_factor"], center_values["duration_factor"]],
        [1.0, 1.0], rtol=0.0, atol=1.0e-14,
    ):
        raise AssertionError("design-slice star is not at the balanced scaling (1,1)")
    if abs(center_values["residual_strain_after_600s"]
           - float(controls["balanced"]["residual_strain_after_600s"])) > 1.0e-10:
        raise AssertionError("design-slice center and refined balanced history disagree")
    with (RESULTS / "observable_design_landscape_center.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(center_values))
        writer.writeheader()
        writer.writerow(center_values)

    np.savez(
        RESULTS / "observable_design_landscape.npz",
        amplitude_factor=amplitude_factors,
        duration_factor=duration_factors,
        residual_strain=residual,
        input_work_mpa=work,
        peak_stress_mpa=peak_stress,
        maximum_rate_per_s=maximum_rate,
        local_residual_gradient=np.asarray([d_residual_d_amplitude, d_residual_d_duration]),
        local_amplitude_duration_gradient_ratio=gradient_ratio,
    )

    x, y = np.meshgrid(amplitude_factors, duration_factors)
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.7), constrained_layout=True)
    residual_levels = np.linspace(np.min(residual), np.max(residual), 15)
    contour = axes[0].contourf(x, y, residual, levels=residual_levels, cmap="viridis")
    target = axes[0].contour(x, y, residual, levels=[TARGET], colors="white", linewidths=2.0)
    axes[0].clabel(target, fmt={TARGET: "target 0.07"}, fontsize=8)
    axes[0].contour(x, y, residual, levels=[TARGET - 0.002, TARGET + 0.002],
                    colors="#F2C14E", linestyles="--", linewidths=1.2)
    axes[0].plot(1.0, 1.0, marker="*", ms=12, color="#D1495B", zorder=8)
    axes[0].annotate("balanced-weight history", xy=(1.0, 1.0), xytext=(1.06, 0.88),
                     arrowprops={"arrowstyle": "->", "color": "#D1495B"},
                     fontsize=8, color="#20252B")
    axes[0].set_title("(a) Residual strain after 600 s recovery")
    axes[0].set_xlabel("strain-amplitude scale")
    axes[0].set_ylabel("segment-duration scale")
    fig.colorbar(contour, ax=axes[0], label="residual true strain")

    work_levels = np.linspace(np.min(work), np.max(work), 15)
    contour = axes[1].contourf(x, y, work, levels=work_levels, cmap="magma")
    axes[1].contour(x, y, residual, levels=[TARGET], colors="white", linewidths=2.0)
    infeasible = (peak_stress > 180.0) | (maximum_rate > 0.01)
    if np.any(infeasible):
        axes[1].contourf(x, y, infeasible.astype(float), levels=[0.5, 1.5],
                         colors="none", hatches=["////"])
    axes[1].plot(1.0, 1.0, marker="*", ms=12, color="#D1495B", zorder=8)
    axes[1].set_title("(b) Cumulative positive work on the same design slice")
    axes[1].set_xlabel("strain-amplitude scale")
    axes[1].set_ylabel("segment-duration scale")
    fig.colorbar(contour, ax=axes[1], label="cumulative positive work (MPa)")

    for axis in axes:
        axis.set_xlim(amplitude_factors[0], amplitude_factors[-1])
        axis.set_ylim(duration_factors[0], duration_factors[-1])
    fig.savefig(RESULTS / "observable_design_landscape.pdf", bbox_inches="tight")
    fig.savefig(RESULTS / "observable_design_landscape.png", dpi=450, bbox_inches="tight")
    plt.close(fig)
    print({
        "residual_range": [float(np.min(residual)), float(np.max(residual))],
        "work_range_mpa": [float(np.min(work)), float(np.max(work))],
        "peak_stress_max_mpa": float(np.max(peak_stress)),
        "maximum_rate_max_per_s": float(np.max(maximum_rate)),
        "local_amplitude_duration_gradient_ratio": float(gradient_ratio),
        "constraint_active_points": int(np.count_nonzero(infeasible)),
    })


if __name__ == "__main__":
    main()
