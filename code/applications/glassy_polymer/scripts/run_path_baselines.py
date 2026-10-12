"""Baseline reverse, cyclic, and retained-state recovery paths."""

from __future__ import annotations

import csv
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from applications.glassy_polymer.constitutive import initial_state, pc_parameters, pmma_parameters
from applications.glassy_polymer.data import classify_curve, load_all_curves
from applications.glassy_polymer.data_cleaning import strictly_increasing_time
from applications.glassy_polymer.protocols import run_strain_history, run_stress_history


OUTPUT = Path(__file__).resolve().parents[1] / "results"


def terminal_state(stacked):
    return jax.tree_util.tree_map(lambda value: value[-1], stacked)


def history_dt(strain, rate, engineering):
    coordinate = 1.0 - np.exp(-strain) if engineering else strain
    return np.maximum(np.abs(np.diff(coordinate, prepend=coordinate[:1])) / rate, 1.0e-5)


def score(predicted, measured):
    error = np.asarray(predicted) - np.asarray(measured)
    return float(np.sqrt(np.mean(error * error))), float(np.max(np.abs(error)))


def simulate_strain_curve(curve, params, substeps=100, scheme="vumat"):
    meta = classify_curve(curve)
    strain = np.asarray(curve.x)
    engineering = meta.material == "PC"
    dt = history_dt(strain, meta.nominal_rate or (0.001 if engineering else 0.0003), engineering)
    result = run_strain_history(
        jnp.asarray(strain), jnp.asarray(dt), params, initial_state(params),
        traction_free=True, substeps=substeps, scheme=scheme,
    )
    return np.asarray(-result.stress[:, 1, 1]), result


def main():
    OUTPUT.mkdir(exist_ok=True)
    curves = load_all_curves()
    groups = []
    for material, family in [("PMMA", "reverse"), ("PMMA", "cyclic"), ("PC", "cyclic")]:
        subset = [c for c in curves if classify_curve(c).material == material and classify_curve(c).family == family]
        subset.sort(key=lambda c: classify_curve(c).target or 0.0)
        params = pmma_parameters() if material == "PMMA" else pc_parameters()
        outputs, rows = [], []
        for curve in subset:
            prediction, _ = simulate_strain_curve(curve, params)
            rmse, maximum = score(prediction, curve.y)
            meta = classify_curve(curve)
            rows.append((material, family, meta.target, len(curve.x), rmse, maximum, bool(np.all(np.isfinite(prediction)))))
            outputs.append((meta.target, np.asarray(curve.x), np.asarray(curve.y), prediction))
        groups.append((material, family, outputs, rows))

    all_rows = [row for _, _, _, rows in groups for row in rows]
    with (OUTPUT / "reverse_cyclic_baseline.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("material", "family", "target_strain", "points", "rmse_mpa", "max_error_mpa", "finite"))
        writer.writerows(all_rows)

    for material, family, outputs, _ in groups:
        cols = 3
        rows_count = int(np.ceil(len(outputs) / cols))
        fig, axes = plt.subplots(rows_count, cols, figsize=(11, 3.3 * rows_count), squeeze=False)
        for axis, (target, strain, measured, predicted) in zip(axes.flat, outputs):
            axis.plot(strain, measured, "o", ms=2.1, label="experiment")
            axis.plot(strain, predicted, "-", lw=1.35, label="JAX/VUMAT")
            axis.set_title(f"target={target if target is not None else 'progressive'}")
            axis.set_xlabel("compressive true strain"); axis.set_ylabel("compressive stress (MPa)"); axis.grid(alpha=.2)
        for axis in axes.flat[len(outputs):]: axis.axis("off")
        if outputs: axes.flat[0].legend(frameon=False)
        fig.tight_layout(); fig.savefig(OUTPUT / f"{material.lower()}_{family}_baseline.png", dpi=220); fig.savefig(OUTPUT / f"{material.lower()}_{family}_baseline.pdf"); plt.close(fig)

    recovery_rows, recovery_outputs = [], []
    recovery_curves = [c for c in curves if classify_curve(c).family == "recovery"]
    loading_curves = [c for c in curves if classify_curve(c).family in {"load_unload", "load_unload_recovery"}]
    for recovery in recovery_curves:
        meta = classify_curve(recovery)
        params = pmma_parameters() if meta.material == "PMMA" else pc_parameters()
        candidates = [c for c in loading_curves if classify_curve(c).material == meta.material
                      and classify_curve(c).target is not None and abs(classify_curve(c).target - meta.target) < 1.0e-8]
        candidates.sort(key=lambda c: classify_curve(c).family != "load_unload_recovery")
        if not candidates:
            continue
        loading = candidates[0]
        history_substeps = 100 if meta.material == "PC" and (meta.target or 0.0) >= 0.6 else 20
        _, loading_result = simulate_strain_curve(loading, params, substeps=history_substeps, scheme="umat")
        state = terminal_state(loading_result.state)
        residual_strain = float(-jnp.log(state.F[1, 1]))
        residual_lateral = float(jnp.log(state.F[0, 0]))
        time = strictly_increasing_time(np.asarray(recovery.x))
        dt = np.maximum(np.diff(time, prepend=time[:1]), 1.0e-5)
        zero = jnp.zeros(len(time))
        recovered = run_stress_history(
            zero, jnp.asarray(dt), params, state, strain0=residual_strain,
            lateral0=residual_lateral, scheme="umat", substeps=100,
        )
        prediction = residual_strain - np.asarray(recovered.strain)
        rmse, maximum = score(prediction, recovery.y)
        recovery_rows.append((meta.material, meta.target, len(time), rmse, maximum, bool(np.all(np.isfinite(prediction)))))
        recovery_outputs.append((meta.material, meta.target, time, np.asarray(recovery.y), prediction))
    with (OUTPUT / "recovery_baseline.csv").open("w", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(("material", "prestrain", "points", "rmse_strain", "max_error_strain", "finite")); writer.writerows(recovery_rows)
    fig, axes = plt.subplots(1, max(len(recovery_outputs), 1), figsize=(4.2 * max(len(recovery_outputs), 1), 3.5), squeeze=False)
    for axis, (material, target, time, measured, predicted) in zip(axes.flat, recovery_outputs):
        axis.plot(time, measured, "o", ms=3, label="experiment"); axis.plot(time, predicted, "-", lw=1.5, label="retained-state JAX")
        axis.set_title(f"{material}, prestrain={target:g}"); axis.set_xlabel("recovery time (s)"); axis.set_ylabel("recovered true strain"); axis.grid(alpha=.2)
    if recovery_outputs: axes.flat[0].legend(frameon=False)
    fig.tight_layout(); fig.savefig(OUTPUT / "recovery_baseline.png", dpi=220); fig.savefig(OUTPUT / "recovery_baseline.pdf"); plt.close(fig)
    print(f"strain paths={len(all_rows)} recovery={len(recovery_rows)}")


if __name__ == "__main__":
    main()
