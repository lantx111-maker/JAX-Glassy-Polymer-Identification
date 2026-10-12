"""Observable retained-response design with refined and perturbed validation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from applications.glassy_polymer.constitutive import pmma_parameters
from applications.glassy_polymer.inverse_design import (
    decode_physical_history,
    observable_retained_response,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "observable_inverse_design"
TARGET = 0.07

mpl.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 8.5,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
})


def logit_fraction(value, upper):
    fraction = np.clip(np.asarray(value) / upper, 1.0e-5, 1.0 - 1.0e-5)
    return np.log(fraction / (1.0 - fraction))


def accepted_pmma_parameters():
    params = pmma_parameters()
    path = ROOT / "results" / "canonical_calibration" / "pmma_parameters.csv"
    decision_path = ROOT / "results" / "canonical_calibration" / "pmma_decision.json"
    if not path.exists():
        return params
    if decision_path.exists() and not json.loads(decision_path.read_text())["accepted"]:
        return params
    rows = list(csv.DictReader(path.open()))
    accepted = next(
        (row for row in rows if row["parameter"] == "exact_training_family_nrmse"),
        None,
    )
    if accepted is None or accepted["accepted"].lower() != "true":
        return params
    values = {
        row["parameter"]: float(row["candidate"])
        for row in rows if row["parameter"] in params._fields
    }
    return params._replace(**values)


def encode_start(knots, durations):
    duration_fraction = (np.asarray(durations) - 20.0) / (400.0 - 20.0)
    duration_fraction = np.clip(duration_fraction, 1.0e-5, 1.0 - 1.0e-5)
    return np.concatenate((
        logit_fraction(knots, 0.55),
        np.log(duration_fraction / (1.0 - duration_fraction)),
    ))


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    params = accepted_pmma_parameters()
    profiles = (
        ("minimum_work", 5.0e-4, 0.0),
        ("balanced", 1.0e-4, 2.0e-3),
        ("minimum_time", 1.0e-5, 2.0e-1),
    )
    starts = (
        encode_start([0.12, 0.04, 0.18, 0.06], [180, 120, 220, 160]),
        encode_start([0.15, 0.08, 0.12, 0.05], [80, 260, 100, 240]),
        encode_start([0.08, 0.16, 0.05, 0.10], [300, 70, 280, 90]),
    )
    trial_rows = []
    selected = []

    value_and_grad = jax.jit(jax.value_and_grad(
        lambda raw, work_weight, time_weight: observable_retained_response(
            raw, params, TARGET, n_per_segment=5, loading_substeps=16,
            recovery_steps=16, recovery_substeps=16,
            work_weight=work_weight, time_weight=time_weight,
        ).objective,
        argnums=0,
    ))

    for profile, work_weight, time_weight in profiles:

        def scipy_objective(raw):
            value, gradient = value_and_grad(
                jnp.asarray(raw), jnp.asarray(work_weight), jnp.asarray(time_weight)
            )
            value = float(value)
            gradient = np.asarray(gradient, dtype=float)
            if not np.isfinite(value) or not np.all(np.isfinite(gradient)):
                return 1.0e8, np.zeros_like(raw)
            return value, gradient

        candidates = []
        for start_index, start in enumerate(starts):
            result = minimize(
                scipy_objective, start, jac=True, method="L-BFGS-B",
                bounds=[(-6.0, 6.0)] * 8,
                options={"maxiter": 30, "ftol": 1.0e-10, "gtol": 1.0e-7},
            )
            check = observable_retained_response(
                jnp.asarray(result.x), params, TARGET, n_per_segment=5,
                loading_substeps=16, recovery_steps=16, recovery_substeps=16,
                work_weight=work_weight, time_weight=time_weight,
            )
            candidates.append((result, check))
            trial_rows.append((
                profile, start_index, float(check.objective),
                float(check.residual_strain), float(check.reload_stiffness),
                float(check.peak_stress), float(check.input_work),
                float(check.net_work), float(check.cumulative_absolute_work),
                float(check.maximum_rate), float(check.time[-1]),
                float(check.total_elapsed_time),
                result.success, result.nit, *np.asarray(result.x),
            ))

        finite_candidates = [
            item for item in candidates if np.isfinite(float(item[1].objective))
        ]
        if not finite_candidates:
            raise RuntimeError(f"No finite inverse-design start for {profile}")
        optimization, _ = min(
            finite_candidates, key=lambda item: float(item[1].objective)
        )
        refined = observable_retained_response(
            jnp.asarray(optimization.x), params, TARGET, n_per_segment=5,
            loading_substeps=24, recovery_steps=24, recovery_substeps=24,
            work_weight=work_weight, time_weight=time_weight,
        )
        selected.append((profile, work_weight, time_weight, optimization.x, refined))

    with (OUTPUT / "observable_design_multistart.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow((
            "profile", "start", "objective", "residual_strain_after_600s",
            "reload_stiffness_mpa", "peak_stress_mpa", "input_work_mpa",
            "net_work_mpa", "cumulative_absolute_work_mpa",
            "maximum_rate_per_s", "loading_program_time_s", "total_elapsed_time_s",
            "optimizer_success",
            "iterations", *(f"raw_{i + 1}" for i in range(8)),
        ))
        writer.writerows(trial_rows)

    validation_rows = []
    robustness: dict[str, list[float]] = {}
    perturbations = (
        ("nominal", params, 24),
        ("nominal_16_substeps", params, 16),
        ("nominal_32_substeps", params, 32),
        ("fast_soft", params._replace(gamma0=params.gamma0 * 1.02, s0=params.s0 * 0.98), 24),
        ("slow_hard", params._replace(gamma0=params.gamma0 * 0.98, s0=params.s0 * 1.02), 24),
    )
    for profile, work_weight, time_weight, raw, _ in selected:
        robustness[profile] = []
        for perturbation, local_params, refined_substeps in perturbations:
            check = observable_retained_response(
                jnp.asarray(raw), local_params, TARGET, n_per_segment=5,
                loading_substeps=refined_substeps, recovery_steps=24,
                recovery_substeps=refined_substeps,
                work_weight=work_weight, time_weight=time_weight,
            )
            robustness[profile].append(float(check.residual_strain))
            validation_rows.append((
                profile, perturbation, TARGET, float(check.residual_strain),
                float(check.reload_stiffness), float(check.peak_stress),
                float(check.input_work), float(check.net_work),
                float(check.cumulative_absolute_work), float(check.maximum_rate),
                float(check.time[-1]), float(check.total_elapsed_time),
                refined_substeps, refined_substeps,
            ))
    with (OUTPUT / "observable_design_refined_robustness.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow((
            "profile", "parameter_case", "target_residual_strain",
            "residual_strain_after_600s", "reload_stiffness_mpa",
            "peak_stress_mpa", "input_work_mpa", "net_work_mpa",
            "cumulative_absolute_work_mpa", "maximum_rate_per_s",
            "loading_program_time_s", "total_elapsed_time_s",
            "loading_substeps", "recovery_substeps",
        ))
        writer.writerows(validation_rows)

    with (OUTPUT / "observable_design_controls.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow((
            "profile", "knot_1", "knot_2", "knot_3", "knot_4",
            "duration_1_s", "duration_2_s", "duration_3_s", "duration_4_s",
            "loading_program_time_s", "total_elapsed_time_s",
            "residual_strain_after_600s", "reload_stiffness_mpa",
            "peak_stress_mpa", "input_work_mpa", "net_work_mpa",
            "cumulative_absolute_work_mpa", "maximum_rate_per_s",
        ))
        for profile, _, _, raw, result in selected:
            decoded_time, decoded_strain = decode_physical_history(
                jnp.asarray(raw), n_per_segment=5
            )
            knot_index = np.asarray([5, 10, 15, 20])
            knots = np.asarray(decoded_strain)[knot_index]
            durations = np.diff(np.asarray(decoded_time)[np.asarray([0, 5, 10, 15, 20])])
            writer.writerow((
                profile, *knots, *durations, float(result.time[-1]),
                float(result.total_elapsed_time),
                float(result.residual_strain), float(result.reload_stiffness),
                float(result.peak_stress), float(result.input_work),
                float(result.net_work), float(result.cumulative_absolute_work),
                float(result.maximum_rate),
            ))

    fig, axes = plt.subplots(2, 2, figsize=(9.3, 6.7), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(selected)))
    for color, (profile, _, _, _, result) in zip(colors, selected):
        label = profile.replace("minimum_work", "work-weighted").replace(
            "minimum_time", "time-weighted").replace("_", " ")
        axes[0, 0].plot(np.asarray(result.time), np.asarray(result.strain), "-",
                        lw=1.5, color=color, label=label)
        axes[0, 0].plot(np.asarray(result.time)[::5], np.asarray(result.strain)[::5],
                        "o", ms=3.2, color=color)
    axes[0, 0].set_xlabel("elapsed time during loading program (s)")
    axes[0, 0].set_ylabel("compressive true strain")
    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[0, 0].grid(alpha=0.25)

    names = [item[0].replace("minimum_work", "work\nweighted").replace(
        "minimum_time", "time\nweighted").replace("_", "\n") for item in selected]
    residuals = [float(item[4].residual_strain) for item in selected]
    axes[0, 1].axhline(TARGET, color="0.45", ls="--", label="target")
    axes[0, 1].plot(range(len(names)), residuals, "o", color="#2266aa")
    axes[0, 1].set_xticks(range(len(names)), names)
    axes[0, 1].set_ylabel("residual strain after 600 s")
    axes[0, 1].grid(alpha=0.25)

    work = [abs(float(item[4].input_work)) for item in selected]
    duration = [float(item[4].time[-1]) for item in selected]
    axes[1, 0].scatter(duration, work, c=colors, s=55)
    for name, x, y in zip(names, duration, work):
        axes[1, 0].annotate(name.replace("\n", " "), (x, y), xytext=(4, 4),
                            textcoords="offset points", fontsize=7)
    axes[1, 0].set_xlabel("loading-program duration (s)")
    axes[1, 0].set_ylabel("cumulative positive work (MPa)")
    axes[1, 0].grid(alpha=0.25)

    positions = np.arange(len(selected))
    for index, (profile, values) in enumerate(robustness.items()):
        axes[1, 1].plot([index] * len(values), values, "o", color=colors[index])
        axes[1, 1].plot([index - 0.12, index + 0.12], [np.mean(values)] * 2,
                        color=colors[index], lw=2)
    axes[1, 1].axhline(TARGET, color="0.45", ls="--")
    axes[1, 1].set_xticks(positions, names)
    axes[1, 1].set_ylabel("refined/perturbed residual strain")
    axes[1, 1].grid(alpha=0.25)
    fig.savefig(OUTPUT / "observable_retained_response_design.pdf")
    fig.savefig(OUTPUT / "observable_retained_response_design.png", dpi=450)
    plt.close(fig)
    print(validation_rows)


if __name__ == "__main__":
    main()
