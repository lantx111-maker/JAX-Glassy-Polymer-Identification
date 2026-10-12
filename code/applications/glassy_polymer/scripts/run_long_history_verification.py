"""Nested implicit-gradient verification and a two-parameter synthetic twin."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from applications.glassy_polymer.calibration import replace_log_ratio_block
from applications.glassy_polymer.constitutive import pmma_parameters
from applications.glassy_polymer.objectives import (
    ProtocolResidual,
    path_integration_weights,
    protocol_balanced_loss,
)
from applications.glassy_polymer.scripts.calibrate_full_history import (
    REGULARIZATION,
    build_experiments,
    predict,
)


OUTPUT = Path(__file__).resolve().parents[1] / "results" / "long_history_verification"
NAMES = ("gamma0", "s0")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    base = pmma_parameters()
    experiments = build_experiments("PMMA")
    creep = next(item for item in experiments if item.family == "creep" and item.target == 42.0)
    unloading = next(item for item in experiments if item.family == "load_unload" and item.target == 0.2)

    six_names = ("gamma0", "m0", "s0", "h2", "s1", "s2")
    theta0 = jnp.zeros(len(six_names))
    direction = jnp.asarray([0.37, -0.41, 0.29, 0.13, -0.51, 0.57])
    direction = direction / jnp.linalg.norm(direction)

    def long_objective(theta, checkpoint):
        params = replace_log_ratio_block(base, six_names, theta)
        prediction = predict(creep, params, REGULARIZATION, checkpoint)
        residual = (prediction - jnp.asarray(creep.y)) / max(np.max(creep.y), 0.02)
        return jnp.mean(residual * residual)

    direct_gradient = jax.grad(lambda value: long_objective(value, False))(theta0)
    checkpoint_gradient = jax.grad(lambda value: long_objective(value, True))(theta0)
    adjoint_directional = float(jnp.vdot(checkpoint_gradient, direction))
    direct_difference = float(
        jnp.linalg.norm(direct_gradient - checkpoint_gradient)
        / jnp.maximum(jnp.linalg.norm(direct_gradient), 1.0e-30)
    )
    gradient_rows = []
    for step in (1.0e-2, 3.0e-3, 1.0e-3, 3.0e-4, 1.0e-4):
        plus = float(long_objective(theta0 + step * direction, True))
        minus = float(long_objective(theta0 - step * direction, True))
        finite_difference = (plus - minus) / (2.0 * step)
        error = abs(finite_difference - adjoint_directional) / max(
            abs(finite_difference), abs(adjoint_directional), 1.0e-14
        )
        gradient_rows.append((step, adjoint_directional, finite_difference,
                              error, direct_difference))
    with (OUTPUT / "nested_implicit_gradient_audit.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("directional_step", "implicit_checkpoint_adjoint",
                         "central_difference", "relative_error",
                         "direct_vs_checkpoint_relative_error"))
        writer.writerows(gradient_rows)

    truth_theta = jnp.log(jnp.asarray([1.08, 0.95]))
    truth = replace_log_ratio_block(base, NAMES, truth_theta)
    twin_measurements = {
        "creep": predict(creep, truth, REGULARIZATION, False),
        "load_unload": predict(unloading, truth, REGULARIZATION, False),
    }
    twin_scales = {
        "creep": max(float(np.max(np.abs(np.asarray(twin_measurements["creep"])))), 0.02),
        "load_unload": 110.0,
    }

    def twin_objective(theta):
        params = replace_log_ratio_block(base, NAMES, theta)
        blocks = []
        for experiment in (creep, unloading):
            measurement = twin_measurements[experiment.family]
            scale = twin_scales[experiment.family]
            blocks.append(ProtocolResidual(
                "PMMA", experiment.family, experiment.family,
                experiment.curve_id, experiment.label,
                predict(experiment, params, REGULARIZATION, True), measurement,
                jnp.asarray(path_integration_weights(experiment.x)), scale,
            ))
        return protocol_balanced_loss(blocks)[0]

    value_gradient = jax.jit(jax.value_and_grad(twin_objective))
    trace = []

    def scipy_objective(theta):
        value, gradient = value_gradient(jnp.asarray(theta))
        trace.append((len(trace) + 1, float(value), *np.exp(theta)))
        return float(value), np.asarray(gradient, dtype=float)

    result = minimize(
        scipy_objective, np.zeros(len(NAMES)), jac=True, method="L-BFGS-B",
        bounds=[(-np.log(1.3), np.log(1.3))] * len(NAMES),
        options={"maxiter": 25, "ftol": 1.0e-14, "gtol": 1.0e-9},
    )
    recovered_ratios = np.exp(result.x)
    truth_ratios = np.exp(np.asarray(truth_theta))
    relative_errors = np.abs(recovered_ratios - truth_ratios) / truth_ratios
    with (OUTPUT / "synthetic_twin_parameters.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("parameter", "truth_ratio", "recovered_ratio", "relative_error"))
        for name, truth_ratio, recovered, error in zip(
                NAMES, truth_ratios, recovered_ratios, relative_errors):
            writer.writerow((name, truth_ratio, recovered, error))
        writer.writerow(("optimizer_success", "", result.success, result.message))
        writer.writerow(("final_objective", "", result.fun, ""))
    with (OUTPUT / "synthetic_twin_trace.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("evaluation", "loss", *[f"{name}_ratio" for name in NAMES]))
        writer.writerows(trace)
    summary = {
        "finest_directional_step": gradient_rows[-1][0],
        "finest_directional_relative_error": gradient_rows[-1][3],
        "direct_vs_checkpoint_relative_error": direct_difference,
        "maximum_twin_parameter_relative_error": float(np.max(relative_errors)),
        "twin_final_objective": float(result.fun),
        "optimizer_reported_success": bool(result.success),
        "optimizer_message": str(result.message),
        "optimizer_status_interpretation": (
            "the quantitative recovery gate controls; a line-search status is "
            "not required after the objective reaches numerical zero"
        ),
    }
    summary["passes_declared_gate"] = bool(
        summary["finest_directional_relative_error"] < 1.0e-6
        and summary["direct_vs_checkpoint_relative_error"] < 1.0e-10
        and summary["maximum_twin_parameter_relative_error"] < 1.0e-5
        and summary["twin_final_objective"] < 1.0e-12
    )
    (OUTPUT / "long_history_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print("gradient", gradient_rows)
    print("twin", result.success, truth_ratios, recovered_ratios, relative_errors, result.fun)
    print(json.dumps(summary, indent=2))
    if not summary["passes_declared_gate"]:
        raise SystemExit("long-history derivative/synthetic-twin gate failed")


if __name__ == "__main__":
    main()
