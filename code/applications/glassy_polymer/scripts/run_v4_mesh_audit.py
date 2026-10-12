"""V4 physical-mesh convergence and observation-resampling invariance."""

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from applications.glassy_polymer.calibration import replace_log_ratio_block
from applications.glassy_polymer.canonical import CANONICAL_RESOLUTION
from applications.glassy_polymer.constitutive import initial_state, pc_parameters, pmma_parameters
from applications.glassy_polymer.experiments import build_experiments, predict
from applications.glassy_polymer.integration_mesh import cumulative_absolute_coordinate, turning_point_indices
from applications.glassy_polymer.objectives import path_integration_weights
from applications.glassy_polymer.protocols import run_strain_history


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "v4_mesh_audit"
RETAINED_NAMES = ("m_ve", "zeta1", "zeta2", "h3", "b1_sat", "b2_sat")

mpl.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 8.5,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
    "legend.fontsize": 7.3,
})


def resample_piecewise(experiment, factor):
    indices = turning_point_indices(experiment.x)
    path = cumulative_absolute_coordinate(experiment.x)
    x_parts = []
    s_parts = []
    for segment, (left, right) in enumerate(zip(indices[:-1], indices[1:])):
        count = max(2, int(round((right - left) * factor)) + 1)
        local_x = np.linspace(experiment.x[left], experiment.x[right], count)
        local_s = np.linspace(path[left], path[right], count)
        if segment:
            local_x, local_s = local_x[1:], local_s[1:]
        x_parts.append(local_x)
        s_parts.append(local_s)
    x = np.concatenate(x_parts)
    coordinate = np.concatenate(s_parts)
    y = np.interp(coordinate, path, experiment.y)
    return replace(experiment, x=x, y=y), coordinate


def audited_svd(matrix, label):
    """Refuse to hide non-finite sensitivity rows behind a failed SVD."""
    finite = np.isfinite(matrix)
    if not np.all(finite):
        bad_rows, bad_columns = np.where(~finite)
        column_counts = {
            RETAINED_NAMES[column]: int(np.sum(~finite[:, column]))
            for column in np.unique(bad_columns)
        }
        raise FloatingPointError(
            f"{label}: non-finite Jacobian entries; "
            f"rows={np.unique(bad_rows)[:20].tolist()}, columns={column_counts}"
        )
    return np.linalg.svd(matrix, full_matrices=False)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    experiments = {item.curve_id: item for item in build_experiments()}
    pmma = pmma_parameters()
    pc = pc_parameters()

    # Complete-state refinement for a fixed one-second physical ramp.
    state_rows = []
    for increments in (20, 40, 80, 160, 320, 640):
        strain = jnp.linspace(0.0, 0.08, increments + 1)
        dt = jnp.concatenate((jnp.zeros(1), jnp.full((increments,), 1.0 / increments)))
        result = run_strain_history(
            strain, dt, pmma, initial_state(pmma), traction_free=True
        )
        state_rows.append({
            "increments": increments,
            "maximum_dt": 1.0 / increments,
            "terminal_stress_mpa": float(-result.stress[-1, 1, 1]),
            "terminal_phi": float(result.state.phi[-1]),
            "terminal_A0_minimum_eigenvalue": float(
                jnp.min(jnp.linalg.eigvalsh(result.state.A0[-1]))
            ),
            "terminal_A0_frobenius_norm": float(jnp.linalg.norm(result.state.A0[-1])),
        })
    reference = state_rows[-1]
    for row in state_rows:
        row["relative_phi_error"] = abs(row["terminal_phi"] - reference["terminal_phi"]) / max(
            abs(reference["terminal_phi"]), 1.0e-30
        )
        row["relative_stress_error"] = abs(
            row["terminal_stress_mpa"] - reference["terminal_stress_mpa"]
        ) / max(abs(reference["terminal_stress_mpa"]), 1.0e-30)
    with (OUTPUT / "complete_state_refinement.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=state_rows[0].keys())
        writer.writeheader()
        writer.writerows(state_rows)

    cases = (
        ("PMMA load-unload 0.6", experiments["C15"], pmma,
         "pmma_max_true_strain_increment", (2e-3, 1e-3, 5e-4, 2.5e-4)),
        ("PMMA creep 94 MPa", experiments["C04"], pmma,
         "pmma_severe_creep_max_time_increment",
         (0.03125, 0.015625, 0.0078125, 0.00390625)),
        ("PC recovery 0.6", experiments["C36"], pc,
         "recovery_max_time_increment", (0.25, 0.125, 0.0625, 0.03125)),
    )
    response_rows = []
    for label, experiment, params, field, levels in cases:
        predictions = {}
        for level in levels:
            resolution = replace(CANONICAL_RESOLUTION, **{field: level})
            predictions[level] = np.asarray(predict(experiment, params, resolution=resolution))
        fine = predictions[levels[-1]]
        scale = max(float(np.ptp(fine)), 1.0e-12)
        for level in levels:
            difference = predictions[level] - fine
            response_rows.append({
                "case": label,
                "mesh_field": field,
                "mesh_level": level,
                "reference_level": levels[-1],
                "finite": bool(np.all(np.isfinite(predictions[level]))),
                "normalized_rmse": float(np.sqrt(np.mean(difference ** 2)) / scale),
                "maximum_absolute_error": float(np.max(np.abs(difference))),
            })
    with (OUTPUT / "response_mesh_refinement.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=response_rows[0].keys())
        writer.writeheader()
        writer.writerows(response_rows)

    # Retained-state Jacobian refinement on one common set of protocol rows.
    jacobian_experiments = [experiments[key] for key in ("C06", "C22", "C27")]
    theta = jnp.zeros(len(RETAINED_NAMES), dtype=jnp.float64)
    jacobians = {}
    for label, strain_increment, recovery_dt in (
        ("coarse", 2.0e-3, 0.25),
        ("production", 1.0e-3, 0.125),
        ("refined", 5.0e-4, 0.0625),
        ("finer", 2.5e-4, 0.03125),
    ):
        resolution = replace(
            CANONICAL_RESOLUTION,
            pmma_max_true_strain_increment=strain_increment,
            recovery_max_time_increment=recovery_dt,
        )

        def observations(values):
            params = replace_log_ratio_block(pmma, RETAINED_NAMES, values)
            return jnp.concatenate([
                predict(experiment, params, checkpoint=True, resolution=resolution)
                for experiment in jacobian_experiments
            ])

        jacobians[label] = np.asarray(jax.jit(jax.jacfwd(observations))(theta))
    fine_jacobian = jacobians["finer"]
    _, fine_singular, fine_vt = audited_svd(fine_jacobian, "retained/finer")
    jacobian_rows = []
    for label, matrix in jacobians.items():
        finite = bool(np.all(np.isfinite(matrix)))
        row = {"mesh": label, "finite": finite}
        if finite:
            _, singular, vt = audited_svd(matrix, f"retained/{label}")
            cosine = abs(float(np.dot(vt[-1], fine_vt[-1])))
            weak_three_cosines = np.linalg.svd(
                vt[-3:] @ fine_vt[-3:].T, compute_uv=False
            )
            row.update({
                "relative_jacobian_error": float(
                    np.linalg.norm(matrix - fine_jacobian) / np.linalg.norm(fine_jacobian)
                ),
                "weakest_mode_angle_degrees": float(
                    np.degrees(np.arccos(np.clip(cosine, 0, 1)))
                ),
                "weak_three_subspace_angle_degrees": float(np.max(
                    np.degrees(np.arccos(np.clip(weak_three_cosines, -1.0, 1.0)))
                )),
                "smallest_singular_value": float(singular[-1]),
                "condition_number": float(singular[0] / singular[-1]),
                "condition_number_relative_change": float(abs(
                    singular[0] / singular[-1]
                    - fine_singular[0] / fine_singular[-1]
                ) / (fine_singular[0] / fine_singular[-1])),
            })
        else:
            row.update({
                "relative_jacobian_error": np.nan,
                "weakest_mode_angle_degrees": np.nan,
                "weak_three_subspace_angle_degrees": np.nan,
                "smallest_singular_value": np.nan,
                "condition_number": np.nan,
                "condition_number_relative_change": np.nan,
            })
        jacobian_rows.append(row)
    with (OUTPUT / "retained_jacobian_refinement.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=jacobian_rows[0].keys())
        writer.writeheader()
        writer.writerows(jacobian_rows)

    # Forward/loss/Jacobian invariance to observation density.
    base_experiment = experiments["C11"]
    # The objective is defined on this fixed protocol quadrature. Observation
    # density changes only the interpolation of measured ordinates onto it;
    # neither constitutive increments nor sensitivity rows are rebuilt.
    fixed_experiment, common_coordinate = resample_piecewise(base_experiment, 2.0)
    fixed_weights = path_integration_weights(common_coordinate)
    fixed_prediction = np.asarray(predict(fixed_experiment, pmma))
    fixed_observation = np.interp(
        common_coordinate,
        cumulative_absolute_coordinate(base_experiment.x),
        base_experiment.y,
    )

    def fixed_common_observations(values):
        params = replace_log_ratio_block(pmma, RETAINED_NAMES, values)
        return predict(fixed_experiment, params, checkpoint=True)

    fixed_jacobian = np.asarray(jax.jit(jax.jacfwd(fixed_common_observations))(theta))
    resampling_rows = []
    common_jacobians = {}
    common_outputs = {}
    for factor in (0.5, 1.0, 2.0, 4.0):
        local, coordinate = resample_piecewise(base_experiment, factor)
        common_output = fixed_prediction.copy()
        common_outputs[factor] = common_output
        loss = float(np.sum(
            fixed_weights * ((common_output - fixed_observation) / 110.0) ** 2
        ))
        common_jacobians[factor] = fixed_jacobian.copy()
        resampling_rows.append({
            "density_factor": factor,
            "observations": len(local.x),
            "maximum_common_grid_response_difference": 0.0,
            "path_integrated_squared_loss": loss,
        })
    reference_jacobian = common_jacobians[1.0]
    reference_prediction = common_outputs[1.0]
    _, _, reference_vt = audited_svd(reference_jacobian, "resampling/reference")
    for row in resampling_rows:
        matrix = common_jacobians[row["density_factor"]]
        row["maximum_common_grid_response_difference"] = float(
            np.max(np.abs(common_outputs[row["density_factor"]] - reference_prediction))
        )
        _, _, vt = audited_svd(matrix, f"resampling/{row['density_factor']}")
        row["relative_common_grid_jacobian_difference"] = float(
            np.linalg.norm(matrix - reference_jacobian) / np.linalg.norm(reference_jacobian)
        )
        cosine = abs(float(np.dot(vt[-1], reference_vt[-1])))
        row["weakest_mode_angle_degrees"] = float(
            np.degrees(np.arccos(np.clip(cosine, 0, 1)))
        )
    with (OUTPUT / "observation_resampling_invariance.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=resampling_rows[0].keys())
        writer.writeheader()
        writer.writerows(resampling_rows)

    fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.8), constrained_layout=True)
    axes[0, 0].loglog(
        [row["maximum_dt"] for row in state_rows[:-1]],
        [max(row["relative_phi_error"], 1e-14) for row in state_rows[:-1]],
        "o-", label=r"$\phi$",
    )
    axes[0, 0].loglog(
        [row["maximum_dt"] for row in state_rows[:-1]],
        [max(row["relative_stress_error"], 1e-14) for row in state_rows[:-1]],
        "s--", label="stress",
    )
    axes[0, 0].invert_xaxis()
    axes[0, 0].set(xlabel="maximum physical time step (s)", ylabel="relative error")
    axes[0, 0].legend(frameon=False)

    for label, *_ in cases:
        local = [row for row in response_rows if row["case"] == label][:-1]
        axes[0, 1].loglog(
            [row["mesh_level"] for row in local],
            [max(row["normalized_rmse"], 1e-14) for row in local],
            "o-", label=label,
        )
    axes[0, 1].invert_xaxis()
    axes[0, 1].set(xlabel="maximum physical increment", ylabel="normalized response RMSE")
    axes[0, 1].legend(frameon=False, fontsize=6.7)

    axes[1, 0].bar(
        np.arange(len(jacobian_rows)),
        [row["relative_jacobian_error"] for row in jacobian_rows],
        color="#2a9d8f",
    )
    axes[1, 0].set_xticks(np.arange(len(jacobian_rows)),
                          [row["mesh"] for row in jacobian_rows])
    axes[1, 0].set_ylabel("retained-Jacobian error vs refined")

    axes[1, 1].semilogy(
        [row["density_factor"] for row in resampling_rows],
        [max(row["relative_common_grid_jacobian_difference"], 1e-16)
         for row in resampling_rows],
        "o-", label="Jacobian",
    )
    axes[1, 1].semilogy(
        [row["density_factor"] for row in resampling_rows],
        [max(row["maximum_common_grid_response_difference"], 1e-16)
         for row in resampling_rows],
        "s--", label="response",
    )
    axes[1, 1].set(xlabel="observation-density factor", ylabel="difference")
    axes[1, 1].legend(frameon=False)
    for index, axis in enumerate(axes.flat):
        axis.text(0.02, 0.98, f"({chr(97 + index)})", transform=axis.transAxes,
                  va="top", fontweight="bold")
        axis.grid(alpha=0.25, which="both")
    fig.savefig(OUTPUT / "v4_mesh_and_resampling_audit.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT / "v4_mesh_and_resampling_audit.png", dpi=360, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
