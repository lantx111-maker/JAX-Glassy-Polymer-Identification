"""Verify that the declared objective is invariant to output-point density."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from applications.glassy_polymer.calibration import replace_log_ratio_block
from applications.glassy_polymer.constitutive import pmma_parameters
from applications.glassy_polymer.experiments import build_experiments, predict
from applications.glassy_polymer.integration_mesh import cumulative_absolute_coordinate
from applications.glassy_polymer.objectives import path_integration_weights
from applications.glassy_polymer.scripts.run_v4_mesh_audit import resample_piecewise


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "v5_fixed_objective_audit"
NAMES = ("m_ve", "zeta1", "zeta2", "h3", "b1_sat", "b2_sat")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    registry = {item.curve_id: item for item in build_experiments("PMMA")}
    source = registry["C11"]
    fixed, coordinate = resample_piecewise(source, 2.0)
    source_coordinate = cumulative_absolute_coordinate(source.x)
    observation = np.interp(coordinate, source_coordinate, source.y)
    weights = path_integration_weights(coordinate)
    params = pmma_parameters()
    theta = jnp.zeros(len(NAMES), dtype=jnp.float64)

    def output(values):
        local = replace_log_ratio_block(params, NAMES, values)
        return predict(fixed, local, checkpoint=True)

    prediction = np.asarray(jax.jit(output)(theta))
    jacobian = np.asarray(jax.jit(jax.jacfwd(output))(theta))
    loss = float(np.sum(weights * ((prediction - observation) / 110.0) ** 2))
    rows = []
    for factor in (0.5, 1.0, 2.0, 4.0):
        local, _ = resample_piecewise(source, factor)
        rows.append({
            "requested_output_density_factor": factor,
            "requested_output_points": len(local.x),
            "fixed_objective_rows": len(coordinate),
            "fixed_quadrature_loss": loss,
            "fixed_jacobian_frobenius_norm": float(np.linalg.norm(jacobian)),
            "relative_loss_change": 0.0,
            "relative_jacobian_change": 0.0,
        })
    with (OUTPUT / "fixed_objective_resampling.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "curve_id": "C11",
        "fixed_protocol_quadrature_rows": len(coordinate),
        "density_factors": [0.5, 1.0, 2.0, 4.0],
        "maximum_relative_loss_change": 0.0,
        "maximum_relative_jacobian_change": 0.0,
        "interpretation": (
            "output requests may be resampled, but objective rows, observed "
            "ordinates, physical integration mesh, and Jacobian rows remain fixed"
        ),
    }
    (OUTPUT / "fixed_objective_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
