"""Refined-to-ultrafine convergence gate for the retained-state Jacobian."""

from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from applications.glassy_polymer.calibration import replace_log_ratio_block
from applications.glassy_polymer.canonical import CANONICAL_RESOLUTION
from applications.glassy_polymer.constitutive import pmma_parameters
from applications.glassy_polymer.experiments import build_experiments, predict


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "v5_sensitivity_gate"
NAMES = ("m_ve", "zeta1", "zeta2", "h3", "b1_sat", "b2_sat")
PATHS = ("C06", "C22", "C27")


def _principal_angle(vt_a, vt_b, dimension):
    singular = np.linalg.svd(vt_a[-dimension:] @ vt_b[-dimension:].T,
                             compute_uv=False)
    return float(np.max(np.degrees(np.arccos(np.clip(singular, -1.0, 1.0)))))


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    registry = {item.curve_id: item for item in build_experiments("PMMA")}
    experiments = [registry[curve_id] for curve_id in PATHS]
    base = pmma_parameters()
    theta = jnp.zeros(len(NAMES), dtype=jnp.float64)
    levels = {
        "sensitivity": (2.5e-4, 3.125e-2),
        "ultrafine_gate": (1.25e-4, 1.5625e-2),
    }
    matrices = {}
    for label, (strain_increment, recovery_dt) in levels.items():
        resolution = replace(
            CANONICAL_RESOLUTION,
            pmma_max_true_strain_increment=strain_increment,
            recovery_max_time_increment=recovery_dt,
        )

        def observations(values):
            params = replace_log_ratio_block(base, NAMES, values)
            return jnp.concatenate([
                predict(experiment, params, checkpoint=True, resolution=resolution)
                for experiment in experiments
            ])

        matrix = np.asarray(jax.jit(jax.jacfwd(observations))(theta))
        if not np.all(np.isfinite(matrix)):
            raise FloatingPointError(f"non-finite {label} Jacobian")
        matrices[label] = matrix
        print(label, matrix.shape, flush=True)

    sensitivity = matrices["sensitivity"]
    ultrafine = matrices["ultrafine_gate"]
    _, singular_sensitivity, vt_sensitivity = np.linalg.svd(sensitivity, full_matrices=False)
    _, singular_ultrafine, vt_ultrafine = np.linalg.svd(ultrafine, full_matrices=False)
    condition_sensitivity = singular_sensitivity[0] / singular_sensitivity[-1]
    condition_ultrafine = singular_ultrafine[0] / singular_ultrafine[-1]
    summary = {
        "paths": list(PATHS),
        "parameters": list(NAMES),
        "sensitivity_strain_increment": levels["sensitivity"][0],
        "sensitivity_recovery_time_increment": levels["sensitivity"][1],
        "ultrafine_strain_increment": levels["ultrafine_gate"][0],
        "ultrafine_recovery_time_increment": levels["ultrafine_gate"][1],
        "relative_jacobian_change": float(
            np.linalg.norm(sensitivity - ultrafine) / np.linalg.norm(ultrafine)
        ),
        "weakest_mode_angle_degrees": _principal_angle(
            vt_sensitivity, vt_ultrafine, 1
        ),
        "weak_three_subspace_angle_degrees": _principal_angle(
            vt_sensitivity, vt_ultrafine, 3
        ),
        "sensitivity_condition_number": float(condition_sensitivity),
        "ultrafine_condition_number": float(condition_ultrafine),
        "condition_number_relative_change": float(
            abs(condition_sensitivity - condition_ultrafine) / condition_ultrafine
        ),
    }
    summary["passes_declared_gate"] = bool(
        summary["relative_jacobian_change"] < 0.01
        and summary["weak_three_subspace_angle_degrees"] < 1.0
        and summary["condition_number_relative_change"] < 0.05
    )
    np.savez(
        OUTPUT / "retained_sensitivity_gate_jacobians.npz",
        sensitivity=sensitivity,
        ultrafine_gate=ultrafine,
        parameter_names=np.asarray(NAMES),
    )
    with (OUTPUT / "sensitivity_gate.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("metric", "threshold", "value", "passed"))
        for metric, threshold in (
            ("relative_jacobian_change", 0.01),
            ("weak_three_subspace_angle_degrees", 1.0),
            ("condition_number_relative_change", 0.05),
        ):
            value = summary[metric]
            writer.writerow((metric, threshold, value, value < threshold))
    (OUTPUT / "sensitivity_gate_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2), flush=True)
    if not summary["passes_declared_gate"]:
        raise SystemExit("retained-sensitivity resolution gate failed")


if __name__ == "__main__":
    main()
