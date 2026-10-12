"""Independent Taylor audit for the SPD matrix-power custom JVP."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from applications.glassy_polymer.constitutive import spd_power


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "v5_matrix_power_audit"
POWERS = (0.2, 0.5, 0.9, 1.0)
STEPS = (1.0e-2, 3.0e-3, 1.0e-3, 3.0e-4, 1.0e-4, 3.0e-5, 1.0e-5)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    angle = 0.47
    rotation = np.asarray([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])
    cases = {
        "repeated_identity": np.eye(3),
        "near_repeated": rotation @ np.diag([0.8, 0.8 + 1.0e-10, 2.1]) @ rotation.T,
        "rotated_repeated": rotation @ np.diag([1.4, 1.4, 2.7]) @ rotation.T,
        "random_spd": np.asarray([
            [1.4, 0.13, -0.04], [0.13, 0.92, 0.07], [-0.04, 0.07, 1.8]
        ]),
    }
    direction = np.asarray([
        [0.12, 0.03, -0.02], [0.03, -0.08, 0.04], [-0.02, 0.04, 0.16]
    ])
    direction /= np.linalg.norm(direction)
    rows = []
    nontrivial_slopes = []
    roundoff_limited_cases = 0
    for case, tensor in cases.items():
        for power in POWERS:
            value, tangent_a = jax.jvp(
                spd_power,
                (jnp.asarray(tensor), jnp.asarray(power)),
                (jnp.asarray(direction), jnp.asarray(0.0)),
            )
            _, tangent_p = jax.jvp(
                spd_power,
                (jnp.asarray(tensor), jnp.asarray(power)),
                (jnp.zeros((3, 3)), jnp.asarray(1.0)),
            )
            value = np.asarray(value)
            for variable, tangent in (("A", tangent_a), ("p", tangent_p)):
                tangent = np.asarray(tangent)
                remainders = []
                for step in STEPS:
                    if variable == "A":
                        perturbed = np.asarray(spd_power(
                            jnp.asarray(tensor + step * direction), jnp.asarray(power)
                        ))
                    else:
                        perturbed = np.asarray(spd_power(
                            jnp.asarray(tensor), jnp.asarray(power + step)
                        ))
                    remainder = np.linalg.norm(
                        perturbed - value - step * tangent
                    ) / max(np.linalg.norm(value), 1.0e-30)
                    remainders.append(remainder)
                    rows.append({
                        "case": case, "power": power,
                        "differentiated_variable": variable,
                        "step": step, "normalized_taylor_remainder": remainder,
                    })
                # Fit only the pre-roundoff portion h >= 1e-4.
                fit_mask = np.asarray(STEPS) >= 1.0e-4
                slope = np.polyfit(
                    np.log(np.asarray(STEPS)[fit_mask]),
                    np.log(np.maximum(np.asarray(remainders)[fit_mask], 1.0e-300)),
                    1,
                )[0]
                if max(remainders) > 1.0e-12:
                    nontrivial_slopes.append(float(slope))
                else:
                    roundoff_limited_cases += 1
                for row in rows[-len(STEPS):]:
                    row["fitted_taylor_slope"] = slope
    with (OUTPUT / "matrix_power_taylor.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    remainders = [row["normalized_taylor_remainder"] for row in rows
                  if row["step"] == 1.0e-5]
    summary = {
        "cases": list(cases),
        "powers": list(POWERS),
        "differentiated_variables": ["A", "p"],
        "minimum_nontrivial_fitted_taylor_slope": float(min(nontrivial_slopes)),
        "maximum_nontrivial_fitted_taylor_slope": float(max(nontrivial_slopes)),
        "roundoff_limited_exact_cases": roundoff_limited_cases,
        "maximum_normalized_remainder_at_1e-5": float(max(remainders)),
        "all_values_finite": bool(all(np.isfinite(row["normalized_taylor_remainder"])
                                       for row in rows)),
    }
    (OUTPUT / "matrix_power_taylor_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))
    passed = (
        summary["all_values_finite"]
        and 1.95 < summary["minimum_nontrivial_fitted_taylor_slope"]
        and summary["maximum_nontrivial_fitted_taylor_slope"] < 2.05
    )
    if not passed:
        raise SystemExit("SPD matrix-power Taylor gate failed")


if __name__ == "__main__":
    main()
