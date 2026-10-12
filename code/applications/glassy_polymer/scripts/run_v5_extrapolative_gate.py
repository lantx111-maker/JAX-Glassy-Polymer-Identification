"""Upgrade the extrapolative 0.070 schedules after the 32--48 gate fails."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from applications.glassy_polymer.constitutive import pmma_parameters
from applications.glassy_polymer.inverse_design import observable_retained_response
from applications.glassy_polymer.scripts.plot_observable_design_landscape import scaled_controls
from applications.glassy_polymer.scripts.run_v4_feasible_design import encode


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "v4_feasible_design" / "feasible_history_refinement.csv"
OUTPUT = ROOT / "results" / "v5_extrapolative_gate"
TARGET = 0.070
PRODUCTION_LEVEL = 96
REFERENCE_LEVEL = 128


def evaluate(raw, params, level):
    return observable_retained_response(
        raw, params, TARGET, n_per_segment=5,
        loading_substeps=level, recovery_steps=level,
        recovery_substeps=level, work_weight=0.0, time_weight=0.0,
    )


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    params = pmma_parameters()
    all_rows = list(csv.DictReader(SOURCE.open()))
    source_rows = [
        row for row in all_rows
        if abs(float(row["target"]) - TARGET) < 1.0e-12
        and int(row["mesh_level"]) == 32
    ]
    if len(source_rows) != 3:
        raise RuntimeError("expected three level-32 extrapolative schedules")

    @jax.jit
    def production_residual(raw):
        return evaluate(raw, params, PRODUCTION_LEVEL).residual_strain

    selected = []
    for row in source_rows:
        profile = row["profile"]
        base_raw = encode(row)
        lower, upper = 0.95, 1.08
        lower_value = float(production_residual(jnp.asarray(
            scaled_controls(base_raw, lower, 1.0)
        )))
        upper_value = float(production_residual(jnp.asarray(
            scaled_controls(base_raw, upper, 1.0)
        )))
        if not lower_value <= TARGET <= upper_value:
            raise RuntimeError(
                f"level-{PRODUCTION_LEVEL} target not bracketed for {profile}: "
                f"[{lower_value}, {upper_value}]"
            )
        for _ in range(24):
            midpoint = 0.5 * (lower + upper)
            candidate = scaled_controls(base_raw, midpoint, 1.0)
            value = float(production_residual(jnp.asarray(candidate)))
            if value < TARGET:
                lower = midpoint
            else:
                upper = midpoint
        factor = 0.5 * (lower + upper)
        raw = scaled_controls(base_raw, factor, 1.0)
        selected.append((profile, factor, raw))
        print(profile, factor, flush=True)

    rows = []
    for profile, factor, raw in selected:
        for level in (64, PRODUCTION_LEVEL, REFERENCE_LEVEL):
            result = evaluate(jnp.asarray(raw), params, level)
            rows.append({
                "target": TARGET,
                "profile": profile,
                "amplitude_correction_from_level32": factor,
                "mesh_level": level,
                "residual_strain": float(result.residual_strain),
                "target_error": float(result.residual_strain) - TARGET,
                "positive_axial_cauchy_work_mpa": float(result.input_work),
                "peak_stress_mpa": float(result.peak_stress),
                "loading_program_time_s": float(result.loading_program_time),
            })
            print(profile, level, rows[-1]["residual_strain"], flush=True)
    with (OUTPUT / "extrapolative_refinement.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    old_changes = []
    new_changes = []
    for profile, _, _ in selected:
        old32 = float(next(row["residual_strain"] for row in all_rows
                           if abs(float(row["target"]) - TARGET) < 1.0e-12
                           and row["profile"] == profile
                           and int(row["mesh_level"]) == 32))
        old48 = float(next(row["residual_strain"] for row in all_rows
                           if abs(float(row["target"]) - TARGET) < 1.0e-12
                           and row["profile"] == profile
                           and int(row["mesh_level"]) == 48))
        new96 = next(row["residual_strain"] for row in rows
                     if row["profile"] == profile
                     and row["mesh_level"] == PRODUCTION_LEVEL)
        new128 = next(row["residual_strain"] for row in rows
                      if row["profile"] == profile
                      and row["mesh_level"] == REFERENCE_LEVEL)
        old_changes.append(abs(old32 - old48))
        new_changes.append(abs(new96 - new128))
    summary = {
        "target": TARGET,
        "status": "extrapolative_secondary_case",
        "old_level32_to_level48_maximum_change": max(old_changes),
        "old_gate_passed": max(old_changes) < 2.0e-4,
        "upgraded_production_level": PRODUCTION_LEVEL,
        "independent_reference_level": REFERENCE_LEVEL,
        "upgraded_maximum_change": max(new_changes),
        "upgraded_gate_threshold": 2.0e-4,
        "upgraded_gate_passed": max(new_changes) < 2.0e-4,
    }
    (OUTPUT / "extrapolative_gate_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2), flush=True)
    if summary["old_gate_passed"] or not summary["upgraded_gate_passed"]:
        raise SystemExit("extrapolative refinement gate failed")


if __name__ == "__main__":
    main()
