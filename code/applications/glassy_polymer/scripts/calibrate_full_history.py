"""Independent, full-resolution, protocol-balanced PC/PMMA calibration.

Every selected curve enters at its native point count. Curves are averaged
within protocol families and protocol-family losses are then averaged, so a
dense cyclic trace cannot overwhelm a sparse recovery trace. Optimization uses
C1-regularized switches and checkpointed reverse-mode differentiation through
the complete history. Candidate parameters are accepted only after independent
exact-switch, full-resolution forward validation.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from applications.glassy_polymer.calibration import (
    bounded_adam,
    replace_log_ratio_block,
)
from applications.glassy_polymer.constitutive import (
    ConstitutiveRegularization,
    MaterialParameters,
    pc_parameters,
    pmma_parameters,
)
from applications.glassy_polymer.experiments import (
    Experiment,
    build_experiments,
    exact_metrics,
    predict,
    residual_blocks,
)
from applications.glassy_polymer.objectives import (
    protocol_balanced_loss,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "canonical_calibration"
REGULARIZATION = ConstitutiveRegularization(0.02, 4.0e-7)


BLOCKS = {
    "PMMA": (
        ("flow_hardening", ("gamma0", "m0", "s0", "h2", "s1", "s2"),
         {"load_unload", "creep"}),
        ("retained_state", ("m_ve", "zeta1", "zeta2", "h3", "b1_sat", "b2_sat"),
         {"reverse", "cyclic", "load_unload_recovery", "recovery"}),
        ("joint", ("gamma0", "m0", "s0", "h2", "s1", "s2", "m_ve",
                   "zeta1", "zeta2", "h3", "b1_sat", "b2_sat"), None),
    ),
    "PC": (
        ("rate_relaxation", ("gamma0", "m0", "s0", "s1", "s2", "m_ve"),
         {"relaxation"}),
        ("retained_state", ("zeta1", "zeta2", "h3", "b1_sat", "b2_sat"),
         {"load_unload_recovery", "recovery"}),
        ("joint", ("gamma0", "m0", "s0", "s1", "s2", "m_ve", "zeta1",
                   "zeta2", "h3", "b1_sat", "b2_sat"), None),
    ),
}


def material_loss(experiments: list[Experiment], params: MaterialParameters,
                  regularization: ConstitutiveRegularization = REGULARIZATION,
                  checkpoint: bool = True) -> jax.Array:
    blocks = residual_blocks(experiments, params, regularization, checkpoint)
    return protocol_balanced_loss(blocks)[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--material", choices=("PC", "PMMA"), required=True)
    parser.add_argument("--iterations", type=int, default=6)
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    material = args.material
    original = pmma_parameters() if material == "PMMA" else pc_parameters()
    current = original
    experiments = build_experiments(material)
    training = [experiment for experiment in experiments if experiment.split == "training"]
    validation = [experiment for experiment in experiments if experiment.split == "validation"]
    test = [experiment for experiment in experiments if experiment.split == "test"]
    diagnostic = [experiment for experiment in experiments if experiment.split == "diagnostic"]
    print(
        material, "curves", len(experiments), "training", len(training),
        "validation", len(validation), "test", len(test),
        "diagnostic", len(diagnostic), flush=True,
    )
    trace_rows = []

    for stage_name, names, families in BLOCKS[material]:
        stage_experiments = training if families is None else [
            experiment for experiment in training if experiment.family in families
        ]
        theta0 = jnp.zeros(len(names), dtype=jnp.float64)
        lower = np.full(len(names), -np.log(1.35))
        upper = np.full(len(names), np.log(1.35))

        def objective(theta):
            params = replace_log_ratio_block(current, names, theta)
            data_loss = material_loss(stage_experiments, params)
            regularizer = 1.0e-3 * jnp.mean((theta - theta0) ** 2)
            return data_loss + regularizer

        value_and_grad = jax.jit(jax.value_and_grad(objective))
        initial_value = float(objective(theta0))
        candidate_theta, trace = bounded_adam(
            value_and_grad, np.asarray(theta0), lower, upper,
            iterations=args.iterations, learning_rate=1.5e-3,
        )
        candidate_value = float(objective(jnp.asarray(candidate_theta)))
        accepted = np.isfinite(candidate_value) and candidate_value < initial_value
        if accepted:
            current = replace_log_ratio_block(current, names, jnp.asarray(candidate_theta))
        for iteration, value in trace:
            trace_rows.append((material, stage_name, iteration, value,
                               initial_value, candidate_value, accepted))
        print(stage_name, initial_value, candidate_value, "accepted", accepted, flush=True)

    baseline_rows, baseline_summary = exact_metrics(experiments, original)
    candidate_rows, candidate_summary = exact_metrics(experiments, current)
    training_keys = [key for key in baseline_summary if key[0] == "training"]
    validation_keys = [key for key in baseline_summary if key[0] == "validation"]
    test_keys = [key for key in baseline_summary if key[0] == "test"]
    baseline_training = float(np.mean([baseline_summary[key] for key in training_keys]))
    candidate_training = float(np.mean([candidate_summary[key] for key in training_keys]))
    baseline_validation = float(np.mean([baseline_summary[key] for key in validation_keys]))
    candidate_validation = float(np.mean([candidate_summary[key] for key in validation_keys]))
    baseline_test = float(np.mean([baseline_summary[key] for key in test_keys]))
    candidate_test = float(np.mean([candidate_summary[key] for key in test_keys]))
    final_accepted = (
        candidate_training < baseline_training
        and candidate_validation < baseline_validation
    )
    final = current if final_accepted else original

    prefix = material.lower()
    with (OUTPUT / f"{prefix}_trace.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("material", "stage", "iteration", "loss", "initial_loss",
                         "candidate_loss", "stage_accepted"))
        writer.writerows(trace_rows)
    with (OUTPUT / f"{prefix}_parameters.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("parameter", "baseline", "candidate", "accepted"))
        for name in original._fields:
            writer.writerow((name, float(getattr(original, name)),
                             float(getattr(current, name)), final_accepted))
        writer.writerow(("exact_training_family_nrmse", baseline_training,
                         candidate_training, final_accepted))
        writer.writerow(("exact_validation_family_nrmse", baseline_validation,
                         candidate_validation, final_accepted))
        writer.writerow(("post_selection_audit_family_nrmse", baseline_test,
                         candidate_test, final_accepted))
    with (OUTPUT / f"{prefix}_validation.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("curve_id", "family", "target", "rate", "split", "points",
                         "baseline_rmse", "baseline_nrmse", "candidate_rmse",
                         "candidate_nrmse", "finite", "label"))
        for baseline, candidate in zip(baseline_rows, candidate_rows):
            writer.writerow((*baseline[:6], baseline[6], baseline[7],
                             candidate[6], candidate[7], baseline[8] and candidate[8],
                             baseline[9]))
    print("exact training", baseline_training, candidate_training, flush=True)
    print("exact validation", baseline_validation, candidate_validation, flush=True)
    print("post-selection audit", baseline_test, candidate_test, flush=True)
    print("FINAL ACCEPTED", final_accepted, flush=True)


if __name__ == "__main__":
    main()
