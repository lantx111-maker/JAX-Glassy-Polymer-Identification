"""Gauge-fixed, exact-operator, full-history PC/PMMA joint inversion.

Every optimization stage uses the same complete training set and frozen
family--record--observable objective. Only the parameter update mask changes.
Validation determines acceptance before any test/diagnostic response is run.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from applications.glassy_polymer.calibration import replace_log_ratio_block
from applications.glassy_polymer.canonical import (
    CANONICAL_OPERATOR_VERSION, CANONICAL_REGULARIZATION, SENSITIVITY_RESOLUTION,
)
from applications.glassy_polymer.constitutive import pc_parameters, pmma_parameters
from applications.glassy_polymer.experiments import build_experiments, predict
from applications.glassy_polymer.objectives import path_integration_weights, pseudo_huber_squared


APP = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = APP / "results" / "gauge_fixed_calibration"
BLOCKS = {
    "PMMA": (("m0", "s0", "h2", "s1", "s2"),
             ("m_ve", "zeta1", "zeta2", "h3", "b1_sat", "b2_sat")),
    "PC": (("m0", "s0", "s1", "s2", "m_ve"),
           ("zeta1", "zeta2", "h3", "b1_sat", "b2_sat")),
}


def stamp():
    return datetime.now(timezone.utc).isoformat()


def json_safe(value):
    if isinstance(value, jax.Array):
        return json_safe(np.asarray(value))
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def write_json(path, payload):
    path.write_text(json.dumps(json_safe(payload), indent=2, allow_nan=False) + "\n")


def write_csv(path, records):
    if not records:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def family_record(experiment):
    return ("load_unload_recovery" if experiment.compound_test_id else experiment.family,
            experiment.compound_test_id or experiment.curve_id)


def scales_and_weights(experiments):
    families = sorted({family_record(experiment)[0] for experiment in experiments})
    record_sets = {family: {family_record(experiment)[1] for experiment in experiments
                            if family_record(experiment)[0] == family}
                   for family in families}
    counts = {}
    for experiment in experiments:
        key = family_record(experiment)
        counts[key] = counts.get(key, 0) + 1
    weights = {}
    scales = {}
    for experiment in experiments:
        family, record = family_record(experiment)
        weights[experiment.curve_id] = 1.0 / (len(families) * len(record_sets[family])
                                              * counts[(family, record)])
        scales[experiment.curve_id] = (max(float(np.max(np.abs(experiment.y))), 0.02)
                                       if experiment.family in {"creep", "recovery"}
                                       else 110.0)
    if experiments and not np.isclose(sum(weights.values()), 1.0):
        raise AssertionError("Training hierarchy does not sum to unit mass")
    return weights, scales


@dataclass
class Evaluation:
    theta: np.ndarray
    objective: float
    data_loss: float
    gradient: np.ndarray
    predictions: dict
    curve_losses: dict


class Problem:
    def __init__(self, material, output, prior_weight):
        self.material = material
        self.output = output
        self.prior_weight = prior_weight
        self.reference = pmma_parameters() if material == "PMMA" else pc_parameters()
        self.names = tuple(name for block in BLOCKS[material] for name in block)
        self.experiments = build_experiments(material)
        self.training = [experiment for experiment in self.experiments if experiment.split == "training"]
        self.weights, self.scales = scales_and_weights(self.training)
        self.functions = {}
        self.forward_functions = {}
        self.evaluation_count = 0
        self.timings = []
        self.last = None
        if "gamma0" in self.names or len(set(self.names)) != len(self.names):
            raise AssertionError("The reference rate must be fixed and parameter columns unique")
        # A compound physical record may not cross data roles.
        roles = {}
        for experiment in self.experiments:
            record = family_record(experiment)[1]
            roles.setdefault(record, set()).add(experiment.split)
        if any(len(value) != 1 for value in roles.values()):
            raise AssertionError("A physical record crosses frozen data roles")

    def parameters(self, theta):
        return replace_log_ratio_block(self.reference, self.names, theta)

    def _training_function(self, experiment):
        if experiment.curve_id not in self.functions:
            weights = jnp.asarray(path_integration_weights(experiment.x))
            measurement = jnp.asarray(experiment.y)
            scale = self.scales[experiment.curve_id]

            def loss_with_response(theta):
                response = predict(experiment, self.parameters(theta),
                                   CANONICAL_REGULARIZATION, True, SENSITIVITY_RESOLUTION)
                loss = jnp.sum(weights * pseudo_huber_squared((response - measurement) / scale))
                return loss, response

            self.functions[experiment.curve_id] = jax.jit(
                jax.value_and_grad(loss_with_response, has_aux=True))
        return self.functions[experiment.curve_id]

    def evaluate(self, theta):
        theta = np.asarray(theta, dtype=float)
        if self.last is not None and np.array_equal(theta, self.last.theta):
            return self.last
        self.evaluation_count += 1
        data_loss = 0.0
        gradient = np.zeros(len(self.names))
        predictions, curve_losses = {}, {}
        for experiment in self.training:
            first = experiment.curve_id not in self.functions
            start = time.monotonic()
            (value, response), tangent = self._training_function(experiment)(jnp.asarray(theta))
            value, response, tangent = float(value), np.asarray(response), np.asarray(tangent)
            elapsed = time.monotonic() - start
            if first:
                print(f"{self.material} compile+first gradient {experiment.curve_id}: {elapsed:.2f}s", flush=True)
                self.timings.append({"curve_id": experiment.curve_id,
                                     "compile_and_first_gradient_seconds": elapsed})
                write_csv(self.output / "compilation_timings.csv", self.timings)
            if not (np.isfinite(value) and np.all(np.isfinite(response))
                    and np.all(np.isfinite(tangent))):
                raise FloatingPointError(f"{experiment.curve_id}: non-finite response, loss, or gradient")
            factor = self.weights[experiment.curve_id]
            data_loss += factor * value
            gradient += factor * tangent
            predictions[experiment.curve_id] = response
            curve_losses[experiment.curve_id] = value
        objective = data_loss + self.prior_weight * float(np.mean(theta * theta))
        gradient += 2.0 * self.prior_weight * theta / len(theta)
        self.last = Evaluation(theta.copy(), objective, data_loss, gradient, predictions, curve_losses)
        return self.last

    def forward(self, experiment, theta):
        if experiment.curve_id not in self.forward_functions:
            self.forward_functions[experiment.curve_id] = jax.jit(
                lambda values: predict(experiment, self.parameters(values),
                                       CANONICAL_REGULARIZATION, True, SENSITIVITY_RESOLUTION))
        return np.asarray(self.forward_functions[experiment.curve_id](jnp.asarray(theta)))

    def metrics(self, experiments, theta, predictions=None):
        weights, scales = scales_and_weights(experiments)
        records, prediction_rows = [], []
        aggregate_nrmse = 0.0
        all_finite = True
        for experiment in experiments:
            response = (predictions[experiment.curve_id] if predictions is not None
                        else self.forward(experiment, theta))
            finite = bool(np.all(np.isfinite(response)))
            all_finite &= finite
            rmse = float(np.sqrt(np.mean((response - experiment.y) ** 2)))
            nrmse = rmse / scales[experiment.curve_id]
            aggregate_nrmse += weights[experiment.curve_id] * nrmse
            family, record = family_record(experiment)
            records.append({"curve_id": experiment.curve_id, "split": experiment.split,
                            "family": experiment.family, "balance_family": family,
                            "record_id": record, "observations": len(experiment.y),
                            "record_observable_weight": weights[experiment.curve_id],
                            "scale": scales[experiment.curve_id], "rmse": rmse,
                            "nrmse": nrmse, "finite": finite})
            for index, (coordinate, measured, predicted) in enumerate(zip(experiment.x, experiment.y, response)):
                prediction_rows.append({"curve_id": experiment.curve_id, "point": index,
                                        "coordinate": float(coordinate), "measurement": float(measured),
                                        "prediction": float(predicted)})
        return aggregate_nrmse, all_finite, records, prediction_rows


def provenance(problem):
    files = {str(path.relative_to(APP)): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in sorted(APP.glob("*.py"))}
    files["protocol_splits.json"] = hashlib.sha256((APP / "protocol_splits.json").read_bytes()).hexdigest()
    files["scripts/calibrate_gauge_fixed.py"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    records = []
    for experiment in problem.experiments:
        digest = hashlib.sha256()
        for array in (experiment.x, experiment.y, experiment.loading_x, experiment.loading_y):
            digest.update(b"None" if array is None else np.asarray(array).tobytes())
        digest.update(repr((experiment.target, experiment.rate, experiment.loading_rate,
                            experiment.control, experiment.split)).encode())
        records.append({"curve_id": experiment.curve_id, "split": experiment.split,
                        "record_id": family_record(experiment)[1],
                        "source_file": str(experiment.source_path),
                        "source_sha256": hashlib.sha256(experiment.source_path.read_bytes()).hexdigest(),
                        "analysis_input_sha256": digest.hexdigest(), "observations": len(experiment.x)})
    return {"created_utc": stamp(), "operator_version": CANONICAL_OPERATOR_VERSION,
            "core_source_sha256": files, "records": records,
            "jax_version": jax.__version__, "numpy_version": np.__version__,
            "python_version": platform.python_version(), "platform": platform.platform(),
            "float64": bool(jax.config.jax_enable_x64),
            "resolution": vars(SENSITIVITY_RESOLUTION)}


def optimize(problem, initial, args, deadline):
    theta = initial.theta.copy()
    current = initial
    limit = np.log(args.ratio_bound)
    stages = (("flow_or_relaxation", BLOCKS[problem.material][0], args.block_steps),
              ("retained_evolution", BLOCKS[problem.material][1], args.block_steps),
              ("joint", problem.names, args.joint_steps))
    trace, stage_results = [], []
    for stage, names, iterations in stages:
        indices = np.asarray([problem.names.index(name) for name in names])
        second_moment = np.zeros(len(theta))
        stage_start = current.objective
        completed = 0
        reason = "iteration_budget_completed"
        for iteration in range(1, iterations + 1):
            # Reserve time for validation, the frozen decision, and post-selection reports.
            if time.monotonic() > deadline:
                reason = "optimization_wall_budget_reached"
                break
            gradient = current.gradient
            second_moment = 0.9 * second_moment + 0.1 * gradient * gradient
            denominator = np.sqrt(second_moment / (1.0 - 0.9 ** iteration)) + 1.0e-8
            direction = np.zeros(len(theta))
            direction[indices] = -gradient[indices] / denominator[indices]
            direction /= max(float(np.max(np.abs(direction))), 1.0)
            projected_gradient = theta - np.clip(theta - gradient, -limit, limit)
            if float(np.max(np.abs(projected_gradient[indices]))) < 1.0e-8:
                reason = "small_projected_block_gradient"
                break
            accepted_step = False
            for backtrack in range(args.max_backtracks):
                step = args.learning_rate * (0.5 ** backtrack)
                proposal = np.clip(theta + step * direction, -limit, limit)
                delta = proposal - theta
                error = ""
                trial = None
                try:
                    trial = problem.evaluate(proposal)
                    accepted_step = trial.objective <= current.objective + 1.0e-4 * float(np.dot(gradient, delta))
                except FloatingPointError as exception:
                    error = str(exception)
                trace.append({"stage": stage, "iteration": iteration,
                              "backtrack": backtrack, "step_size": step,
                              "old_objective": current.objective,
                              "trial_objective": float("nan") if trial is None else trial.objective,
                              "trial_data_loss": float("nan") if trial is None else trial.data_loss,
                              "accepted_step": accepted_step, "error": error,
                              "gradient_inf_norm": float(np.max(np.abs(gradient))),
                              "active_block_projected_gradient_inf_norm": float(np.max(np.abs(projected_gradient[indices]))),
                              "elapsed_seconds": time.monotonic() - args.run_start})
                write_csv(problem.output / "optimization_trace.csv", trace)
                if accepted_step:
                    theta, current = proposal, trial
                    completed += 1
                    write_json(problem.output / "optimization_checkpoint.json", {
                        "stage": stage, "iteration": iteration, "names": problem.names,
                        "theta": theta, "objective": current.objective, "data_loss": current.data_loss,
                        "parameters": problem.parameters(jnp.asarray(theta))._asdict()})
                    print(f"{problem.material} {stage} {iteration}/{iterations}: "
                          f"loss={current.objective:.9g}, step={step:.3g}", flush=True)
                    break
            if not accepted_step:
                reason = "no_finite_descent_step"
                break
        stage_results.append({"stage": stage, "free_parameters": names,
                              "requested_steps": iterations, "accepted_steps": completed,
                              "initial_objective": stage_start, "final_objective": current.objective,
                              "termination": reason})
        write_json(problem.output / "stages.json", stage_results)
    return current, stage_results


def run_material(material, args):
    output = args.output / material.lower()
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    problem = Problem(material, output, args.prior_weight)
    write_json(output / "provenance.json", provenance(problem))
    write_json(output / "preregistered_analysis.json", {
        "created_utc": stamp(), "free_parameters": problem.names,
        "fixed_reference_rate": float(problem.reference.gamma0),
        "global_log_ratio_bound": np.log(args.ratio_bound),
        "all_stages_use_identical_training_records_and_weights": True,
        "training_weights": problem.weights, "training_scales": problem.scales,
        "robust_transition": 2.0, "prior_weight": args.prior_weight,
        "block_steps": args.block_steps, "joint_steps": args.joint_steps,
        "learning_rate": args.learning_rate, "max_backtracks": args.max_backtracks,
        "acceptance": "finite candidate; training robust data loss, training family-record NRMSE, and validation family-record NRMSE all strictly below baseline; test and diagnostic are not used",
        "directional_check_step": 1.0e-4, "directional_check_relative_tolerance": 1.0e-3,
        "scope": "Conditional local calibration; historical source overlap means post-selection data are not blind external validation."})
    theta0 = np.zeros(len(problem.names))
    baseline = problem.evaluate(theta0)
    write_json(output / "baseline_parameters.json", problem.reference._asdict())
    print(f"{material} initial complete training gradient: {time.monotonic()-start:.1f}s; "
          f"objective={baseline.objective:.9g}", flush=True)

    direction = np.cos(np.arange(1, len(theta0) + 1, dtype=float))
    direction /= np.linalg.norm(direction)
    h = 1.0e-4
    forward = problem.evaluate(theta0 + h * direction).objective
    backward = problem.evaluate(theta0 - h * direction).objective
    fd = (forward - backward) / (2.0 * h)
    ad = float(np.dot(baseline.gradient, direction))
    relative_error = abs(fd - ad) / max(abs(fd), abs(ad), 1.0e-12)
    derivative_pass = bool(relative_error < 1.0e-3)
    write_json(output / "directional_derivative_check.json", {
        "point": "nominal reference", "direction": direction, "step": h,
        "finite_difference": fd, "implicit_history_adjoint": ad,
        "relative_error": relative_error, "tolerance": 1.0e-3,
        "passed": derivative_pass, "includes_prior": True})
    if not derivative_pass:
        raise RuntimeError(f"{material}: nominal full-history directional check failed ({relative_error})")
    print(f"{material} full-history gradient check rel.error={relative_error:.3e}", flush=True)

    candidate, stages = optimize(problem, baseline, args,
                                 min(args.deadline - args.reporting_reserve, start + args.material_opt_seconds))
    validation = [experiment for experiment in problem.experiments if experiment.split == "validation"]
    summaries = {}
    for model, evaluation in (("baseline", baseline), ("candidate", candidate)):
        for split, experiments in (("training", problem.training), ("validation", validation)):
            metric, finite, records, predictions = problem.metrics(
                experiments, evaluation.theta,
                evaluation.predictions if split == "training" else None)
            summaries[f"{model}_{split}"] = {"family_record_nrmse": metric, "all_finite": finite}
            write_csv(output / f"{model}_{split}_metrics.csv", records)
            write_csv(output / f"{model}_{split}_predictions.csv", predictions)
    accepted = bool(candidate.data_loss < baseline.data_loss
                    and summaries["candidate_training"]["all_finite"]
                    and summaries["candidate_validation"]["all_finite"]
                    and summaries["candidate_training"]["family_record_nrmse"] < summaries["baseline_training"]["family_record_nrmse"]
                    and summaries["candidate_validation"]["family_record_nrmse"] < summaries["baseline_validation"]["family_record_nrmse"])
    params = problem.parameters(jnp.asarray(candidate.theta))
    if float(params.gamma0) != float(problem.reference.gamma0):
        raise AssertionError("The reference-rate gauge was altered")
    active_bounds = [name for name, value in zip(problem.names, candidate.theta)
                     if abs(abs(value) - np.log(args.ratio_bound)) < 1.0e-10]
    decision = {"frozen_utc_before_test": stamp(), "material": material,
                "optimization_completed": all(item["termination"] == "iteration_budget_completed" for item in stages),
                "validation_accepted": accepted, "selected_parameters": "candidate" if accepted else "baseline",
                "baseline_data_loss": baseline.data_loss, "candidate_data_loss": candidate.data_loss,
                "baseline_regularized_objective": baseline.objective, "candidate_regularized_objective": candidate.objective,
                "candidate_gradient_inf_norm": float(np.max(np.abs(candidate.gradient))),
                "active_bounds": active_bounds, "gamma0_fixed": float(params.gamma0),
                "metrics": summaries, "stages": stages,
                "test_used_for_selection": False, "diagnostic_used_for_selection": False,
                "convergence_claim": "A bounded finite-step local update; iteration completion is not a stationarity or global-optimality certificate."}
    write_json(output / "candidate_parameters.json", params._asdict())
    write_json(output / "selected_parameters.json", (params if accepted else problem.reference)._asdict())
    write_json(output / "decision_pretest.json", decision)
    decision_hash = hashlib.sha256((output / "decision_pretest.json").read_bytes()).hexdigest()
    print(f"{material} DECISION FROZEN BEFORE TEST: accepted={accepted}; {summaries}", flush=True)

    for split in ("test", "diagnostic"):
        experiments = [experiment for experiment in problem.experiments if experiment.split == split]
        if not experiments:
            continue
        for model, evaluation in (("baseline", baseline), ("candidate", candidate)):
            metric, finite, records, predictions = problem.metrics(experiments, evaluation.theta)
            summaries[f"{model}_{split}"] = {"family_record_nrmse": metric, "all_finite": finite}
            write_csv(output / f"{model}_{split}_metrics.csv", records)
            write_csv(output / f"{model}_{split}_predictions.csv", predictions)
    if decision_hash != hashlib.sha256((output / "decision_pretest.json").read_bytes()).hexdigest():
        raise AssertionError("The pre-test decision changed during post-selection evaluation")
    final_summary = {**decision, "metrics_including_postselection": summaries,
                     "decision_pretest_sha256": decision_hash,
                     "wall_seconds": time.monotonic() - start,
                     "objective_evaluations": problem.evaluation_count,
                     "reported_utc": stamp()}
    write_json(output / "summary.json", final_summary)
    return final_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--material", choices=("PMMA", "PC", "both"), default="both")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--block-steps", type=int, default=4)
    parser.add_argument("--joint-steps", type=int, default=12)
    parser.add_argument("--ratio-bound", type=float, default=1.25)
    parser.add_argument("--learning-rate", type=float, default=0.0125)
    parser.add_argument("--prior-weight", type=float, default=1.0e-3)
    parser.add_argument("--max-backtracks", type=int, default=6)
    parser.add_argument("--max-seconds", type=float, default=1800.0)
    parser.add_argument("--material-opt-seconds", type=float, default=700.0)
    parser.add_argument("--reporting-reserve", type=float, default=300.0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.run_start = time.monotonic()
    args.deadline = args.run_start + args.max_seconds
    materials = ("PMMA", "PC") if args.material == "both" else (args.material,)
    results = {}
    for material in materials:
        try:
            results[material] = run_material(material, args)
        except Exception as error:
            failure = {"status": "failed", "error": f"{type(error).__name__}: {error}",
                       "reported_utc": stamp(), "elapsed_seconds": time.monotonic() - args.run_start}
            write_json(args.output / f"{material.lower()}_failure.json", failure)
            results[material] = failure
            print(f"{material} FAILED: {failure['error']}", flush=True)
        write_json(args.output / "run_summary.json", results)
    print(f"Results archived at {args.output}", flush=True)
    if any(result.get("status") == "failed" for result in results.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
