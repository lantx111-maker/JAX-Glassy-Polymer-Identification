"""Resume a checkpointed complete-history calibration.

The calculation reuses the declared problem and remaining optimization
schedule. Missing second moments are reset explicitly; this is not a
bitwise continuation of uninterrupted RMSProp."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import resource
import shutil
import time

import jax.numpy as jnp
import numpy as np

import calibrate_gauge_fixed as engine


def read_json(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_file_app_suffix(value):
    """Keep the complete APP-relative path, discarding only a host prefix."""
    if not isinstance(value, str):
        raise RuntimeError("A provenance source_file must be a path string")
    parts = Path(value).parts
    if ".." in parts:
        raise RuntimeError("A provenance source_file contains path traversal")
    anchors = [index for index in range(len(parts) - 1)
               if parts[index:index + 2] == ("applications", "glassy_polymer")]
    if len(anchors) != 1:
        raise RuntimeError("A provenance source_file has no unique APP path anchor")
    suffix = parts[anchors[0] + 2:]
    if not suffix:
        raise RuntimeError("A provenance source_file has no APP-relative filename")
    return "/".join(suffix)


def assert_resume_provenance_matches(current, archived):
    """Assert the original REFINED scientific invariants after directory relocation.

    Runtime descriptors (creation time, platform, etc.) remain descriptive, as
    in the initial driver. Every record field, including both content hashes,
    data role, record identity and observation count, is compared unchanged;
    source_file alone is compared by its full APP-relative suffix. Inputs are
    not mutated. This function does not evaluate any constitutive history.
    """
    for key in ("core_source_sha256", "operator_version", "resolution", "float64"):
        if key not in current or key not in archived or current[key] != archived[key]:
            raise RuntimeError(f"Original/current provenance mismatch: {key}")
    current_records, archived_records = current.get("records"), archived.get("records")
    if (not isinstance(current_records, list) or not isinstance(archived_records, list)
            or len(current_records) != len(archived_records)):
        raise RuntimeError("Original/current provenance mismatch: records")
    for index, (current_record, archived_record) in enumerate(zip(current_records, archived_records)):
        if not isinstance(current_record, dict) or not isinstance(archived_record, dict):
            raise RuntimeError(f"Invalid provenance record at index {index}")
        left, right = dict(current_record), dict(archived_record)
        left["source_file"] = _source_file_app_suffix(left.get("source_file"))
        right["source_file"] = _source_file_app_suffix(right.get("source_file"))
        if left != right:
            raise RuntimeError(f"Original/current provenance mismatch: records[{index}]")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume-from", type=Path, required=True,
                        help="Completed reporting directory of the interrupted material run")
    parser.add_argument("--output", type=Path, required=True,
                        help="Fresh parent directory; the material subdirectory must not exist")
    parser.add_argument("--max-seconds", type=float, default=6000.0)
    parser.add_argument("--reporting-reserve", type=float, default=300.0)
    args = parser.parse_args()
    source = args.resume_from.resolve()
    original = read_json(source / "summary.json")
    registration = read_json(source / "preregistered_analysis.json")
    checkpoint = read_json(source / "optimization_checkpoint.json")
    old_provenance = read_json(source / "provenance.json")
    old_stages = read_json(source / "stages.json")
    material = original["material"]
    args.block_steps = int(registration["block_steps"])
    args.joint_steps = int(registration["joint_steps"])
    args.ratio_bound = float(np.exp(registration["global_log_ratio_bound"]))
    args.prior_weight = float(registration["prior_weight"])
    args.learning_rate = float(registration["learning_rate"])
    args.max_backtracks = int(registration["max_backtracks"])
    args.material_opt_seconds = args.max_seconds - args.reporting_reserve
    args.run_start = time.monotonic()
    args.deadline = args.run_start + args.max_seconds
    prior_counts = {item["stage"]: item["accepted_steps"] for item in old_stages}
    required = {"flow_or_relaxation": args.block_steps,
                "retained_evolution": args.block_steps, "joint": args.joint_steps}
    remaining = {stage: required[stage] - prior_counts[stage] for stage in required}
    if any(value < 0 for value in remaining.values()) or not any(remaining.values()):
        raise ValueError("The archived stage counts do not define an unfinished schedule")
    if any(item["termination"] not in {"iteration_budget_completed", "optimization_wall_budget_reached"}
           for item in old_stages):
        raise ValueError("Only a budget interruption, not a failed descent, can be resumed here")
    original_driver = Path(engine.__file__).resolve()
    expected_driver_hash = old_provenance["core_source_sha256"]["scripts/calibrate_gauge_fixed.py"]
    if sha(original_driver) != expected_driver_hash:
        raise RuntimeError("The original computational driver changed; refusing continuation")
    metadata = {
        "source_directory": str(source), "declared_utc": engine.stamp(),
        "source_checkpoint_sha256": sha(source / "optimization_checkpoint.json"),
        "source_preregistration_sha256": sha(source / "preregistered_analysis.json"),
        "source_decision_pretest_sha256": sha(source / "decision_pretest.json"),
        "source_driver_sha256": expected_driver_hash,
        "resume_driver_sha256": sha(Path(__file__)),
        "previous_accepted_steps": prior_counts, "fixed_remaining_schedule": remaining,
        "optimizer_second_moments_reset": True,
        "continuation_equivalence": "Same objective, nominal reference, bounds, learning rate, and accepted checkpoint; second moments were not archived and are reset. Not bitwise equivalent to an uninterrupted run.",
        "prior_test_results_already_reported": True,
        "test_used_to_change_objective_bounds_parameters_or_schedule": False,
        "baseline_and_nominal_directional_check": "Recomputed, not substituted from previous files.",
    }
    original_provenance = engine.provenance
    original_write_json = engine.write_json

    def checked_provenance(problem):
        current = original_provenance(problem)
        assert_resume_provenance_matches(current, old_provenance)
        if (list(problem.names) != registration["free_parameters"]
                or problem.weights != registration["training_weights"]
                or problem.scales != registration["training_scales"]
                or float(problem.reference.gamma0) != registration["fixed_reference_rate"]):
            raise RuntimeError("Original reference, fixed rate, training weights, or names changed")
        if problem.reference._asdict() != read_json(source / "baseline_parameters.json"):
            raise RuntimeError("The nominal reference parameter vector changed")
        shutil.copy2(original_driver, problem.output / "COMPUTATION_SOURCE_initial.py")
        shutil.copy2(Path(__file__), problem.output / "COMPUTATION_SOURCE_resume.py")
        shutil.copy2(source / "optimization_checkpoint.json", problem.output / "inherited_checkpoint.json")
        shutil.copy2(source / "optimization_trace.csv", problem.output / "inherited_optimization_trace.csv")
        shutil.copy2(source / "stages.json", problem.output / "inherited_stages.json")
        current["resume"] = metadata
        original_write_json(problem.output / "resume_registration.json", metadata)
        return current

    def augmented_write_json(path, payload):
        if path.name in {"preregistered_analysis.json", "decision_pretest.json", "summary.json"}:
            payload = {**payload, "resume": metadata}
        return original_write_json(path, payload)

    def resume_optimize(problem, baseline, unused_args, deadline):
        if list(problem.names) != checkpoint["names"]:
            raise RuntimeError("Checkpoint parameter ordering changed")
        theta = np.asarray(checkpoint["theta"], dtype=float)
        limit = np.log(args.ratio_bound)
        if not np.all(np.isfinite(theta)) or np.any(np.abs(theta) > limit + 1e-12):
            raise ValueError("The inherited checkpoint is not a finite feasible point")
        current = problem.evaluate(theta)
        differences = {key: abs(getattr(current, key) - checkpoint[key])
                       for key in ("objective", "data_loss")}
        if max(differences.values()) > 1e-10:
            raise RuntimeError(f"Checkpoint objective did not reproduce: {differences}")
        engine.write_json(problem.output / "checkpoint_reproduction.json", differences)
        blocks = (("flow_or_relaxation", engine.BLOCKS[material][0]),
                  ("retained_evolution", engine.BLOCKS[material][1]),
                  ("joint", problem.names))
        trace, stages = [], []
        for stage, names in blocks:
            indices = np.asarray([problem.names.index(name) for name in names])
            second_moment = np.zeros(len(theta))
            completed = 0
            start_objective = current.objective
            reason = "iteration_budget_completed"
            for local_iteration in range(1, remaining[stage] + 1):
                if time.monotonic() > deadline:
                    reason = "optimization_wall_budget_reached"
                    break
                iteration = prior_counts[stage] + local_iteration
                gradient = current.gradient
                second_moment = 0.9 * second_moment + 0.1 * gradient * gradient
                denominator = np.sqrt(second_moment / (1 - 0.9 ** local_iteration)) + 1e-8
                direction = np.zeros(len(theta))
                direction[indices] = -gradient[indices] / denominator[indices]
                direction /= max(float(np.max(np.abs(direction))), 1.0)
                projected = theta - np.clip(theta - gradient, -limit, limit)
                if np.max(np.abs(projected[indices])) < 1e-8:
                    reason = "small_projected_block_gradient"
                    break
                accepted = False
                for backtrack in range(args.max_backtracks):
                    begin = time.monotonic()
                    step = args.learning_rate * 0.5 ** backtrack
                    proposal = np.clip(theta + step * direction, -limit, limit)
                    trial, error = None, ""
                    try:
                        trial = problem.evaluate(proposal)
                        accepted = trial.objective <= current.objective + 1e-4 * float(np.dot(gradient, proposal - theta))
                    except FloatingPointError as exception:
                        error = str(exception)
                    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    rss_mb = rss / (1024 ** 2 if platform.system() == "Darwin" else 1024)
                    elapsed = time.monotonic() - begin
                    trace.append({"stage": stage, "iteration": iteration, "backtrack": backtrack,
                                  "step_size": step, "old_objective": current.objective,
                                  "trial_objective": np.nan if trial is None else trial.objective,
                                  "trial_data_loss": np.nan if trial is None else trial.data_loss,
                                  "accepted_step": accepted, "error": error,
                                  "gradient_inf_norm": float(np.max(np.abs(gradient))),
                                  "active_block_projected_gradient_inf_norm": float(np.max(np.abs(projected[indices]))),
                                  "elapsed_seconds": time.monotonic() - args.run_start,
                                  "evaluation_seconds": elapsed, "peak_rss_mb": rss_mb})
                    engine.write_csv(problem.output / "optimization_trace.csv", trace)
                    if accepted:
                        theta, current = proposal, trial
                        completed += 1
                        engine.write_json(problem.output / "optimization_checkpoint.json", {
                            "stage": stage, "iteration": iteration, "names": problem.names,
                            "theta": theta, "objective": current.objective, "data_loss": current.data_loss,
                            "parameters": problem.parameters(jnp.asarray(theta))._asdict(),
                            "second_moment": second_moment, "moment_iteration": local_iteration})
                        print(f"{material} resumed {stage} {iteration}/{required[stage]}: "
                              f"loss={current.objective:.9g}, eval={elapsed:.1f}s, peakRSS={rss_mb:.0f}MB", flush=True)
                        break
                if not accepted:
                    reason = "no_finite_descent_step"
                    break
            stages.append({"stage": stage, "free_parameters": names,
                           "requested_steps": required[stage],
                           "accepted_steps": prior_counts[stage] + completed,
                           "previous_accepted_steps": prior_counts[stage],
                           "resumed_accepted_steps": completed, "resumed_requested_steps": remaining[stage],
                           "initial_objective": next(item["initial_objective"] for item in old_stages if item["stage"] == stage),
                           "resume_initial_objective": start_objective,
                           "final_objective": current.objective, "termination": reason})
            engine.write_json(problem.output / "stages.json", stages)
        return current, stages

    engine.provenance = checked_provenance
    engine.write_json = augmented_write_json
    engine.optimize = resume_optimize
    print(f"{material} fixed continuation schedule: {remaining}; original outputs preserved", flush=True)
    summary = engine.run_material(material, args)
    engine.write_json(args.output / f"{material.lower()}_resume_summary.json", {**summary, "resume": metadata})


if __name__ == "__main__":
    main()
