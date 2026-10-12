"""Replay converged physical IFT sites and archive branch-resolved diagnostics.

No constitutive or control equation is replaced. The history recorder retains
the same nested substep operations while exposing their normally discarded
interior states. Counts identify distinct converged material-step sites, not
AD execution multiplicities or temporary outer-Newton iterates.

``--mode all`` replays the mother, domain and diagonal-design suites;
``--mode cross-probes`` separately adds the two crossed probe combinations.
Use an explicit fresh ``--output`` for a new computational source version.
Existing archives are immutable and retain their own COMPUTATION_SOURCE.py.
"""
from __future__ import annotations

import os
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")

import argparse
from contextlib import contextmanager
import csv
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import shutil
import time

import jax
import jax.numpy as jnp
import numpy as np

from applications.glassy_polymer import constitutive as c
from applications.glassy_polymer import experiments as ex
from applications.glassy_polymer import inverse_design as design
from applications.glassy_polymer import protocols as p
from applications.glassy_polymer.canonical import (
    CANONICAL_OPERATOR_VERSION, CANONICAL_REGULARIZATION as REG,
    CANONICAL_RESOLUTION, CanonicalResolution,
)
from applications.glassy_polymer.scripts.run_v4_feasible_design import encode

APP = Path(__file__).resolve().parents[1]
OUTPUT = APP / "results/readme_v9_derivative_guards"
TOL_STRESS_MPA = 1e-10
TOL_RATE_PER_S = 1e-10


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    def clean(obj):
        if isinstance(obj, dict):
            return {str(k): clean(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [clean(v) for v in obj]
        if isinstance(obj, np.ndarray):
            return clean(obj.tolist())
        if isinstance(obj, (np.integer, np.bool_)):
            return obj.item()
        if isinstance(obj, (float, np.floating)):
            return float(obj) if np.isfinite(obj) else None
        return obj
    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    if rows:
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def primary_denominator(q, F, old, dt, params):
    """Same residual, branch predicate and q differentiation as _implicit_q0_jvp."""
    return jax.lax.cond(
        q > 1e-14,
        lambda _: jax.grad(lambda value: c._q0_stress_residual(value, F, old, dt, params, REG))(q),
        lambda _: jax.grad(lambda value: c._q0_residual(value, F, old, dt, params, REG))(q),
        None,
    )


def primary_residual(q, F, old, dt, params):
    return jax.lax.cond(q > 1e-14,
                        lambda _: c._q0_stress_residual(q, F, old, dt, params, REG),
                        lambda _: c._q0_residual(q, F, old, dt, params, REG), None)


def _record_history(kind, values, dt, params, state0, *, substeps=1,
                    scheme="umat", regularization=REG, checkpoint=False,
                    time0=0.0, previous_strain0=None, lateral0=None,
                    strain0=0.0, previous_target0=0.0, traction_free=False,
                    implicit_control=True, audit_control=False):
    """Identical nested physical scan, additionally returning every substep."""
    del audit_control
    values = jnp.asarray(values, dtype=jnp.float64)
    dt = jnp.broadcast_to(jnp.asarray(dt, dtype=jnp.float64), values.shape)
    clock = jnp.where(jnp.asarray(time0) != 0, jnp.asarray(time0), state0.time)
    state0 = state0._replace(time=clock)
    fractions = (jnp.arange(substeps, dtype=jnp.float64) + 1) / substeps
    if kind == "strain":
        previous = values[0] if previous_strain0 is None else jnp.asarray(previous_strain0)
        eta0 = 0.5 * previous if lateral0 is None else jnp.asarray(lateral0)
        eps0 = previous
    else:
        previous = jnp.asarray(previous_target0)
        eps0 = jnp.asarray(strain0)
        eta0 = jnp.asarray(0.0 if lateral0 is None else lateral0)

    def step(carry, inputs):
        state, clock, eps, eta, previous = carry
        value, step_dt = inputs
        sub_dt = step_dt / substeps
        controls = previous + fractions * (value - previous)

        def substep(inner, control):
            old, t, e, lateral = inner
            end = t + sub_dt
            if kind == "strain":
                if traction_free:
                    stress, new, diag, next_eta = p.traction_free_uniaxial_step(
                        control, old, sub_dt, end, params, lateral,
                        scheme, regularization, True)
                else:
                    stress, new, diag = p._advance_material(
                        p.isochoric_uniaxial_F(control), old, sub_dt, end,
                        params, scheme, regularization)
                    next_eta = 0.5 * control
                next_eps = control
            else:
                next_eps, next_eta, stress, new, diag = p.stress_controlled_uniaxial_step(
                    control, old, sub_dt, end, params, e, lateral,
                    scheme, regularization, implicit_control, True)
            return (new, end, next_eps, next_eta), (next_eps, stress, new, diag)

        fn = jax.checkpoint(substep) if checkpoint else substep
        (new, end, next_eps, next_eta), all_steps = jax.lax.scan(
            fn, (state, clock, eps, eta), controls)
        return (new, end, next_eps, next_eta, value), all_steps

    fn = jax.checkpoint(step) if checkpoint else step
    _, all_steps = jax.lax.scan(fn, (state0, clock, eps0, eta0, previous), (values, dt))
    endpoint = lambda array: array[:, -1]
    endpoints = jax.tree_util.tree_map(endpoint, all_steps)
    full = jax.tree_util.tree_map(lambda array: array.reshape((-1,) + array.shape[2:]), all_steps)
    history = (p.HistoryResult(endpoints[1], endpoints[2], endpoints[3]) if kind == "strain"
               else p.StressHistoryResult(*endpoints))
    metadata = (state0, jnp.repeat(dt / substeps, substeps), full[2], full[3],
                jnp.asarray(1 if kind == "strain" else 2, dtype=jnp.int32))
    return history, metadata


@contextmanager
def recorder():
    """Patch only local module bindings, restoring them immediately after tracing."""
    saved = [(module, module.run_strain_history, module.run_stress_history)
             for module in (ex, design)]
    captured = []

    def strain(*args, **kwargs):
        result, info = _record_history("strain", *args, **kwargs)
        captured.append(info)
        return result

    def stress(*args, **kwargs):
        result, info = _record_history("stress", *args, **kwargs)
        captured.append(info)
        return result

    for module, _, _ in saved:
        module.run_strain_history, module.run_stress_history = strain, stress
    try:
        yield captured
    finally:
        for module, strain_fn, stress_fn in saved:
            module.run_strain_history, module.run_stress_history = strain_fn, stress_fn


def captured_prediction(experiment, params, resolution):
    with recorder() as histories:
        prediction = ex.predict(experiment, params, checkpoint=True, resolution=resolution)
    return prediction, tuple(histories)


def captured_design(raw, params, state_level, probe_substeps):
    with recorder() as histories:
        result = design.observable_retained_response(
            raw, params, .020, n_per_segment=5, loading_substeps=state_level,
            recovery_steps=state_level, recovery_substeps=state_level,
            work_weight=0., time_weight=0.)
    state = result.recovered_state
    strain = result.residual_strain + jnp.linspace(0., .10, 81)
    times = jnp.linspace(0., 100., 81)
    probe, info = _record_history(
        "strain", strain, jnp.diff(times, prepend=times[:1]), params, state,
        traction_free=True, substeps=probe_substeps, checkpoint=True,
        previous_strain0=result.residual_strain, lateral0=jnp.log(state.F[0, 0]))
    histories.append(info)
    return (-probe.stress[:, 1, 1], tuple(histories), state)


def summarize(label, captured, params, folder, batch_size):
    rows, arrays, offset, previous_terminal, previous_branch = [], {}, 0, None, None
    failures = []
    evaluate = jax.jit(jax.vmap(lambda q, F, old, dt: (
        primary_denominator(q, F, old, dt, params), primary_residual(q, F, old, dt, params))))
    for phase, (start, dt, states, diagnostics, control_kind) in enumerate(captured):
        start, states, diagnostics = jax.tree_util.tree_map(np.asarray, (start, states, diagnostics))
        dt = np.asarray(dt)
        q = diagnostics.q0
        old = jax.tree_util.tree_map(lambda first, series: np.concatenate((first[None], series[:-1])), start, states)
        denominators, residuals = [], []
        for left in range(0, len(q), batch_size):
            right = min(left + batch_size, len(q))
            slope, residual = evaluate(jnp.asarray(q[left:right]), jnp.asarray(states.F[left:right]),
                                       jax.tree_util.tree_map(lambda x: jnp.asarray(x[left:right]), old),
                                       jnp.asarray(dt[left:right]))
            denominators.append(np.asarray(slope)); residuals.append(np.asarray(residual))
        slope, repeated_residual = np.concatenate(denominators), np.concatenate(residuals)
        branch = q > 1e-14
        guarded = ~(np.abs(slope) > 1e-14)
        finite = np.isfinite(slope) & np.isfinite(diagnostics.residual)
        connected = previous_terminal is not None and all(
            np.array_equal(a, b) for a, b in zip(start, previous_terminal))
        boundary_switch = int(connected and previous_branch != bool(branch[0]))
        previous_terminal = jax.tree_util.tree_map(lambda value: value[-1], states)
        previous_branch = bool(branch[-1])
        b1_error = states.b1 - (params.b1_sat + (params.E - params.b1_sat) * np.exp(-params.h3 * states.gamma_p))
        b2_error = states.b2 - (params.b2_sat + (params.E - params.b2_sat) * np.exp(-params.h3 * states.gamma_p))
        payload = dict(time_s=states.time, dt_s=dt, q0_per_s=q, q1_per_s=diagnostics.q1,
                       q2_per_s=diagnostics.q2, stress_branch=branch,
                       residual_branch_units=diagnostics.residual,
                       denominator_branch_units=slope, denominator_unit_replacement=guarded,
                       control_min_singular_value_mpa=diagnostics.control_min_singular_value,
                       control_min_singular_value_over_E=diagnostics.control_min_singular_value / params.E,
                       control_residual_mpa=diagnostics.control_residual_norm,
                       control_condition_number=diagnostics.control_condition_number,
                       b1_closed_error_mpa=b1_error, b2_closed_error_mpa=b2_error)
        arrays.update({f"phase{phase}_{key}": value for key, value in payload.items()})
        for active, name, unit, tolerance in ((True, "stress", "MPa*s", TOL_STRESS_MPA),
                                               (False, "rate", "dimensionless", TOL_RATE_PER_S)):
            mask = branch == active
            selected = slope[mask]
            rows.append({"path": label, "phase": phase, "branch": name,
                         "sites": int(mask.sum()), "guard_replacements": int(np.sum(guarded & mask)),
                         "nonfinite_denominators": int(np.sum(mask & ~np.isfinite(slope))),
                         "minimum_abs_denominator": float(np.min(np.abs(selected))) if selected.size else None,
                         "denominator_unit": unit,
                         "max_abs_residual": float(np.max(np.abs(diagnostics.residual[mask]))) if mask.any() else None,
                         "residual_unit": "MPa" if active else "s^-1", "residual_tolerance": tolerance,
                         "max_residual_tolerance_ratio": float(np.max(np.abs(diagnostics.residual[mask])) / tolerance) if mask.any() else None,
                         "phase_branch_switches": int(np.sum(branch[1:] != branch[:-1])),
                         "connected_boundary_switch": boundary_switch,
                         "control_kind": "scalar_eta" if int(control_kind) == 1 else "eps_c_eta",
                         "control_min_singular_value_mpa": float(np.min(diagnostics.control_min_singular_value)),
                         "control_min_singular_value_over_E": float(np.min(diagnostics.control_min_singular_value) / params.E),
                         "control_scalar_tangent_guard_sites": int(np.sum(~(np.abs(diagnostics.control_min_singular_value) > 1e-12))) if int(control_kind) == 1 else 0,
                         "max_control_residual_mpa": float(np.max(diagnostics.control_residual_norm)),
                         "residual_replay_max_abs_difference": float(np.max(np.abs(repeated_residual - diagnostics.residual))),
                         "max_b1_closed_error_mpa": float(np.max(np.abs(b1_error))),
                         "max_b2_closed_error_mpa": float(np.max(np.abs(b2_error)))})
        for index in np.flatnonzero(guarded | ~finite):
            failures.append({"path": label, "phase": phase, "physical_index": int(index),
                             "time_s": float(states.time[index]), "dt_s": float(dt[index]),
                             "branch": "stress" if branch[index] else "rate", "q0_per_s": float(q[index]),
                             "denominator": float(slope[index]), "unit_replacement": bool(guarded[index])})
        offset += len(q)
    np.savez_compressed(folder / f"{label}_step_logs.npz", **arrays)
    write_csv(folder / f"{label}_branch_summary.csv", rows)
    write_json(folder / f"{label}_failures.json", failures)
    summary = {"path": label, "physical_increment_sites": offset,
               "derivative_denominator_replacements": sum(row["guard_replacements"] for row in rows),
               "all_selected_denominators_finite": all(row["nonfinite_denominators"] == 0 for row in rows),
               "branch_switches_including_connected_stages": sum(row["phase_branch_switches"] + row["connected_boundary_switch"] for row in rows[::2]),
               "maximum_dimensionless_residual_tolerance_ratio": max(row["max_residual_tolerance_ratio"] or 0. for row in rows),
               "maximum_abs_b_closed_error_mpa": max(max(row["max_b1_closed_error_mpa"], row["max_b2_closed_error_mpa"]) for row in rows),
               "branches": rows}
    return summary


def run_cross_probes(folder, batch_size):
    """Replay missing probe substeps from unchanged, archived complete states."""
    sources = {
        "state_archive": APP / "results/v8_reload_refinement/reload_refinement_states_and_envelopes.npz",
        "state_runs": APP / "results/v8_reload_refinement/reload_refinement_runs.csv",
        "controls": APP / "results/v4_feasible_design/feasible_history_refinement.csv",
        "cross_trajectories": APP / "results/readme_v9_probe_cross/probe_cross_trajectories.csv",
        "cross_summary": APP / "results/readme_v9_probe_cross/probe_cross_summary.json",
    }
    source_hashes = {key: sha(path) for key, path in sources.items()}
    with np.load(sources["state_archive"], allow_pickle=False) as stored:
        arrays = {key: stored[key].copy() for key in stored.files}
    with sources["state_runs"].open() as handle:
        runs = {(int(row["state_level"]), row["profile"]): row for row in csv.DictReader(handle)}
    with sources["cross_trajectories"].open() as handle:
        references = list(csv.DictReader(handle))
    with sources["controls"].open() as handle:
        controls = [row for row in csv.DictReader(handle)
                    if abs(float(row["target"]) - .020) < 1e-12 and int(row["mesh_level"]) == 32]
    if len(controls) != 3 or len({row["profile"] for row in controls}) != 3:
        raise ValueError("Expected the three predeclared retained-state histories")
    observation_time = arrays["time_s"]
    additional = arrays["additional_true_strain"]
    np.testing.assert_array_equal(observation_time, np.linspace(0., 100., 81))
    np.testing.assert_array_equal(additional, np.linspace(0., .10, 81))
    dt = np.diff(observation_time, prepend=observation_time[:1])
    params = c.pmma_parameters()
    summaries = []
    for level, substeps in ((32, 8), (48, 4)):
        evaluate = jax.jit(lambda state, residual: _record_history(
            "strain", residual + jnp.asarray(additional), jnp.asarray(dt), params, state,
            traction_free=True, substeps=substeps, checkpoint=True,
            previous_strain0=residual, lateral0=jnp.log(state.F[0, 0])))
        for control in controls:
            profile = control["profile"]
            label = f"cross_state{level}_probe{substeps}_{profile}"
            summary_file = folder / f"{label}_summary.json"
            if summary_file.exists():
                archived = json.loads(summary_file.read_text())
                if archived["probe_source_sha256"] != source_hashes:
                    raise ValueError("Archived crossed-probe source binding changed")
                summaries.append(archived)
                continue
            state = c.MaterialState(*(jnp.asarray(arrays[f"state{level}_{profile}_{name}"])
                                      for name in c.MaterialState._fields))
            residual = float(runs[(level, profile)]["residual_strain"])
            np.testing.assert_allclose(-np.log(float(state.F[1, 1])), residual, rtol=0., atol=2e-15)
            if not all(np.isfinite(np.asarray(value)).all() for value in state):
                raise FloatingPointError("Non-finite inherited complete state")
            selected = sorted((row for row in references if int(row["state_level"]) == level
                               and int(row["probe_substeps"]) == substeps
                               and row["profile"] == profile), key=lambda row: int(row["point"]))
            if [int(row["point"]) for row in selected] != list(range(81)):
                raise ValueError("Crossed-probe reference rows are not the frozen 81-point trace")
            np.testing.assert_array_equal([float(row["time_s"]) for row in selected], observation_time)
            np.testing.assert_array_equal([float(row["additional_true_strain"]) for row in selected], additional)
            expected = np.asarray([float(row["compressive_cauchy_stress_mpa"]) for row in selected])
            print(f"replay {label} (probe only, unchanged complete retained state)", flush=True)
            started = time.monotonic()
            result, metadata = evaluate(state, jnp.asarray(residual))
            response = np.asarray(-result.stress[:, 1, 1])
            np.savez_compressed(folder / f"{label}_observations.npz", prediction=response,
                                reference_prediction=expected, time_s=observation_time,
                                additional_true_strain=additional)
            summary = summarize(label, (metadata,), params, folder, batch_size)
            summary.update(state_level=level, probe_substeps=substeps, profile=profile,
                           phase_labels=["main_reload_0.10_from_archived_complete_state"],
                           prehistory_recomputed=False, complete_state_reused_without_modification=True,
                           probe_source_sha256=source_hashes,
                           maximum_abs_reference_trajectory_difference_mpa=float(np.max(np.abs(response - expected))),
                           reference_trajectory_agrees=bool(np.allclose(response, expected, rtol=1e-10, atol=5e-9)),
                           prediction_all_finite=bool(np.isfinite(response).all()),
                           wall_seconds=time.monotonic() - started)
            write_json(summary_file, summary)
            summaries.append(summary)
            print(json.dumps({key: value for key, value in summary.items() if key != "branches"}), flush=True)
        jax.clear_caches(); gc.collect()
    if source_hashes != {key: sha(path) for key, path in sources.items()}:
        raise ValueError("An inherited probe source changed during this audit")
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("mother", "domain", "design", "all", "cross-probes"), default="all")
    parser.add_argument("--curves", nargs="*")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--batch-size", type=int, default=2048)
    args = parser.parse_args()
    if args.mode == "cross-probes" and args.output == OUTPUT:
        args.output = OUTPUT / "crossed_probes"
    args.output.mkdir(parents=True, exist_ok=True)
    base = json.loads((APP / "results/v9_creep_continuation/final_mechanism_geometry.json").read_text())
    resolution = CanonicalResolution(**base["resolution"])
    core = {path.name: sha(path) for path in APP.glob("*.py")}
    signature = {"core_source_sha256": core, "protocol_splits_sha256": sha(APP / "protocol_splits.json"),
                 "driver_sha256": sha(Path(__file__)), "operator": CANONICAL_OPERATOR_VERSION,
                 "mother_matrix_sha256": sha(APP / base["source_array"]),
                 "mother_resolution": asdict(resolution), "domain_resolution": asdict(CANONICAL_RESOLUTION),
                 "float64": bool(jax.config.jax_enable_x64),
                 "count_scope": "Distinct converged physical increment IFT sites; not repeated AD executions or temporary outer-Newton trials.",
                 "q_branch": "q>1e-14: stress; otherwise rate", "guard": "not(abs(denominator)>1e-14), includes equality and non-finite comparisons",
                 "units": {"stress_denominator": "MPa*s", "rate_denominator": "dimensionless",
                           "control_variables": ["eps_c (dimensionless)", "eta (dimensionless)"],
                           "control_residuals": ["sigma_00 (MPa)", "-sigma_11-target (MPa)"],
                           "control_jacobian": "MPa", "stress_control_jacobian_diagonal_addition": "1e-12 MPa"},
                 "residual_tolerances": {"stress_MPa": TOL_STRESS_MPA, "rate_per_s": TOL_RATE_PER_S},
                 "thread_settings": {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "XLA_FLAGS")}}
    definition_path = args.output / "audit_definition.json"
    if definition_path.exists():
        if json.loads(definition_path.read_text()) != signature:
            raise ValueError("Existing audit has different frozen source/settings; use a fresh output directory.")
    else:
        write_json(definition_path, signature)
        shutil.copyfile(__file__, args.output / "COMPUTATION_SOURCE.py")
    jobs = []
    if args.mode in {"mother", "domain", "all"}:
        if args.mode in {"mother", "all"}:
            selected = set(row["curve_id"] for row in json.loads((APP / "results/v8_full_geometry_audit/row_manifest.json").read_text()))
            jobs.extend(("mother_final", experiment, resolution) for experiment in ex.build_experiments("PMMA") if experiment.curve_id in selected)
        if args.mode in {"domain", "all"}:
            jobs.extend(("domain44", experiment, CANONICAL_RESOLUTION) for experiment in ex.build_experiments())
    summaries = []
    if args.mode == "cross-probes":
        summaries.extend(run_cross_probes(args.output, args.batch_size))
    for group, experiment, local_resolution in jobs:
        if args.curves and experiment.curve_id not in args.curves:
            continue
        label = group + "_" + experiment.curve_id
        summary_file = args.output / f"{label}_summary.json"
        if summary_file.exists():
            print(f"already archived {label}; not overwritten", flush=True)
            summaries.append(json.loads(summary_file.read_text()))
            continue
        started = time.monotonic()
        params = c.pmma_parameters() if experiment.material == "PMMA" else c.pc_parameters()
        print(f"replay {label}", flush=True)
        prediction, captured = jax.jit(lambda: captured_prediction(experiment, params, local_resolution))()
        prediction = np.asarray(prediction)
        np.savez_compressed(args.output / f"{label}_observations.npz", prediction=prediction,
                            x=experiment.x, measured=experiment.y)
        forward_seconds = time.monotonic() - started
        print(f"forward {label} {forward_seconds:.1f}s; diagnose physical roots", flush=True)
        summary = summarize(label, captured, params, args.output, args.batch_size)
        summary.update(material=experiment.material, family=experiment.family,
                       source_sha256=sha(experiment.source_path),
                       forward_seconds=forward_seconds, wall_seconds=time.monotonic() - started,
                       prediction_all_finite=bool(np.isfinite(prediction).all()), resolution=asdict(local_resolution))
        write_json(summary_file, summary)
        summaries.append(summary)
        print(json.dumps({key: value for key, value in summary.items() if key != "branches"}), flush=True)
        jax.clear_caches(); gc.collect()
    if args.mode in {"design", "all"}:
        source = APP / "results/v4_feasible_design/feasible_history_refinement.csv"
        with source.open() as handle:
            rows = [row for row in csv.DictReader(handle) if abs(float(row["target"]) - .020) < 1e-12 and int(row["mesh_level"]) == 32]
        with np.load(APP / "results/v8_reload_refinement/reload_refinement_states_and_envelopes.npz") as archive:
            reference = {name: archive[name].copy() for name in archive.files}
        params = c.pmma_parameters()
        for level, substeps in ((32, 4), (48, 8)):
            for row in rows:
                label = f"selected_state{level}_probe{substeps}_{row['profile']}"
                if (args.output / f"{label}_summary.json").exists():
                    summaries.append(json.loads((args.output / f"{label}_summary.json").read_text()))
                    continue
                print(f"replay {label}", flush=True)
                started = time.monotonic()
                response, histories, final_state = jax.jit(lambda raw: captured_design(raw, params, level, substeps))(jnp.asarray(encode(row)))
                response = np.asarray(response)
                differences = {name: float(np.max(np.abs(np.asarray(value) - reference[f"state{level}_{row['profile']}_{name}"]))) for name, value in zip(final_state._fields, final_state)}
                np.savez_compressed(args.output / f"{label}_observations.npz", prediction=response, time_s=np.linspace(0, 100, 81))
                summary = summarize(label, histories, params, args.output, args.batch_size)
                summary.update(source_controls_sha256=sha(source), state_level=level, probe_substeps=substeps,
                               archived_retained_state_max_abs_difference_by_field=differences,
                               wall_seconds=time.monotonic() - started,
                               phase_labels=["loading", "transition_and_zero_stress_recovery", "aux_stiffness_0.002", "aux_reload_0.02", "main_reload_0.10"],
                               prediction_all_finite=bool(np.isfinite(response).all()))
                write_json(args.output / f"{label}_summary.json", summary)
                summaries.append(summary)
                print(json.dumps({key: value for key, value in summary.items() if key != "branches"}), flush=True)
                jax.clear_caches(); gc.collect()
    write_json(args.output / f"{args.mode}_run_summary.json", summaries)
    # Compact dimensional and nondimensional evidence for plotting without
    # replaying histories or combining MPa with inverse seconds.
    completed = [json.loads(path.read_text()) for path in sorted(args.output.glob("*_summary.json"))
                 if not path.name.endswith("run_summary.json")]
    compact = []
    for item in completed:
        branches = item["branches"]
        maxima = {branch: max((row["max_abs_residual"] or 0.) for row in branches
                              if row["branch"] == branch) for branch in ("stress", "rate")}
        compact.append({"path": item["path"], "physical_increment_sites": item["physical_increment_sites"],
                        "stress_residual_max_mpa": maxima["stress"],
                        "rate_residual_max_per_s": maxima["rate"],
                        "stress_residual_ratio": maxima["stress"] / TOL_STRESS_MPA,
                        "rate_residual_ratio": maxima["rate"] / TOL_RATE_PER_S,
                        "maximum_dimensionless_ratio": item["maximum_dimensionless_residual_tolerance_ratio"],
                        "denominator_replacements": item["derivative_denominator_replacements"],
                        "control_min_singular_value_over_E": min(row["control_min_singular_value_over_E"] for row in branches)})
    write_csv(args.output / "completed_path_residuals.csv", compact)


if __name__ == "__main__":
    main()
