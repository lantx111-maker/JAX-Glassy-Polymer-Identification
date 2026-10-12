"""Frozen experiment registry and canonical full-history prediction.

This module is the only bridge between the read-only experimental corpus and
the differentiable material-point operator.  Calibration scripts must consume
``Experiment`` records from here rather than implementing protocol-dependent
UMAT/VUMAT choices locally.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .canonical import (
    CANONICAL_REGULARIZATION,
    CANONICAL_RESOLUTION,
    CANONICAL_SCHEME,
)
from .constitutive import (
    ConstitutiveRegularization,
    MaterialParameters,
    initial_state,
)
from .data import classify_curve, load_all_curves
from .data_cleaning import strictly_increasing_time
from .integration_mesh import strain_control_mesh, time_control_mesh
from .objectives import ProtocolResidual, path_integration_weights
from .protocols import run_strain_history, run_stress_history


ROOT = Path(__file__).resolve().parent
SPLIT_FILE = ROOT / "protocol_splits.json"
VALID_SPLITS = ("training", "validation", "test", "diagnostic")


@dataclass(frozen=True)
class Experiment:
    curve_id: str
    material: str
    family: str
    control: str
    label: str
    target: float | None
    rate: float | None
    x: np.ndarray
    y: np.ndarray
    split: str
    source_path: Path
    compound_test_id: str | None = None
    loading_x: np.ndarray | None = None
    loading_y: np.ndarray | None = None
    loading_rate: float | None = None


class ExperimentAudit(NamedTuple):
    increments: jax.Array
    finite: jax.Array
    max_abs_root_residual: jax.Array
    max_control_residual: jax.Array
    min_control_singular_value: jax.Array
    max_control_condition_number: jax.Array
    maximum_root_iterations: jax.Array
    root_fallback_activations: jax.Array
    active_primary_increments: jax.Array
    max_q0: jax.Array
    max_q1: jax.Array
    max_q2: jax.Array
    min_eigenvalue_A0: jax.Array
    min_eigenvalue_A1: jax.Array
    min_eigenvalue_A2: jax.Array
    min_determinant_A0: jax.Array
    max_determinant_A0: jax.Array
    min_determinant_A1: jax.Array
    max_determinant_A1: jax.Array
    min_determinant_A2: jax.Array
    max_determinant_A2: jax.Array
    min_determinant_Fp: jax.Array
    max_determinant_Fp: jax.Array
    minimum_total_resistance: jax.Array
    minimum_b1: jax.Array
    maximum_b1: jax.Array
    minimum_b2: jax.Array
    maximum_b2: jax.Array
    minimum_kappa1: jax.Array
    maximum_kappa1: jax.Array
    minimum_kappa2: jax.Array
    maximum_kappa2: jax.Array
    minimum_gent_margin: jax.Array
    matrix_log_floor_activations: jax.Array
    gent_floor_activations: jax.Array
    terminal_time: jax.Array


def _finite_minimum(values: jax.Array) -> jax.Array:
    finite = jnp.isfinite(values)
    return jnp.where(jnp.any(finite), jnp.min(jnp.where(finite, values, jnp.inf)), jnp.nan)


def _finite_maximum(values: jax.Array) -> jax.Array:
    finite = jnp.isfinite(values)
    return jnp.where(jnp.any(finite), jnp.max(jnp.where(finite, values, -jnp.inf)), jnp.nan)


def _summarize_histories(histories: list, params: MaterialParameters) -> ExperimentAudit:
    states = jax.tree_util.tree_map(
        lambda *values: jnp.concatenate(values, axis=0),
        *(history.state for history in histories),
    )
    diagnostics = jax.tree_util.tree_map(
        lambda *values: jnp.concatenate(values, axis=0),
        *(history.diagnostics for history in histories),
    )
    stresses = jnp.concatenate([history.stress for history in histories], axis=0)
    eig0 = jnp.linalg.eigvalsh(states.A0)
    eig1 = jnp.linalg.eigvalsh(states.A1)
    eig2 = jnp.linalg.eigvalsh(states.A2)
    det0 = jnp.linalg.det(states.A0)
    det1 = jnp.linalg.det(states.A1)
    det2 = jnp.linalg.det(states.A2)
    det_fp = jnp.linalg.det(states.Fp)
    determinant_F = jnp.linalg.det(states.F)
    F_distortional = states.F / jnp.cbrt(determinant_F)[:, None, None]
    B_distortional = F_distortional @ jnp.swapaxes(F_distortional, -1, -2)
    gent_margin = 1.0 - (jnp.trace(B_distortional, axis1=-2, axis2=-1) - 3.0) / params.I_m
    all_values = [
        stresses, *states, diagnostics.q0, diagnostics.q1, diagnostics.q2,
        diagnostics.residual, diagnostics.control_residual_norm,
    ]
    finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(value)) for value in all_values]))
    return ExperimentAudit(
        jnp.asarray(states.time.shape[0]), finite,
        _finite_maximum(jnp.abs(diagnostics.residual)),
        _finite_maximum(diagnostics.control_residual_norm),
        _finite_minimum(diagnostics.control_min_singular_value),
        _finite_maximum(diagnostics.control_condition_number),
        _finite_maximum(diagnostics.newton_iterations),
        jnp.asarray(0),
        jnp.count_nonzero(diagnostics.q0 > 0.0),
        _finite_maximum(diagnostics.q0),
        _finite_maximum(diagnostics.q1),
        _finite_maximum(diagnostics.q2),
        _finite_minimum(eig0), _finite_minimum(eig1), _finite_minimum(eig2),
        _finite_minimum(det0), _finite_maximum(det0),
        _finite_minimum(det1), _finite_maximum(det1),
        _finite_minimum(det2), _finite_maximum(det2),
        _finite_minimum(det_fp), _finite_maximum(det_fp),
        _finite_minimum(params.s0 + states.kappa1 + states.kappa2),
        _finite_minimum(states.b1), _finite_maximum(states.b1),
        _finite_minimum(states.b2), _finite_maximum(states.b2),
        _finite_minimum(states.kappa1), _finite_maximum(states.kappa1),
        _finite_minimum(states.kappa2), _finite_maximum(states.kappa2),
        _finite_minimum(gent_margin),
        (jnp.count_nonzero(eig0 <= 1.0e-12)
         + jnp.count_nonzero(eig1 <= 1.0e-12)
         + jnp.count_nonzero(eig2 <= 1.0e-12)),
        jnp.count_nonzero(gent_margin <= 1.0e-8),
        states.time[-1],
    )


def _matches(value: float | None, candidates: list[float]) -> bool:
    return value is not None and any(abs(value - candidate) < 1.0e-10 for candidate in candidates)


def _pair_matches(rate: float | None, target: float | None,
                  candidates: list[list[float]]) -> bool:
    return rate is not None and target is not None and any(
        abs(rate - pair[0]) < 1.0e-12 and abs(target - pair[1]) < 1.0e-10
        for pair in candidates
    )


def frozen_split(material: str, family: str, target: float | None,
                 rate: float | None, policy: dict) -> str:
    entry = policy[material][family]
    for split in VALID_SPLITS:
        if entry.get(f"{split}_all", False):
            return split
        if _pair_matches(rate, target, entry.get(f"{split}_pairs", [])):
            return split
        if _matches(target, entry.get(split, [])):
            return split
    raise ValueError(
        f"Curve is absent from frozen split policy: {material}/{family}, "
        f"target={target}, rate={rate}"
    )


def _compound_id(material: str, family: str, target: float | None) -> str | None:
    if family not in {"load_unload_recovery", "recovery"} or target is None:
        return None
    return f"{material}-load-unload-recovery-{target:g}"


def build_experiments(material: str | None = None) -> list[Experiment]:
    """Load unmodified response values and assign frozen record-level roles.

    Source workbooks remain byte-for-byte unchanged.  Response-value outliers
    are retained and flagged in the QC manifest; the robust objective controls
    their influence.  Deterministic monotone repair is applied only to time
    coordinates at the point where a time-controlled protocol is integrated.
    """
    policy = json.loads(SPLIT_FILE.read_text())
    curves = load_all_curves()
    experiments: list[Experiment] = []
    for index, curve in enumerate(curves, start=1):
        meta = classify_curve(curve)
        if material is not None and meta.material != material:
            continue
        split = frozen_split(
            meta.material, meta.family, meta.target, meta.nominal_rate, policy
        )
        curve_x = np.asarray(curve.x, dtype=float)
        curve_y = np.asarray(curve.y, dtype=float)
        loading_x = loading_y = None
        loading_rate = None
        if meta.family == "recovery":
            candidates = [
                candidate for candidate in curves
                if classify_curve(candidate).material == meta.material
                and classify_curve(candidate).family in {
                    "load_unload", "load_unload_recovery"
                }
                and classify_curve(candidate).target is not None
                and meta.target is not None
                and abs(classify_curve(candidate).target - meta.target) < 1.0e-10
            ]
            candidates.sort(
                key=lambda candidate: classify_curve(candidate).family
                != "load_unload_recovery"
            )
            if not candidates:
                raise ValueError(f"No loading record found for {curve.path}")
            loading = candidates[0]
            loading_x = np.asarray(loading.x, dtype=float)
            loading_y = np.asarray(loading.y, dtype=float)
            loading_rate = classify_curve(loading).nominal_rate
        experiments.append(Experiment(
            curve_id=f"C{index:02d}",
            material=meta.material,
            family=meta.family,
            control=meta.control,
            label=curve.path.name,
            target=meta.target,
            rate=meta.nominal_rate,
            x=np.asarray(curve_x, dtype=float),
            y=np.asarray(curve_y, dtype=float),
            split=split,
            source_path=curve.path,
            compound_test_id=_compound_id(meta.material, meta.family, meta.target),
            loading_x=loading_x,
            loading_y=loading_y,
            loading_rate=loading_rate,
        ))
    return experiments


def history_dt(strain: np.ndarray, rate: float, engineering: bool) -> np.ndarray:
    coordinate = 1.0 - np.exp(-strain) if engineering else strain
    return np.maximum(
        np.abs(np.diff(coordinate, prepend=coordinate[:1])) / rate, 1.0e-5
    )


def _stress_ramp(target: float, params: MaterialParameters, state,
                 strain0: float = 0.0, lateral0: float = 0.0,
                 previous_target: float = 0.0,
                 regularization: ConstitutiveRegularization = CANONICAL_REGULARIZATION,
                 checkpoint: bool = False, steps: int = 20,
                 duration: float = 1.0e-5,
                 audit_control: bool = False,
                 return_result: bool = False,
                 scheme: str = CANONICAL_SCHEME):
    """Resolve an observation-independent fast stress transition."""
    targets = jnp.linspace(previous_target, target, steps + 1)[1:]
    result = run_stress_history(
        targets, jnp.full((steps,), duration / steps), params, state,
        strain0=strain0, lateral0=lateral0, scheme=scheme,
        substeps=1, regularization=regularization, checkpoint=checkpoint,
        implicit_control=True, time0=state.time,
        previous_target0=previous_target,
        audit_control=audit_control,
    )
    terminal_state = jax.tree_util.tree_map(lambda value: value[-1], result.state)
    terminal_strain = result.strain[-1]
    terminal_lateral = jnp.log(terminal_state.F[0, 0])
    outputs = (terminal_state, terminal_strain, terminal_lateral)
    return (*outputs, result) if return_result else outputs


def predict(experiment: Experiment, params: MaterialParameters,
            regularization: ConstitutiveRegularization = CANONICAL_REGULARIZATION,
            checkpoint: bool = False,
            resolution=CANONICAL_RESOLUTION,
            _return_audit: bool = False,
            scheme: str = CANONICAL_SCHEME,
            _return_histories: bool = False):
    """Predict one curve with the single canonical UMAT-ordered operator."""
    def finish(prediction, histories):
        if _return_audit and _return_histories:
            return prediction, _summarize_histories(histories, params), histories
        if _return_audit:
            return prediction, _summarize_histories(histories, params)
        if _return_histories:
            return prediction, histories
        return prediction

    if experiment.family == "creep":
        time = strictly_increasing_time(experiment.x)
        ramp_state, ramp_strain, ramp_lateral, ramp_result = _stress_ramp(
            float(experiment.target), params, initial_state(params),
            regularization=regularization, checkpoint=checkpoint,
            duration=resolution.initial_time_increment,
            audit_control=_return_audit, return_result=True,
            scheme=scheme,
        )
        mesh = time_control_mesh(
            float(time[-1]), resolution.creep_time_increment(
                experiment.material, float(experiment.target)
            ),
            initial_dt=resolution.initial_time_increment,
        )
        hold_dt = np.diff(mesh.time)
        if hold_dt.size:
            hold_result = run_stress_history(
                jnp.full(hold_dt.shape, experiment.target), jnp.asarray(hold_dt),
                params, ramp_state, strain0=ramp_strain, lateral0=ramp_lateral,
                scheme=scheme, substeps=1,
                regularization=regularization, checkpoint=checkpoint,
                implicit_control=True, time0=ramp_state.time,
                previous_target0=float(experiment.target),
                audit_control=_return_audit,
            )
            simulated = jnp.concatenate((ramp_strain[None], hold_result.strain))
            histories = [ramp_result, hold_result]
        else:
            simulated = ramp_strain[None]
            histories = [ramp_result]
        prediction = jnp.interp(jnp.asarray(time), jnp.asarray(mesh.time), simulated)
        return finish(prediction, histories)
    if experiment.family == "relaxation":
        time = strictly_increasing_time(experiment.x)
        engineering_target = 1.0 - np.exp(-experiment.target)
        loading_end = min(float(time[-1]), engineering_target / experiment.rate)
        # The loading ramp is strain-resolved; the hold is time-resolved.  Both
        # meshes are graded afresh at their common stage boundary.
        loading_max_dt = min(
            resolution.maximum_time_increment(experiment.material),
            resolution.maximum_strain_increment(experiment.material)
            / experiment.rate,
        )
        loading_mesh = time_control_mesh(
            loading_end, loading_max_dt,
            initial_dt=resolution.initial_time_increment,
        )
        mesh_time = loading_mesh.time
        if float(time[-1]) > loading_end:
            hold_mesh = time_control_mesh(
                float(time[-1]) - loading_end,
                resolution.maximum_time_increment(experiment.material),
                initial_dt=resolution.initial_time_increment,
            )
            mesh_time = np.concatenate((
                loading_mesh.time,
                loading_end + hold_mesh.time[1:],
            ))
        mesh_dt = np.diff(mesh_time, prepend=mesh_time[:1])
        engineering_strain = np.minimum(experiment.rate * mesh_time, engineering_target)
        strain = -np.log(np.maximum(1.0 - engineering_strain, 1.0e-8))
        result = run_strain_history(
            jnp.asarray(strain), jnp.asarray(mesh_dt), params, initial_state(params),
            traction_free=True, scheme=scheme,
            substeps=1,
            regularization=regularization, checkpoint=checkpoint,
            audit_control=_return_audit,
        )
        prediction = jnp.interp(
            jnp.asarray(time), jnp.asarray(mesh_time), -result.stress[:, 1, 1]
        )
        return finish(prediction, [result])
    if experiment.family == "recovery":
        loading = experiment.loading_x
        peak = int(np.argmax(loading))
        zero_candidates = np.flatnonzero(
            np.abs(experiment.loading_y[peak + 1:]) <= 0.1
        )
        if zero_candidates.size:
            loading = loading[:peak + 2 + int(zero_candidates[0])]
        loading_mesh = strain_control_mesh(
            loading,
            experiment.loading_rate
            or (0.001 if experiment.material == "PC" else 0.0003),
            experiment.material == "PC",
            resolution.maximum_strain_increment(experiment.material),
            resolution.initial_time_increment,
        )
        loading_result = run_strain_history(
            jnp.asarray(loading_mesh.strain), jnp.asarray(loading_mesh.dt), params,
            initial_state(params), traction_free=True, scheme=scheme,
            substeps=1,
            regularization=regularization, checkpoint=checkpoint,
            audit_control=_return_audit,
        )
        state = jax.tree_util.tree_map(lambda value: value[-1], loading_result.state)
        residual_strain = -jnp.log(state.F[1, 1])
        residual_lateral = jnp.log(state.F[0, 0])
        previous_target = -loading_result.stress[-1, 1, 1]
        loading_time = float(np.sum(loading_mesh.dt))
        time = strictly_increasing_time(experiment.x)
        recovery_mesh = time_control_mesh(
            float(time[-1]), resolution.recovery_max_time_increment,
            initial_dt=resolution.initial_time_increment,
        )
        recovery_state, recovery_strain0, recovery_lateral0, ramp_result = _stress_ramp(
            0.0, params, state, strain0=residual_strain,
            lateral0=residual_lateral, previous_target=previous_target,
            regularization=regularization, checkpoint=checkpoint,
            duration=resolution.initial_time_increment,
            audit_control=_return_audit, return_result=True,
            scheme=scheme,
        )
        hold_dt = np.diff(recovery_mesh.time)
        if hold_dt.size:
            recovered = run_stress_history(
                jnp.zeros(hold_dt.shape), jnp.asarray(hold_dt), params, recovery_state,
                strain0=recovery_strain0, lateral0=recovery_lateral0,
                scheme=scheme, substeps=1,
                regularization=regularization, checkpoint=checkpoint,
                implicit_control=True, time0=recovery_state.time,
                previous_target0=0.0,
                audit_control=_return_audit,
            )
            recovery_strain = jnp.concatenate((recovery_strain0[None], recovered.strain))
            histories = [loading_result, ramp_result, recovered]
        else:
            recovery_strain = recovery_strain0[None]
            histories = [loading_result, ramp_result]
        prediction = jnp.interp(
            jnp.asarray(time), jnp.asarray(recovery_mesh.time),
            recovery_strain0 - recovery_strain,
        )
        zero_indices = np.flatnonzero(np.isclose(time, 0.0))
        if zero_indices.size:
            prediction = prediction.at[jnp.asarray(zero_indices)].set(0.0)
        return finish(prediction, histories)

    rate = experiment.rate or (0.001 if experiment.material == "PC" else 0.0003)
    mesh = strain_control_mesh(
        experiment.x, rate, experiment.material == "PC",
        resolution.maximum_strain_increment(experiment.material),
        resolution.initial_time_increment,
    )
    result = run_strain_history(
        jnp.asarray(mesh.strain), jnp.asarray(mesh.dt), params, initial_state(params),
        traction_free=True, scheme=scheme,
        substeps=1,
        regularization=regularization, checkpoint=checkpoint,
        audit_control=_return_audit,
    )
    prediction = jnp.interp(
        jnp.asarray(mesh.observation_coordinate), jnp.asarray(mesh.path_coordinate),
        -result.stress[:, 1, 1],
    )
    return finish(prediction, [result])


def predict_with_audit(experiment: Experiment, params: MaterialParameters,
                       regularization: ConstitutiveRegularization = CANONICAL_REGULARIZATION,
                       checkpoint: bool = False,
                       resolution=CANONICAL_RESOLUTION,
                       scheme: str = CANONICAL_SCHEME):
    """Return the prediction and pathwise state/root/control diagnostics."""
    return predict(
        experiment, params, regularization, checkpoint, resolution,
        _return_audit=True, scheme=scheme,
    )


def predict_with_histories(experiment: Experiment, params: MaterialParameters,
                           regularization: ConstitutiveRegularization = CANONICAL_REGULARIZATION,
                           checkpoint: bool = False,
                           resolution=CANONICAL_RESOLUTION,
                           scheme: str = CANONICAL_SCHEME):
    """Return response and internal histories for verification-only audits."""
    return predict(
        experiment, params, regularization, checkpoint, resolution,
        scheme=scheme, _return_histories=True,
    )


def residual_blocks(experiments: list[Experiment], params: MaterialParameters,
                    regularization: ConstitutiveRegularization,
                    checkpoint: bool) -> list[ProtocolResidual]:
    blocks: list[ProtocolResidual] = []
    for experiment in experiments:
        prediction = predict(experiment, params, regularization, checkpoint)
        scale = (
            max(float(np.max(np.abs(experiment.y))), 0.02)
            if experiment.family in {"creep", "recovery"} else 110.0
        )
        blocks.append(ProtocolResidual(
            experiment.material,
            experiment.family,
            ("load_unload_recovery" if experiment.compound_test_id else experiment.family),
            experiment.compound_test_id or experiment.curve_id,
            experiment.curve_id,
            prediction, jnp.asarray(experiment.y),
            jnp.asarray(path_integration_weights(experiment.x)), scale,
        ))
    return blocks


def exact_metrics(experiments: list[Experiment], params: MaterialParameters):
    rows = []
    record_values: dict[tuple[str, str, str], list[float]] = {}
    for experiment in experiments:
        prediction = np.asarray(predict(experiment, params))
        error = prediction - experiment.y
        rmse = float(np.sqrt(np.mean(error * error)))
        scale = (
            max(float(np.max(np.abs(experiment.y))), 0.02)
            if experiment.family in {"creep", "recovery"} else 110.0
        )
        nrmse = rmse / scale
        rows.append((
            experiment.curve_id, experiment.family, experiment.target,
            experiment.rate, experiment.split, len(experiment.x), rmse, nrmse,
            bool(np.all(np.isfinite(prediction))), experiment.label,
        ))
        balance_family = (
            "load_unload_recovery" if experiment.compound_test_id
            else experiment.family
        )
        record_id = experiment.compound_test_id or experiment.curve_id
        record_values.setdefault(
            (experiment.split, balance_family, record_id), []
        ).append(nrmse)
    family_values: dict[tuple[str, str], list[float]] = {}
    for (split, family, _), values in record_values.items():
        family_values.setdefault((split, family), []).append(float(np.mean(values)))
    summary = {
        key: float(np.mean(values)) for key, values in family_values.items()
    }
    return rows, summary
