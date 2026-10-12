"""Retained-state-aware inverse loading design."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .constitutive import MaterialParameters, initial_state
from .canonical import CANONICAL_REGULARIZATION, CANONICAL_SCHEME
from .protocols import run_strain_history, run_stress_history


class DesignResult(NamedTuple):
    objective: jax.Array
    strain: jax.Array
    terminal_stress: jax.Array
    terminal_plastic_strain: jax.Array
    retained_state: jax.Array


class ObservableDesignResult(NamedTuple):
    """Observable retained-response design and its physical constraints."""

    objective: jax.Array
    time: jax.Array
    strain: jax.Array
    residual_strain: jax.Array
    reload_stiffness: jax.Array
    peak_stress: jax.Array
    input_work: jax.Array
    maximum_rate: jax.Array
    retained_state_norm: jax.Array
    net_work: jax.Array
    cumulative_absolute_work: jax.Array
    rate_jump_penalty: jax.Array
    loading_program_time: jax.Array
    total_elapsed_time: jax.Array
    terminal_phi: jax.Array
    terminal_kappa1: jax.Array
    terminal_kappa2: jax.Array
    terminal_b1: jax.Array
    terminal_b2: jax.Array
    terminal_A0_eigenvalues: jax.Array
    terminal_A1_eigenvalues: jax.Array
    terminal_A2_eigenvalues: jax.Array
    reload_stress_at_two_percent: jax.Array
    recovered_state: object


def decode_piecewise_linear(raw_controls: jax.Array, n_per_segment: int,
                            maximum_strain: float) -> jax.Array:
    knots = maximum_strain * jax.nn.sigmoid(raw_controls)
    knots = jnp.concatenate((jnp.zeros(1), knots))
    fractions = jnp.linspace(0.0, 1.0, n_per_segment, endpoint=False)
    segments = [a + fractions * (b - a) for a, b in zip(knots[:-1], knots[1:])]
    return jnp.concatenate(segments + [knots[-1:]])


def retained_state_objective(raw_controls: jax.Array, params: MaterialParameters,
                             target_residual_strain: float,
                             n_per_segment: int = 20,
                             maximum_strain: float = 0.8,
                             step_time: float = 1.0) -> DesignResult:
    strain = decode_piecewise_linear(raw_controls, n_per_segment, maximum_strain)
    history = run_strain_history(
        strain, step_time, params, initial_state(params), traction_free=True,
        substeps=20, scheme="vumat",
    )
    terminal = jnp.sqrt(jnp.sum(jnp.log(jnp.diag(history.state.Fp[-1])) ** 2))
    retained = jnp.linalg.norm(history.state.A0[-1] - jnp.eye(3))
    terminal_stress = -history.stress[-1, 1, 1]
    mismatch = terminal - target_residual_strain
    smoothness = jnp.mean(jnp.diff(strain, n=2) ** 2)
    objective = mismatch ** 2 + 1.0e-3 * smoothness + 1.0e-5 * terminal_stress ** 2
    return DesignResult(objective, strain, terminal_stress, terminal, retained)


def decode_physical_history(raw_controls: jax.Array, n_per_segment: int = 8,
                            maximum_strain: float = 0.55,
                            minimum_duration: float = 20.0,
                            maximum_duration: float = 400.0,
                            ) -> tuple[jax.Array, jax.Array]:
    """Decode four strain knots and four durations into a physical-time path."""
    if raw_controls.shape[0] != 8:
        raise ValueError("observable history design requires eight controls")
    knots = maximum_strain * jax.nn.sigmoid(raw_controls[:4])
    knots = jnp.concatenate((jnp.zeros(1), knots))
    durations = minimum_duration + (maximum_duration - minimum_duration) * jax.nn.sigmoid(
        raw_controls[4:]
    )
    fraction = jnp.linspace(0.0, 1.0, n_per_segment, endpoint=False)
    time_segments = []
    strain_segments = []
    elapsed = jnp.asarray(0.0)
    for start, end, duration in zip(knots[:-1], knots[1:], durations):
        time_segments.append(elapsed + fraction * duration)
        strain_segments.append(start + fraction * (end - start))
        elapsed = elapsed + duration
    return (
        jnp.concatenate(time_segments + [elapsed[None]]),
        jnp.concatenate(strain_segments + [knots[-1:]]),
    )


def observable_retained_response(
    raw_controls: jax.Array,
    params: MaterialParameters,
    target_residual_strain: float,
    target_reload_stiffness: float | None = None,
    n_per_segment: int = 8,
    loading_substeps: int = 8,
    recovery_steps: int = 12,
    recovery_substeps: int = 8,
    recovery_time: float = 600.0,
    stress_transition_time: float = 37.5,
    maximum_strain: float = 0.55,
    maximum_rate: float = 0.01,
    maximum_stress: float = 180.0,
    work_weight: float = 1.0e-5,
    time_weight: float = 0.0,
) -> ObservableDesignResult:
    """Design an executable history for an observable zero-stress residual.

    The programmed strain path uses seconds and true strain.  It is followed
    by a zero-stress recovery dwell and a small reload probe.  The residual
    strain and reload tangent are therefore experimentally observable; no
    internal variable is used as the target.
    """
    time, strain = decode_physical_history(
        raw_controls, n_per_segment=n_per_segment,
        maximum_strain=maximum_strain,
    )
    dt = jnp.diff(time, prepend=time[:1])
    dt = dt.at[1:].set(jnp.maximum(dt[1:], 1.0e-5))
    loading = run_strain_history(
        strain, dt, params, initial_state(params), traction_free=True,
        scheme=CANONICAL_SCHEME, substeps=loading_substeps, checkpoint=True,
        regularization=CANONICAL_REGULARIZATION,
    )
    state = jax.tree_util.tree_map(lambda value: value[-1], loading.state)
    current_strain = -jnp.log(state.F[1, 1])
    current_lateral = jnp.log(state.F[0, 0])
    current_stress = -loading.stress[-1, 1, 1]
    zero_stress_time = recovery_time - stress_transition_time
    if zero_stress_time <= 0.0:
        raise ValueError("recovery_time must exceed stress_transition_time")
    # Stage II is a fixed-duration ramp from the terminal compressive stress to
    # zero; Stage III then holds zero stress for the remainder of the stated
    # 600 s transition--recovery interval.  Its physical duration therefore
    # does not change when the numerical resolution is refined.
    recovery_dt = jnp.concatenate((
        jnp.asarray([stress_transition_time]),
        jnp.full((recovery_steps,), zero_stress_time / recovery_steps),
    ))
    recovery = run_stress_history(
        jnp.zeros(recovery_steps + 1), recovery_dt, params, state,
        strain0=current_strain, lateral0=current_lateral,
        scheme=CANONICAL_SCHEME, substeps=recovery_substeps,
        checkpoint=True, time0=time[-1], previous_target0=current_stress,
        regularization=CANONICAL_REGULARIZATION,
    )
    recovered_state = jax.tree_util.tree_map(lambda value: value[-1], recovery.state)
    residual = recovery.strain[-1]
    recovered_lateral = jnp.log(recovered_state.F[0, 0])

    probe = jnp.asarray([residual, residual + 2.0e-3])
    reload = run_strain_history(
        probe, jnp.asarray([0.0, 2.0]), params, recovered_state,
        traction_free=True, scheme=CANONICAL_SCHEME, substeps=4,
        checkpoint=True, time0=time[-1] + recovery_time,
        previous_strain0=residual, lateral0=recovered_lateral,
        regularization=CANONICAL_REGULARIZATION,
    )
    reload_stress = -reload.stress[:, 1, 1]
    reload_stiffness = (reload_stress[-1] - reload_stress[0]) / 2.0e-3
    larger_probe = jnp.asarray([residual, residual + 2.0e-2])
    larger_reload = run_strain_history(
        larger_probe, jnp.asarray([0.0, 20.0]), params, recovered_state,
        traction_free=True, scheme=CANONICAL_SCHEME, substeps=8,
        checkpoint=True, time0=time[-1] + recovery_time,
        previous_strain0=residual, lateral0=recovered_lateral,
        regularization=CANONICAL_REGULARIZATION,
    )
    reload_stress_at_two_percent = -larger_reload.stress[-1, 1, 1]

    axial_stress = -loading.stress[:, 1, 1]
    rate = jnp.abs(jnp.diff(strain) / jnp.maximum(jnp.diff(time), 1.0e-12))
    peak_stress = jnp.max(jnp.abs(axial_stress))
    max_rate = jnp.max(rate)
    work_increments = 0.5 * (axial_stress[1:] + axial_stress[:-1]) * jnp.diff(strain)
    net_work = jnp.sum(work_increments)
    input_work = jnp.sum(jnp.maximum(work_increments, 0.0))
    cumulative_absolute_work = jnp.sum(jnp.abs(work_increments))
    retained = jnp.linalg.norm(recovered_state.A0 - jnp.eye(3))

    residual_mismatch = (residual - target_residual_strain) / 0.01
    stiffness_mismatch = jnp.asarray(0.0)
    if target_reload_stiffness is not None:
        stiffness_mismatch = (reload_stiffness - target_reload_stiffness) / 200.0
    rate_violation = jax.nn.relu(max_rate / maximum_rate - 1.0)
    stress_violation = jax.nn.relu(peak_stress / maximum_stress - 1.0)
    physical_rates = jnp.diff(strain) / jnp.maximum(jnp.diff(time), 1.0e-12)
    rate_jump_penalty = jnp.mean(
        (jnp.diff(physical_rates) / maximum_rate) ** 2
    )
    objective = (
        residual_mismatch ** 2
        + stiffness_mismatch ** 2
        + 10.0 * rate_violation ** 2
        + 10.0 * stress_violation ** 2
        + 1.0e-2 * rate_jump_penalty
        + work_weight * input_work
        + time_weight * time[-1] / 1000.0
    )
    return ObservableDesignResult(
        objective, time, strain, residual, reload_stiffness, peak_stress,
        input_work, max_rate, retained, net_work, cumulative_absolute_work,
        rate_jump_penalty, time[-1], time[-1] + recovery_time + 2.0,
        recovered_state.phi, recovered_state.kappa1, recovered_state.kappa2,
        recovered_state.b1, recovered_state.b2,
        jnp.linalg.eigvalsh(recovered_state.A0),
        jnp.linalg.eigvalsh(recovered_state.A1),
        jnp.linalg.eigvalsh(recovered_state.A2),
        reload_stress_at_two_percent,
        recovered_state,
    )
