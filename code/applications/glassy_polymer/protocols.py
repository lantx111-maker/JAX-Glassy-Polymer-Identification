"""Loading protocols for material-point simulations."""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .constitutive import (
    EXACT_REGULARIZATION,
    ConstitutiveRegularization,
    MaterialParameters,
    MaterialState,
    StepDiagnostics,
    update_material,
    update_material_legacy,
)


class HistoryResult(NamedTuple):
    stress: jax.Array
    state: MaterialState
    diagnostics: StepDiagnostics


class UniaxialStepResult(NamedTuple):
    stress: jax.Array
    state: MaterialState
    diagnostics: StepDiagnostics
    lateral_log_strain: jax.Array


class StressHistoryResult(NamedTuple):
    strain: jax.Array
    stress: jax.Array
    state: MaterialState
    diagnostics: StepDiagnostics


def _advance_material(F, state, dt, time, params, scheme, regularization):
    """Route the isolated lineage map without admitting it as canonical."""
    if scheme == "legacy":
        return update_material_legacy(F, state, dt, time, params, regularization)
    return update_material(F, state, dt, time, params, scheme, regularization)


def isochoric_uniaxial_F(compressive_true_strain: jax.Array) -> jax.Array:
    """Initial protocol: positive input denotes axial compression."""
    axial = jnp.exp(-compressive_true_strain)
    lateral = jnp.exp(0.5 * compressive_true_strain)
    return jnp.diag(jnp.array([lateral, axial, lateral]))


def uniaxial_F(compressive_true_strain: jax.Array,
               lateral_log_strain: jax.Array) -> jax.Array:
    """Axis 1 is loaded; axes 0 and 2 are traction-free lateral directions."""
    return jnp.diag(jnp.array([
        jnp.exp(lateral_log_strain),
        jnp.exp(-compressive_true_strain),
        jnp.exp(lateral_log_strain),
    ]))


def traction_free_uniaxial_step(compressive_true_strain: jax.Array,
                                state: MaterialState, dt: jax.Array,
                                time: jax.Array, params: MaterialParameters,
                                lateral_guess: jax.Array,
                                scheme: str = "umat",
                                regularization: ConstitutiveRegularization = EXACT_REGULARIZATION,
                                audit_control: bool = False,
                                ) -> UniaxialStepResult:
    """Advance a step while enforcing zero lateral Cauchy traction implicitly."""
    def lateral_residual(eta):
        stress, _, _ = _advance_material(
            uniaxial_F(compressive_true_strain, eta), state, dt, time, params,
            scheme, regularization,
        )
        return stress[0, 0]

    def solve(f, guess):
        def body(_, eta):
            value, slope = jax.value_and_grad(f)(eta)
            safe = jnp.where(jnp.abs(slope) > 1.0e-10, slope, 1.0)
            increment = jnp.clip(value / safe, -0.02, 0.02)
            return eta - increment
        return jax.lax.fori_loop(0, 20, body, guess)

    def tangent_solve(g, y):
        response = g(jnp.ones_like(y))
        return y / jnp.where(jnp.abs(response) > 1.0e-12, response, 1.0)

    eta = jax.lax.custom_root(lateral_residual, lateral_guess, solve, tangent_solve)
    stress, state_new, diagnostics = _advance_material(
        uniaxial_F(compressive_true_strain, eta), state, dt, time, params,
        scheme, regularization,
    )
    lateral_residual_value = stress[0, 0]
    if audit_control:
        lateral_slope = jax.grad(lateral_residual)(eta)
        diagnostics = diagnostics._replace(
            control_residual_norm=jnp.abs(lateral_residual_value),
            control_min_singular_value=jnp.abs(lateral_slope),
            control_condition_number=jnp.asarray(1.0, dtype=jnp.float64),
        )
    else:
        diagnostics = diagnostics._replace(
            control_residual_norm=jnp.abs(lateral_residual_value),
        )
    return UniaxialStepResult(stress, state_new, diagnostics, eta)


def run_strain_history(strain: jax.Array, dt: jax.Array,
                       params: MaterialParameters,
                       state0: MaterialState,
                       traction_free: bool = False,
                       substeps: int = 1,
                       scheme: str = "umat",
                       regularization: ConstitutiveRegularization = EXACT_REGULARIZATION,
                       checkpoint: bool = False,
                       time0: float = 0.0,
                       previous_strain0: float | None = None,
                       lateral0: float | None = None,
                       audit_control: bool = False,
                       ) -> HistoryResult:
    strain = jnp.asarray(strain, dtype=jnp.float64)
    dt = jnp.broadcast_to(jnp.asarray(dt, dtype=jnp.float64), strain.shape)
    clock0 = jnp.where(
        jnp.asarray(time0, dtype=jnp.float64) != 0.0,
        jnp.asarray(time0, dtype=jnp.float64),
        jnp.asarray(state0.time, dtype=jnp.float64),
    )
    state0 = state0._replace(time=clock0)

    fractions = (jnp.arange(substeps, dtype=jnp.float64) + 1.0) / substeps

    def step(carry, inputs):
        state, time, eta, previous_eps = carry
        eps, step_dt = inputs
        sub_dt = step_dt / substeps
        sub_strain = previous_eps + fractions * (eps - previous_eps)

        def substep(subcarry, sub_eps):
            substate, subtime, subeta = subcarry
            time_new = subtime + sub_dt
            if traction_free:
                result = traction_free_uniaxial_step(
                    sub_eps, substate, sub_dt, time_new, params, subeta, scheme,
                    regularization, audit_control,
                )
                stress, state_new, diag, eta_new = result
            else:
                stress, state_new, diag = _advance_material(
                    isochoric_uniaxial_F(sub_eps), substate, sub_dt, time_new,
                    params, scheme, regularization,
                )
                eta_new = 0.5 * sub_eps
            return (state_new, time_new, eta_new), (stress, state_new, diag)

        differentiable_substep = jax.checkpoint(substep) if checkpoint else substep
        (state_new, time_new, eta_new), outputs = jax.lax.scan(
            differentiable_substep, (state, time, eta), sub_strain
        )
        stress_all, state_all, diag_all = outputs
        stress = jax.tree_util.tree_map(lambda value: value[-1], stress_all)
        state_output = jax.tree_util.tree_map(lambda value: value[-1], state_all)
        diag = jax.tree_util.tree_map(lambda value: value[-1], diag_all)
        return (state_new, time_new, eta_new, eps), (stress, state_output, diag)

    differentiable_step = jax.checkpoint(step) if checkpoint else step
    previous_initial = strain[0] if previous_strain0 is None else jnp.asarray(previous_strain0)
    lateral_initial = (
        0.5 * previous_initial if lateral0 is None else jnp.asarray(lateral0)
    )
    (_, _, _, _), (stress, states, diagnostics) = jax.lax.scan(
        differentiable_step,
        (state0, clock0, lateral_initial, previous_initial),
        (strain, dt),
    )
    return HistoryResult(stress, states, diagnostics)


def stress_controlled_uniaxial_step(target_compressive_stress: jax.Array,
                                    state: MaterialState, dt: jax.Array,
                                    time: jax.Array, params: MaterialParameters,
                                    strain_guess: jax.Array,
                                    lateral_guess: jax.Array,
                                    scheme: str = "umat",
                                    regularization: ConstitutiveRegularization = EXACT_REGULARIZATION,
                                    implicit_control: bool = True,
                                    audit_control: bool = False,
                                    ):
    """Implicitly enforce axial target stress and zero lateral traction."""
    initial = jnp.stack((strain_guess, lateral_guess))
    if implicit_control:
        control = _implicit_stress_control(
            target_compressive_stress, state, dt, time, params, initial,
            scheme, regularization, 25,
        )
    else:
        control = _solve_stress_control(
            target_compressive_stress, state, dt, time, params, initial,
            scheme, regularization, 25,
        )
    eps, eta = control
    stress, state_new, diagnostics = _advance_material(
        uniaxial_F(eps, eta), state, dt, time, params, scheme,
        regularization,
    )
    residual_norm = jnp.linalg.norm(jnp.stack((
        stress[0, 0], -stress[1, 1] - target_compressive_stress,
    )))
    if audit_control:
        control_jacobian = jax.jacfwd(
            lambda value: _stress_control_residual(
                value, target_compressive_stress, state, dt, time, params,
                scheme, regularization,
            )
        )(control)
        singular = jnp.linalg.svd(control_jacobian, compute_uv=False)
        diagnostics = diagnostics._replace(
            control_residual_norm=residual_norm,
            control_min_singular_value=singular[-1],
            control_condition_number=singular[0] / jnp.maximum(singular[-1], 1.0e-30),
        )
    else:
        diagnostics = diagnostics._replace(control_residual_norm=residual_norm)
    return eps, eta, stress, state_new, diagnostics


def _stress_control_residual(control, target, state, dt, time, params,
                             scheme, regularization):
    eps, eta = control
    stress, _, _ = _advance_material(
        uniaxial_F(eps, eta), state, dt, time, params, scheme, regularization
    )
    return jnp.stack((stress[0, 0], -stress[1, 1] - target))


def _solve_stress_control(target, state, dt, time, params, initial,
                          scheme, regularization, max_iter):
    def body(_, control):
        residual = lambda value: _stress_control_residual(
            value, target, state, dt, time, params, scheme, regularization
        )
        value = residual(control)
        jacobian = jax.jacfwd(residual)(control)
        increment = jnp.linalg.solve(jacobian + 1.0e-12 * jnp.eye(2), value)
        increment = jnp.clip(increment, -0.05, 0.05)
        return control - increment
    return jax.lax.fori_loop(0, max_iter, body, initial)


@partial(jax.custom_jvp, nondiff_argnums=(6, 7, 8))
def _implicit_stress_control(target, state, dt, time, params, initial,
                             scheme, regularization, max_iter):
    return _solve_stress_control(
        target, state, dt, time, params, initial, scheme,
        regularization, max_iter,
    )


@_implicit_stress_control.defjvp
def _implicit_stress_control_jvp(scheme, regularization, max_iter,
                                 primals, tangents):
    target, state, dt, time, params, initial = primals
    target_dot, state_dot, dt_dot, time_dot, params_dot, _ = tangents
    control = _implicit_stress_control(
        target, state, dt, time, params, initial,
        scheme, regularization, max_iter,
    )

    def residual_inputs(target_value, state_value, dt_value, time_value,
                        params_value):
        return _stress_control_residual(
            control, target_value, state_value, dt_value, time_value,
            params_value, scheme, regularization,
        )

    _, residual_dot = jax.jvp(
        residual_inputs, (target, state, dt, time, params),
        (target_dot, state_dot, dt_dot, time_dot, params_dot),
    )
    jacobian = jax.jacfwd(
        lambda value: _stress_control_residual(
            value, target, state, dt, time, params, scheme, regularization
        )
    )(control)
    control_dot = jnp.linalg.solve(
        jacobian + 1.0e-12 * jnp.eye(2), -residual_dot
    )
    return control, control_dot


def run_stress_history(target_compressive_stress: jax.Array, dt: jax.Array,
                       params: MaterialParameters, state0: MaterialState,
                       strain0: float = 0.0, lateral0: float = 0.0,
                       scheme: str = "umat", substeps: int = 1,
                       regularization: ConstitutiveRegularization = EXACT_REGULARIZATION,
                       checkpoint: bool = False,
                       implicit_control: bool = True,
                       time0: float = 0.0,
                       previous_target0: float = 0.0,
                       audit_control: bool = False,
                       ) -> StressHistoryResult:
    targets = jnp.asarray(target_compressive_stress, dtype=jnp.float64)
    dt = jnp.broadcast_to(jnp.asarray(dt, dtype=jnp.float64), targets.shape)
    clock0 = jnp.where(
        jnp.asarray(time0, dtype=jnp.float64) != 0.0,
        jnp.asarray(time0, dtype=jnp.float64),
        jnp.asarray(state0.time, dtype=jnp.float64),
    )
    state0 = state0._replace(time=clock0)

    fractions = (jnp.arange(substeps, dtype=jnp.float64) + 1.0) / substeps

    def step(carry, inputs):
        state, time, eps, eta, previous_target = carry
        target, step_dt = inputs
        sub_dt = step_dt / substeps
        sub_targets = previous_target + fractions * (target - previous_target)

        def substep(subcarry, subtarget):
            substate, subtime, subeps, subeta = subcarry
            time_new = subtime + sub_dt
            eps_new, eta_new, stress, state_new, diag = stress_controlled_uniaxial_step(
                subtarget, substate, sub_dt, time_new, params, subeps, subeta,
                scheme, regularization, implicit_control, audit_control,
            )
            return (state_new, time_new, eps_new, eta_new), (eps_new, stress, state_new, diag)

        differentiable_substep = jax.checkpoint(substep) if checkpoint else substep
        (state_new, time_new, eps_new, eta_new), outputs = jax.lax.scan(
            differentiable_substep, (state, time, eps, eta), sub_targets
        )
        strain_all, stress_all, states_all, diagnostics_all = outputs
        endpoint = lambda value: jax.tree_util.tree_map(lambda item: item[-1], value)
        return ((state_new, time_new, eps_new, eta_new, target),
                (strain_all[-1], stress_all[-1], endpoint(states_all), endpoint(diagnostics_all)))

    differentiable_step = jax.checkpoint(step) if checkpoint else step
    (_, _, _, _, _), (strain, stress, states, diagnostics) = jax.lax.scan(
        differentiable_step,
        (state0, clock0, jnp.asarray(strain0),
         jnp.asarray(lateral0), jnp.asarray(previous_target0)),
        (targets, dt),
    )
    return StressHistoryResult(strain, stress, states, diagnostics)
