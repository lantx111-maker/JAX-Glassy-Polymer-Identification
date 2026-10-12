"""Finite-strain isothermal VE-VP material-point model.

The implementation follows the equations and update ordering in Lan et al.'s
retained-state model.  It intentionally contains no experiment-specific loading
logic; loading/control protocols live in ``protocols.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy as jsp

jax.config.update("jax_enable_x64", True)

Array = jax.Array


@dataclass(frozen=True)
class ConstitutiveRegularization:
    """Differentiation-only regularization of constitutive switches.

    ``macaulay_width`` has stress units (MPa). Zero recovers the exact
    positive-part operation. ``activation_rate`` is retained in archived
    configurations for backward compatibility; the V4 continuous-branch
    operator does not use the discontinuous source-UMAT gate.
    """

    macaulay_width: float = 0.0
    activation_rate: float = 0.0


EXACT_REGULARIZATION = ConstitutiveRegularization()


class MaterialParameters(NamedTuple):
    E: Array
    nu: Array
    G_R: Array
    I_m: Array
    gamma0: Array
    m0: Array
    alpha_p: Array
    s0: Array
    kappa1_sat: Array
    h1: Array
    d: Array
    kappa2_sat: Array
    h2: Array
    s1: Array
    s2: Array
    m_ve: Array
    b: Array
    zeta1: Array
    zeta2: Array
    h3: Array
    b1_sat: Array
    b2_sat: Array


class MaterialState(NamedTuple):
    F: Array
    Fp: Array
    A0: Array
    A1: Array
    A2: Array
    kappa1: Array
    kappa2: Array
    gamma_p: Array
    phi: Array
    b1: Array
    b2: Array
    time: Array


class StepDiagnostics(NamedTuple):
    q0: Array
    q1: Array
    q2: Array
    residual: Array
    newton_iterations: Array
    tau0: Array
    control_residual_norm: Array
    control_min_singular_value: Array
    control_condition_number: Array


def pc_parameters() -> MaterialParameters:
    return MaterialParameters(
        E=2800.0, nu=0.37, G_R=14.5, I_m=6.3,
        gamma0=1.7e-3, m0=0.017, alpha_p=0.116, s0=6.3,
        kappa1_sat=26.8, h1=180.0, d=9.3,
        kappa2_sat=100.0, h2=0.3,
        s1=7.0, s2=12.0, m_ve=0.35,
        b=19.0, zeta1=30.0, zeta2=0.013,
        h3=30.0, b1_sat=1000.0, b2_sat=400.0,
    )


def pmma_parameters() -> MaterialParameters:
    return MaterialParameters(
        E=3200.0, nu=0.33, G_R=14.0, I_m=8.0,
        gamma0=1.7e-3, m0=0.04, alpha_p=0.2, s0=27.0,
        kappa1_sat=26.0, h1=150.0, d=8.5,
        kappa2_sat=150.0, h2=0.36,
        s1=13.0, s2=28.0, m_ve=0.14,
        b=13.0, zeta1=39.0, zeta2=0.05,
        h3=40.0, b1_sat=600.0, b2_sat=160.0,
    )


def initial_state(params: MaterialParameters) -> MaterialState:
    eye = jnp.eye(3, dtype=jnp.float64)
    zero = jnp.asarray(0.0, dtype=jnp.float64)
    return MaterialState(eye, eye, eye, eye, eye, zero, zero, zero, zero,
                         jnp.asarray(params.E), jnp.asarray(params.E), zero)


def _sym(a: Array) -> Array:
    return 0.5 * (a + a.T)


def running_rate_average(phi: Array, rate: Array, dt: Array,
                         start_time: Array, time_regularization: float,
                         ) -> Array:
    """Exact step for ``dphi/dt=(rate-phi)/(t+t_eps)`` at constant rate."""
    end_time = start_time + dt
    return ((start_time + time_regularization) * phi + dt * rate) / (
        end_time + time_regularization
    )


def _dev(a: Array) -> Array:
    return a - jnp.trace(a) * jnp.eye(3, dtype=a.dtype) / 3.0


def _spd_function(a: Array, fn, floor: float = 1.0e-12) -> Array:
    vals, vecs = jnp.linalg.eigh(_sym(a))
    vals = jnp.maximum(vals, floor)
    return (vecs * fn(vals)) @ vecs.T


@jax.custom_jvp
def spd_log(a: Array) -> Array:
    """Matrix logarithm of a 3x3 SPD tensor with a repeated-root-safe JVP."""
    vals, vecs = jnp.linalg.eigh(_sym(a))
    vals = jnp.maximum(vals, 1.0e-12)
    return (vecs * jnp.log(vals)) @ vecs.T


@spd_log.defjvp
def _spd_log_jvp(primals, tangents):
    (a,), (a_dot,) = primals, tangents
    raw_vals, vecs = jnp.linalg.eigh(_sym(a))
    floor = jnp.asarray(1.0e-12, dtype=raw_vals.dtype)
    vals = jnp.maximum(raw_vals, floor)
    out = (vecs * jnp.log(vals)) @ vecs.T
    ai = raw_vals[:, None]
    aj = raw_vals[None, :]
    vi = vals[:, None]
    vj = vals[None, :]

    # For positive eigenvalues use log1p(r)/r, which remains accurate when
    # two roots nearly coincide.  A fourth-order series avoids the removable
    # r=0 singularity.  If the defensive eigenvalue floor is active, the JVP
    # differentiates the actual clipped spectral function rather than an
    # unclipped logarithm.
    r = (vi - vj) / vj
    small = jnp.abs(r) < 1.0e-4
    series = (1.0 - 0.5 * r + r * r / 3.0
              - r * r * r / 4.0 + r ** 4 / 5.0) / vj
    safe_r = jnp.where(small, 1.0, r)
    regular = jnp.log1p(safe_r) / (vj * safe_r)
    positive_difference = jnp.where(small, series, regular)

    raw_delta = ai - aj
    raw_safe = jnp.where(jnp.abs(raw_delta) > 1.0e-30, raw_delta, 1.0)
    clipped_secant = (jnp.log(vi) - jnp.log(vj)) / raw_safe
    both_clipped = (ai <= floor) & (aj <= floor)
    one_clipped = (ai <= floor) | (aj <= floor)
    divided_difference = jnp.where(
        both_clipped,
        0.0,
        jnp.where(one_clipped, clipped_secant, positive_difference),
    )
    local_dot = vecs.T @ _sym(a_dot) @ vecs
    out_dot = vecs @ (divided_difference * local_dot) @ vecs.T
    return _sym(out), _sym(out_dot)


@jax.custom_jvp
def spd_power(a: Array, power: Array) -> Array:
    """Spectral power of a 3x3 SPD tensor with an explicit Fr\'echet JVP.

    The matrix and exponent derivatives are evaluated together.  A stable
    divided difference is used when eigenvalues coalesce, so the derivative
    does not depend on an arbitrary eigenbasis inside a repeated eigenspace.
    """
    vals, vecs = jnp.linalg.eigh(_sym(a))
    vals = jnp.maximum(vals, 1.0e-12)
    return _sym((vecs * vals ** power) @ vecs.T)


@spd_power.defjvp
def _spd_power_jvp(primals, tangents):
    (a, power), (a_dot, power_dot) = primals, tangents
    raw_vals, vecs = jnp.linalg.eigh(_sym(a))
    floor = jnp.asarray(1.0e-12, dtype=raw_vals.dtype)
    vals = jnp.maximum(raw_vals, floor)
    powered = vals ** power
    out = (vecs * powered) @ vecs.T

    ai = raw_vals[:, None]
    aj = raw_vals[None, :]
    vi = vals[:, None]
    vj = vals[None, :]
    r = (vi - vj) / vj
    small = jnp.abs(r) < 1.0e-4
    # ((1+r)^p-1)/r through fourth order; at r=0 this is exactly p.
    series = (power
              + 0.5 * power * (power - 1.0) * r
              + power * (power - 1.0) * (power - 2.0) * r ** 2 / 6.0
              + power * (power - 1.0) * (power - 2.0)
              * (power - 3.0) * r ** 3 / 24.0
              + power * (power - 1.0) * (power - 2.0)
              * (power - 3.0) * (power - 4.0) * r ** 4 / 120.0)
    safe_r = jnp.where(small, 1.0, r)
    regular = jnp.expm1(power * jnp.log1p(safe_r)) / safe_r
    positive_difference = vj ** (power - 1.0) * jnp.where(
        small, series, regular
    )

    raw_delta = ai - aj
    raw_safe = jnp.where(jnp.abs(raw_delta) > 1.0e-30, raw_delta, 1.0)
    clipped_secant = (vi ** power - vj ** power) / raw_safe
    both_clipped = (ai <= floor) & (aj <= floor)
    one_clipped = (ai <= floor) | (aj <= floor)
    divided_difference = jnp.where(
        both_clipped,
        0.0,
        jnp.where(one_clipped, clipped_secant, positive_difference),
    )

    local_dot = vecs.T @ _sym(a_dot) @ vecs
    matrix_dot = vecs @ (divided_difference * local_dot) @ vecs.T
    exponent_dot = (vecs * (powered * jnp.log(vals))) @ vecs.T * power_dot
    return _sym(out), _sym(matrix_dot + exponent_dot)


def sym_expm(a: Array) -> Array:
    return jsp.linalg.expm(_sym(a))


def elastic_mandel(F: Array, Fp: Array, params: MaterialParameters) -> Array:
    Fe = F @ jnp.linalg.inv(Fp)
    C = _sym(Fe.T @ Fe)
    Ee = 0.5 * spd_log(C)
    G = params.E / (2.0 * (1.0 + params.nu))
    K = params.E / (3.0 * (1.0 - 2.0 * params.nu))
    return 2.0 * G * _dev(Ee) + K * jnp.trace(Ee) * jnp.eye(3)


def mandel_to_cauchy(F: Array, Fp: Array, M: Array,
                     params: MaterialParameters) -> Array:
    Fe = F @ jnp.linalg.inv(Fp)
    J = jnp.linalg.det(Fe)
    # Compute the same isotropic Hencky stress directly in the spatial frame;
    # this avoids differentiating polar eigenvectors at repeated stretches.
    be = _sym(Fe @ Fe.T)
    ee = 0.5 * spd_log(be)
    G = params.E / (2.0 * (1.0 + params.nu))
    K = params.E / (3.0 * (1.0 - 2.0 * params.nu))
    tau = 2.0 * G * _dev(ee) + K * jnp.trace(ee) * jnp.eye(3)
    del M  # retained in the interface to document the Mandel/spatial pairing
    return _sym(tau) / J


def network_cauchy(F: Array, params: MaterialParameters) -> Array:
    J = jnp.linalg.det(F)
    Fdis = F / jnp.cbrt(J)
    Bdis = _sym(Fdis @ Fdis.T)
    denom = 1.0 - (jnp.trace(Bdis) - 3.0) / params.I_m
    denom = jnp.maximum(denom, 1.0e-8)
    return params.G_R * _dev(Bdis) / (J * denom)


def _driving(M: Array, A: Array, modulus: Array, alpha_p: Array) -> tuple[Array, Array, Array]:
    Mb = modulus * spd_log(A)
    eff = _dev(M - Mb)
    squared_tau = 0.5 * jnp.sum(eff * eff)
    # The tiny norm floor keeps the scalar invariant differentiable at the
    # origin; the rate direction itself is still set exactly to zero below the
    # declared tolerance.
    regularized_tau = jnp.sqrt(jnp.maximum(squared_tau, 1.0e-30))
    active_direction = squared_tau > 1.0e-28
    tau = jnp.where(active_direction, regularized_tau, 0.0)
    drive = tau + alpha_p * jnp.trace(M) / 3.0
    safe_tau = jnp.maximum(tau, 1.0e-15)
    trial_direction = eff / (jnp.sqrt(2.0) * safe_tau)
    # At zero deviatoric driving stress the direction is not unique.  The
    # constitutive rate tensor is defined as zero without evaluating a
    # normalized direction, avoiding a fictitious preferred direction.
    N = jnp.where(active_direction, trial_direction, jnp.zeros_like(eff))
    return drive, tau, N


def _positive_part(value: Array, width: float) -> Array:
    """Exact or zero-subthreshold-flow C1 Macaulay regularization."""
    if width <= 0.0:
        return jnp.maximum(value, 0.0)
    width_array = jnp.asarray(width, dtype=value.dtype)
    transition = value * value / (2.0 * width_array)
    return jnp.where(
        value <= 0.0,
        0.0,
        jnp.where(value < width_array, transition, value - 0.5 * width_array),
    )


def _positive_power_rate(drive: Array, resistance: Array, gamma0: Array,
                         exponent: Array, macaulay_width: float = 0.0) -> Array:
    positive = _positive_part(drive, macaulay_width)
    resistance = jnp.maximum(resistance, 1.0e-12)
    # log form avoids intermediate overflow for small rate-sensitivity exponents.
    log_rate = jnp.log(gamma0) + jnp.log(jnp.maximum(positive / resistance, 1.0e-300)) / exponent
    return jnp.where(positive > 0.0, jnp.exp(jnp.minimum(log_rate, 50.0)), 0.0)


def _kappas(q: Array, state: MaterialState, F: Array, dt: Array,
            params: MaterialParameters) -> tuple[Array, Array]:
    """Frozen-coefficient scalar updates for constant q and end-step lambda."""
    target1 = params.kappa1_sat * jnp.exp(-params.d * state.gamma_p)
    k1 = target1 + (state.kappa1 - target1) * jnp.exp(-dt * params.h1 * q)
    lam = jnp.sqrt(jnp.sum(F * F) / 3.0)
    coefficient2 = dt * params.h2 * (lam - 1.0) * q
    k2 = (params.kappa2_sat
          + (state.kappa2 - params.kappa2_sat) * jnp.exp(-coefficient2))
    return k1, k2


def _legacy_kappas(q: Array, state: MaterialState, F: Array, dt: Array,
                    params: MaterialParameters) -> tuple[Array, Array]:
    """Forward-Euler trial functions retained only for source-code parity."""
    target1 = params.kappa1_sat * jnp.exp(-params.d * state.gamma_p)
    k1 = state.kappa1 + dt * params.h1 * (target1 - state.kappa1) * q
    lam = jnp.sqrt(jnp.sum(F * F) / 3.0)
    k2 = (state.kappa2 + dt * params.h2 * (lam - 1.0)
          * (params.kappa2_sat - state.kappa2) * q)
    return k1, k2


def _legacy_q0_stress_residual(q: Array, F: Array, state: MaterialState,
                               dt: Array, params: MaterialParameters,
                               regularization: ConstitutiveRegularization,
                               ) -> Array:
    Mtr = elastic_mandel(F, state.Fp, params)
    drive, tau, _ = _driving(Mtr, state.A0, params.b * params.zeta1,
                             params.alpha_p)
    effective_drive = jnp.where(
        tau > 0.0, _positive_part(drive, regularization.macaulay_width), 0.0
    )
    G = params.E / (2.0 * (1.0 + params.nu))
    k1, k2 = _legacy_kappas(q, state, F, dt, params)
    return (effective_drive - G * dt * q
            - (params.s0 + k1 + k2) * (q / params.gamma0) ** params.m0)


def _legacy_solve_q0(F: Array, state: MaterialState, dt: Array,
                     params: MaterialParameters,
                     regularization: ConstitutiveRegularization,
                     max_iter: int = 80) -> tuple[Array, Array, Array, Array, Array]:
    """Historical damped-Newton solve used to audit the Fortran lineage."""
    Mtr = elastic_mandel(F, state.Fp, params)
    drive, tau, direction = _driving(
        Mtr, state.A0, params.b * params.zeta1, params.alpha_p
    )
    q = _positive_power_rate(
        jnp.where(tau > 0.0, drive, 0.0),
        params.s0 + state.kappa1 + state.kappa2,
        params.gamma0, params.m0, regularization.macaulay_width,
    )
    q = jnp.clip(q, 0.0, 1.0e6)

    def body(_, value):
        residual = lambda trial: _legacy_q0_stress_residual(
            trial, F, state, dt, params, regularization
        )
        r, slope = jax.value_and_grad(residual)(value)
        safe_slope = jnp.where(jnp.abs(slope) > 1.0e-14, slope, 1.0)
        proposal = jnp.clip(value - r / safe_slope, 0.0, 1.0e6)
        return 0.5 * value + 0.5 * proposal

    q = jax.lax.fori_loop(0, max_iter, body, q)
    residual = _legacy_q0_stress_residual(
        q, F, state, dt, params, regularization
    )
    return q, residual, jnp.asarray(max_iter), tau, direction


def _q0_residual(q: Array, F: Array, state: MaterialState, dt: Array,
                 params: MaterialParameters,
                 regularization: ConstitutiveRegularization) -> Array:
    Mtr = elastic_mandel(F, state.Fp, params)
    drive, tau, _ = _driving(Mtr, state.A0, params.b * params.zeta1, params.alpha_p)
    effective_drive = jnp.where(
        tau > 0.0, _positive_part(drive, regularization.macaulay_width), 0.0
    )
    G = params.E / (2.0 * (1.0 + params.nu))
    k1, k2 = _kappas(q, state, F, dt, params)
    resistance = params.s0 + k1 + k2
    # Rate-space residual is algebraically equivalent to the stress-space
    # power-law equation for an active branch, but has a finite unit slope at
    # q=0. This removes the artificial q**m derivative singularity (m<1)
    # without changing the converged positive root.
    rate = _positive_power_rate(
        effective_drive - G * dt * q, resistance, params.gamma0, params.m0,
        regularization.macaulay_width,
    )
    return q - rate


def _q0_stress_residual(q: Array, F: Array, state: MaterialState, dt: Array,
                        params: MaterialParameters,
                        regularization: ConstitutiveRegularization) -> Array:
    """Code-authoritative stress-space equation used by the primal Newton solve."""
    Mtr = elastic_mandel(F, state.Fp, params)
    drive, tau, _ = _driving(Mtr, state.A0, params.b * params.zeta1, params.alpha_p)
    effective_drive = jnp.where(
        tau > 0.0, _positive_part(drive, regularization.macaulay_width), 0.0
    )
    G = params.E / (2.0 * (1.0 + params.nu))
    k1, k2 = _kappas(q, state, F, dt, params)
    return (effective_drive - G * dt * q
            - (params.s0 + k1 + k2) * (q / params.gamma0) ** params.m0)


@partial(jax.custom_jvp, nondiff_argnums=(4, 5))
def _implicit_q0(F: Array, state: MaterialState, dt: Array,
                 params: MaterialParameters,
                 regularization: ConstitutiveRegularization,
                 max_iter: int) -> Array:
    """Solve the scalar VP equation; differentiate its converged residual.

    The active stress-space residual has a physical bracket: its value is
    non-negative at ``q=0`` and non-positive once the elastic trial driving
    stress has been exhausted by ``G dt q``.  Bisection on that bracket is
    deliberately used instead of an undamped Newton iteration.  The latter
    can jump between the very flat ``q**m`` branch and a clipped artificial
    rate when ``m << 1``. The solve follows the admissible root selected by
    this bracket and update procedure; global root uniqueness is not assumed.
    The implicit derivative is valid on a differentiable selected branch
    with nonzero residual derivative with respect to ``q``.
    """
    Mtr = elastic_mandel(F, state.Fp, params)
    drive, tau, _ = _driving(Mtr, state.A0, params.b * params.zeta1, params.alpha_p)
    effective_drive = jnp.where(
        tau > 0.0, _positive_part(drive, regularization.macaulay_width), 0.0
    )
    q_rate = _positive_power_rate(
        effective_drive, params.s0 + state.kappa1 + state.kappa2,
        params.gamma0, params.m0, 0.0,
    )
    q_rate = jnp.clip(q_rate, 0.0, 1.0e6)

    def positive_step(_):
        G = params.E / (2.0 * (1.0 + params.nu))
        exhausted = effective_drive / jnp.maximum(G * dt, 1.0e-30)
        # At this bound the elastic part of the driving stress is exhausted;
        # it is also the first physically admissible sign-changing bracket.
        # A rate-law predictor can be enormous for m << 1 and must not expand
        # the bracket into the reverse-kappa2 overflow domain.
        upper = jnp.maximum(exhausted, 1.0e-300)
        log_lower = jnp.log(jnp.asarray(1.0e-300, dtype=dt.dtype))
        log_upper = jnp.log(upper)

        def body(_, bracket):
            lower_log, upper_log = bracket
            midpoint_log = 0.5 * (lower_log + upper_log)
            midpoint = jnp.exp(midpoint_log)
            value = _q0_stress_residual(
                midpoint, F, state, dt, params, regularization
            )
            return jax.lax.cond(
                value > 0.0,
                lambda __: (midpoint_log, upper_log),
                lambda __: (lower_log, midpoint_log),
                operand=None,
            )

        lower_log, upper_log = jax.lax.fori_loop(
            0, max_iter, body,
            (log_lower, log_upper),
        )
        return jnp.exp(0.5 * (lower_log + upper_log))

    return jax.lax.cond(
        (effective_drive > 0.0) & (dt > 0.0) & (q_rate > 0.0),
        positive_step,
        lambda _: jnp.where(effective_drive > 0.0, q_rate, 0.0),
        operand=None,
    )


@_implicit_q0.defjvp
def _implicit_q0_jvp(regularization, max_iter, primals, tangents):
    F, state, dt, params = primals
    F_dot, state_dot, dt_dot, params_dot = tangents
    q = _implicit_q0(F, state, dt, params, regularization, max_iter)

    def linearize(residual_function):
        def residual_inputs(F_value, state_value, dt_value, params_value):
            return residual_function(
                q, F_value, state_value, dt_value, params_value, regularization
            )
        _, residual_dot = jax.jvp(
            residual_inputs, (F, state, dt, params),
            (F_dot, state_dot, dt_dot, params_dot),
        )
        residual_slope = jax.grad(
            lambda value: residual_function(
                value, F, state, dt, params, regularization
            )
        )(q)
        return residual_dot, residual_slope

    # Match the primal stress-space residual on every active root. Exactly at
    # q=0, where q**m has an infinite derivative for m<1, use the algebraically
    # equivalent rate-space residual with finite unit slope.
    residual_dot, residual_slope = jax.lax.cond(
        q > 1.0e-14,
        lambda _: linearize(_q0_stress_residual),
        lambda _: linearize(_q0_residual),
        operand=None,
    )
    safe_slope = jnp.where(jnp.abs(residual_slope) > 1.0e-14,
                           residual_slope, 1.0)
    return q, -residual_dot / safe_slope


def _solve_q0(F: Array, state: MaterialState, dt: Array, params: MaterialParameters,
              max_iter: int = 60,
              regularization: ConstitutiveRegularization = EXACT_REGULARIZATION,
              ) -> tuple[Array, Array, Array, Array, Array]:
    Mtr = elastic_mandel(F, state.Fp, params)
    _, tau, N = _driving(Mtr, state.A0, params.b * params.zeta1, params.alpha_p)
    q = _implicit_q0(F, state, dt, params, regularization, max_iter)
    residual = jax.lax.cond(
        q > 1.0e-14,
        lambda value: _q0_stress_residual(
            value, F, state, dt, params, regularization
        ),
        lambda value: _q0_residual(
            value, F, state, dt, params, regularization
        ),
        q,
    )
    iterations = jnp.asarray(max_iter)
    return q, residual, iterations, tau, N


def _explicit_q0(state: MaterialState, params: MaterialParameters,
                 regularization: ConstitutiveRegularization = EXACT_REGULARIZATION,
                 ) -> tuple[Array, Array, Array]:
    """Production VUMAT flow rule, evaluated from the increment-start state."""
    M_start = elastic_mandel(state.F, state.Fp, params)
    drive, tau, direction = _driving(
        M_start, state.A0, params.b * params.zeta1, params.alpha_p
    )
    q = jnp.where(tau > 0.0, _positive_power_rate(
        drive, params.s0 + state.kappa1 + state.kappa2,
        params.gamma0, params.m0, regularization.macaulay_width,
    ), 0.0)
    return q, tau, direction


def update_material(F: Array, state: MaterialState, dt: Array, time: Array,
                    params: MaterialParameters,
                    scheme: str = "umat",
                    regularization: ConstitutiveRegularization = EXACT_REGULARIZATION,
                    ) -> tuple[Array, MaterialState, StepDiagnostics]:
    """Advance one material point under a prescribed total deformation gradient."""
    dt = jnp.asarray(dt, dtype=jnp.float64)
    # The constitutive clock is part of the restartable state.  ``time`` is
    # retained in the call signature for control-root compatibility, but the
    # state clock is authoritative and advances autonomously by ``dt``.
    time = jnp.asarray(time, dtype=jnp.float64)
    start_time = jnp.asarray(state.time, dtype=jnp.float64)
    end_time = start_time + dt
    Mtr = elastic_mandel(F, state.Fp, params)
    if scheme == "umat":
        q0, residual, nit, tau0, N0 = _solve_q0(
            F, state, dt, params, regularization=regularization
        )
    elif scheme == "vumat":
        q0, tau0, N0 = _explicit_q0(state, params, regularization)
        residual, nit = jnp.asarray(0.0), jnp.asarray(0)
    else:
        raise ValueError(f"Unknown integration scheme: {scheme}")

    # The reference UMAT evaluates the VE trial stresses at increment start.
    M_start = elastic_mandel(state.F, state.Fp, params)
    drive1, _, N1 = _driving(M_start, state.A1, state.b1, params.alpha_p)
    drive2, _, N2 = _driving(M_start, state.A2, state.b2, params.alpha_p)
    q1 = _positive_power_rate(
        drive1, params.s1, params.gamma0, params.m_ve,
        regularization.macaulay_width,
    )
    q2 = _positive_power_rate(
        drive2, params.s2, params.gamma0, params.m_ve,
        regularization.macaulay_width,
    )
    scale = 1.0 / jnp.sqrt(2.0)
    D0, D1, D2 = scale * q0 * N0, scale * q1 * N1, scale * q2 * N2
    if scheme == "umat":
        # The stabilized canonical operator advances the common inelastic map
        # with every active branch.  The source UMAT's q0-only gate is retained
        # exclusively in ``update_material_legacy``: it is discontinuous when
        # q1 or q2 remains active and can make the outer traction root absent.
        Fp_new = sym_expm(dt * (D0 + D1 + D2)) @ state.Fp
    else:
        Fp_new = (jnp.eye(3) + dt * (D0 + D1 + D2)) @ state.Fp
    k1, k2 = _kappas(q0, state, F, dt, params)
    gamma_p = state.gamma_p + dt * q0

    lnA0 = spd_log(state.A0)
    time_regularization = 1.0e-5 if scheme == "umat" else 1.0e-4
    # Exact increment for dphi/dt=(q0-phi)/(t+t_eps) when the converged q0 is
    # held constant over the material step.  Since
    # d[(t+t_eps)phi]/dt=q0, this update removes the singular explicit-Euler
    # first step while preserving the continuous running-rate definition.
    phi = running_rate_average(
        state.phi, q0, dt, start_time, time_regularization
    )
    # SPD-preserving frozen-coefficient Strang split of
    # A0_dot=D0 A0+A0 D0-c A0 log(A0).  The relaxation subflow is the
    # spectral power A -> A**exp(-c dt), and the transport subflow is an
    # exact congruence.  Recoverable branches use their exact congruence
    # subflows.  These maps remove forward-Euler loss of positive definiteness.
    recovery_coefficient = (
        params.zeta1 * q0
        + params.zeta2 * jnp.exp(-state.gamma_p) * state.phi
        * (params.b * params.zeta1)
    )
    half_power = jnp.exp(-0.5 * dt * recovery_coefficient)
    A0_half = spd_power(state.A0, half_power)
    flow0 = sym_expm(dt * D0)
    A0_transport = _sym(flow0 @ A0_half @ flow0.T)
    A0 = spd_power(A0_transport, half_power)
    flow1 = sym_expm(dt * D1)
    flow2 = sym_expm(dt * D2)
    A1 = _sym(flow1 @ state.A1 @ flow1.T)
    A2 = _sym(flow2 @ state.A2 @ flow2.T)
    b1 = params.b1_sat + (state.b1 - params.b1_sat) * jnp.exp(-dt * params.h3 * q0)
    b2 = params.b2_sat + (state.b2 - params.b2_sat) * jnp.exp(-dt * params.h3 * q0)

    new_state = MaterialState(
        F, Fp_new, A0, A1, A2, k1, k2, gamma_p, phi, b1, b2, end_time
    )
    M = elastic_mandel(F, Fp_new, params)
    stress = mandel_to_cauchy(F, Fp_new, M, params) + network_cauchy(F, params)
    nan = jnp.asarray(jnp.nan, dtype=jnp.float64)
    diag = StepDiagnostics(q0, q1, q2, residual, nit, tau0, nan, nan, nan)
    return stress, new_state, diag


def update_material_legacy(
        F: Array, state: MaterialState, dt: Array, time: Array,
        params: MaterialParameters,
        regularization: ConstitutiveRegularization = EXACT_REGULARIZATION,
        ) -> tuple[Array, MaterialState, StepDiagnostics]:
    """Historical UMAT-ordered Euler map, isolated for lineage parity only.

    Scientific predictions use :func:`update_material`.  Keeping this map
    explicit prevents a compiled-source parity claim from being silently
    transferred to the V4 stabilized discrete operator.
    """
    dt = jnp.asarray(dt, dtype=jnp.float64)
    time = jnp.asarray(time, dtype=jnp.float64)
    q0, residual, nit, tau0, N0 = _legacy_solve_q0(
        F, state, dt, params, regularization
    )
    M_start = elastic_mandel(state.F, state.Fp, params)
    drive1, _, N1 = _driving(M_start, state.A1, state.b1, params.alpha_p)
    drive2, _, N2 = _driving(M_start, state.A2, state.b2, params.alpha_p)
    q1 = _positive_power_rate(
        drive1, params.s1, params.gamma0, params.m_ve,
        regularization.macaulay_width,
    )
    q2 = _positive_power_rate(
        drive2, params.s2, params.gamma0, params.m_ve,
        regularization.macaulay_width,
    )
    scale = 1.0 / jnp.sqrt(2.0)
    D0, D1, D2 = scale * q0 * N0, scale * q1 * N1, scale * q2 * N2
    candidate = sym_expm(dt * (D0 + D1 + D2)) @ state.Fp
    Fp = jnp.where(q0 > 0.0, candidate, state.Fp)
    k1, k2 = _legacy_kappas(q0, state, F, dt, params)
    gamma_p = state.gamma_p + dt * q0
    lnA0 = spd_log(state.A0)
    # The source UMAT divides by the increment-start total time.  This is the
    # lineage behavior responsible for the first-step singular amplification;
    # it is retained here only so the parity audit remains literal.
    phi = state.phi + dt * (q0 - state.phi) / (state.time + 1.0e-5)
    recovery = (
        params.zeta1 * q0 * (state.A0 @ lnA0)
        + params.zeta2 * jnp.exp(-state.gamma_p) * state.phi
        * (params.b * params.zeta1) * (state.A0 @ lnA0)
    )
    A0 = _sym(state.A0 + dt * (
        D0 @ state.A0 + state.A0 @ D0 - recovery
    ))
    A1 = _sym(state.A1 + dt * (D1 @ state.A1 + state.A1 @ D1))
    A2 = _sym(state.A2 + dt * (D2 @ state.A2 + state.A2 @ D2))
    b1 = state.b1 + dt * params.h3 * (params.b1_sat - state.b1) * q0
    b2 = state.b2 + dt * params.h3 * (params.b2_sat - state.b2) * q0
    new_state = MaterialState(
        F, Fp, A0, A1, A2, k1, k2, gamma_p, phi, b1, b2, state.time + dt
    )
    M = elastic_mandel(F, Fp, params)
    stress = mandel_to_cauchy(F, Fp, M, params) + network_cauchy(F, params)
    nan = jnp.asarray(jnp.nan, dtype=jnp.float64)
    diagnostics = StepDiagnostics(
        q0, q1, q2, residual, nit, tau0, nan, nan, nan
    )
    return stress, new_state, diagnostics
