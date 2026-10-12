"""Finite-change checks for the analytical scale gauge and inherited clock."""

import jax
import jax.numpy as jnp
import numpy as np

from applications.glassy_polymer.constitutive import (
    initial_state, pc_parameters, pmma_parameters, update_material,
)


@jax.jit
def _advance_output(state, params, start_amplitude, end_amplitude, duration):
    def increment(carry, fraction):
        amplitude = start_amplitude + fraction * (end_amplitude - start_amplitude)
        deformation = jnp.asarray([
            [jnp.exp(.4 * amplitude), .12 * amplitude, 0.0],
            [0.0, jnp.exp(-amplitude), .07 * amplitude],
            [.025 * amplitude, 0.0, jnp.exp(.6 * amplitude)],
        ])
        dt = duration / 32
        stress, updated, diagnostics = update_material(
            deformation, carry, dt, carry.time + dt, params)
        return updated, (stress, diagnostics.residual)

    updated, (stresses, residuals) = jax.lax.scan(
        increment, state, (jnp.arange(32) + 1) / 32)
    return stresses[-1], updated, jnp.max(jnp.abs(residuals))


def _history(params, state=None, first=0, last=16):
    """A non-coaxial load--unload path independent of digitised source data."""
    state = initial_state(params) if state is None else state
    output = []
    for index in range(first, last):
        amplitude = .008 * (index + 1 if index < 10 else 19 - index)
        previous = .008 * (index if index <= 10 else 20 - index)
        dt = .2 + .03 * index
        stress, state, residual = _advance_output(state, params, previous, amplitude, dt)
        assert np.all(np.isfinite(np.asarray(stress)))
        assert float(residual) < 1.e-8
        output.append((stress, state))
    return output


def test_finite_scale_symmetry_preserves_stress_and_covariant_state():
    for factory in (pc_parameters, pmma_parameters):
        base = factory()
        baseline = _history(base)
        for scale in (.5, 2.0):
            primary = scale ** base.m0
            recoverable = scale ** base.m_ve
            transformed = base._replace(
                gamma0=scale * base.gamma0,
                s0=primary * base.s0,
                kappa1_sat=primary * base.kappa1_sat,
                kappa2_sat=primary * base.kappa2_sat,
                s1=recoverable * base.s1,
                s2=recoverable * base.s2,
            )
            for (stress, state), (stress_t, state_t) in zip(baseline, _history(transformed)):
                np.testing.assert_allclose(stress_t, stress, rtol=1.e-9, atol=1.e-8)
                for name in state._fields:
                    factor = primary if name in ("kappa1", "kappa2") else 1.0
                    np.testing.assert_allclose(getattr(state_t, name), factor * getattr(state, name),
                                               rtol=1.e-9, atol=1.e-10)


def test_running_rate_identity_survives_complete_state_restart():
    params = pmma_parameters()
    uninterrupted = _history(params)
    resumed = _history(params, state=uninterrupted[7][1], first=8)
    for (stress, state), (stress_r, state_r) in zip(uninterrupted[8:], resumed):
        np.testing.assert_allclose(stress_r, stress, rtol=1.e-12, atol=1.e-12)
        for value, resumed_value in zip(state, state_r):
            np.testing.assert_allclose(resumed_value, value, rtol=1.e-12, atol=1.e-12)
    for _, state in uninterrupted:
        np.testing.assert_allclose(state.phi, state.gamma_p / (state.time + 1.e-5),
                                   rtol=1.e-12, atol=1.e-14)
