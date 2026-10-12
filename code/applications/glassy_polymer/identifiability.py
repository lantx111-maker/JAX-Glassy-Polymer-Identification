"""Local and profile-oriented parameter-identifiability diagnostics."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


class IdentifiabilityResult(NamedTuple):
    jacobian: jax.Array
    singular_values: jax.Array
    right_singular_vectors: jax.Array
    fisher: jax.Array
    correlation: jax.Array
    effective_rank: jax.Array


def local_identifiability(observation_map, theta: jax.Array,
                          noise_scale: jax.Array | float = 1.0,
                          rank_tolerance: float = 1.0e-6) -> IdentifiabilityResult:
    """Differentiate all observables and expose sloppy parameter combinations."""
    jac = jax.jacrev(observation_map)(theta)
    jac = jac.reshape((-1, theta.size)) / jnp.asarray(noise_scale)
    _, singular_values, vt = jnp.linalg.svd(jac, full_matrices=False)
    fisher = jac.T @ jac
    covariance = jnp.linalg.pinv(fisher, rtol=rank_tolerance)
    std = jnp.sqrt(jnp.maximum(jnp.diag(covariance), 1.0e-30))
    correlation = covariance / (std[:, None] * std[None, :])
    threshold = rank_tolerance * jnp.maximum(singular_values[0], 1.0e-30)
    effective_rank = jnp.sum(singular_values > threshold)
    return IdentifiabilityResult(
        jac, singular_values, vt, fisher, correlation, effective_rank
    )

