"""Joint multi-path calibration with physically bounded parameter transforms."""

from __future__ import annotations

from typing import NamedTuple, Sequence

import jax
import jax.numpy as jnp

from .constitutive import MaterialParameters, initial_state
from .protocols import run_strain_history


class StrainControlledExperiment(NamedTuple):
    strain: jax.Array
    dt: jax.Array
    measured_compressive_stress: jax.Array
    weight: jax.Array
    scheme: str = "vumat"


def positive_parameter_vector(params: MaterialParameters) -> jax.Array:
    """Log coordinates for all positive parameters; nu uses a logit."""
    values = []
    for name, value in zip(params._fields, params):
        value = jnp.asarray(value)
        if name == "nu":
            values.append(jnp.log(value / (0.499 - value)))
        else:
            values.append(jnp.log(jnp.maximum(value, 1.0e-30)))
    return jnp.stack(values)


def material_parameters(theta: jax.Array) -> MaterialParameters:
    values = []
    for name, value in zip(MaterialParameters._fields, theta):
        values.append(0.499 * jax.nn.sigmoid(value) if name == "nu" else jnp.exp(value))
    return MaterialParameters(*values)


def predict_experiment(params: MaterialParameters,
                       experiment: StrainControlledExperiment) -> jax.Array:
    result = run_strain_history(
        experiment.strain, experiment.dt, params, initial_state(params),
        traction_free=True, scheme=experiment.scheme,
    )
    return -result.stress[:, 1, 1]


def joint_residual(theta: jax.Array,
                   experiments: Sequence[StrainControlledExperiment]) -> jax.Array:
    params = material_parameters(theta)
    blocks = []
    for experiment in experiments:
        prediction = predict_experiment(params, experiment)
        scale = jnp.maximum(jnp.sqrt(jnp.mean(experiment.measured_compressive_stress ** 2)), 1.0)
        blocks.append(jnp.sqrt(experiment.weight) *
                      (prediction - experiment.measured_compressive_stress) / scale)
    return jnp.concatenate(blocks)


def joint_loss(theta: jax.Array,
               experiments: Sequence[StrainControlledExperiment]) -> jax.Array:
    residual = joint_residual(theta, experiments)
    # Smooth-L1 limits leverage of digitisation artefacts and isolated outliers.
    return jnp.mean(jnp.sqrt(1.0 + residual * residual) - 1.0)
