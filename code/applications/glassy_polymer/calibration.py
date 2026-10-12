"""Parameter-block transforms and conservative bounded optimization helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from .constitutive import MaterialParameters


@dataclass(frozen=True)
class ParameterBlock:
    name: str
    parameters: tuple[str, ...]
    lower_ratio: float = 0.5
    upper_ratio: float = 2.0


def pack_log_block(params: MaterialParameters, names: Sequence[str]) -> jax.Array:
    return jnp.log(jnp.asarray([getattr(params, name) for name in names]))


def replace_log_block(params: MaterialParameters, names: Sequence[str],
                      log_values: jax.Array) -> MaterialParameters:
    values = jnp.exp(log_values)
    return params._replace(**{name: value for name, value in zip(names, values)})


def replace_log_ratio_block(reference: MaterialParameters, names: Sequence[str],
                            log_ratios: jax.Array) -> MaterialParameters:
    """Apply multiplicative updates while preserving the exact zero update.

    Using ``exp(log(parameter))`` can move a nominal baseline by one floating-
    point ulp, which is enough to select a different branch in a delicate
    stress-control solve. Ratio coordinates guarantee that zero corresponds to
    the bitwise reference value before multiplication.
    """
    ratios = jnp.exp(log_ratios)
    return reference._replace(**{
        name: jnp.asarray(getattr(reference, name)) * ratio
        for name, ratio in zip(names, ratios)
    })


def bounded_adam(value_and_grad, initial: np.ndarray, lower: np.ndarray,
                 upper: np.ndarray, iterations: int = 20,
                 learning_rate: float = 2.0e-3,
                 gradient_clip: float = 1.0) -> tuple[np.ndarray, list[tuple[int, float]]]:
    """Small-step Adam with best-iterate retention for delicate histories."""
    x = np.asarray(initial, dtype=float).copy()
    first = np.zeros_like(x)
    second = np.zeros_like(x)
    best_x = x.copy()
    best_value = np.inf
    trace: list[tuple[int, float]] = []
    for iteration in range(1, iterations + 1):
        value, gradient = value_and_grad(jnp.asarray(x))
        value = float(value)
        gradient = np.asarray(gradient, dtype=float)
        trace.append((iteration, value))
        if np.isfinite(value) and value < best_value:
            best_value, best_x = value, x.copy()
        gradient = np.nan_to_num(gradient, nan=0.0, posinf=gradient_clip,
                                 neginf=-gradient_clip)
        gradient = np.clip(gradient, -gradient_clip, gradient_clip)
        first = 0.9 * first + 0.1 * gradient
        second = 0.999 * second + 0.001 * gradient * gradient
        first_hat = first / (1.0 - 0.9 ** iteration)
        second_hat = second / (1.0 - 0.999 ** iteration)
        step = learning_rate * (0.97 ** (iteration - 1))
        x = np.clip(x - step * first_hat / (np.sqrt(second_hat) + 1.0e-8),
                    lower, upper)
    return best_x, trace
