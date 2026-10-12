"""Full-resolution, curve- and protocol-balanced inverse objectives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class ProtocolResidual:
    """One complete experimental curve in a multi-path objective."""

    material: str
    family: str
    balance_family: str
    record_id: str
    label: str
    prediction: jax.Array
    measurement: jax.Array
    point_weight: jax.Array
    scale: float


def path_integration_weights(coordinate: np.ndarray) -> np.ndarray:
    """Trapezoidal weights on cumulative path length, normalized to one.

    This retains every experimental point while preventing locally dense
    digitization from dominating a curve. Reversals are preserved because the
    path coordinate uses absolute increments rather than sorting the data.
    """
    coordinate = np.asarray(coordinate, dtype=float)
    if coordinate.ndim != 1 or coordinate.size == 0:
        raise ValueError("coordinate must be a non-empty one-dimensional array")
    if coordinate.size == 1:
        return np.ones(1)
    ds = np.abs(np.diff(coordinate))
    positive = ds[ds > 0.0]
    floor = (np.median(positive) if positive.size else 1.0) * 1.0e-6
    ds = np.maximum(ds, floor)
    weights = np.empty_like(coordinate)
    weights[0] = 0.5 * ds[0]
    weights[-1] = 0.5 * ds[-1]
    weights[1:-1] = 0.5 * (ds[:-1] + ds[1:])
    return weights / np.sum(weights)


def pseudo_huber_squared(residual: jax.Array, transition: float = 2.0) -> jax.Array:
    """Twice the pseudo-Huber loss, quadratic for small residuals."""
    scaled = residual / transition
    return 2.0 * transition * transition * (jnp.sqrt(1.0 + scaled * scaled) - 1.0)


def curve_loss(block: ProtocolResidual, robust_transition: float = 2.0) -> jax.Array:
    normalized = (block.prediction - block.measurement) / block.scale
    return jnp.sum(block.point_weight * pseudo_huber_squared(normalized, robust_transition))


def protocol_balanced_loss(blocks: Sequence[ProtocolResidual],
                           robust_transition: float = 2.0) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Average observables within records, then records within families.

    This hierarchy prevents a compound physical experiment from receiving
    twice the mass merely because loading--unloading and recovery were
    digitised as separate response curves.  Only families present in the
    supplied subset enter the outer mean.
    """
    if not blocks:
        raise ValueError("at least one residual block is required")
    families = sorted({block.balance_family for block in blocks})
    family_losses: dict[str, jax.Array] = {}
    for family in families:
        family_blocks = [block for block in blocks if block.balance_family == family]
        records = sorted({block.record_id for block in family_blocks})
        record_losses = []
        for record in records:
            observables = [curve_loss(block, robust_transition) for block in family_blocks
                           if block.record_id == record]
            record_losses.append(jnp.mean(jnp.stack(observables)))
        family_losses[family] = jnp.mean(jnp.stack(record_losses))
    return jnp.mean(jnp.stack(list(family_losses.values()))), family_losses


WeightingMode = Literal["equal_point", "equal_curve", "equal_protocol"]


def multi_path_loss(blocks: Sequence[ProtocolResidual],
                    mode: WeightingMode = "equal_protocol",
                    robust_transition: float = 2.0,
                    ) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Evaluate the three weighting policies used in the sensitivity audit.

    ``equal_point`` gives every sampled observation equal mass, ``equal_curve``
    gives every response curve equal mass after path integration, and
    ``equal_protocol`` averages curves within families before averaging the
    families.  The latter is the declared primary objective.
    """
    if not blocks:
        raise ValueError("at least one residual block is required")
    if mode == "equal_protocol":
        return protocol_balanced_loss(blocks, robust_transition)
    family_losses: dict[str, jax.Array] = {}
    families = sorted({block.balance_family for block in blocks})
    for family in families:
        values = [curve_loss(block, robust_transition) for block in blocks
                  if block.balance_family == family]
        family_losses[family] = jnp.mean(jnp.stack(values))
    if mode == "equal_curve":
        values = [curve_loss(block, robust_transition) for block in blocks]
        return jnp.mean(jnp.stack(values)), family_losses
    if mode == "equal_point":
        point_values = []
        for block in blocks:
            normalized = (block.prediction - block.measurement) / block.scale
            point_values.append(pseudo_huber_squared(normalized, robust_transition))
        return jnp.mean(jnp.concatenate(point_values)), family_losses
    raise ValueError(f"unknown weighting mode: {mode}")
