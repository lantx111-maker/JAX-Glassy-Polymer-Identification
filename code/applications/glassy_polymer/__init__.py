"""Differentiable retained-state constitutive model for glassy polymers."""

from .constitutive import MaterialParameters, MaterialState, initial_state, update_material
from .canonical import CANONICAL_OPERATOR, CANONICAL_OPERATOR_VERSION

__all__ = [
    "CANONICAL_OPERATOR",
    "CANONICAL_OPERATOR_VERSION",
    "MaterialParameters",
    "MaterialState",
    "initial_state",
    "update_material",
]
