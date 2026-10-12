"""Single canonical discrete operator used by every scientific inverse problem.

Legacy UMAT/VUMAT paths remain available in :mod:`constitutive` for parity
audits.  They are not selectable through this module.  The canonical operator
uses the converged, end-of-increment UMAT update ordering for both materials
and every loading protocol; protocol-specific differences are limited to the
control equation and declared substep resolution.
"""

from __future__ import annotations

from dataclasses import dataclass

from .constitutive import ConstitutiveRegularization


CANONICAL_SCHEME = "umat"
CANONICAL_ACTIVATION_RATE = 0.0
CANONICAL_REGULARIZATION = ConstitutiveRegularization(
    macaulay_width=0.0,
    activation_rate=CANONICAL_ACTIVATION_RATE,
)
CANONICAL_OPERATOR_VERSION = (
    "canonical-v4r2-continuous-branches-clock-bracketed-root-exact-phi-spd-refined-mesh"
)


@dataclass(frozen=True)
class CanonicalResolution:
    """Declared physical-mesh policy for the canonical material-point operator.

    Legacy integer substep fields remain available to parity/audit scripts.
    Scientific experiment predictions use the maximum physical increments.
    """

    pmma_strain: int = 20
    pmma_stress: int = 20
    pc_strain: int = 100
    pc_stress: int = 10
    recovery: int = 30
    pmma_max_true_strain_increment: float = 1.0e-3
    pc_max_true_strain_increment: float = 1.0e-3
    pmma_max_time_increment: float = 0.25
    pc_max_time_increment: float = 2.5
    pmma_severe_creep_threshold_mpa: float = 90.0
    pmma_severe_creep_max_time_increment: float = 1.5625e-2
    recovery_max_time_increment: float = 0.125
    initial_time_increment: float = 1.0e-5

    def strain_substeps(self, material: str) -> int:
        return self.pmma_strain if material.upper() == "PMMA" else self.pc_strain

    def stress_substeps(self, material: str) -> int:
        return self.pmma_stress if material.upper() == "PMMA" else self.pc_stress

    def maximum_strain_increment(self, material: str) -> float:
        return (self.pmma_max_true_strain_increment
                if material.upper() == "PMMA" else self.pc_max_true_strain_increment)

    def maximum_time_increment(self, material: str) -> float:
        return (self.pmma_max_time_increment
                if material.upper() == "PMMA" else self.pc_max_time_increment)

    def creep_time_increment(self, material: str, target_mpa: float) -> float:
        """Protocol-severity rule fixed independently of observation sampling."""
        if (material.upper() == "PMMA"
                and target_mpa >= self.pmma_severe_creep_threshold_mpa):
            return min(
                self.pmma_max_time_increment,
                self.pmma_severe_creep_max_time_increment,
            )
        return self.maximum_time_increment(material)


CANONICAL_RESOLUTION = CanonicalResolution()

# Parameter sensitivities and all reported SVDs use a stricter mesh than the
# response-only production layer.  The next halved layer is reserved for the
# independent convergence gate and is not used to improve reported spectra.
SENSITIVITY_RESOLUTION = CanonicalResolution(
    pmma_max_true_strain_increment=2.5e-4,
    recovery_max_time_increment=3.125e-2,
)


@dataclass(frozen=True)
class CanonicalOperator:
    """Immutable definition of the forward map used for inference."""

    version: str = CANONICAL_OPERATOR_VERSION
    scheme: str = CANONICAL_SCHEME
    resolution: CanonicalResolution = CANONICAL_RESOLUTION
    regularization: ConstitutiveRegularization = CANONICAL_REGULARIZATION


CANONICAL_OPERATOR = CanonicalOperator()


def assert_canonical_scheme(scheme: str) -> None:
    if scheme != CANONICAL_SCHEME:
        raise ValueError(
            f"Scientific inverse problems require {CANONICAL_SCHEME!r}; "
            f"{scheme!r} is a legacy parity-only scheme."
        )
