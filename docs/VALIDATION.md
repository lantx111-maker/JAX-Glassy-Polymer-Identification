# Verification of this code snapshot

The editable package installed successfully on 12 October 2026. The synthetic
material-point smoke case completed with finite stress and a finite parameter
derivative. Its finite scale transformation changed stress by 0.0 MPa within
the computed precision.

The default verification command passed five existing numerical checks:

1. Stress-free initialization.
2. Finite and differentiable short loading history.
3. Satisfaction of the coupled stress-control targets.
4. Finite rate–resistance scaling equivalence for PMMA and PC, including state covariance.
5. Complete-state restart and the accumulated-flow/clock identity.

The test environment was Python 3.12.14, JAX/JAXLIB 0.4.23, NumPy 1.26.4,
SciPy 1.11.4, Matplotlib 3.8.0, OpenPyXL 3.1.5, ml_dtypes 0.5.4 and
pytest 7.4.4, using CPU execution and 64-bit arithmetic. The original numerical
archive records Python 3.11.16; this check is not a byte-for-byte rerun of that
entire campaign.

Source hashes, imported analysis helpers and the absence of coordinate/result
payloads were checked before the private upload. The documentation correction
does not alter executable constitutive statements. Installation and these
bounded checks do not replace the full derivative, mesh and calibration
evidence reported in the manuscript.
