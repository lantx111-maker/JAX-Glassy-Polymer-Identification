# External inputs

The data-independent checks run immediately after installation. Curve-based
analyses use the existing `GLASSY_POLYMER_DATA_ROOT` interface:

```bash
export GLASSY_POLYMER_DATA_ROOT="/path/to/your/authorized/data/raw"
```

The loader discovers individual XLSX curves within the original Chinese
material/family directory hierarchy. Column 1 is the source abscissa and
column 2 is the measured response; an optional first-row header is accepted.
Do not change the source file names or folder labels for the article corpus:
the classifier extracts material, loading family, rate and target from them.
`protocol_splits.json` preserves the declared training and validation roles.
`experiments.py` and `data_cleaning.py` implement loading-history reconstruction
and the recorded abscissa treatment; measured ordinates are not silently fitted.

The literature sources are Hasan (1994), Anand and Ames (2006,
doi:10.1016/j.ijplas.2005.07.006), and Wang et al. (2019,
doi:10.1016/j.ijplas.2019.06.013). This repository does not supply their
digitized observations or infer permission to redistribute them.

Geometry continuation uses previous Jacobians and per-history signatures.
The PMMA calibration continuation additionally needs the original seven-step
checkpoint and its decision/provenance files. Resume files are not fabricated
by this repository; rerunning a fresh fit is a different computation from
resuming that archived campaign. Frozen arrays and checkpoint hashes belong
to the separate author numerical archive.

Generated experimental comparisons can contain the supplied observations.
Keep those outputs outside the Git history unless their sharing terms have
been reviewed. The supplied `.gitignore` excludes default result directories.

The original numerical archive keeps its original file names and source
signatures. Existing archived checkpoints must be evaluated with their matching
archived code; descriptive source names and identifiers in this repository do
not replace those signatures. Fresh calculations generate mutually consistent
source signatures using this repository's names. Source-hash verification is
retained and is not bypassed to accept an incompatible checkpoint.
