# Glassy Polymer Constitutive Identification

This repository provides the JAX material-point implementation and analysis
code accompanying **Origins of non-uniqueness in the identification of
internal-variable constitutive models for glassy polymers**. The model is the
isothermal specialization of the constitutive model cited in the manuscript.

The calculations distinguish exact parameter equivalence, weak constraints
under different loading histories, and the state information needed for
subsequent prediction. Both PMMA and PC parameter sets are supplied.

Authors: Zhiyang Ni, Tianxiang Lan, Yaodong Jiang, Peidong Wu, Yueguang Wei.
Corresponding author: Tianxiang Lan, lantianxiang111@163.com.

## Install

Use Python 3.11 for the archived numerical environment; Python 3.10–3.12
is supported by this source package. Calculations use 64-bit CPU execution.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

## Start here

```bash
./reproduce.sh list
./reproduce.sh smoke
./reproduce.sh verify
```

The smoke case simulates a short traction-free loading history, checks a
parameter derivative and the finite rate–resistance transformation, and writes
CSV/JSON outputs. The verification command runs existing numerical tests of
initialization, strain/stress control, exact scaling and complete-state restart.
Neither command needs experimental observations.

## Source and workflows

- `code/applications/glassy_polymer/`: constitutive update, controls, sensitivities,
  objectives, calibration and retained-state design.
- `code/applications/glassy_polymer/scripts/`: scientific analysis drivers and
  their imported helpers, named by the calculation they perform.
- `tests/`: numerical regression checks.
- `cases/`: the data-independent installation case and the manuscript's applied
  four-segment schedules.
- `docs/WORKFLOWS.md`: analysis entry points, prerequisites and expected scope.
- `docs/EXTERNAL_DATA.md`: external observation files and checkpoint requirements.
- `SOURCE_MANIFEST.json`: source origins and SHA-256 hashes.

The published-figure comparisons and calibration cannot be reproduced from
code alone: they also require the digitized literature observations, and some
refinement/continuation drivers require earlier computed arrays or checkpoints.
These inputs are not distributed in this code snapshot. The external-data
loader is retained; users with appropriate access can supply their own copy.
Full computations can take hours on a CPU. An installation check does not
establish that the entire manuscript has been recomputed.

Generated results are excluded from Git. Literature observation coordinates,
original Fortran routines, companion manuscripts and manuscript figure files
are not included. See `NOTICE.md` for source and redistribution scope.

## Licence and citation

The supplied GPL v3 licence is preserved in `LICENSE`. `CITATION.cff` records
the current manuscript title and author order. No article DOI or code-archive
DOI has been assigned in this repository.
