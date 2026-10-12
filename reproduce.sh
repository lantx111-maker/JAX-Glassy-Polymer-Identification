#!/usr/bin/env bash
set -euo pipefail
gp_root="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$gp_root"
gp_python="${GP_PYTHON:-$gp_root/.venv/bin/python}"
if [[ ! -x "$gp_python" ]]; then gp_python="$(command -v python3)"; fi
export PYTHONPATH="$gp_root/code${PYTHONPATH:+:$PYTHONPATH}"
export JAX_ENABLE_X64=1
export JAX_PLATFORM_NAME=cpu
export MPLBACKEND=Agg
export MPLCONFIGDIR="$gp_root/.cache/matplotlib"
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p "$MPLCONFIGDIR"
gp_mode="${1:-list}"
if [[ $# -gt 0 ]]; then shift; fi
case "$gp_mode" in
  list) printf '%s\n' 'smoke: data-independent short history' 'verify: five data-independent numerical checks' 'script NAME [arguments]: scientific driver; see docs/WORKFLOWS.md' ;;
  smoke) "$gp_python" cases/material_point_smoke.py "$@" ;;
  verify) "$gp_python" -m pytest -q \
      tests/test_constitutive.py::test_initial_state_is_stress_free \
      tests/test_constitutive.py::test_short_history_is_finite_and_differentiable \
      tests/test_constitutive.py::test_coupled_stress_control_enforces_targets \
      tests/test_v9_invariance.py "$@" ;;
  script)
    gp_name="${1:?Supply a driver name listed in docs/WORKFLOWS.md}"; shift
    if [[ ! "$gp_name" =~ ^[A-Za-z0-9_]+$ ]]; then printf '%s\n' 'Invalid driver name' >&2; exit 2; fi
    gp_script="$gp_root/code/applications/glassy_polymer/scripts/$gp_name.py"
    if [[ ! -f "$gp_script" ]]; then printf '%s\n' 'Driver not included in this snapshot' >&2; exit 2; fi
    "$gp_python" "$gp_script" "$@" ;;
  *) printf '%s\n' 'Usage: ./reproduce.sh {list|smoke|verify|script NAME [arguments]}' >&2; exit 2 ;;
esac
