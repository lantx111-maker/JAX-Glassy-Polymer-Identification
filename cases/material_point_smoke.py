"""Short synthetic history: installation check, not a full paper rerun."""
from pathlib import Path
import argparse
import csv
import json
import jax
import jax.numpy as jnp
import numpy as np
from applications.glassy_polymer.constitutive import initial_state, pmma_parameters
from applications.glassy_polymer.protocols import run_strain_history

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('outputs/material_point_smoke'))
    args = parser.parse_args()
    params = pmma_parameters()
    strain = jnp.linspace(0., .02, 5)
    dt = jnp.full(5, .2)
    def response(p):
        return run_strain_history(strain, dt, p, initial_state(p),
                                  traction_free=True, substeps=4, scheme='umat')
    baseline = response(params)
    scale = 2.
    a, b = scale ** params.m0, scale ** params.m_ve
    transformed = params._replace(gamma0=scale * params.gamma0,
        s0=a * params.s0, kappa1_sat=a * params.kappa1_sat,
        kappa2_sat=a * params.kappa2_sat, s1=b * params.s1, s2=b * params.s2)
    equivalent = response(transformed)
    sigma = np.asarray(baseline.stress)
    error = float(np.max(np.abs(sigma - np.asarray(equivalent.stress))))
    gradient = float(jax.grad(lambda E: -response(params._replace(E=E)).stress[-1,1,1])(jnp.asarray(params.E)))
    np.testing.assert_allclose(sigma, np.asarray(equivalent.stress), rtol=1e-9, atol=1e-8)
    assert np.all(np.isfinite(sigma)) and np.isfinite(gradient)
    assert float(np.max(np.abs(sigma[:,0,0]))) < 1e-8
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'response.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['compressive_true_strain','compressive_true_stress_MPa'])
        writer.writerows(zip(np.asarray(strain).tolist(), (-sigma[:,1,1]).tolist()))
    report = {'status':'passed', 'case':'synthetic installation check',
        'points':5, 'substeps':4, 'scale_symmetry_max_stress_difference_MPa':error,
        'terminal_stress_derivative_wrt_E':gradient,
        'jax_version':jax.__version__, 'jax_backend':jax.default_backend(),
        'x64':bool(jax.config.jax_enable_x64)}
    (args.output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
