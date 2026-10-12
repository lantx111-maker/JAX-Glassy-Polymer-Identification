# Analysis entry points

Run a driver through `./reproduce.sh script NAME [arguments]`. NAME is its
file stem within `code/applications/glassy_polymer/scripts/`.

| Purpose | Entry | Inputs required |
|---|---|---|
| Installation and short constitutive history | `./reproduce.sh smoke` | None |
| Numerical checks of equivalence and restart | `./reproduce.sh verify` | None |
| Digitized-data audit and manifest | `audit_digitised_data`, `build_data_manifest` | Authorized source workbooks |
| Complete nominal responses and path qualification | `run_path_baselines`, `run_v4_all_path_audit` | Workbooks |
| Sensitivity and fixed observation geometry | `run_v5_sensitivity_gate`, `run_v5_fixed_objective_audit`, `audit_v8_full_geometry` | Workbooks; preceding generated results |
| Independent and coupled mesh comparisons | `audit_v9_geometry_convergence`, `audit_v9_creep_continuation` | Workbooks; preceding Jacobians and signatures |
| Directional derivatives and control safeguards | `audit_readme_v9_final_directions`, `audit_readme_v9_derivative_guards` | Workbooks and specified prior arrays |
| Bounded PC/PMMA calibration | `run_v9_gauge_fixed_joint_inversion` | Workbooks; explicit fresh output directory |
| PMMA checkpoint continuation | `run_v9_gauge_fixed_joint_inversion_resume` | Workbooks and original checkpoint |
| Synthetic loading-history design | `run_inverse_design`, `run_v4_feasible_design` | Nominal parameters and preceding design outputs |
| Subsequent probes and crossed mesh comparisons | `run_v5_follow_on_discriminator`, `audit_v8_reload_refinement`, `audit_readme_v9_probe_cross` | Earlier design outputs and complete saved states |
| Matrix-power derivative check | `run_v5_matrix_power_audit` | None |

Imported helper drivers are included with their source dependencies. Read each
driver's header and argument parser before launching an expensive computation.
They retain fixed numerical identifiers and relative result locations from
the numerical archive. The public code snapshot does not contain the archive's
precomputed prerequisites or promise a one-command reproduction of all figures.

The three four-segment schedules in `cases/matched_history_controls.json`
are copied from the current manuscript table, including its displayed rounding.
They are already amplitude-scaled applied endpoints. Do not apply the amplitude
factor again. A 37.5 s transition to zero stress and 562.5 s recovery follow
loading; those stages are outside the four loading durations. Exact frozen
controls and their state/probe meshes remain in the numerical archive.
