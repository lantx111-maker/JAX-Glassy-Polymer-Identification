# Analysis entry points

Run a driver through `./reproduce.sh script NAME [arguments]`. NAME is its
file stem within `code/applications/glassy_polymer/scripts/`.

| Purpose | Entry | Inputs required |
|---|---|---|
| Installation and short constitutive history | `./reproduce.sh smoke` | None |
| Numerical checks of equivalence and restart | `./reproduce.sh verify` | None |
| Digitized-data audit and manifest | `audit_digitised_data`, `build_data_manifest` | Authorized source workbooks |
| Complete nominal responses and path qualification | `run_path_baselines`, `check_response_paths` | Workbooks |
| Sensitivity and fixed observation geometry | `check_parameter_sensitivities`, `check_observation_geometry`, `compare_initial_geometry_meshes` | Workbooks; preceding generated results |
| Independent and coupled mesh comparisons | `refine_geometry_mesh`, `refine_creep_mesh` | Workbooks; preceding Jacobians and signatures |
| Directional derivatives and control safeguards | `check_directional_derivatives`, `check_derivative_guards` | Workbooks and specified prior arrays |
| Bounded PC/PMMA calibration | `calibrate_gauge_fixed` | Workbooks; explicit fresh output directory |
| PMMA checkpoint continuation | `resume_gauge_fixed_calibration` | Workbooks and original checkpoint |
| Synthetic loading-history design | `run_inverse_design`, `match_residual_histories` | Nominal parameters and preceding design outputs |
| Subsequent probes and crossed mesh comparisons | `compare_subsequent_probes`, `compare_reload_meshes`, `check_crossed_probes` | Earlier design outputs and complete saved states |
| Matrix-power derivative check | `check_matrix_power_derivatives` | None |

Imported helper drivers are included with their source dependencies. Read each
driver's header and argument parser before launching an expensive computation.
They use descriptive names and relative result locations. The code snapshot
does not contain the archive's precomputed prerequisites or promise a
one-command reproduction of all figures.

The three four-segment schedules in `cases/matched_history_controls.json`
are copied from the current manuscript table, including its displayed rounding.
They are already amplitude-scaled applied endpoints. Do not apply the amplitude
factor again. A 37.5 s transition to zero stress and 562.5 s recovery follow
loading; those stages are outside the four loading durations. Exact frozen
controls and their state/probe meshes remain in the numerical archive.
