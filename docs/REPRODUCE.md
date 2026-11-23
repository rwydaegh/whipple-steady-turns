Each row lists a paper item, the command (run from the repository root with `uv run python`), its output and the run time on one CPU core.

| Paper item | Command | Output | Run time |
| --- | --- | --- | --- |
| All figures from stored data | `figures/make_all.py` | `results/figures/` | 1 min |
| Fig. fig:eigs, "Eigenvalues against speed" | `figures/eigenvalues.py` | `results/figures/eigenvalues.pdf` | 30 s |
| Fig. fig:tongue, "Region of stable turns at constant steer torque" | `figures/region.py` | `results/figures/region.pdf` | 10 s |
| Table tab:region (supplement), benchmark bicycle and bicycles A and B | `scripts/tongue.py` | `results/tongue.json` | 2 min |
| Largest stable lean at constant speed, "Constant speed" | `scripts/constant_wheel_speed.py` | `results/constant_wheel_speed.json` | 16 min |
| Results of each measured set | `scripts/measured/analyse.py` | `data/measured/results/` | 1 min per set, 1 h in all |
| Tables tab:measured-sets, tab:measured-riderless, "Measured bicycles" | `scripts/measured/tables.py` | `data/measured/tables.md` | 10 s |
| Table tab:tensors, "Inadmissible front-frame inertia tensors" | `scripts/measured/tensor_corrections.py` | `data/measured/tensor_corrections.json` | 7 min |
| Fig. fig:rule-measured, "Accuracy of the capsize rules" | `figures/rule_accuracy.py` | `results/figures/rule_accuracy.pdf` | 5 s |
| Figs. fig:measured-overview, fig:rider-effect (supplement) | `figures/measured_overview.py`, `figures/rider_effect.py` | `results/figures/` | 5 s each |
| Theorem prop:fold, "Capsize condition in a turn" | `scripts/capsize/theorem1_checks.py`, `scripts/capsize_condition.py` | `data/capsize/theorem1_checks.json` | 10 s |
| Eq. twothirds, 68 loading paths, "Torque limit on the loading path" | `scripts/capsize/loading_paths.py` (`--quick`: 2 min) | `data/capsize/loading_paths.json` | 10 min |
| Fig. fig:loading, torque limit and torque ramps | `scripts/capsize/torque_ramp.py`, then `figures/loading_path.py` | `results/figures/loading_path.pdf` | 1.5 min |
| Table tab:rule, "Capsize rule: order and accuracy" | `scripts/capsize/rule_accuracy.py`, `scripts/capsize/random_designs.py` | `data/capsize/` | 5 min |
| Theorem thm:countersteer counts and the order of the two folds | `scripts/countersteer/grid_scan.py` (`--quick`: 1 min), then `grid_report.py` | `data/countersteer/grid_summary.json` | 36 min |
| "Accuracy of the Hopf curve" and "Speed of the tip" | `scripts/weave/hopf_accuracy.py` (`--quick`: 1 min) | `data/weave/hopf_accuracy.json` | 6 min |
| Sect. "Criticality of the weave onset", Fig. fig:l1 | `scripts/lyapunov_coefficient.py` | `results/lyapunov_coefficient.pdf` | 18 min |
| Period doublings at T = -0.418 N m (supplement) | `scripts/period_doubling.py` | `results/period_doubling.json` | 3 min |
| Fig. fig:homoclinic, panels (b) and (c) | `scripts/weave_chaos.py` (short run); full run with `c/hires.c` on many cores | `results/weave_chaos.json` | 6 min (short run) |
| Torque coefficients, Table tab:types | `scripts/types/upright_types.py` | `data/types/upright_types.json` | 20 s |
| Table tab:trails, "Bicycle types" | `scripts/types/critical_trails.py` | `data/types/critical_trails.json` | 2 min |
| "First order in the steer angle at any lean" | `scripts/types/first_order.py` | `data/types/first_order.json` | 5 s |
| Fig. fig:types | `scripts/types/trail_series.py`, `scripts/types/trail_maps.py`, then `figures/types.py` | `results/figures/types.pdf` | 5 min |
| "Exact invariances" | `scripts/design/invariances.py` | `data/design/invariances.json` | 30 s |
| Fig. fig:design-chart, Tables tab:design-chart, tab:design-mass, tab:design-variants, tab:design-tradeoff | `scripts/design/elasticities.py`, `scripts/design/rule_elasticities.py`, then `figures/design_moves.py` | `results/figures/design_moves.pdf` | 12 min |
| Table tab:design-loads | `scripts/design/loads.py` | `data/design/loads.json` | 2.5 min |
| Table tab:design-example, "A design example" | `scripts/design/design_example.py` | `data/design/design_example.json` | 20 s |
| Random designs: design list | `scripts/random_designs/sampler.py` | `data/random_designs/designs.json.gz` | 2 s |
| Random designs: one design per call, run in parallel on many cores | `scripts/random_designs/analyse_design.py --start i --stop j`, then `collect.py` | `data/random_designs/results.jsonl.gz` | 2 min per design, 90 h in all |
| Random designs: tables of the supplement | `scripts/random_designs/analyse.py` | printed | 1 s |
| Closing stiffness k*, steer spring results, "Steer torque with a centring spring" | `scripts/rider/arms_grid.py`, `arms_analyse.py`, `arms_summary.py`, `arms_region.py`, `arms_checks.py`, `arms_simulations.py` | `data/rider/` | 26 min |
| Mass shift results, "Mass shift and steer torque" | `scripts/rider/shift_*.py` | `data/rider/` | 40 min |
| Fig. fig:rider | `figures/rider.py` | `results/figures/rider.pdf` | 5 s |
| Power balance, first integral, drive force at a steady turn, "Energy" | `scripts/energy/identities.py` | `data/energy/identities.json` | 25 s |
| Fig. fig:energy (a, c, d), recovery and capsize, "Energy in disturbed motions" | `scripts/energy/disturbed_motions.py`, then `figures/energy.py` | `results/figures/energy.pdf` | 2.5 min |
| Fig. fig:energy (b), growing weave | `scripts/energy/growing_weave.py` | `data/energy/growing_weave.json` | 3.5 min |
| Upright recovery after Fig. 5.4 of Limebeer and Massaro (2018) (supplement) | `scripts/energy/upright_recovery.py` | `data/energy/upright_recovery.json` | 25 s |
| Speed feedback (supplement, "Constant speed: drive and speed feedback") | `scripts/energy/speed_feedback.py` (`--quick` exists) | `data/energy/speed_feedback.json` | 6 min |
| Fig. fig:sim, kicks from a stable turn | `scripts/simulations/kicks.py`, then `figures/simulations.py` | `results/figures/simulations.pdf` | 15 s |
| Runs near the boundaries, capsize at the torque limit, weave beyond the Hopf curve (supplement) | `scripts/simulations/boundary_runs.py` (`--quick`: 2 min) | `data/simulations/boundary_runs.json` | 4.5 min |
| Fig. fig:cain, Table tab:cain, "Comparison with the measurements of Cain and Perkins" | `scripts/experiments/cain_perkins.py`, then `figures/cain_perkins.py` | `results/figures/cain_perkins.pdf` | 3.5 min |
| Main numbers recomputed with separate solvers | `scripts/checks/main_results.py` (`--quick`: 2.5 min) | `data/checks/independent.json` | 5 min |
| Tests against the published values | `uv run pytest` | | 1 min |
