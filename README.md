# Stability of steady turns of the Whipple bicycle

This code computes the steady turns of the nonlinear Whipple bicycle at constant steer torque and their stability, and it reproduces the figures, tables and numbers of the paper listed below.

## Install and run

The project uses [uv](https://docs.astral.sh/uv/) and Python 3.10 or newer, and it runs on a CPU.

```
uv sync
uv run python figures/make_all.py
uv run pytest
```

The figures are written to `results/figures/` from the stored results in `data/`. [docs/REPRODUCE.md](docs/REPRODUCE.md) lists the command, output file and run time for each paper item.

## Layout

- `whipple/`: the model, the stability analysis, the capsize rules and the parameter sets
- `figures/`: one script per paper figure
- `scripts/measured/`, `scripts/weave/`: measured bicycles and the weave limit
- `scripts/capsize/`, `scripts/countersteer/`, `scripts/types/`: capsize limit, countersteer and bicycle types
- `scripts/design/`, `scripts/random_designs/`: design moves and the random designs
- `scripts/rider/`: steer spring and rider mass shift
- `scripts/energy/`, `scripts/simulations/`, `scripts/experiments/`: energy, time simulations and the measurements of Cain and Perkins (2012)
- `scripts/checks/`: recomputation of the main numbers with separate solvers
- `scripts/*.py`, `c/`: criticality of the weave onset, the model tests and earlier results
- `data/`: small result files that the figures and tables read

## Conventions

Axes and signs follow Meijaard et al. (2007), with positive lean and steer to the right. The speed V is the speed of the rear contact point, and "constant speed" means a constant rear-wheel speed.

## Data, licence and citation

The measured bicycle and rider parameters in `whipple/data/` derive from the BicycleParameters data of J. K. Moore (BSD-2-Clause, see `whipple/data/LICENSE-BicycleParameters.txt`) and from tables of Hess et al. (2012), Schwab et al. (2013) and Moore (2012). The code is under the MIT License (`LICENSE`).

AI tools helped to develop much of the derivations and of the code; the tests and a recomputation with separate solvers check the results.

Please cite the paper, R. Wydaeghe and D. J. N. Limebeer, Stability of steady turns of the Whipple bicycle, manuscript in preparation (2026), and the software as given in `CITATION.cff`.
