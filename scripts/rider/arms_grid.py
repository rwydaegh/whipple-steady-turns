"""Exact quartic data of steady turns on (speed, lean) grids for the steer spring and damper (Sect. 13.2, "Steer torque with a centring spring"; supplementary material, "Centring spring: further results")."""  # noqa: E501

import argparse
import time
from pathlib import Path

import numpy as np
from arms_model import Upright, all_sets, family, numerators, pencils

from whipple.model import Whipple

REPO = Path(__file__).resolve().parents[2]
GRIDS = REPO / "results" / "rider" / "arms_grids"
MEASURED = REPO / "data" / "measured" / "results"


def grid_path(name, fine=False):
    return GRIDS / f"{name}{'_fine' if fine else ''}.npz"


def rigid_results(name):
    """Stored region of the set without arms (scripts/measured), or None."""
    import gzip
    import json

    f = MEASURED / f"{name}.json.gz"
    if not f.exists():
        return None
    with gzip.open(f, "rt") as fh:
        return json.load(fh)


def make_grid(name, p, fine=False):
    """Speeds and leans of the grid of one set (range from its region without arms)."""
    u = Upright(p)
    res = rigid_results(name)
    top, v_lo = 16.0, 0.72 * u.V_a3
    if res:
        top = max(res[pl].get("max_lean_deg", 16.0) for pl in ("free", "constant"))
        v_lo = min(v_lo, 0.9 * min(res[pl].get("BT_V", u.V_a3) for pl in ("free", "constant")))
    phi_max = min(1.25 * top + 3.0, 40.0)
    v_hi = 1.03 * u.v_c if u.v_c == u.v_c else 12.0
    if fine:
        return np.arange(v_lo, v_hi, 0.025), np.radians(np.arange(0.1, phi_max + 1e-9, 0.1))
    return np.linspace(v_lo, v_hi, 60), np.radians(np.arange(0.25, phi_max + 1e-9, 0.25))


def compute(p, Vs, phis):
    """delta, T, A and Lpp of the turns at the grid points (nan where no turn was found)."""
    bike = Whipple(p)
    nV, nP = len(Vs), len(phis)
    out = dict(delta=np.full((nV, nP), np.nan), T=np.full((nV, nP), np.nan))
    for pl in ("free", "constant"):
        out[f"A_{pl}"] = np.full((5, nV, nP), np.nan)
        out[f"L_{pl}"] = np.full((3, nV, nP), np.nan)
    for i, V in enumerate(Vs):
        for j, d in enumerate(family(bike, V, phis)):
            try:
                pen, turn = pencils(bike, phis[j], d)
            except ValueError:
                break
            out["delta"][i, j], out["T"][i, j] = d, turn.T
            for pl in ("free", "constant"):
                out[f"A_{pl}"][:, i, j], out[f"L_{pl}"][:, i, j] = numerators(*pen[pl])
    return out


def run(name, p, fine=False):
    t0 = time.time()
    Vs, phis = make_grid(name, p, fine)
    out = compute(p, Vs, phis)
    GRIDS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(grid_path(name, fine), V=Vs, phi=phis, **out)
    n = np.isfinite(out["delta"]).sum()
    print(f"{name}: {len(Vs)} x {len(phis)} turns, {n} found, {time.time() - t0:.0f} s", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--quick", action="store_true", help="benchmark, coarse grid only")
    ap.add_argument("--fine", action="store_true", help="fine grid of the benchmark")
    ap.add_argument("--sets", nargs="*", help="names of the parameter sets")
    a = ap.parse_args()
    sets = all_sets()
    if a.fine:
        run("benchmark", sets["benchmark"], fine=True)
        return
    names = ["benchmark"] if a.quick else (a.sets or list(sets))
    for n in names:
        run(n, sets[n])


if __name__ == "__main__":
    main()
