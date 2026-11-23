"""Stable turns with the steer spring and damper for the benchmark and the 23 measured sets (Sect. 13.2, "Steer torque with a centring spring"; supplementary material, "Centring spring: further results", Table tab:arms-lit)."""  # noqa: E501

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from arms_grid import grid_path
from arms_model import (
    ArmQuartic,
    Upright,
    all_sets,
    delta_at,
    hurwitz3,
    largest_lean_rule,
    quartic,
    tip_exact,
)
from scipy import ndimage

from whipple.model import Whipple

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "rider" / "arms_results"

KFRAC = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
KBEYOND = (1.02, 1.1, 1.5, 2.0, 5.0, 20.0, 200.0)
CFRAC = (0.1, 0.25, 0.5, 0.75, 0.9)
LITERATURE = {  # (k in N m/rad, c in N m s/rad), Table tab:arms-lit
    "Pick and Cole, relaxed": (3.2, 0.72),
    "Bulsink": (5.0, 0.9),
    "Pick and Cole, tense": (59.4, 1.81),
    "Cossalter et al.": (1053.4, 19.28),
}


def load(name, fine=False):
    return np.load(grid_path(name, fine))


def region(g, plant, k=0.0, c=0.0, Ia=0.0):
    """Quartic coefficients, D3 and the mask of stable turns on the grid."""
    a = quartic(g[f"A_{plant}"], g[f"L_{plant}"], k, c, Ia)
    D3 = hurwitz3(a)
    with np.errstate(invalid="ignore"):
        ok = (a[0] > 0) & (a[2] > 0) & (a[3] > 0) & (D3 > 0)
    return a, D3, ok


def tongue(g, plant, k=0.0, c=0.0, Ia=0.0, full=False):
    """Largest lean, area and lowest speed of the stable region with the lowest lean.

    The top boundary is refined by linear interpolation of the test that changes sign and by a
    parabola in V. With full=True the tests and the mask are returned as well.
    """
    V, phi = g["V"], g["phi"]
    a, D3, ok = region(g, plant, k, c, Ia)
    out = dict(k=k, c=c, Ia=Ia, n_stable=int(ok.sum()), phi_grid_max=math.degrees(phi[-1]))
    if not ok.any():
        out.update(lean=0.0, area=0.0, n_tongue=0, other_min_lean=None)
        return (out, a, D3, ok) if full else out
    lab, n = ndimage.label(ok)
    jmin = [np.where(lab == i + 1)[1].min() for i in range(n)]
    t = int(np.argmin(jmin)) + 1
    tm = lab == t
    others = [
        math.degrees(phi[jmin[i]]) for i in range(n) if i + 1 != t and (lab == i + 1).sum() > 2
    ]
    tests = (a[3], D3, a[2], a[0])
    top = np.full(len(V), np.nan)
    for i in range(len(V)):
        js = np.flatnonzero(tm[i])
        if not len(js):
            continue
        j = js.max()
        top[i] = phi[j]
        if j + 1 < len(phi):
            cr = [
                phi[j] + (phi[j + 1] - phi[j]) * f[i, j] / (f[i, j] - f[i, j + 1])
                for f in tests
                if np.isfinite(f[i, j + 1]) and f[i, j] > 0 >= f[i, j + 1]
            ]
            if cr:
                top[i] = min(cr)
    i = int(np.nanargmax(top))
    lean, Vtop = top[i], V[i]
    if 0 < i < len(V) - 1 and np.isfinite(top[i - 1]) and np.isfinite(top[i + 1]):
        y0, y1, y2 = top[i - 1], top[i], top[i + 1]
        den = y0 - 2 * y1 + y2
        if den < 0:
            s = 0.5 * (y0 - y2) / den
            lean = y1 - 0.25 * (y0 - y2) * s
            Vtop = V[i] + s * (V[1] - V[0])
    cols = np.flatnonzero(tm.any(axis=1))
    dA = (V[1] - V[0]) * math.degrees(phi[1] - phi[0])
    out.update(
        lean=math.degrees(lean),
        V_top=float(Vtop),
        area=float(tm.sum() * dA),
        n_tongue=int(tm.sum()),
        V_min=float(V[cols.min()]),
        V_max=float(V[cols.max()]),
        min_lean=math.degrees(phi[min(jmin)]),
        other_min_lean=(min(others) if others else None),
    )
    return (out, a, D3, tm) if full else out


def exact_tip(bike, aq, u, plant, k, c):
    """Exact tip by Newton, started at the tip of the order-phi^2 coefficients."""
    t2 = aq.tip_arms(plant, k, c)
    if t2 is None:
        return None
    V, phi = t2
    try:
        d = delta_at(bike, phi, V, u.lin.steer_ratio(V) * phi)
    except (RuntimeError, ValueError):
        return None
    return tip_exact(bike, plant, k, c, guess=(phi, d))


def bisect(f, hi, n=26):
    """Largest x in [0, hi] with f(x) not None (bisection)."""
    lo = 0.0
    for _ in range(n):
        mid = 0.5 * (lo + hi)
        if f(mid) is None:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def run(name, p):
    t0 = time.time()
    g = load(name)
    u = Upright(p)
    bike = Whipple(p)
    aq = ArmQuartic(p)
    k0m, kcm, sym = u.meijaard_form()
    a1c, a2c = u.corner()
    out = dict(
        name=name,
        V_a3=u.V_a3,
        v_c=u.v_c,
        v_w=u.v_w(),
        k0=u.k0,
        kappa_c=u.kappa_c,
        k_star=u.k_star,
        c_star=u.c_star,
        Ia_star=u.Ia_star(0.0),
        corner_a1=a1c,
        corner_a2=a2c,
        meijaard_form_error=max(abs(k0m - u.k0), abs(kcm - u.kappa_c)),
        S_A=sym["S_A"],
        mT=sym["mT"],
        h=sym["h"],
    )
    Vs = np.linspace(0.6 * u.V_a3, 1.05 * u.v_c, 3001)
    out["k_star_eigen_scan"] = bisect(lambda k: u.band_numeric(k, Vs=Vs), 3 * u.k_star)
    out["c_star_eigen_scan"] = bisect(lambda c: u.band_numeric(0.0, c, Vs=Vs), 3 * u.c_star)
    for plant in ("free", "constant"):
        r = dict(lean0_grid=tongue(g, plant)["lean"])
        rows = []
        for f in KFRAC:
            k = f * u.k_star
            t = tongue(g, plant, k)
            t["frac"], t["band"] = f, u.band(k)
            te = t["tip_exact"] = exact_tip(bike, aq, u, plant, k, 0.0)
            t["lean_star"] = max(t["lean"], te["phi_deg"] if te else 0.0)
            t["tip_limited"] = bool(te and te["phi_deg"] >= t["lean"])
            for order in (1, 2):
                t[f"rule_order{order}"] = largest_lean_rule(p, k, plant, order)
            rows.append(t)
        r["k_rows"] = rows
        r["beyond"] = [dict(frac=f, **tongue(g, plant, f * u.k_star)) for f in KBEYOND]
        crow = []
        for f in CFRAC:
            c = f * u.c_star
            t = tongue(g, plant, 0.0, c)
            t["frac"], t["band"] = f, u.band(0.0, c)
            te = t["tip_exact"] = exact_tip(bike, aq, u, plant, 0.0, c)
            t["lean_star"] = max(t["lean"], te["phi_deg"] if te else 0.0)
            crow.append(t)
        r["c_rows"] = crow
        r["c_beyond"] = [
            dict(frac=f, **tongue(g, plant, 0.0, f * u.c_star)) for f in (1.05, 1.5, 3.0)
        ]
        r["literature"] = {
            key: dict(band=u.band(k, c), **tongue(g, plant, k, c))
            for key, (k, c) in LITERATURE.items()
        }
        out[plant] = r
    OUT.mkdir(parents=True, exist_ok=True)
    safe = name.replace("+", "_").replace("(", "").replace(")", "")
    (OUT / f"{safe}.json").write_text(json.dumps(out, indent=1, default=float))
    print(
        f"{name:32s} k* {u.k_star:6.3f} c* {u.c_star:6.2f} lean0 free "
        f"{out['free']['lean0_grid']:.2f}  {time.time() - t0:.0f} s",
        flush=True,
    )
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--quick", action="store_true", help="benchmark only")
    ap.add_argument("--sets", nargs="*", help="names of the parameter sets")
    a = ap.parse_args()
    sets = all_sets()
    names = ["benchmark"] if a.quick else (a.sets or list(sets))
    for n in names:
        run(n, sets[n])


if __name__ == "__main__":
    main()
