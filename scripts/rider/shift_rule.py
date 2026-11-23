"""Hands-free capsize rule eq. rider-rule with the fitted coefficient b21 (supplementary material, "Mass shift: further results")."""  # noqa: E501

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import fsolve
from shift_model import Whipple, delta_for_speed, params_of, tag, totals, yB_from_Delta
from shift_tongue import DATA as TONGUES
from shift_tongue import SETS

from whipple import capsize_rule as cr
from whipple.linear import Linear

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "data" / "rider" / "shift_rule"
MONO = [(i, n - i) for n in (1, 3, 5) for i in range(n + 1)]  # odd terms phi_G^i Delta^j


def fit(b, p, lin, V, a=0.06):
    """Least-squares fit of T(phi_G, Delta) at fixed V; coefficients c_ij of phi_G^i Delta^j."""
    pos = np.linspace(a / 4, a, 4)
    pts = np.r_[-pos[::-1], pos]
    X, y = [], []
    r = lin.steer_ratio(V)
    for phiG, D in itertools.product(pts, np.r_[pts / 1.5, 0.0]):
        bb = b.with_params(yB=yB_from_Delta(p, D))
        d = delta_for_speed(bb, phiG - D, V, r * phiG if abs(phiG) > 1e-9 else 1e-6)
        y.append(bb.turn(phiG - D, d).T)
        X.append([phiG**i * D**j for i, j in MONO])
    c, *_ = np.linalg.lstsq(np.array(X), np.array(y), rcond=None)
    return {f"c{i}{j}": float(v) for (i, j), v in zip(MONO, c)}


def rule_constant(c, form):
    """Hands-free capsize lean at constant speed: "cubic" phi_G^2 = -tau1/(3 tau3), or "next"."""
    t1, t3, t5 = c["c10"], c["c30"], c["c50"]
    if form == "cubic":
        x = -t1 / (3 * t3)
        return math.sqrt(x) if x > 0 else math.nan
    k = 2 * c["c21"] / c["c01"]
    roots = np.roots([5 * t5 - k * t3, 3 * t3 - k * t1, t1])
    roots = roots[(np.abs(roots.imag) < 1e-12) & (roots.real > 0)].real
    return math.sqrt(roots.min()) if len(roots) else math.nan


def factor(c):
    """beta21 + sqrt(1 + beta21^2) and beta21 = 2 b21 tau1 / (9 tau3 g S_A)."""
    beta = 2 * c["c21"] * c["c10"] / (9 * c["c30"] * c["c01"])
    return beta + math.sqrt(1 + beta * beta), beta


def poly_fold(c):
    """T = 0 and dT/dphi_G = 0 with the full degree-5 fit."""

    def F(x):
        ph, D = x
        T = sum(c[f"c{i}{j}"] * ph**i * D**j for i, j in MONO)
        dT = sum(i * c[f"c{i}{j}"] * ph ** (i - 1) * D**j for i, j in MONO if i > 0)
        return [T, dT]

    x0 = rule_constant(c, "cubic")
    if not x0 == x0:
        return math.nan
    sol, _, ier, _ = fsolve(F, [x0, -c["c10"] * x0 / c["c01"] / 3], full_output=True)
    return sol[0] if ier == 1 else math.nan


def exact_fold(name, plant):
    r = json.loads((TONGUES / f"{tag(name)}_f1_{plant}.json").read_text())
    V, P = np.array(r["fold"]["V"]), np.array(r["fold"]["phiG"])
    o = np.argsort(V)
    return V[o], P[o], r["largest"]["phiG"], r["bt"]["V"]


def run(name):
    p = params_of(name)
    b, lin = Whipple(p), Linear(p)
    SA = totals(p)[2]
    vc = lin.capsize_speed()
    ex = {pl: exact_fold(name, pl) for pl in ("constant", "free")}
    rows = []
    for V in np.linspace(min(ex["constant"][3], ex["free"][3]), 0.99 * vc, 30):
        c = fit(b, p, lin, V)
        fac, beta = factor(c)
        f1 = cr.phi_c_free(p, V, order=1)
        row = dict(
            V=V,
            **c,
            gSA=p["g"] * SA,
            beta21=beta,
            factor=fac,
            constant_cubic=rule_constant(c, "cubic"),
            constant_factor=rule_constant(c, "cubic") * fac,
            constant_next=rule_constant(c, "next"),
            constant_poly=poly_fold(c),
            free_first_order=f1,
            free_factor=f1 * fac,
        )
        for pl in ("constant", "free"):
            Vf, Pf = ex[pl][:2]
            row[f"{pl}_exact"] = float(np.interp(V, Vf, Pf)) if Vf[0] <= V <= Vf[-1] else math.nan
        rows.append(row)
    summ = {}
    for key in (
        "constant_cubic",
        "constant_factor",
        "constant_next",
        "constant_poly",
        "free_first_order",
        "free_factor",
    ):
        L = ex[key.split("_")[0]][2]
        vals = np.array([r[key] for r in rows])
        i = int(np.nanargmax(vals))
        summ[key] = dict(
            largest_deg=math.degrees(vals[i]),
            V=rows[i]["V"],
            exact_largest_deg=math.degrees(L["phiG"]),
            exact_V=L["V"],
            error_deg=math.degrees(vals[i] - L["phiG"]),
        )
        print(name, key, f"error {summ[key]['error_deg']:+.3f} deg", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{tag(name)}.json").write_text(
        json.dumps(dict(bike=name, v_c=vc, rows=rows, summary=summ), indent=1) + "\n"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sets", nargs="*", default=list(SETS))
    for name in ap.parse_args().sets:
        run(name)


if __name__ == "__main__":
    main()
