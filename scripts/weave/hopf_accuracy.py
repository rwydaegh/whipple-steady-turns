"""Accuracy of the Hopf curve from the quartic to order phi^2 and with the exact a4 (Sect. 8.2,
"Accuracy of the Hopf curve").
"""

import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
from scipy.optimize import brentq, newton  # noqa: E402

from whipple import stability as st  # noqa: E402
from whipple.bicycles import BICYCLES, measured_sets  # noqa: E402
from whipple.linear import Linear  # noqa: E402
from whipple.model import Whipple  # noqa: E402
from whipple.quartic import Quartic  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "data" / "weave" / "hopf_accuracy.json"
EXTRA = ("Silver", "Yellowrev")  # open regions, reported apart
DEG = math.degrees


def test_sets():
    sets = dict(BICYCLES)
    sets.update({k: p for k, p in measured_sets("riderless").items() if k != "Balanceassistv1"})
    return sets


def hopf_curve(bike, plant, ds_max):
    """Exact Hopf curve, (lean, steer) points from upright to where a3 or a4 changes sign."""
    lin = Linear(bike.p)
    vw, phi0 = lin.weave_speed((2.0, 7.0)), 2e-3

    def test(x):
        return st.hurwitz(st.characteristic(bike, *x, plant))[0]

    def point(V):
        return np.array([phi0, st.delta_for_speed(bike, phi0, V, phi0 * lin.steer_ratio(V), plant)])

    x0 = point(brentq(lambda V: test(point(V)), vw - 0.1, vw + 0.1, xtol=1e-12))

    def stop(x):
        a = st.characteristic(bike, *x, plant)
        return a[2] < 0 or a[3] < 0 or x[0] > 0.45

    c = st.continue_curve(test, x0, np.array([1.0, 0.0]), stop, ds_max=ds_max)
    return c[:-1], st.characteristic(bike, *c[-1], plant)


def steer_at(bike, phi, V, guess):
    """Steer of the turn with lean phi and contact speed V (secant method)."""

    def res(d):
        return bike.p["rR"] * math.sqrt(bike.turn_spin_torque(phi, d)[0]) - V

    return newton(res, guess, x1=guess * (1 + 1e-4) + 1e-7, tol=1e-13, maxiter=100)


def exact_a4(bike, phi, delta, plant):
    if plant == "free":
        return st.capsize_formula(bike, phi, delta)  # Theorem 1
    return st.characteristic(bike, phi, delta, plant)[3]


def hybrid_lean(bike, q, plant, V, phi_e, delta_e):
    """Root in lean of a1 a2 a3 - a3^2 - a1^2 a4 at speed V: a1..a3 to order phi^2, a4 exact."""

    def H(phi):
        d = steer_at(bike, phi, V, delta_e * phi / phi_e)
        a = q.a(V, phi, plant)
        return a[0] * a[1] * a[2] - a[2] ** 2 - a[0] ** 2 * exact_a4(bike, phi, d, plant)

    for w in (0.01, 0.025, 0.05):
        lo, hi = max(phi_e - w, 1e-3), phi_e + w
        try:
            if H(lo) * H(hi) < 0:
                return brentq(H, lo, hi, xtol=1e-10)
        except (RuntimeError, ValueError):
            continue
    return math.nan


def eigen_lean(law, plant, V):
    """Lean where the weave eigenvalue series Re(lambda + Lambda phi^2) at fixed V is zero."""
    t = law.terms(V)[0]
    Lam = t["Lambda"] - t["energy_speed"] - (t["spin_coupling"] if plant == "constant" else 0)
    x = -t["lam"].real / Lam.real
    return math.sqrt(x) if x > 0 else math.nan


def hybrid_tip(bike, q, plant, bt):
    """Point of the exact fold curve where a3 to order phi^2 is zero."""

    def F(x):
        V = bike.turn(*x).V
        return np.array([exact_a4(bike, *x, plant), q.a(V, x[0], plant)[2]])

    x = st.newton_system(F, bt)
    t = bike.turn(*x)
    return t.V, DEG(t.phi)


def analyse(bike, q, plant, npts, ds_max):
    curve, a = hopf_curve(bike, plant, ds_max)
    out = {}
    if a[2] < 0:  # the curve ends where a3 = a4 = 0: a Bogdanov-Takens point
        bt = st.bogdanov_takens(bike, curve[-1], plant)
        tb = bike.turn(*bt)
        curve = np.vstack([curve, bt])
        out["bt"] = [tb.V, DEG(tb.phi)]
        try:
            out["hybrid_tip"] = list(hybrid_tip(bike, q, plant, bt))
        except (RuntimeError, ValueError, np.linalg.LinAlgError):
            out["hybrid_tip"] = None
    idx = np.unique(np.linspace(0, len(curve) - 1, npts).round().astype(int))
    rows = []
    for x in curve[idx]:
        t = st.trimmed_turn(bike, *x, plant)
        if x[0] < 0.01:
            continue
        V = t.V
        o2 = q.hopf_x(V, plant)
        rows.append(
            [
                V,
                DEG(x[0]),
                DEG(math.sqrt(o2)) if o2 > 0 else math.nan,
                DEG(hybrid_lean(bike, q, plant, V, *x)),
                DEG(eigen_lean(q.law, plant, V)),
            ]
        )
    r = np.array(rows)
    err = np.abs(r[:, 2:] - r[:, [1]])
    for k, name in enumerate(("order2", "hybrid", "eigen")):
        e = err[:, k]
        i = int(np.nanargmax(e))
        out[name] = dict(max_err=float(e[i]), at_V=float(r[i, 0]), n_nan=int(np.isnan(e).sum()))
    out["largest_hopf_lean"] = float(r[:, 1].max())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="fewer points, benchmark and Gyro only")
    args = ap.parse_args()
    sets = test_sets()
    if args.quick:
        sets = {k: sets[k] for k in ("benchmark", "Gyro")}
    npts, ds = (15, 1e-2) if args.quick else (60, 6e-3)
    res = {}
    for name, p in sets.items():
        bike, q = Whipple(p), Quartic(p)
        res[name] = {pl: analyse(bike, q, pl, npts, ds) for pl in ("free", "constant")}
        for pl, d in res[name].items():
            print(
                name,
                pl,
                {k: round(d[k]["max_err"], 3) for k in ("order2", "hybrid", "eigen")},
                "BT",
                d.get("bt"),
                "hybrid tip",
                d.get("hybrid_tip"),
                flush=True,
            )

    def worst(names, plant, key):
        k = max(names, key=lambda n: res[n][plant][key]["max_err"])
        return [res[k][plant][key]["max_err"], k]

    main13 = [n for n in res if n not in EXTRA]
    summ = {}
    for plant in ("free", "constant"):
        tips = [
            abs(res[n][plant]["hybrid_tip"][0] / res[n][plant]["bt"][0] - 1)
            for n in main13
            if res[n][plant].get("hybrid_tip")
        ]
        eig = [res[n][plant]["eigen"]["max_err"] for n in main13]
        summ[plant] = {
            "order2_max_13_sets": worst(main13, plant, "order2"),
            "hybrid_max_13_sets": worst(main13, plant, "hybrid"),
            "hybrid_max_with_Silver_Yellowrev": worst(list(res), plant, "hybrid"),
            "hybrid_tip_speed_error_max_13_sets": max(tips) if tips else None,
            "eigen_error_range_13_sets": [min(eig), max(eig)],
        }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not args.quick:
        OUT.write_text(json.dumps(dict(summary=summ, sets=res), indent=1) + "\n")
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
