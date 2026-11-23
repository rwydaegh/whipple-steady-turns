"""Items (a) to (e) and V_min (supplement, "First order in the steer angle at any lean")."""

import argparse
import math
import time

import numpy as np
from common import all_sets, write_json
from scipy.optimize import minimize_scalar

from whipple.linear import Linear
from whipple.model import Whipple

H = 1e-30
LEANS = (2, 5, 10, 20, 30, 45, 60, 75)


def trim(bike, phi, delta):
    z = np.zeros(3, complex if np.iscomplexobj(np.array([phi, delta])) else float)
    _, M, Fg, _ = bike.parts(phi, delta, z)
    one = z.copy()
    one[2] = 1.0
    Fc = bike.parts(phi, delta, one)[2] - Fg
    S = -Fg[0] / Fc[0]
    return S, -Fg[1] - S * Fc[1], M, Fc


def first_order(p):
    """Largest relative deviation of (a) to (e) over the leans 2 to 75 deg."""
    b = Whipple(p)
    lin = Linear(p)
    g, rR = lin.g, p["rR"]
    Z, SA, ka, kb = -lin.K0[0, 0], -lin.K0[0, 1], lin.K2[0, 1], lin.K2[1, 1]
    U00 = b.potential(0.0, 0.0)
    tau_inf = g * (Z * kb / ka - SA)
    w = dict(a=0.0, b=0.0, c=0.0, d=0.0, e=0.0)
    for phi in np.radians(LEANS):
        _, _, M, Fc = trim(b, phi, 1j * H)
        dU = b.potential(phi, 1j * H).imag / H
        w["a"] = max(
            w["a"], abs(b.potential(phi, 0.0) - U00 - g * Z * (math.cos(phi) - 1)) / (g * Z)
        )
        w["b"] = max(w["b"], abs(dU + g * SA * math.sin(phi)) / (g * SA))
        dFc = Fc[:2].imag / H / rR**2
        w["c"] = max(w["c"], abs(dFc[0] + ka) / ka, abs(dFc[1] + kb) / kb)
        dms = M[2, 2].imag / H / rR**2
        w["d"] = max(w["d"], abs(dms + 2 * ka * math.tan(phi)) / (2 * ka * math.tan(phi)))
        T = trim(b, phi, 1e-7)[1]
        w["e"] = max(w["e"], abs(T - tau_inf * math.sin(phi)) / abs(tau_inf))
    return w


def v_min(p, degs=(10, 20, 30, 45)):
    """Lowest speed of a turn at fixed lean, where dS/d(delta) = 0."""
    b = Whipple(p)
    out = {}
    for deg in degs:
        phi = math.radians(deg)

        def S(d, phi=phi):
            try:
                s = trim(b, phi, d)[0]
            except (RuntimeError, np.linalg.LinAlgError):
                return 1e9
            return s if s > 0 else 1e9

        ds = np.linspace(0.02, 1.5, 150)
        k = int(np.argmin([S(d) for d in ds]))
        r = minimize_scalar(
            S,
            bounds=(ds[max(k - 1, 0)], ds[min(k + 1, 149)]),
            method="bounded",
            options=dict(xatol=1e-10),
        )
        out[deg] = p["rR"] * math.sqrt(r.fun)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="benchmark, A and B only")
    args = ap.parse_args()
    t0 = time.time()
    sets = all_sets()
    names = ["benchmark", "A", "B"] if args.quick else list(sets)
    rows = {}
    for n in names:
        rows[n] = first_order(sets[n])
        print(n, {k: f"{v:.1e}" for k, v in rows[n].items()}, flush=True)
    worst = {k: max(r[k] for r in rows.values()) for k in "abcde"}
    vm = v_min(sets["benchmark"])
    print("largest deviations", worst, "benchmark V_min", vm)
    write_json(
        "first_order.json",
        dict(
            leans_deg=LEANS, largest=worst, sets=rows, benchmark_v_min=vm, seconds=time.time() - t0
        ),
        quick=args.quick,
    )


if __name__ == "__main__":
    main()
