"""tab:trails and paths between measured bicycles (supplement, "Bicycle types")."""

import argparse
import math
import time

import numpy as np
from common import all_sets, fold_fn, poly_data, upright, write_json
from scipy.optimize import brentq

from whipple import capsize_rule as cr
from whipple.model import Whipple


def residual(make, x, plant, h=2e-4):
    """(F, F_phi, F_delta) at x = (phi, delta, s): zero where the fold curve has a saddle."""
    bike = Whipple(make(x[2]))

    def F(a, b):
        return fold_fn(bike, a, b, plant)

    return np.array(
        [
            F(x[0], x[1]),
            (F(x[0] + h, x[1]) - F(x[0] - h, x[1])) / (2 * h),
            (F(x[0], x[1] + h) - F(x[0], x[1] - h)) / (2 * h),
        ]
    )


def saddle(make, x0, plant, hs, tol=1e-9, maxiter=30):
    x = np.array(x0, float)
    for _ in range(maxiter):
        r = residual(make, x, plant)
        J = np.empty((3, 3))
        for k, hk in enumerate((2e-4, 2e-4, hs)):
            e = np.zeros(3)
            e[k] = hk
            J[:, k] = (residual(make, x + e, plant) - residual(make, x - e, plant)) / (2 * hk)
        dx = -np.linalg.solve(J, r)
        x = x + min(1.0, 0.1 / max(abs(dx[0]), 1e-12), 0.05 / max(abs(dx[1]), 1e-12)) * dx
        if np.abs(dx).max() < tol:
            return x
    raise RuntimeError("saddle Newton did not converge")


def critical(make, lo, hi):
    """Local criterion (tau3(v_c) = 0 or D_free(v_c) = 0) and the opening of the region."""
    res = {}
    for plant, key in (("constant", "tau3"), ("free", "D_free")):
        s1 = brentq(lambda s, key=key: upright(make(s))[key], lo, hi, xtol=1e-7)
        u = upright(make(s1))
        ser = cr.series(make(s1))
        if plant == "constant":
            dD = 3 * u["dtau3"]
        else:

            def D(v, ser=ser):
                q = ser.with_derivatives(v)
                return 3 * q["tau3"] - 2 * q["h2"] * q["dtau1"] + q["dh2"] * q["tau1"]

            dD = (D(u["v_c"] + 1e-3) - D(u["v_c"] - 1e-3)) / 2e-3
        xs = -u["dtau1"] / dD
        res[plant] = dict(local=s1, v_c=u["v_c"])
        for scale in (1.0, 0.8, 0.6, 1.2):
            phi = math.sqrt(xs) * scale
            d = u["r1"] * phi + u["r3"] * phi**3 + u["r5"] * phi**5
            try:
                x = saddle(make, [phi, d, s1], plant, hs=1e-4 * max(abs(s1), 0.1))
            except (RuntimeError, np.linalg.LinAlgError):
                continue
            bike = Whipple(make(x[2]))
            t = bike.turn(x[0], x[1])
            if lo - 0.5 * (hi - lo) < x[2] < hi + 0.5 * (hi - lo) and x[0] > math.radians(1):
                res[plant].update(
                    opening=x[2], lean_deg=math.degrees(x[0]), V=t.V, ay_g=t.ay / 9.81
                )
                break
        print(plant, res[plant], flush=True)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="benchmark trail only")
    args = ap.parse_args()
    t0 = time.time()
    B = all_sets()

    def line(a, b):
        return lambda s: {k: (1 - s) * B[a][k] + s * B[b][k] for k in B[a]}

    cases = {"benchmark trail": (lambda s: dict(B["benchmark"], c=float(s)), 0.10, 0.25)}
    if not args.quick:
        cases["A trail"] = (lambda s: dict(B["A"], c=float(s)), 0.20, 0.45)
        cases["Yellow to Yellowrev"] = (line("Yellow", "Yellowrev"), 0.0, 0.9)
        cases["Crescendo to Silver"] = (line("Crescendo", "Silver"), 0.0, 1.0)
    out = {}
    for name, (make, lo, hi) in cases.items():
        print(name, flush=True)
        out[name] = critical(make, lo, hi)
    if not args.quick:
        yy = line("Yellow", "Yellowrev")
        out["Yellow to Yellowrev"]["no capsize speed above"] = brentq(
            lambda s: poly_data(yy(s))["r_c"], 0.5, 1.0, xtol=1e-6
        )
        swap = {
            "Silver with the trail of Crescendo": dict(B["Silver"], c=B["Crescendo"]["c"]),
            "Silver with the wheelbase of Crescendo": dict(B["Silver"], w=B["Crescendo"]["w"]),
            "Crescendo with the trail of Silver": dict(B["Crescendo"], c=B["Silver"]["c"]),
            "Silver": B["Silver"],
        }
        out["tau3 at v_c"] = {k: upright(p)["tau3"] for k, p in swap.items()}
        print(out["tau3 at v_c"])
    out["seconds"] = time.time() - t0
    write_json("critical_trails.json", out, quick=args.quick)


if __name__ == "__main__":
    main()
