"""fig:types (a, b), tau3 and fold tops against trail (Sect. 10, "Closed and open regions")."""

import math
import time

import numpy as np
from common import BICYCLES, clean, fold_top, upright, write_json
from scipy.optimize import brentq

from whipple import capsize_rule as cr
from whipple.model import Whipple

TRAILS = {
    "benchmark": (0.08, 0.10, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19),
    "A": (0.0832912572138, 0.16, 0.24, 0.30, 0.34),
}


def tau3_curve(p):
    s = cr.series(p)
    V = np.linspace(3.0, 12.0, 181)
    t3 = np.array([s.coefficients(v)["tau3"] for v in V])
    k = np.flatnonzero(np.sign(t3[:-1]) != np.sign(t3[1:]))[0]
    V3 = brentq(lambda v: s.coefficients(v)["tau3"], V[k], V[k + 1])
    return dict(c=p["c"], V=V, tau3=t3, V3=V3, v_c=s.v_c, tau3_at_vc=s.coefficients(s.v_c)["tau3"])


def tops(name):
    """Exact free-rolling fold tops, each started from the top before (first from the rule)."""
    base = clean(BICYCLES[name])
    out, x = [], None
    for c in TRAILS[name]:
        p = dict(base, c=c)
        bike = Whipple(p)
        if x is None:
            r = cr.largest_lean_rule(p, "free", 2)
            co = cr.series(p).coefficients(r["V"])
            x = (r["phi"], co["r1"] * r["phi"] + co["r3"] * r["phi"] ** 3)
        t = fold_top(bike, "free", x)
        x = (t.phi * 1.05, t.delta)
        out.append(dict(c=c, lean_deg=math.degrees(t.phi), V=t.V))
        print(f"{name} c = {c:.4f}: top {math.degrees(t.phi):.3f} deg at {t.V:.3f} m/s", flush=True)
    return out


def main():
    t0 = time.time()
    base = clean(BICYCLES["benchmark"])
    curves = [tau3_curve(dict(base, c=c)) for c in (0.08, 0.20)]
    V3 = [upright(dict(base, c=c))["V3"] for c in np.arange(0.08, 0.2501, 0.01)]
    out = dict(
        tau3=curves,
        V3_range_trail_80_to_250mm=[min(V3), max(V3)],
        tops={n: tops(n) for n in TRAILS},
        seconds=time.time() - t0,
    )
    for d in curves:
        print(
            f"c = {d['c']}: V3 {d['V3']:.3f}, v_c {d['v_c']:.3f}, tau3(v_c) {d['tau3_at_vc']:+.3f}"
        )
    print("V3 for trails 80 to 250 mm:", out["V3_range_trail_80_to_250mm"])
    write_json("trail_series.json", out)


if __name__ == "__main__":
    main()
