"""Base sets, largest stable lean, tip and elasticities for Sect. 12, "Design rules"."""

import gzip
import json
import math
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402
from moves import with_rider  # noqa: E402
from scipy.optimize import brentq  # noqa: E402

from whipple import stability as st  # noqa: E402
from whipple.bicycles import BICYCLES, measured_sets  # noqa: E402
from whipple.linear import Linear  # noqa: E402
from whipple.model import Whipple  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "design"
G = 9.81
PLANTS = ("free", "constant")

_RIDER = measured_sets("rider")
# base name: (parameters, name of the stored result in data/measured/results)
BASES = {
    "benchmark": (dict(BICYCLES["benchmark"]), "benchmark"),
    "A_rider": (with_rider(BICYCLES["A"], 10.0), "A_rider(invented)"),
    "Crescendo+Jason": (_RIDER["Crescendo+Jason"], "Crescendo+Jason"),
    "Browser+Jason": (_RIDER["Browser+Jason"], "Browser+Jason"),
}
BASE_LABELS = {
    "benchmark": "Benchmark",
    "A_rider": "A + rider",
    "Crescendo+Jason": "Crescendo + Jason",
    "Browser+Jason": "Browser + Jason",
}


def stored(name):
    with gzip.open(REPO / "data" / "measured" / "results" / f"{name}.json.gz", "rt") as f:
        return json.load(f)


def fold_test(bike, plant):
    return lambda x: st.characteristic(bike, *x, plant)[3]


def lean_star(p, plant, guess):
    """Top of the fold curve: (lean deg, V, point (phi, delta), ok). ok: the point is a fold top
    with the three other roots stable (a4 = 0, a1 > 0, a3 > 0, a1 a2 > a3)."""
    bike = Whipple(p)
    x = st.boundary_extremum(bike, fold_test(bike, plant), lambda t: t.phi, guess, plant)
    t = st.trimmed_turn(bike, *x, plant)
    a = st.characteristic(bike, *x, plant)
    ok = bool(abs(a[3]) < 1e-8 and a[0] > 0 and a[2] > 0 and a[0] * a[1] - a[2] > 0)
    return math.degrees(t.phi), t.V, x, ok


def largest_ay(p, plant, guess):
    """Largest lateral acceleration on the fold curve: (a_y / g, V, point)."""
    bike = Whipple(p)
    x = st.boundary_extremum(bike, fold_test(bike, plant), lambda t: t.ay, guess, plant)
    t = bike.turn(*x)
    return t.ay / G, t.V, x


def tip(p, plant, guess):
    """Bogdanov-Takens tip (a3 = a4 = 0): (V, lean deg, point)."""
    bike = Whipple(p)
    x = st.bogdanov_takens(bike, guess, plant)
    t = bike.turn(*x)
    return t.V, math.degrees(t.phi), x


def weave_speed(p, bracket=(2.0, 8.0)):
    return Linear(p).weave_speed(bracket)


def capsize_speed(p):
    return Linear(p).capsize_speed()


PHI_T = math.radians(5.0)


def torque_per_g(p, V, phi=PHI_T):
    """Steer torque per g of lateral acceleration of the turn at 5 deg lean and speed V (N m)."""
    bike, lin = Whipple(p), Linear(p)
    g0 = phi * lin.steer_ratio(V)
    try:
        d = st.delta_for_speed(bike, phi, V, g0)
    except RuntimeError:

        def res(dd):
            t = bike.turn(phi, dd)
            return (t.V if t else 0.0) - V

        d = brentq(res, 0.5 * g0, 2.0 * g0, xtol=1e-15)
    t = bike.turn(phi, d)
    return t.T / (t.ay / G)


class Base:
    """A base set with its start points: the fold top of each plant and the free-rolling tip."""

    def __init__(self, name):
        self.name = name
        self.p, key = BASES[name]
        res = stored(key)
        self.guess = {}
        for plant in PLANTS:
            fold = np.array(res[plant]["curves"]["fold"])
            self.guess[plant] = fold[int(np.argmax(fold[:, 0]))]
        self.guess["tip"] = np.array(res["free"]["curves"]["hopf"])[-1]
        self.guess["ay"] = self.guess["free"]
        self.values = self.evaluate(self.p)
        for plant in PLANTS:
            self.guess[plant] = self.values[f"x_{plant}"]
        self.guess["tip"] = self.values["x_tip"]

    def evaluate(self, p, quantities=("lean_free", "lean_constant", "tip_V", "v_w", "v_c", "T3")):
        """Largest stable lean (both plants), tip speed, v_w, v_c, |T/a_y| at 3 m/s of p."""
        out = {}
        for plant in PLANTS:
            if f"lean_{plant}" in quantities:
                d, V, x, ok = lean_star(p, plant, self.guess[plant])
                out.update({f"lean_{plant}": d, f"V_{plant}": V, f"x_{plant}": x,
                            f"ok_{plant}": ok})  # fmt: skip
        if "tip_V" in quantities:
            out["tip_V"], out["tip_deg"], out["x_tip"] = tip(p, "free", self.guess["tip"])
        if "v_w" in quantities:
            out["v_w"] = weave_speed(p)
        if "v_c" in quantities:
            out["v_c"] = capsize_speed(p)
        if "T3" in quantities:
            out["T3"] = abs(torque_per_g(p, 3.0))
        return out


QUANTITIES = ("lean_free", "lean_constant", "tip_V", "v_w", "v_c", "T3")


def elasticity(base, move, kind, step):
    """Central log difference of each quantity. kind "pct": d ln q / d ln f with f = exp(+-step);
    kind "cm": 100 d ln q / ds (% per cm) with s = +-step cm."""
    if kind == "pct":
        pp, pm, den = move(base.p, math.exp(step)), move(base.p, math.exp(-step)), 2 * step
    else:
        pp, pm, den = move(base.p, step), move(base.p, -step), 2 * step / 100
    a, b = base.evaluate(pp), base.evaluate(pm)
    out = {q: (math.log(a[q]) - math.log(b[q])) / den for q in QUANTITIES}
    out["ok"] = all(a[f"ok_{pl}"] and b[f"ok_{pl}"] for pl in PLANTS)
    return out


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, default=_plain) + "\n")
    print("wrote", path.relative_to(REPO))


def _plain(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    raise TypeError(type(x))
