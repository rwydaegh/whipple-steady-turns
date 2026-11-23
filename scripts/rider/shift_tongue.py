"""Exact boundary of the stable turns with steer torque only (f = 0) or hands free with the shift alone (f = 1) (Sect. 13.3, "Mass shift and steer torque", Fig. fig:rider(b); supplementary material, "Mass shift: further results")."""  # noqa: E501

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, newton
from shift_model import Whipple, hurwitz, params_of, quartic, tag, totals

from whipple import stability as st
from whipple.linear import Linear

REPO = Path(__file__).resolve().parents[2]
FULL = REPO / "results" / "rider" / "shift_tongues"
DATA = REPO / "data" / "rider" / "shift_tongues"
SETS = ("benchmark", "Browser+Jason", "Rigid+Jason", "Crescendo+Jason")


class Mix:
    """Turns kept by f T + (1 - f) g S_A Delta = 0 (f = 0: torque only, f = 1: shift only)."""

    def __init__(self, p, f, plant):
        self.p, self.f, self.plant = p, f, plant
        self.b = Whipple(p)
        self.mT, self.h, SA = totals(p)
        self.gSA = p["g"] * SA
        self.W = 0.0

    def Delta(self, W):
        return math.atan(self.p["mB"] * W / (self.mT * self.h))

    def trim(self, phi, delta):
        if self.f == 0:
            return 0.0

        def c(W):
            T = self.b.with_params(yB=W).turn_spin_torque(phi, delta)[1]
            return self.f * T + (1 - self.f) * self.gSA * self.Delta(W)

        W0 = self.W if math.isfinite(self.W) else 0.0
        W1 = W0 + 1e-4
        c0, c1 = c(W0), c(W1)
        for _ in range(40):  # secant
            if c1 == c0:
                break
            W0, c0, W1 = W1, c1, W1 - c1 * (W1 - W0) / (c1 - c0)
            c1 = c(W1)
            if abs(W1 - W0) < 1e-15 or abs(c1) < 1e-13:
                break
        if not math.isfinite(W1):
            raise RuntimeError("trim failed")
        self.W = W1
        return W1

    def bike(self, phi, delta):
        W = self.trim(phi, delta)
        return self.b.with_params(yB=W) if W else self.b

    def char(self, x):
        try:
            return quartic(self.bike(*x), x[0], x[1], self.plant)
        except (RuntimeError, ValueError, ZeroDivisionError, np.linalg.LinAlgError):
            self.W = 0.0
            return np.full(4, np.nan)

    def info(self, x):
        bb = self.bike(*x)
        t = bb.turn(*x)
        D = self.Delta(bb.p["yB"])
        a = quartic(bb, x[0], x[1], self.plant)
        return dict(
            phi=x[0],
            delta=x[1],
            V=t.V,
            T=t.T,
            yB=bb.p["yB"],
            Delta=D,
            phiG=x[0] + D,
            ay=t.ay,
            a=list(a),
            D3=hurwitz(a),
        )

    def speed(self, x):
        bb = self.bike(*x)
        s2 = bb.turn_spin_torque(*x)[0]
        return self.p["rR"] * math.sqrt(s2) if s2 > 0 else float("nan")


def compute(p, f, plant, phi_max=0.40, ds_max=3e-3):
    m = Mix(p, f, plant)
    lin = Linear(p)
    vw, vc = lin.weave_speed(), lin.capsize_speed()
    phi0 = 2e-3

    def hopf_test(x):
        return hurwitz(m.char(x))

    def fold_test(x):
        return m.char(x)[3]

    def point(V):
        d = newton(lambda d: m.speed([phi0, d]) - V, phi0 * lin.steer_ratio(V), tol=1e-14)
        return np.array([phi0, d])

    def start(test, v):
        return point(brentq(lambda V: test(point(V)), v - 0.1, v + 0.1, xtol=1e-12))

    up = np.array([1.0, 0.0])
    hopf = st.continue_curve(
        hopf_test,
        start(hopf_test, vw),
        up,
        stop=lambda x: m.char(x)[2] < 0 or x[0] > phi_max,
        ds_max=ds_max,
    )
    x0 = start(fold_test, vc)
    a3_up = m.char(x0)[2]
    fold = st.continue_curve(fold_test, x0, up, stop=lambda x: x[0] > phi_max, ds_max=ds_max)
    a3 = np.array([m.char(x)[2] for x in fold])
    ks = np.flatnonzero((np.sign(a3) != np.sign(a3_up)) & np.isfinite(a3))
    bt = None
    if len(ks):
        k = int(ks[0])
        try:
            bt = st.newton_system(lambda x: m.char(x)[2:], fold[k])
        except (RuntimeError, np.linalg.LinAlgError):
            bt = fold[k]  # first fold point past the sign change of a3
            print("tip: Newton failed, first fold point past the sign change of a3 used")
        fold = np.vstack([fold[:k], bt])
        hopf = np.vstack([hopf[:-1], bt])
    hopf_i = [m.info(x) for x in hopf]
    fold_i = [m.info(x) for x in fold]

    def refine(test, x, quantity):
        def F(y):
            gr = st.gradient(test, y, 1e-6)
            t = np.array([-gr[1], gr[0]])
            qg = st.gradient(lambda z: m.info(z)[quantity], y, 1e-6)
            return np.array([test(y) / np.hypot(*gr), t @ qg / np.hypot(*t)])

        return st.newton_system(F, np.asarray(x, float), h=1e-6, tol=1e-9)

    best = {}
    for quantity in ("phiG", "phi"):
        cand = []
        for name, curve, info, test in (
            ("hopf", hopf, hopf_i, hopf_test),
            ("fold", fold, fold_i, fold_test),
        ):
            i = int(np.argmax([r[quantity] for r in info]))
            x = curve[i]
            if 0 < i < len(curve) - 1:
                try:
                    x = refine(test, x, quantity)
                except (RuntimeError, np.linalg.LinAlgError):
                    pass
            r = m.info(x)
            r["curve"] = name
            cand.append(r)
        best[quantity] = max(cand, key=lambda r: r[quantity])
    return dict(
        f=f,
        plant=plant,
        vw=vw,
        vc=vc,
        bt=m.info(bt) if bt is not None else None,
        hopf=hopf_i,
        fold=fold_i,
        largest=best,
    )


def compact(res):
    """Boundary points (V, phi, phi_G, Delta, T) and the tip and largest-lean points."""

    def cols(rows):
        return {k: [round(r[k], 7) for r in rows] for k in ("V", "phi", "phiG", "Delta", "T")}

    keep = ("phi", "delta", "V", "T", "yB", "Delta", "phiG", "ay", "curve")
    return dict(
        bike=res["bike"],
        f=res["f"],
        plant=res["plant"],
        hopf=cols(res["hopf"]),
        fold=cols(res["fold"]),
        bt={k: v for k, v in res["bt"].items() if k in keep} if res["bt"] else None,
        largest={q: {k: v for k, v in r.items() if k in keep} for q, r in res["largest"].items()},
    )


def run(name, f, plant):
    t0 = time.time()
    res = compute(params_of(name), f, plant)
    res["bike"] = name
    file = f"{tag(name)}_f{f:g}_{plant}.json"
    FULL.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    (FULL / file).write_text(json.dumps(res, indent=1))
    (DATA / file).write_text(json.dumps(compact(res)) + "\n")
    L = res["largest"]["phiG"]
    print(
        f"{name} f={f:g} {plant}: largest phi_G {math.degrees(L['phiG']):.3f} deg "
        f"(frame {math.degrees(L['phi']):.3f}) at {L['V']:.3f} m/s; {time.time() - t0:.0f} s",
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="benchmark, free rolling only")
    ap.add_argument("--sets", nargs="*", default=list(SETS))
    ap.add_argument("--f", nargs="*", type=float, default=[0.0, 1.0])
    ap.add_argument("--plants", nargs="*", default=["free", "constant"])
    ap.add_argument(
        "--compact",
        action="store_true",
        help="only rewrite data/ from the stored files in results/",
    )
    a = ap.parse_args()
    if a.compact:
        for file in sorted(FULL.glob("*.json")):
            DATA.mkdir(parents=True, exist_ok=True)
            (DATA / file.name).write_text(json.dumps(compact(json.loads(file.read_text()))) + "\n")
        return
    sets, plants = (["benchmark"], ["free"]) if a.quick else (a.sets, a.plants)
    for name in sets:
        for plant in plants:
            for f in a.f:
                run(name, f, plant)


if __name__ == "__main__":
    main()
