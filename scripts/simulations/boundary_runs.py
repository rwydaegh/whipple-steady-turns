"""Runs near the boundaries, torque limit, weave, 16 turns, two trails (supplementary Sect. 4)."""

import argparse
import math
import time

import numpy as np
from common import lean_speed, run, save, turn_at
from matplotlib.path import Path as MPath
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from whipple import stability as st
from whipple.bicycles import BENCHMARK
from whipple.model import Whipple
from whipple.weave import TorqueFamily, weave_start

G = 9.81
bike = Whipple(BENCHMARK)


def kind_of(b, tr):
    ev = st.eigenvalues(b, tr.phi, tr.delta)
    leading = ev[np.argmax(ev.real)]
    kind = "stable" if leading.real < 0 else "weave" if abs(leading.imag) > 1e-6 else "capsize"
    return kind, leading, ev


def turn_ay(V, ay_g):
    """Turn with rear-contact speed V and lateral acceleration ay_g * g."""
    R = V * V / (ay_g * G)
    phi = brentq(lambda p: turn_at(bike, V, math.degrees(p)).R - R, 0.05, 0.35, xtol=1e-14)
    return turn_at(bike, V, math.degrees(phi))


def boundary_ay(V):
    """a_y / g of the Hopf and fold points at speed V on the small-lean side of the tip."""

    def test(lean, k):
        tr = turn_at(bike, V, lean)
        return st.hurwitz(st.characteristic(bike, tr.phi, tr.delta))[k]

    out = []
    for k, lo, hi in ((0, 8.0, 10.5), (1, 10.6, 13.0)):
        lean = brentq(lambda x, k=k: test(x, k), lo, hi, xtol=1e-12)
        out.append(turn_at(bike, V, lean).ay / G)
    return out


def envelope_rate(t, dev, t_from=3.0, lo=1e-9):
    m = (t > t_from) & (dev < 1e-3) & (dev > lo)
    tt, lg = t[m], np.log(dev[m])
    env = [
        (
            tt[(tt >= a) & (tt < a + 2)][np.argmax(lg[(tt >= a) & (tt < a + 2)])],
            lg[(tt >= a) & (tt < a + 2)].max(),
        )
        for a in np.arange(tt[0], tt[-1], 2.0)
        if np.any((tt >= a) & (tt < a + 2))
    ]
    env = np.array(env)
    return float(np.polyfit(env[:, 0], env[:, 1], 1)[0]) if len(env) > 3 else None


def near_boundaries(quick):
    """Turns at 3.6 m/s inside and just outside the region, kicked, at constant torque."""
    ayH, ayF = boundary_ay(3.6)
    out = dict(boundary_ay_g=[ayH, ayF], runs=[])
    t_end = 60.0 if quick else 200.0
    for ay, kick, tag in (
        (0.20, 0.05, "inside"),
        (ayH - 0.01, 1e-6, "beyond the weave curve"),
        (ayF + 0.01, 1e-6, "beyond the fold curve"),
    ):
        tr = turn_ay(3.6, ay)
        y0 = np.array([tr.phi, tr.delta, kick, 0.0, tr.s])

        def fall(t, y, tr=tr):
            return 0.25 - abs(y[0] - tr.phi)

        fall.terminal = True
        sol = solve_ivp(
            lambda t, y, T=tr.T: bike.field(y, T),
            (0, t_end),
            y0,
            method="DOP853",
            rtol=1e-11,
            atol=1e-14,
            events=fall,
            max_step=0.02,
        )
        dev = np.hypot(sol.y[2], sol.y[3])
        late = None
        if tag == "inside":
            m = sol.t > 30.0
            late = envelope_rate(sol.t[m], dev[m], 30.0, 1e-13)
        end = bike.turn(sol.y[0, -1], sol.y[1, -1])
        r = dict(
            tag=tag,
            ay_g=ay,
            lean_deg=math.degrees(tr.phi),
            T=tr.T,
            kick=kick,
            eigenvalues=sorted(st.eigenvalues(bike, tr.phi, tr.delta), key=lambda x: x.real),
            t_end=sol.t[-1],
            left_turn=bool(sol.status == 1),
            lean_end_deg=math.degrees(sol.y[0, -1]),
            envelope_rate=envelope_rate(sol.t, dev, lo=1e-9 if kick > 1e-3 else 1e-5),
            envelope_rate_after_30s=late,
            end_turn_eigenvalues=sorted(
                st.eigenvalues(bike, end.phi, end.delta), key=lambda x: x.real
            )
            if end
            else None,
        )
        out["runs"].append(r)
        print(r, flush=True)
    return out


def level_turn(E0, lean_deg, guess):
    phi = math.radians(lean_deg)
    d = brentq(
        lambda d: bike.energy(bike.turn_state(bike.turn(phi, d))) - E0, guess * 0.6, guess * 1.6
    )
    return bike.turn(phi, d)


def stable_polygon():
    tg = st.tongue(bike, "free")
    pts = [(bike.turn(*x).V, math.degrees(x[0])) for x in tg.hopf]
    pts += [(bike.turn(*x).V, math.degrees(x[0])) for x in tg.fold][::-1]
    return MPath(np.array(pts))


def torque_limit(poly):
    tr = turn_at(bike, 4.5, 8.0)
    E0 = bike.energy(bike.turn_state(tr))
    targets = [9.0, 10.0, 11.0, 11.8]
    Ts = [level_turn(E0, x, tr.delta * x / 8).T for x in targets]
    ev = []
    for k, T in enumerate(Ts):
        ev += [(3.0 + 8.5 * k, "T", T), (3.0 + 8.5 * k + 4.5, "kick", -0.15)]
    t_ramp, rate = 3.0 + 8.5 * len(Ts), -0.004

    def ramp(t_stop):
        n = int((t_stop - t_ramp) / 0.25 + 1e-9)
        return [(t_ramp + 0.25 * j, "T", Ts[-1] + rate * 0.25 * j) for j in range(1, n + 1)]

    r = run(bike, bike.turn_state(tr), tr.T, 75.0, ev + ramp(75.0), fps=30)
    lean, V = lean_speed(bike, r)
    inside = poly.contains_points(np.c_[V, lean])
    k = np.flatnonzero((r["t"] > t_ramp) & ~inside)
    k_out = next(i for i in k if not inside[i:].any() or (~inside[i : i + 30]).all())
    t_cross = float(r["t"][k_out])
    r = run(bike, bike.turn_state(tr), tr.T, 75.0, ev + ramp(t_cross), fps=30)
    lean, V = lean_speed(bike, r)
    out = dict(
        start_T=tr.T,
        target_T=Ts,
        t_cross=t_cross,
        lean_cross_deg=lean[k_out],
        V_cross=V[k_out],
        T_hold=r["T"][-1],
        fell=r["fell"],
        V_at_12deg=V[np.argmax(lean >= 12)],
        max_H_drift=max(s["H_drift"] for s in r["segs"]),
    )
    for x in (8.0, 11.0, 12.0):
        t = level_turn(E0, x, tr.delta * x / 8)
        out[f"slowest eigenvalue at {x:g} deg on the level"] = max(
            st.eigenvalues(bike, t.phi, t.delta).real
        )
    print("torque limit", out, flush=True)
    return out


def weave_steps():
    tr = turn_at(bike, 3.6, 10.5)
    E0 = bike.energy(bike.turn_state(tr))
    targets = [10.0, 9.5, 9.0, 8.0]
    turns = [level_turn(E0, x, tr.delta * x / 10.5) for x in targets]
    ev = [(3.0 + 9.0 * k, "T", t.T) for k, t in enumerate(turns)] + [(7.5, "kick", -0.1)]
    r = run(bike, bike.turn_state(tr), tr.T, 48.0, ev, fps=30)
    lean, V = lean_speed(bike, r)
    m = r["t"] > r["t"][-1] - 10
    out = dict(
        start_T=tr.T,
        target_T=[t.T for t in turns],
        fell=r["fell"],
        weave_real_part=[max(st.eigenvalues(bike, t.phi, t.delta).real) for t in turns],
        last10s_lean_deg=[lean[m].min(), lean[m].max()],
        last10s_V=[V[m].min(), V[m].max()],
        max_H_drift=max(s["H_drift"] for s in r["segs"]),
    )
    print("weave steps", out, flush=True)
    return out


def period_two(quick):
    T = -0.4179855714497167
    fam = TorqueFamily(bike, T, 0.159, 0.119)
    phi_h = fam.hopf(0.139, 0.179)
    phi_f, _ = fam.fold(phi_h, phi_h + 0.1)
    y0, _ = weave_start(fam, 610.75 + 794.1195, phi_h, phi_f)
    tr = fam.turn(y0[0])
    sol = solve_ivp(
        lambda t, z: bike.field(z, T),
        (0, 30.0 if quick else 100.0),
        np.r_[y0, tr.s],
        method="DOP853",
        rtol=1e-10,
        atol=1e-12,
    )
    r = run(bike, sol.y[:, -1], T, 18.0)
    lean, V = lean_speed(bike, r)
    out = dict(
        turn=dict(V=tr.V, lean_deg=math.degrees(tr.phi)),
        lean_deg=[lean.min(), lean.max()],
        V=[V.min(), V.max()],
        fell=r["fell"],
    )
    print("period two", out, flush=True)
    return out


FLEET = [
    (4.5, 10.0),
    (4.0, 10.0),
    (3.6, 10.6),
    (4.3, 11.8),
    (4.8, 9.0),
    (4.5, 8.0),
    (4.2, 9.0),
    (5.0, 9.5),
    (3.4, 8.0),
    (3.0, 7.0),
    (3.8, 7.5),
    (3.2, 9.5),
    (5.0, 15.5),
    (4.5, 14.5),
    (5.5, 15.0),
    (4.0, 14.0),
]


def fleet(quick):
    out = []
    for V0, lean0 in FLEET[:: 4 if quick else 1]:
        tr = turn_at(bike, V0, lean0)
        kind, leading, _ = kind_of(bike, tr)
        r = run(bike, bike.turn_state(tr), tr.T, 25.0, [(0.5, "kick", 0.05)], fps=10)
        lean, _ = lean_speed(bike, r)
        out.append(
            dict(
                V=V0,
                lean_deg=lean0,
                R=tr.R,
                kind=kind,
                leading=leading,
                fell=r["fell"],
                lean_range_deg=[lean.min(), lean.max()],
                lean_range_last10s_deg=[lean[r["t"] > 15].min(), lean[r["t"] > 15].max()]
                if r["fell"] is None
                else None,
                lean_end_deg=lean[-1],
            )
        )
        print("fleet", out[-1], flush=True)
    return out


def trails(quick):
    out = {}
    for c in (0.08, 0.11):
        b = Whipple(dict(BENCHMARK, c=c))
        tr = turn_at(b, 4.5, 14.0)
        _, leading, _ = kind_of(b, tr)
        r = run(b, b.turn_state(tr), tr.T, 20.0, [(2.0, "kick", 0.1)], fps=10)
        d = dict(T=tr.T, leading=leading, fell=r["fell"])
        if not quick:
            tg = st.tongue(b, "free")
            big = st.largest_stable(b, tg, lambda x: x.phi)
            d.update(largest_stable_lean_deg=math.degrees(big.phi), at_V=big.V)
        out[f"trail {c * 1000:.0f} mm"] = d
        print("trail", c, d, flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="shorter runs, four of the sixteen turns")
    quick = ap.parse_args().quick
    t0 = time.time()
    out = dict(near_boundaries=near_boundaries(quick))
    if not quick:
        out["torque_limit"] = torque_limit(stable_polygon())
    out["weave_steps"] = weave_steps()
    out["period_two"] = period_two(quick)
    out["fleet"] = fleet(quick)
    out["trails"] = trails(quick)
    print(f"{time.time() - t0:.0f} s")
    save("boundary_runs" + ("_quick" if quick else ""), out)


if __name__ == "__main__":
    main()
