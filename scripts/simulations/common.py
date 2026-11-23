"""Time simulations with steer-torque steps and kicks (supplementary "Method and checks")."""

import json
import math
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
from scipy.integrate import solve_ivp  # noqa: E402
from scipy.optimize import brentq  # noqa: E402

from whipple.output import _plain  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "simulations"
KW = dict(method="DOP853", rtol=1e-10, atol=1e-12)
FPS = 30
LEAN_STOP = math.radians(45)


def save(name, data):
    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / f"{name}.json"
    path.write_text(json.dumps(_plain(data)) + "\n")
    print("wrote", path.relative_to(REPO))


def turn_at(bike, V, lean_deg):
    """Steady turn with rear-contact speed V and lean lean_deg (bracketed root in the steer)."""
    phi = math.radians(lean_deg)
    d0 = bike.p["w"] * 9.81 * math.tan(phi) / (V * V * math.cos(bike.p["lam"]))

    def res(d):
        s2 = bike.turn_spin_torque(phi, d)[0]
        return bike.p["rR"] * math.sqrt(s2) - V if s2 > 0 else 50.0

    return bike.turn(phi, brentq(res, d0 * 0.5, d0 * 2.0, xtol=1e-15, rtol=1e-15))


def rhs(bike, T):
    """Five model states, then x, y of the rear-wheel centre, yaw and the two wheel angles."""

    def f(t, y):
        z = y[:5]
        try:
            qd, ud = bike.accelerations(z[0], z[1], z[2:5], T)
        except (np.linalg.LinAlgError, ValueError, RuntimeError):
            return np.full(10, np.nan)  # trial step far beyond the fall: the solver shrinks it
        c, s = math.cos(y[7]), math.sin(y[7])
        return np.r_[z[2:4], ud, c * qd[0] - s * qd[1], s * qd[0] + c * qd[1], qd[3], qd[7], qd[8]]

    return f


def run(bike, z0, T0, t_end, events=(), lean_stop=LEAN_STOP, y0_extra=None, fps=FPS):
    """events: (time, 'T', value) or (time, 'kick', lean-rate step); output at fps."""
    y = np.r_[z0, np.zeros(5) if y0_extra is None else y0_extra]
    frames = np.arange(0, t_end + 1e-9, 1 / fps)
    events = sorted(events, key=lambda e: e[0])
    bounds = [0.0] + [e[0] for e in events] + [t_end]
    T = T0
    ts, Ys, Ts, segs = [], [], [], []

    def fall(t, y):
        return min(lean_stop - abs(y[0]), math.radians(60) - abs(y[1]))

    fall.terminal = True
    fell = None
    for i in range(len(bounds) - 1):
        a, b = bounds[i], bounds[i + 1]
        last = i == len(bounds) - 2
        sel = frames[(frames >= a - 1e-9) & ((frames < b - 1e-9) | last)]
        if b > a:
            sol = solve_ivp(rhs(bike, T), (a, b), y, t_eval=sel, events=[fall], **KW)
            if len(sol.t):
                H = [bike.energy(r[:5]) - T * r[1] for r in sol.y.T]
                segs.append(dict(t0=a, t1=sol.t[-1], T=T, H_drift=np.ptp(H)))
            ts.append(sol.t)
            Ys.append(sol.y.T)
            Ts.append(np.full(len(sol.t), T))
            if sol.status == -1:
                raise RuntimeError(f"integration failed at t = {sol.t[-1]}: {sol.message}")
            if sol.status == 1:
                ts.append(sol.t_events[0])
                Ys.append(sol.y_events[0])
                Ts.append([T])
                fell = float(sol.t_events[0][0])
                break
            y = sol.y[:, -1].copy()
        if i < len(events):
            _, kind, val = events[i]
            if kind == "T":
                T = val
            else:
                y[2] += val
    return dict(t=np.concatenate(ts), Y=np.vstack(Ys), T=np.concatenate(Ts), segs=segs, fell=fell)


def contacts(bike, y):
    """Rear and front contact points (model axes: x forward, y right, z down)."""
    P = bike.p
    phi, delta, psi = y[0], y[1], y[7]
    theta = bike.pitch(phi, delta)

    def rot(axis, a):
        c, s = math.cos(a), math.sin(a)
        return {
            0: np.array([[1, 0, 0], [0, c, -s], [0, s, c]]),
            1: np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
            2: np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]),
        }[axis]

    R = rot(2, psi) @ rot(0, phi) @ rot(1, theta)
    S = rot(1, P["lam"]) @ rot(2, delta) @ rot(1, -P["lam"])
    rear = np.array([y[5], y[6], -P["rR"] * math.cos(phi)])
    pivot = np.array([P["w"] + P["c"], 0.0, P["rR"]])
    front = rear + R @ (pivot + S @ (np.array([P["w"], 0, P["rR"] - P["rF"]]) - pivot))
    ez = np.array([0.0, 0.0, 1.0])

    def to_ground(axle, r):
        return r * (ez - axle * axle[2]) / math.sqrt(1 - axle[2] ** 2)

    return rear + to_ground(R[:, 1], P["rR"]), front + to_ground((R @ S)[:, 1], P["rF"])


def fit_circle(xy):
    A = np.c_[2 * xy, np.ones(len(xy))]
    cx, cy, k = np.linalg.lstsq(A, (xy**2).sum(1), rcond=None)[0]
    r = math.sqrt(k + cx * cx + cy * cy)
    return (cx, cy), r, np.abs(np.hypot(xy[:, 0] - cx, xy[:, 1] - cy) - r).max()


def lean_speed(bike, r):
    return np.degrees(r["Y"][:, 0]), bike.p["rR"] * np.abs(r["Y"][:, 4])
