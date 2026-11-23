"""Helpers of the type scripts: upright data, P4, stability maps and fold tops."""

import gzip
import json
import math
from pathlib import Path

import numpy as np
from numpy.polynomial import Polynomial as Poly
from scipy import ndimage
from scipy.optimize import brentq

from whipple import capsize_rule as cr
from whipple import stability as st
from whipple.bicycles import BICYCLES, measured_sets
from whipple.linear import Linear
from whipple.model import PARAMETERS, Whipple

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "types"
RESULTS = REPO / "data" / "measured" / "results"


def clean(p):
    return {k: float(p[k]) for k in PARAMETERS if k in p}


def stored(name):
    f = RESULTS / f"{name}.json.gz"
    if not f.exists():
        return None
    with gzip.open(f, "rt") as fh:
        return json.load(fh)


def all_sets():
    """Benchmark, A, B, A with the invented rider and the measured sets, by name."""
    out = {k: clean(BICYCLES[k]) for k in ("benchmark", "A", "B")}
    out["A_rider(invented)"] = clean(stored("A_rider(invented)")["params"])
    out.update({k: clean(v) for k, v in measured_sets("rider").items()})
    out.update({k: clean(v) for k, v in measured_sets("riderless").items()})
    return out


def _plain(x):
    return x.tolist() if hasattr(x, "tolist") else float(x)


def write_json(name, data, quick=False):
    """Write data/types/<name>, or results/types/<name> for a --quick run."""
    folder = REPO / "results" / "types" if quick else DATA
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_text(json.dumps(data, indent=1, default=_plain) + "\n")
    print("wrote", (folder / name).relative_to(REPO))


def poly_data(p):
    """Constants of tau1 = tau1_inf (1 - r/r_c) and tau3 = P4(r), with r3, V3 and checks."""
    s = cr.series(p)
    lin, g = s.lin, s.lin.g
    K0, K2 = lin.K0, lin.K2
    Z, SA, ka, kb = -K0[0, 0], -K0[0, 1], K2[0, 1], K2[1, 1]
    a = kb / ka
    b = SA * (a - math.sin(p["lam"])) / Z
    u = {(i, 4 - i): s.U[i, 4 - i] for i in range(5)}
    c = {(i, 3 - i): -s.Fc[i, 3 - i, :] / s.rR**2 for i in range(4)}
    r = Poly([0.0, 1.0])
    Gp = sum(i * u[i, j] * r**j for (i, j) in u if i > 0)
    Gd = sum(j * u[i, j] * r ** (j - 1) for (i, j) in u if j > 0)
    pre = (g / ka) * (Z + SA * r)
    Gp = Gp + pre * (c[2, 1][0] + c[1, 2][0] * r + c[0, 3][0] * r**2)
    Gd = Gd + pre * (c[2, 1][1] + c[1, 2][1] * r + c[0, 3][1] * r**2)
    P4 = -(a + b * r) * Gp + Gd
    coef = np.zeros(5)
    coef[: len(P4.coef)] = P4.coef
    tau1_inf = g * (Z * a - SA)
    rc = (SA / Z - a) / b
    err3 = 0.0
    for V in np.linspace(2.5, 14.0, 12):
        co = s.coefficients(V)
        err3 = max(err3, abs(P4(lin.steer_ratio(V)) - co["tau3"]) / (abs(co["tau3"]) + 1e-3))
    roots = sorted(x.real for x in P4.roots() if abs(x.imag) < 1e-10 and x.real > 0)
    out = dict(
        Z=Z,
        S_A=SA,
        k_a=ka,
        k_b=kb,
        a=a,
        b=b,
        tau1_inf=tau1_inf,
        r_c=rc,
        coef=coef,
        b0_check=-tau1_inf / 6,
        err_tau3=err3,
        c30=float(np.abs(c[3, 0]).max()),
        c21=float(np.abs(c[2, 1]).max()),
        v_c=s.v_c,
        tau3_vc=float(P4(rc)),
        V_r=math.sqrt(g * SA / ka) if SA > 0 else math.nan,
    )
    if roots:
        out["r3"] = roots[0]
        out["V3"] = math.sqrt((g / ka) * (Z / roots[0] + SA))
    return out


def upright(p):
    """Weave and capsize speeds, V3, tau3(v_c) and D_free = 3 tau3 - 2 h2 tau1' at v_c."""
    s = cr.series(p)
    vc, vw = s.v_c, s.weave_speed()
    out = dict(v_w=vw, v_c=vc)
    if vc == vc:
        c = s.with_derivatives(vc)
        out.update(
            tau3=c["tau3"],
            tau5=c["tau5"],
            h2=c["h2"],
            dtau1=c["dtau1"],
            dtau3=c["dtau3"],
            r1=c["r1"],
            r3=c["r3"],
            r5=c["r5"],
        )
        out["D_free"] = 3 * c["tau3"] - 2 * c["h2"] * c["dtau1"]
    vs = np.linspace(1.0, 25.0, 481)
    t3 = np.array([s.coefficients(v)["tau3"] for v in vs])
    k = np.flatnonzero(t3[:-1] * t3[1:] < 0)
    f = lambda v: s.coefficients(v)["tau3"]  # noqa: E731
    out["V3"] = brentq(f, vs[k[0]], vs[k[0] + 1]) if len(k) else math.nan
    return out


def inertia_admissible(p):
    """False if a body has a non-positive mass, height or principal inertia, or breaks the
    triangle inequality of the principal inertias."""
    for b in "RBHF":
        if b in "RF":
            inertia = np.diag([p[f"I{b}xx"], p[f"I{b}yy"], p[f"I{b}xx"]])
            height = p[f"r{b}"]
        else:
            inertia = np.array(
                [
                    [p[f"I{b}xx"], 0, p[f"I{b}xz"]],
                    [0, p[f"I{b}yy"], 0],
                    [p[f"I{b}xz"], 0, p[f"I{b}zz"]],
                ]
            )
            height = -p[f"z{b}"]
        ev = np.linalg.eigvalsh(inertia)
        if ev[0] <= 0 or ev[0] + ev[1] - ev[2] < -1e-12 or p[f"m{b}"] <= 0 or height <= 0:
            return False
    return True


# ---- stability maps in (speed, lean) --------------------------------------------------------


def hurwitz3(a):
    """Third Hurwitz determinant a1 a2 a3 - a3^2 - a1^2 a4: zero on the weave Hopf curve."""
    return a[..., 0] * a[..., 1] * a[..., 2] - a[..., 2] ** 2 - a[..., 0] ** 2 * a[..., 3]


def stable_component(Vs, re, v_w, v_c):
    """Stable cells joined to the stable upright speed range (4-connectivity)."""
    lab, _ = ndimage.label(re < 0)
    hi = v_c if v_c == v_c else Vs[-1]
    ids = set(lab[np.flatnonzero((Vs > v_w) & (Vs < hi)), 0]) - {0}
    return np.isin(lab, list(ids))


def solve_delta(bike, phi, V, guess, tol=1e-13, maxiter=25):
    """Steer of the turn at lean phi and speed V; Newton with steps of at most 0.05 rad."""
    s2t = (V / bike.p["rR"]) ** 2
    d = guess
    for _ in range(maxiter):
        s2 = bike.turn_spin_torque(phi, d + 1e-30j)[0]
        f, df = s2.real - s2t, s2.imag / 1e-30
        if not np.isfinite(f) or df == 0:
            return math.nan
        step = f / df
        if abs(step) > 0.05:
            step = math.copysign(0.05, step)
        d -= step
        if abs(step) < tol:
            return d
    return math.nan


def stability_map(p, Vs, phis_deg):
    """Coefficients a1..a4 and the largest real part of the eigenvalues, free rolling."""
    bike = Whipple(p)
    lin = Linear(bike.p)
    nV, nP = len(Vs), len(phis_deg)
    a_free = np.full((nV, nP, 4), np.nan)
    re_free = np.full((nV, nP), np.nan)
    for i, V in enumerate(Vs):
        prev = []
        for j, pd in enumerate(phis_deg):
            phi = math.radians(pd)
            if len(prev) >= 2:
                (p1, d1), (p2, d2) = prev[-2], prev[-1]
                guess = d2 + (d2 - d1) * (phi - p2) / (p2 - p1)
            elif prev:
                guess = prev[-1][1] * phi / prev[-1][0]
            else:
                guess = lin.steer_ratio(V) * phi
            d = solve_delta(bike, phi, V, guess)
            if not d == d or d <= 0:
                break
            turn = bike.turn(phi, d)
            if turn is None or abs(turn.V - V) > 1e-8:
                break
            prev.append((phi, d))
            a = st.characteristic(bike, phi, d, "free")
            a_free[i, j] = a
            re_free[i, j] = np.roots(np.r_[1.0, a]).real.max()
    return a_free, re_free


# ---- exact fold curve in the chart (lean, steer) ------------------------------------------------


def fold_fn(bike, phi, delta, plant):
    """det d(T, S)/d(phi, delta) (constant speed) or det d(T, H)/d(phi, delta) (free rolling)."""
    h = 1e-20

    def f(a, b):
        S, T = bike.turn_spin_torque(a, b)
        if plant == "constant":
            return np.array([T, S])
        return np.array([T, 0.5 * bike.mass_matrix(a, b)[2, 2] * S + bike.potential(a, b) - T * b])

    dp = f(phi + 1j * h, delta).imag / h
    dd = f(phi, delta + 1j * h).imag / h
    return dp[0] * dd[1] - dd[0] * dp[1]


def fold_top(bike, plant, x0, h=1e-4):
    """Top of the exact fold curve near x0 = (phi, delta): the Turn there."""
    x0 = np.asarray(x0, float)
    e = np.array([0.0, h])

    def f(x):
        return fold_fn(bike, x[0], x[1], plant)

    scale = abs(f(x0 + [0.02, 0.0]) - f(x0 - [0.02, 0.0])) / 0.04

    def F(x):
        return np.array([f(x), (f(x + e) - f(x - e)) / (2 * h)]) / scale

    return bike.turn(*st.newton_system(F, x0, h=1e-5, tol=1e-9, maxiter=30))
