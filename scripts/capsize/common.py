"""Helpers of the capsize scripts: parameter sets, trims of a turn, folds and paths."""

import gzip
import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from whipple import stability as st
from whipple.bicycles import BICYCLES, measured_sets
from whipple.linear import Linear
from whipple.model import PARAMETERS, Whipple

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "capsize"
H_STEP = 1e-30


def clean(p):
    return {k: float(p[k]) for k in PARAMETERS if k in p}


def invented_rider():
    """Bicycle A with an invented rider, as stored with the measured results."""
    f = REPO / "data" / "measured" / "results" / "A_rider(invented).json.gz"
    with gzip.open(f, "rt") as fh:
        return json.load(fh)["params"]


def all_sets(extra=True):
    """The 42 parameter sets of the stability grids, by name."""
    out = {k: clean(BICYCLES[k]) for k in ("benchmark", "A", "B")}
    out["A_rider(invented)"] = clean(invented_rider())
    out.update({k: clean(v) for k, v in measured_sets("rider").items()})
    out.update({k: clean(v) for k, v in measured_sets("riderless").items()})
    if extra:
        out["benchmark_c150"] = dict(out["benchmark"], c=0.15)
        out["benchmark_c200"] = dict(out["benchmark"], c=0.20)
    return out


def _plain(x):
    return x.tolist() if hasattr(x, "tolist") else float(x)


def write_json(name, data, quick=False):
    """Write data/capsize/<name>, or results/capsize/<name> for a --quick run."""
    folder = REPO / "results" / "capsize" if quick else DATA
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_text(json.dumps(data, indent=1, default=_plain) + "\n")
    print("wrote", (folder / name).relative_to(REPO))


# ---- trims of the turn through q = (lean, steer) -----------------------------------------------


def statics(bike, phi, delta):
    """U, M (3x3), F_g, F_c at (phi, delta); complex arguments allowed."""
    cplx = np.iscomplexobj(np.array([phi, delta]))
    z = np.zeros(3, complex if cplx else float)
    _, M, Fg, _ = bike.parts(phi, delta, z)
    one = z.copy()
    one[2] = 1.0
    Fc = bike.parts(phi, delta, one)[2] - Fg
    return bike.potential(phi, delta), M, Fg, Fc


def trim(bike, phi, delta):
    """S = s^2, T, H = m_s S / 2 + U - T delta and E = H + T delta."""
    U, M, Fg, Fc = statics(bike, phi, delta)
    S = -Fg[0] / Fc[0]
    T = -Fg[1] - S * Fc[1]
    E = 0.5 * M[2, 2] * S + U
    return np.array([S, T, E - T * delta, E])


def trim_grad(bike, phi, delta):
    """Values (S, T, H, E) and their gradients in (phi, delta), shapes 4 and 4x2."""
    v = trim(bike, phi, delta)
    gp = trim(bike, phi + 1j * H_STEP, delta).imag / H_STEP
    gd = trim(bike, phi, delta + 1j * H_STEP).imag / H_STEP
    return v.real, np.column_stack([gp, gd])


def a4_both(bike, phi, delta):
    """a4 of both plants from Theorem 1 and the constant-speed condition, with the prefactors."""
    U, M, Fg, Fc = statics(bike, phi, delta)
    v, G = trim_grad(bike, phi, delta)
    S, T, H, E = v
    detMqq, detM = np.linalg.det(M[:2, :2]), np.linalg.det(M)
    mu_c, mu_f = -Fc[0] / detMqq, -2 * Fc[0] / detM
    dTS = G[1, 0] * G[0, 1] - G[1, 1] * G[0, 0]
    dTH = G[1, 0] * G[2, 1] - G[1, 1] * G[2, 0]
    V = bike.p["rR"] * math.sqrt(S) if S > 0 else math.nan
    return dict(
        S=S,
        V=V,
        T=T,
        H=H,
        E=E,
        mu_constant=mu_c,
        mu_free=mu_f,
        a4_constant=mu_c * dTS,
        a4_free=mu_f * dTH,
        dTS=dTS,
        dTH=dTH,
    )


def a4_constant(bike, phi, delta):
    return a4_both(bike, phi, delta)["a4_constant"]


def _speed(bike, phi, d):
    try:
        S = bike.turn_spin_torque(phi, d)[0]
    except (RuntimeError, np.linalg.LinAlgError):
        return math.nan
    return bike.p["rR"] * math.sqrt(S) if S > 0 else math.nan


def delta_at(bike, lin, phi, V, guess=None):
    """Steer of the turn with lean phi and contact speed V on the normal branch, or nan."""
    if guess is None or not np.isfinite(guess) or guess <= 0:
        guess = phi * lin.steer_ratio(V)
    if not guess > 0:
        return math.nan
    lo = hi = guess
    flo = fhi = _speed(bike, phi, guess) - V
    if not np.isfinite(flo):
        return math.nan
    n = 0
    while flo < 0 and n < 60:
        hi, fhi = lo, flo
        lo /= 1.25
        flo = _speed(bike, phi, lo) - V
        n += 1
        if not np.isfinite(flo):
            return math.nan
    while fhi > 0 and n < 120:
        lo, flo = hi, fhi
        hi *= 1.25
        if hi > 1.5:
            return math.nan
        fhi = _speed(bike, phi, hi) - V
        n += 1
        if not np.isfinite(fhi):
            return math.nan
    if flo < 0 or fhi > 0:
        return math.nan
    if flo == 0 or fhi == 0:
        return lo if flo == 0 else hi
    return float(brentq(lambda d: _speed(bike, phi, d) - V, lo, hi, xtol=1e-15, rtol=1e-14))


def upright(p):
    """Linear model, capsize speed v_c and weave speed v_w."""
    lin = Linear(p)
    K0, K2 = lin.K0, lin.K2
    x = -lin.g * np.linalg.det(K0) / (K0[0, 0] * K2[1, 1] - K0[1, 0] * K2[0, 1])
    vc = math.sqrt(x) if x > 0 else math.nan

    def growth(v):
        lam = lin.eigenvalues(v)
        osc = lam[np.abs(lam.imag) > 1e-9]
        return osc.real.max() if len(osc) else math.nan

    vs = np.linspace(0.5, 1.5 * vc if vc == vc else 15.0, 400)
    g = np.array([growth(v) for v in vs])
    k = np.flatnonzero((g[:-1] > 0) & (g[1:] <= 0))
    vw = brentq(growth, vs[k[0]], vs[k[0] + 1], xtol=1e-12) if len(k) else math.nan
    return dict(lin=lin, v_c=vc, v_w=vw)


def constant_speed_fold(bike, lin, V, phi_max=0.6):
    """First fold at constant contact speed V: (lean, T), or None."""
    phis = np.linspace(0.01, phi_max, 120)
    vals = []
    for x in phis:
        d = delta_at(bike, lin, x, V)
        vals.append(a4_constant(bike, x, d) if np.isfinite(d) else np.nan)
    k = np.flatnonzero(np.array(vals) < 0)
    if not len(k):
        return None
    fc = brentq(
        lambda x: a4_constant(bike, x, delta_at(bike, lin, x, V)),
        phis[k[0] - 1],
        phis[k[0]],
        xtol=1e-12,
    )
    return fc, a4_both(bike, fc, delta_at(bike, lin, fc, V))["T"]


# ---- paths from straight running ----------------------------------------------------------------


def path(bike, lin, v0, kind, phi_end=0.6):
    """Curve of constant H (kind "H") or loading path dE = T d(delta) (kind "load") from v0."""
    phi0 = 1e-3
    d0 = delta_at(bike, lin, phi0, v0)

    def rhs(phi, y):
        v, G = trim_grad(bike, phi, y[0])
        if kind == "H":
            return [-G[2, 0] / G[2, 1]]
        return [-G[3, 0] / (G[3, 1] - v[1])]

    def stop(phi, y):
        return trim_grad(bike, phi, y[0])[0][0] - 1e-6

    stop.terminal = True
    return solve_ivp(
        rhs,
        (phi0, phi_end),
        [d0],
        rtol=1e-11,
        atol=1e-13,
        dense_output=True,
        events=stop,
        max_step=0.01,
    )


def first_fold_on_path(bike, sol):
    """First zero of a4 (free rolling) and, found apart, the first extremum of T on the path."""

    def a4(phi):
        return a4_both(bike, phi, float(sol.sol(phi)[0]))["a4_free"]

    def torque(x):
        return trim_grad(bike, x, float(sol.sol(x)[0]))[0][1]

    def dT(phi, h=1e-5):
        return (torque(phi + h) - torque(phi - h)) / (2 * h)

    xs = np.linspace(sol.t[0] + 2e-5, sol.t[-1] - 2e-5, 240)
    out = {}
    for key, f in (("a4", a4), ("dT", dT)):
        vals = np.array([f(x) for x in xs])
        k = np.flatnonzero(np.sign(vals[1:]) != np.sign(vals[0]))
        out[key] = brentq(f, xs[k[0]], xs[k[0] + 1], xtol=1e-12) if len(k) else None
    if out["a4"] is not None:
        d = float(sol.sol(out["a4"])[0])
        a = a4_both(bike, out["a4"], d)
        out.update(T=a["T"], V=a["V"], delta=d)
    return out


def simulate_ramp(bike, v0, rate, plant, phi_stop):
    """Full equations from straight running at v0 with the torque T = -rate t."""
    s0 = -v0 / bike.p["rR"]

    def free(t, z):
        return bike.field(z, -rate * t)

    def constant(t, x):
        _, Mr, f, _ = bike.parts(x[0], x[1], np.r_[x[2], x[3], s0])
        return np.r_[x[2:4], np.linalg.solve(Mr[:2, :2], f[:2] + np.array([0.0, -rate * t]))]

    def stop(t, z):
        return z[0] - phi_stop

    stop.terminal = True
    z0 = [0.0] * (5 if plant == "free" else 4)
    if plant == "free":
        z0[4] = s0
    return solve_ivp(
        free if plant == "free" else constant,
        (0.0, 5.0 / rate),
        z0,
        rtol=1e-8,
        atol=1e-10,
        events=stop,
        dense_output=True,
        max_step=0.25,
    )


# ---- exact fold curve in the chart (lean, steer) ------------------------------------------------


def det_fold(bike, x, plant):
    """det d(T, S)/d(phi, delta) (constant speed) or det d(T, H)/d(phi, delta) (free rolling)."""
    if plant == "constant":
        a = np.array(bike.turn_spin_torque(x[0] + 1j * H_STEP, x[1])).imag / H_STEP
        b = np.array(bike.turn_spin_torque(x[0], x[1] + 1j * H_STEP)).imag / H_STEP
        return a[1] * b[0] - b[1] * a[0]
    a = st.torque_and_first_integral(bike, x[0] + 1j * H_STEP, x[1]).imag / H_STEP
    b = st.torque_and_first_integral(bike, x[0], x[1] + 1j * H_STEP).imag / H_STEP
    return a[0] * b[1] - b[0] * a[1]


def fold_top(bike, plant, x0, h=1e-4):
    """Top of the exact fold curve near x0 = (phi, delta): the Turn there."""
    x0 = np.asarray(x0, float)
    e = np.array([0.0, h])
    scale = abs(det_fold(bike, x0 + [0.02, 0.0], plant) - det_fold(bike, x0 - [0.02, 0.0], plant))
    scale /= 0.04

    def F(x):
        f0 = det_fold(bike, x, plant)
        fd = (det_fold(bike, x + e, plant) - det_fold(bike, x - e, plant)) / (2 * h)
        return np.array([f0, fd]) / scale

    x = st.newton_system(F, x0, h=1e-5, tol=1e-9, maxiter=30)
    return bike.turn(*x)


def rule_fold_top(p, plant):
    """Exact fold top, started from the second-order rule: Turn, or None."""
    from whipple import capsize_rule as cr

    r = cr.largest_lean_rule(p, plant, 2)
    if not r["phi"] == r["phi"]:
        return None
    c = cr.series(p).coefficients(r["V"])
    phi = r["phi"]
    t = fold_top(Whipple(p), plant, (phi, c["r1"] * phi + c["r3"] * phi**3))
    return t if abs(t.phi - phi) < math.radians(3.0) else None
