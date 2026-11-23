"""Helpers of the countersteer scripts: parameter sets, trims and turns at a speed."""

import gzip
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

from whipple import stability as st
from whipple.bicycles import BICYCLES, measured_sets
from whipple.linear import Linear
from whipple.model import PARAMETERS, Whipple

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "countersteer"
RAW = REPO / "results" / "countersteer"
H_STEP = 1e-30
WIDE = ("Silver", "benchmark_c150", "benchmark_c200")


def clean(p):
    return {k: float(p[k]) for k in PARAMETERS if k in p}


def all_sets():
    """The 42 parameter sets of the stability grids, by name."""
    out = {k: clean(BICYCLES[k]) for k in ("benchmark", "A", "B")}
    with gzip.open(REPO / "data/measured/results/A_rider(invented).json.gz", "rt") as fh:
        out["A_rider(invented)"] = clean(json.load(fh)["params"])
    out.update({k: clean(v) for k, v in measured_sets("rider").items()})
    out.update({k: clean(v) for k, v in measured_sets("riderless").items()})
    out["benchmark_c150"] = dict(out["benchmark"], c=0.15)
    out["benchmark_c200"] = dict(out["benchmark"], c=0.20)
    return out


def _plain(x):
    return x.tolist() if hasattr(x, "tolist") else float(x)


def write_json(name, data, folder=DATA):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_text(json.dumps(data, indent=1, default=_plain) + "\n")
    print("wrote", (folder / name).relative_to(REPO))


def statics(bike, phi, delta):
    """U, M (3x3), F_g, F_c at (phi, delta); complex arguments allowed."""
    z = np.zeros(3, complex if np.iscomplexobj(np.array([phi, delta])) else float)
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
    v = trim(bike, phi, delta)
    gp = trim(bike, phi + 1j * H_STEP, delta).imag / H_STEP
    gd = trim(bike, phi, delta + 1j * H_STEP).imag / H_STEP
    return v.real, np.column_stack([gp, gd])


def R_matrix(bike, phi, delta, S):
    """R = d(-U_q + S f_c)/dq at fixed S."""

    def r(p, d):
        _, _, Fg, Fc = statics(bike, p, d)
        return Fg[:2] + S * Fc[:2]

    return np.column_stack(
        [r(phi + 1j * H_STEP, delta).imag / H_STEP, r(phi, delta + 1j * H_STEP).imag / H_STEP]
    )


def a4_both(bike, phi, delta):
    """a4 of both plants from the static formulas, and the signs that the theorems use."""
    U, M, Fg, Fc = statics(bike, phi, delta)
    v, G = trim_grad(bike, phi, delta)
    S, T, H, E = v
    R = R_matrix(bike, phi, delta, S)
    detMqq, detM = np.linalg.det(M[:2, :2]), np.linalg.det(M)
    dTS = G[1, 0] * G[0, 1] - G[1, 1] * G[0, 0]
    dTH = G[1, 0] * G[2, 1] - G[1, 1] * G[2, 0]
    Sd, Sp = G[0, 1], G[0, 0]
    return dict(
        S=S,
        V=bike.p["rR"] * math.sqrt(S) if S > 0 else math.nan,
        T=T,
        H=H,
        E=E,
        R=R,
        Fcphi=Fc[0],
        Uphi=-Fg[0],
        a4_constant=-Fc[0] / detMqq * dTS,
        a4_free=-2 * Fc[0] / detM * dTH,
        T_phi_S=G[1, 0] - G[1, 1] * Sp / Sd,
        T_S=G[1, 1] / Sd,
        H_phi_S=G[2, 0] - G[2, 1] * Sp / Sd,
        H_S=G[2, 1] / Sd,
    )


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
    return solve_decreasing(lambda d: _speed(bike, phi, d) - V, guess, 0.0, 1.5, 1.25, 1e-14)


def solve_decreasing(fun, guess, lo_min=1e-7, hi_max=1.5, factor=1.2, rtol=1e-13):
    """Root of a function that decreases with its argument; bracket from the guess."""
    lo = hi = guess
    flo = fhi = fun(guess)
    if not np.isfinite(flo):
        return math.nan
    n = 0
    while flo < 0 and n < 80:
        hi, fhi = lo, flo
        lo /= factor
        if lo < lo_min:
            return math.nan
        flo = fun(lo)
        n += 1
        if not np.isfinite(flo):
            return math.nan
    while fhi > 0 and n < 160:
        lo, flo = hi, fhi
        hi *= factor
        if hi > hi_max:
            return math.nan
        fhi = fun(hi)
        n += 1
        if not np.isfinite(fhi):
            return math.nan
    if flo < 0 or fhi > 0:
        return math.nan
    if flo == 0 or fhi == 0:
        return lo if flo == 0 else hi
    return float(brentq(fun, lo, hi, xtol=1e-15, rtol=rtol))


def line_scan(bike, lin, V, phis):
    """Steer of the turns at fixed V over increasing leans, with continuation."""
    out, guess = [], None
    for k, phi in enumerate(phis):
        if guess is None or not np.isfinite(guess):
            guess = phi * lin.steer_ratio(V)
        d = delta_at(bike, lin, phi, V, guess)
        out.append(d)
        if np.isfinite(d):
            if k >= 1 and np.isfinite(out[k - 1]) and k + 1 < len(phis):
                guess = d + (d - out[k - 1]) * (phis[k + 1] - phi) / (phi - phis[k - 1])
            elif k + 1 < len(phis):
                guess = d * phis[k + 1] / phi
        else:
            guess = None
    return np.array(out)


def stable(bike, phi, delta, plant):
    a = st.characteristic(bike, phi, delta, plant)
    return bool(np.all(np.roots(np.r_[1.0, a]).real < 0))


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


__all__ = ["Whipple"]
