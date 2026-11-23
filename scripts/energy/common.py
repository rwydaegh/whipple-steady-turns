"""Shared tools of the energy scripts (Sect. 3, "Energy"; energy split of Eq. Ksplit)."""

import json
import math
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
from scipy.integrate import solve_ivp  # noqa: E402
from scipy.optimize import brentq, fsolve  # noqa: E402

from whipple import stability as st  # noqa: E402
from whipple.bicycles import BENCHMARK  # noqa: E402
from whipple.linear import Linear  # noqa: E402
from whipple.model import Whipple, complex_step  # noqa: E402
from whipple.output import _plain  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "energy"
RESULTS = REPO / "results" / "energy"
KW = dict(method="DOP853", rtol=1e-11, atol=1e-12)


def bench():
    bike = Whipple(BENCHMARK)
    return bike, Linear(bike.p)


def save(name, data, gz=False):
    """Write data/energy/<name>.json (or .json.gz) and return the path."""
    import gzip

    DATA.mkdir(parents=True, exist_ok=True)
    text = json.dumps(_plain(data), indent=None if gz else 1)
    if gz:
        path = DATA / f"{name}.json.gz"
        with gzip.open(path, "wt", compresslevel=9) as f:
            f.write(text)
    else:
        path = DATA / f"{name}.json"
        path.write_text(text + "\n")
    print("wrote", path.relative_to(REPO))
    return path


ENERGIES = ("U", "Kf", "Kl", "E", "WT", "Wd", "H")


def rounded(series, digits=7):
    """Round to significant digits; energies are stored as changes from the start."""
    out = {}
    for k, v in series.items():
        v = np.asarray(v, float)
        if k in ENERGIES:
            v = v - v[0]
        out[k] = [float(f"{x:.{digits}g}") for x in v]
    return out


# steady turns


def _speed(bike, phi, d):
    S = bike.turn_spin_torque(phi, d)[0]
    return bike.p["rR"] * math.sqrt(S) if S > 0 else -1.0


def turn_at(bike, lin, phi, V):
    """Steady turn with lean phi (rad) and rear-contact speed V (m/s) on the small-steer branch.

    On this branch the speed falls when the steer grows at fixed lean, so the root is bracketed
    from the linear guess and found by Brent's method.
    """
    lo = hi = phi * lin.steer_ratio(V)
    while _speed(bike, phi, lo) < V:
        lo /= 1.25
    while _speed(bike, phi, hi) > V:
        hi *= 1.25
    d = brentq(lambda d: _speed(bike, phi, d) - V, lo, hi, xtol=1e-15, rtol=1e-14)
    return bike.turn(phi, d)


def turn_state(t):
    return np.array([t.phi, t.delta, 0.0, 0.0, t.s])


def trim(bike, phi, delta):
    """S = s^2, T, H = m_s S / 2 + U - T delta and E = H + T delta of the turn."""
    S = bike.turn_spin_torque(phi, delta)[0]
    T, H = st.torque_and_first_integral(bike, phi, delta)
    return S, T, H, H + T * delta


def turn_with(bike, T0, H0, guess):
    """Steady turn with steer torque T0 and first integral H0, the end state of free rolling."""

    def residual(x):
        _, T, H, _ = trim(bike, *x)
        return [T - T0, H - H0]

    x = fsolve(residual, guess, xtol=1e-13)
    return bike.turn(*x)


# dynamics


def drive_constant(bike, z, T):
    """Drive force Q that keeps s constant (the exact constraint force), and the lean and steer
    accelerations."""
    _, Mr, f, _ = bike.parts(z[0], z[1], z[2:5])
    qdd = np.linalg.solve(Mr[:2, :2], f[:2] + np.array([0.0, T]))
    return Mr[2, :2] @ qdd - f[2], qdd


def field_free(bike, T):
    return lambda t, z: bike.field(z, T)


def field_constant(bike, T):
    """Constant speed. The state has a sixth entry: the drive work, the integral of Q s."""

    def f(t, z):
        Q, qdd = drive_constant(bike, z, T)
        return np.r_[z[2:4], qdd, 0.0, Q * z[4]]

    return f


def field_feedback(bike, T, G, s0):
    """Speed feedback Q = -G (s - s0)."""

    def f(t, z):
        _, ud = bike.accelerations(z[0], z[1], z[2:5], T, G * (z[4] - s0))
        return np.r_[z[2:4], ud]

    return f


def simulate(f, z0, t_end, n, events=None, max_step=0.02):
    ts = np.linspace(0.0, t_end, n)
    return solve_ivp(f, (0.0, t_end), z0, t_eval=ts, events=events, max_step=max_step, **KW)


def jacobian_free(bike, z, T):
    return complex_step(lambda x: bike.field(x, T), z)


def jacobian_constant(bike, z, T):
    def lateral(x):
        _, Mr, f, _ = bike.parts(x[0], x[1], np.r_[x[2], x[3], z[4]])
        return np.r_[x[2:4], np.linalg.solve(Mr[:2, :2], f[:2] + np.array([0.0, T]))]

    return complex_step(lateral, z[:4])


def weave_mode(J):
    """Eigenvalue and eigenvector of the weave (the oscillatory pair with the largest real part)."""
    lam, vec = np.linalg.eig(J)
    k = int(np.argmax(np.where(np.abs(lam.imag) > 0.5, lam.real, -np.inf)))
    return lam[k], vec[:, k]


def quartics(bike, t):
    """Monic characteristic quartics of free rolling and constant speed at the turn t, and
    det M_qq / det M (so that gamma = G det M_qq / det M)."""
    z = turn_state(t)
    lam = np.linalg.eigvals(jacobian_free(bike, z, t.T))
    lam = lam[np.argsort(np.abs(lam))][1:]  # drop the zero eigenvalue of the first integral
    a_free = np.real(np.poly(lam))
    a_const = np.real(np.poly(np.linalg.eigvals(jacobian_constant(bike, z, t.T))))
    M = bike.mass_matrix(t.phi, t.delta)
    return a_free, a_const, np.linalg.det(M[:2, :2]) / np.linalg.det(M)


def quintic(a_free, a_const, gamma):
    """Monic quintic of the bicycle with speed feedback: lambda chi_H + gamma chi_S (Eq. family)."""
    return np.r_[a_free, 0.0] + gamma * np.r_[0.0, a_const]


# energy and speeds


def energy_parts(bike, z):
    """U, K_f, K_l, E and m_s of the state z."""
    M = bike.mass_matrix(z[0], z[1])
    u = np.asarray(z[2:5], float)
    U = bike.potential(z[0], z[1])
    ms = M[2, 2]
    Kf = 0.5 * (M[2] @ u) ** 2 / ms
    M_eff = M[:2, :2] - np.outer(M[:2, 2], M[2, :2]) / ms
    Kl = 0.5 * u[:2] @ M_eff @ u[:2]
    return dict(U=U, Kf=Kf, Kl=Kl, E=U + Kf + Kl, ms=ms)


def speeds(bike, z):
    """Speeds of the state z (m/s) and the height of the total mass centre (m):
    Vs = r_R |s| (s relative to the rear frame), Vc the rear-contact speed, Vg the speed of the
    total mass centre, Va = r_R |s + pitch rate + yaw rate sin(lean)| (absolute spin of the rear
    wheel about its axle), hg and the yaw rate."""
    rR = bike.p["rR"]
    _, _, jv, _, _, heights, _, B = bike._geometry(z[0], z[1])
    qd = B @ np.asarray(z[2:5], float)  # rates of (x, y, z, yaw, lean, pitch, steer, wheels)
    phi = z[0]
    vcx = qd[0] + qd[3] * rR * math.sin(phi)
    vcy = qd[1] - rR * math.cos(phi) * qd[4]
    m = bike.mass
    vg = np.einsum("b,bki,i->k", m, jv, qd) / m.sum()
    return dict(
        Vs=rR * abs(z[4]),
        Vc=math.hypot(vcx, vcy),
        Vg=float(np.linalg.norm(vg)),
        Va=rR * abs(z[4] + qd[5] + qd[3] * math.sin(phi)),
        hg=float(-heights @ m / m.sum()),
        yaw_rate=qd[3],
    )


def moving_mean(x, n):
    """Mean over n consecutive samples (one weave period), 'valid' part only."""
    return np.convolve(np.asarray(x, float), np.ones(n) / n, mode="valid")
