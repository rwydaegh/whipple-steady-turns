"""Shared functions for the lateral rider mass shift (Sect. 13.3, "Mass shift and steer torque")."""

import math
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402
from scipy.optimize import newton  # noqa: E402

from whipple import stability as st  # noqa: E402
from whipple.bicycles import BICYCLES, measured_sets  # noqa: E402
from whipple.linear import matrices  # noqa: E402
from whipple.model import Whipple  # noqa: E402

PLANTS = ("free", "constant")


def params_of(name):
    """Parameters of the benchmark, bicycle A or B, or a measured bicycle and rider set."""
    return dict(BICYCLES[name]) if name in BICYCLES else measured_sets("rider")[name]


def tag(name):
    return name.replace("+", "_").replace("(", "").replace(")", "")


def totals(p):
    """m_T, h (height of the total mass centre) and S_A of the upright linear model."""
    mT = p["mR"] + p["mB"] + p["mH"] + p["mF"]
    zT = (-p["rR"] * p["mR"] + p["zB"] * p["mB"] + p["zH"] * p["mH"] - p["rF"] * p["mF"]) / mT
    return mT, -zT, -matrices(p)[2][0, 1]


def yB_from_Delta(p, Delta):
    """Lateral offset yB of the rear-frame mass centre for the lean offset Delta (rad)."""
    mT, h, _ = totals(p)
    return mT * h * math.tan(Delta) / p["mB"]


def Delta_from_yB(p, yB):
    mT, h, _ = totals(p)
    return math.atan(p["mB"] * yB / (mT * h))


def hurwitz(a):
    a1, a2, a3, a4 = a
    return a1 * a2 * a3 - a3**2 - a1**2 * a4


def quartic(bike, phi, delta, plant):
    """(a1..a4) of the turn through (phi, delta) of this bicycle at its own steer torque."""
    return st.characteristic(bike, phi, delta, plant)


def delta_for_speed(bike, phi, V, guess):
    """Steer of the turn with frame lean phi and rear-contact speed V (secant)."""

    def res(d):
        s2 = bike.turn_spin_torque(phi, d)[0]
        return bike.p["rR"] * math.sqrt(s2) - V if s2 > 0 else -V

    d = newton(res, guess, x1=guess * (1 + 1e-4) + 1e-7, tol=1e-13, maxiter=100)
    if abs(res(d)) > 1e-9:
        raise RuntimeError("delta_for_speed failed")
    return d


def trim_yB(bike, phi, delta, T0=0.0, guess=0.0):
    """yB that holds the turn through (phi, delta) at the steer torque T0 (secant)."""
    return newton(
        lambda W: bike.with_params(yB=W).turn_spin_torque(phi, delta)[1] - T0,
        guess,
        x1=guess + 1e-3,
        tol=1e-14,
        maxiter=60,
    )


# ---- rear frame with products of inertia IBxy, IByz (rigid sideways translation of a rider) ----
def _rot(axis, a):
    c, s = np.cos(a), np.sin(a)
    if axis == 0:
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


class WhippleAsym(Whipple):
    """Whipple model with the rear-frame products of inertia IBxy and IByz."""

    def __init__(self, params):
        super().__init__(params)
        self.dI = np.array(
            [
                [0.0, self.p.get("IBxy", 0.0), 0.0],
                [self.p.get("IBxy", 0.0), 0.0, self.p.get("IByz", 0.0)],
                [0.0, self.p.get("IByz", 0.0), 0.0],
            ]
        )

    def with_params(self, **changes):
        return WhippleAsym({**self.p, **changes})

    def _geometry(self, phi, delta):
        theta, A, jv, jw, inertia, heights, D, B = super()._geometry(phi, delta)
        if self.dI.any():
            R = _rot(0, phi) @ _rot(1, theta)
            inertia = inertia.astype(np.result_type(inertia, R)).copy()
            inertia[1] = inertia[1] + R @ self.dI @ R.T
        return theta, A, jv, jw, inertia, heights, D, B


def translated(p, y_r, rider):
    """Parameters with a rider part (m_r, x_r, z_r) translated sideways by y_r without rotation."""
    m_r, x_r, z_r = rider
    mB = p["mB"]
    m_f = mB - m_r
    x_f = (mB * p["xB"] - m_r * x_r) / m_f
    z_f = (mB * p["zB"] - m_r * z_r) / m_f
    mu = m_r * m_f / mB
    q = dict(p)
    q["yB"] = m_r * y_r / mB
    q["IBxx"] = p["IBxx"] + mu * y_r**2
    q["IBzz"] = p["IBzz"] + mu * y_r**2
    q["IBxy"] = p.get("IBxy", 0.0) - mu * y_r * (x_r - x_f)
    q["IByz"] = p.get("IByz", 0.0) - mu * y_r * (z_r - z_f)
    return q


# ---- a 75 kg rider added to bicycle A --------------------------------------------------------
def _tensor(xx, yy, zz, xz):
    return np.array([[xx, 0.0, xz], [0.0, yy, 0.0], [xz, 0.0, zz]])


def rider_parts(theta_deg=10.0):
    """75 kg rider: lower body 33 kg, upper body 42 kg as a cylinder (0.75 m by 0.17 m) hinged
    at the hip (0.25, -1.00) and tilted forward by theta. Items (m, x, z, inertia)."""
    lower = (33.0, 0.36, -0.72, _tensor(1.5, 1.2, 0.8, 0.0))
    th = math.radians(theta_deg)
    m, L, r = 42.0, 0.75, 0.17
    u = np.array([math.sin(th), 0.0, -math.cos(th)])
    Ia, Ip = m * r * r / 2, m * (3 * r * r + L * L) / 12
    inertia = Ip * np.eye(3) + (Ia - Ip) * np.outer(u, u)
    upper = (m, 0.25 + L / 2 * u[0], -1.00 + L / 2 * u[2], inertia)
    return [lower, upper]


def with_rider(p, theta_deg=10.0):
    """Parameters of the bicycle with the rider of rider_parts added to the rear frame."""
    parts = [(p["mB"], p["xB"], p["zB"], _tensor(p["IBxx"], p["IByy"], p["IBzz"], p["IBxz"]))]
    parts += rider_parts(theta_deg)
    m = sum(q[0] for q in parts)
    x = sum(q[0] * q[1] for q in parts) / m
    z = sum(q[0] * q[2] for q in parts) / m
    inertia = np.zeros((3, 3))
    for mi, xi, zi, Ii in parts:
        d = np.array([xi - x, 0.0, zi - z])
        inertia += Ii + mi * (d @ d * np.eye(3) - np.outer(d, d))
    return dict(
        p,
        mB=m,
        xB=x,
        zB=z,
        IBxx=inertia[0, 0],
        IByy=inertia[1, 1],
        IBzz=inertia[2, 2],
        IBxz=inertia[0, 2],
    )
