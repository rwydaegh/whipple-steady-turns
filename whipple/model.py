"""Nonlinear Whipple bicycle with a steer torque, its steady turns and its Jacobian.

Axes and signs follow Meijaard et al. (2007): x forward, y right, z down, positive lean and
steer to the right. The state is z = (lean, steer, lean rate, steer rate, s), with s the spin
rate of the rear wheel relative to the rear frame. Forward running has s < 0.
"""

import cmath
import math
from dataclasses import dataclass
from functools import cache

import numpy as np
import sympy as sp

PARAMETERS = (
    "w c lam rR mR IRxx IRyy xB zB mB IBxx IByy IBzz IBxz "
    "xH zH mH IHxx IHyy IHzz IHxz rF mF IFxx IFyy yB g"
).split()
COMPLEX_STEP = 1e-30


def _rot(axis: int, a):
    c, s = sp.cos(a), sp.sin(a)
    if axis == 0:
        return sp.Matrix([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == 1:
        return sp.Matrix([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return sp.Matrix([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _derive():
    """Symbolic kinematics of the four bodies in the coordinates (x, y, z, yaw, lean, pitch, steer,
    rear wheel angle, front wheel angle)."""
    p = dict(zip(PARAMETERS, sp.symbols(PARAMETERS)))
    q = sp.Matrix(sp.symbols("x y z psi phi theta delta sr sf"))
    v = sp.Matrix(sp.symbols("vx vy vz vpsi vphi vtheta vdelta vsr vsf"))
    psi, phi, theta, delta = q[3:7]
    ex, ey, ez = sp.Matrix([1, 0, 0]), sp.Matrix([0, 1, 0]), sp.Matrix([0, 0, 1])

    R = _rot(2, psi) * _rot(0, phi) * _rot(1, theta)
    S = _rot(1, p["lam"]) * _rot(2, delta) * _rot(1, -p["lam"])
    H = R * S
    pivot = sp.Matrix([p["w"] + p["c"], 0, p["rR"]])
    rear = sp.Matrix(q[:3])
    front = rear + R * (pivot + S * (sp.Matrix([p["w"], 0, p["rR"] - p["rF"]]) - pivot))
    frame = rear + R * sp.Matrix([p["xB"], p["yB"], p["zB"] + p["rR"]])
    fork = rear + R * (pivot + S * (sp.Matrix([p["xH"], 0, p["zH"] + p["rR"]]) - pivot))
    centres = [rear, frame, fork, front]

    wb = sp.zeros(3, 9)
    wb[:, 3] = ez
    wb[:, 4] = _rot(2, psi) * ex
    wb[:, 5] = _rot(2, psi) * _rot(0, phi) * ey
    wh = wb.copy()
    wh[:, 6] = R * _rot(1, p["lam"]) * ez
    wr = wb.copy()
    wr[:, 7] = R * ey
    wf = wh.copy()
    wf[:, 8] = H * ey
    jw = [wr, wb, wh, wf]
    jv = [c.jacobian(q) for c in centres]

    def tensor(xx, yy, zz, xz=0):
        return sp.Matrix([[xx, 0, xz], [0, yy, 0], [xz, 0, zz]])

    inertia = [
        R * tensor(p["IRxx"], p["IRyy"], p["IRxx"]) * R.T,
        R * tensor(p["IBxx"], p["IByy"], p["IBzz"], p["IBxz"]) * R.T,
        H * tensor(p["IHxx"], p["IHyy"], p["IHzz"], p["IHxz"]) * H.T,
        H * tensor(p["IFxx"], p["IFyy"], p["IFxx"]) * H.T,
    ]

    def to_contact(axle, r):
        return r * (ez - axle * axle[2]) / sp.sqrt(1 - axle[2] ** 2)

    rho_r, rho_f = to_contact(R * ey, p["rR"]), to_contact(H * ey, p["rF"])
    A = (jv[0] + sp.Matrix.hstack(*[wr[:, i].cross(rho_r) for i in range(9)])).col_join(
        jv[3] + sp.Matrix.hstack(*[wf[:, i].cross(rho_f) for i in range(9)])
    )
    # Differentiate before setting yaw to zero, so that all yaw-rate terms remain.
    bias_v = [(j * v).jacobian(q) * v for j in jv]
    bias_w = [(j * v).jacobian(q) * v for j in jw]
    bias_a = (A * v).jacobian(q) * v
    gap = front[2] + rho_f[2]

    upright = {psi: 0, q[2]: -p["rR"] * sp.cos(phi)}
    geometry = [A, *jv, *jw, *inertia, sp.Matrix([c[2] for c in centres])]
    bias = [*bias_v, *bias_w, bias_a]
    contact = [sp.Matrix([gap, gap.diff(theta)])]
    args = (phi, theta, delta)
    pars = [p[k] for k in PARAMETERS]
    return args, v, pars, upright, geometry, bias, contact


def _flat(mats, subs):
    return [e.subs(subs) for m in mats for e in m]


def _compile(args, exprs):
    real = sp.lambdify(args, exprs, modules=[vars(math)], cse=True)
    cplx = sp.lambdify(args, exprs, modules=[vars(cmath)], cse=True)
    return real, cplx


@cache
def _kernels():
    args, v, pars, upright, geometry, bias, contact = _derive()
    return {
        "geometry": _compile([*args, pars], _flat(geometry, upright)),
        "bias": _compile([*args, list(v), pars], _flat(bias, upright)),
        "contact": _compile([*args, pars], _flat(contact, upright)),
    }


def _call(name, *args):
    real, cplx = _kernels()[name]
    if any(np.iscomplexobj(a) for a in args):
        return np.array(cplx(*args), dtype=complex)
    return np.array(real(*args), dtype=float)


@dataclass(frozen=True)
class Turn:
    """Steady turn through (lean, steer). V, R and a_y refer to the rear contact point."""

    phi: float
    delta: float
    s: float
    T: float
    V: float
    R: float
    ay: float
    yaw_rate: float


class Whipple:
    """Nonlinear Whipple bicycle with a steer torque T and a rear drive torque tau."""

    def __init__(self, params: dict):
        p = {"yB": 0.0, "g": 9.81, **params}
        self.p = p
        self.par = [p[k] for k in PARAMETERS]
        self.mass = np.array([p["mR"], p["mB"], p["mH"], p["mF"]])
        # Total mass plus the wheel spin inertias, the mass in v_E = sqrt(2 E / m_tot).
        self.m_tot = self.mass.sum() + p["IRyy"] / p["rR"] ** 2 + p["IFyy"] / p["rF"] ** 2

    def with_params(self, **changes) -> "Whipple":
        return Whipple({**self.p, **changes})

    def pitch(self, phi, delta):
        """Rear-frame pitch that keeps both wheels on the ground (Newton from upright)."""
        theta = 0.0 * (phi + delta)
        for _ in range(30):
            gap, slope = _call("contact", phi, theta, delta, self.par)
            step = gap / slope
            theta = theta - step
            if abs(step) < 1e-15:
                return theta
        raise RuntimeError("pitch constraint did not converge")

    def _geometry(self, phi, delta):
        theta = self.pitch(phi, delta)
        g = _call("geometry", phi, theta, delta, self.par)
        A = g[:54].reshape(6, 9)
        jv = g[54:162].reshape(4, 3, 9)
        jw = g[162:270].reshape(4, 3, 9)
        inertia = g[270:306].reshape(4, 3, 3)
        heights = g[306:310]
        D = np.vstack([A, np.eye(9)[[4, 6, 7]]])
        B = np.linalg.solve(D, np.vstack([np.zeros((6, 3)), np.eye(3)]))
        return theta, A, jv, jw, inertia, heights, D, B

    def parts(self, phi, delta, u):
        """Velocities qd (9), reduced mass matrix (3x3), reduced force (3) and the basis B (9x3)."""
        theta, A, jv, jw, inertia, heights, D, B = self._geometry(phi, delta)
        qd = B @ u
        b = _call("bias", phi, theta, delta, qd, self.par)
        av, aw, abias = b[:12].reshape(4, 3), b[12:24].reshape(4, 3), b[24:]
        omega = np.einsum("bki,i->bk", jw, qd)
        spin = np.einsum("bij,bj->bi", inertia, omega)
        M = np.einsum("bki,b,bkj->ij", jv, self.mass, jv) + np.einsum(
            "bki,bkl,blj->ij", jw, inertia, jw
        )
        h = np.einsum("bki,b,bk->i", jv, self.mass, av) + np.einsum(
            "bki,bk->i", jw, np.einsum("bij,bj->bi", inertia, aw) + np.cross(omega, spin)
        )
        F = self.p["g"] * np.einsum("bi,b->i", jv[:, 2, :], self.mass)
        c = np.linalg.solve(D, np.r_[-abias, np.zeros(3)])
        return qd, B.T @ M @ B, B.T @ (F - h - M @ c), B

    def mass_matrix(self, phi, delta):
        return self.parts(phi, delta, np.zeros(3))[1]

    def potential(self, phi, delta):
        heights = self._geometry(phi, delta)[5]
        return -self.p["g"] * self.mass @ heights

    def energy(self, z):
        """Kinetic plus potential energy (J) of the state z."""
        Mr = self.mass_matrix(z[0], z[1])
        return 0.5 * z[2:5] @ Mr @ z[2:5] + self.potential(z[0], z[1])

    def accelerations(self, phi, delta, u, T=0.0, tau=0.0):
        """Velocities qd (9) and du/dt (3) at speeds u = (lean rate, steer rate, s)."""
        qd, Mr, f, *_ = self.parts(phi, delta, u)
        return qd, np.linalg.solve(Mr, f + np.array([0.0, T, -tau]))

    def qdd(self, phi, delta, u):
        """Second derivatives of all nine coordinates."""
        qd, Mr, f, B = self.parts(phi, delta, u)
        theta, *_, D, _ = self._geometry(phi, delta)
        b = _call("bias", phi, theta, delta, qd, self.par)
        return B @ np.linalg.solve(Mr, f) + np.linalg.solve(D, np.r_[-b[24:], np.zeros(3)])

    def field(self, z, T=0.0):
        _, ud = self.accelerations(z[0], z[1], z[2:5], T)
        return np.r_[z[2:4], ud]

    def jacobian(self, z, T=0.0):
        """5x5 Jacobian of the field at constant steer torque (complex step)."""
        return complex_step(lambda x: self.field(x, T), z)

    def static_forces(self, phi, delta):
        """Gravity part F_g and centrifugal part F_c of the force at u = (0, 0, s), Eq. (5)."""
        f0 = self.parts(phi, delta, np.zeros(3))[2]
        f1 = self.parts(phi, delta, np.array([0.0, 0.0, 1.0]))[2]
        return f0, f1 - f0

    def turn_spin_torque(self, phi, delta):
        """Squared spin s^2 and steer torque T of the steady turn through (lean, steer), Eq. (6)."""
        Fg, Fc = self.static_forces(phi, delta)
        s2 = -Fg[0] / Fc[0]
        return s2, -Fg[1] - s2 * Fc[1]

    def turn(self, phi, delta) -> Turn | None:
        """Steady turn through (lean, steer) in forward running, or None if s^2 <= 0."""
        s2, T = self.turn_spin_torque(phi, delta)
        if not s2 > 0:
            return None
        B = self._geometry(phi, delta)[-1]
        s = math.copysign(math.sqrt(s2), B[0, 2])
        yaw = B[3, 2] * s
        V = self.p["rR"] * abs(s)
        values = (phi, delta, s, T, V, V / abs(yaw), V * abs(yaw), yaw)
        return Turn(*(float(x) for x in values))

    def turn_state(self, turn: Turn):
        return np.array([turn.phi, turn.delta, 0.0, 0.0, turn.s])

    def friction_demand(self, turn: Turn):
        """Total horizontal ground force divided by the weight, in a steady turn."""
        _, _, jv, *_, B = self._geometry(turn.phi, turn.delta)
        momentum = np.einsum("b,bki,i->k", self.mass, jv, B[:, 2] * turn.s)
        # Each mass centre moves on a circle, so the force is the yaw rate times the momentum.
        return abs(turn.yaw_rate) * np.hypot(*momentum[:2]) / (self.mass.sum() * self.p["g"])


def complex_step(f, x, h=COMPLEX_STEP):
    """Jacobian of f at the real point x by complex-step differentiation."""
    x = np.asarray(x, dtype=float)
    cols = []
    for j in range(len(x)):
        xc = x.astype(complex)
        xc[j] += 1j * h
        cols.append(np.imag(f(xc)) / h)
    return np.column_stack(cols)
