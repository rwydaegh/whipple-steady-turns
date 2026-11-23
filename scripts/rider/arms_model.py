"""Shared functions for the steer spring, damper and inertia of the arms (Sect. 13.2, "Steer torque with a centring spring")."""  # noqa: E501

import math
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402
from scipy.optimize import brentq, minimize_scalar, newton  # noqa: E402

from whipple import capsize_rule as cr  # noqa: E402
from whipple import stability as st  # noqa: E402
from whipple.bicycles import BICYCLES, measured_sets  # noqa: E402
from whipple.linear import Linear  # noqa: E402
from whipple.model import Whipple  # noqa: E402
from whipple.quartic import Quartic, bracket  # noqa: E402

H = 1e-30  # complex step
PLANTS = ("free", "constant")


def all_sets():
    """The benchmark bicycle and the 23 measured bicycle and rider sets, by name."""
    return {"benchmark": dict(BICYCLES["benchmark"]), **measured_sets("rider")}


# ---- exact matrix polynomial of a steady turn ----------------------------------------------------
def pencils(bike: Whipple, phi, delta):
    """{plant: (Me, Ce, Ke)} of the turn through (phi, delta), and the turn (complex step)."""
    turn = bike.turn(phi, delta)
    if turn is None:
        raise ValueError("no forward turn")
    s0 = turn.s
    M = bike.mass_matrix(phi, delta)
    Mqq, m, ms = M[:2, :2], M[:2, 2], M[2, 2]
    u0 = np.array([0, 0, s0], complex)
    C = np.column_stack(
        [np.imag(bike.parts(phi, delta, u0 + 1j * H * np.eye(3)[j])[2]) / H for j in range(2)]
    )
    Cq, cs = C[:2] / s0, C[2] / s0
    q0 = np.array([phi, delta], complex)
    N = np.column_stack(
        [np.imag(bike.parts(*(q0 + 1j * H * np.eye(2)[j]), u0)[2][:2]) / H for j in range(2)]
    )
    fc = bike.static_forces(phi, delta)[1][:2]
    free = (
        Mqq - np.outer(m, m) / ms,
        -s0 * Cq + s0 / ms * (np.outer(m, cs) + 2 * np.outer(fc, m)),
        -N - 2 * s0**2 / ms * np.outer(fc, cs),
    )
    constant = (Mqq, -s0 * Cq, -N)
    return {"free": free, "constant": constant}, turn


def numerators(Me, Ce, Ke):
    """(det Me, A1, A2, A3, A4) of the quartic without arms, and (Me_pp, Ce_pp, Ke_pp)."""
    A = np.array(
        [
            np.linalg.det(Me),
            bracket(Me, Ce),
            np.linalg.det(Ce) + bracket(Me, Ke),
            bracket(Ce, Ke),
            np.linalg.det(Ke),
        ]
    )
    return A, np.array([Me[0, 0], Ce[0, 0], Ke[0, 0]])


def quartic(A, Lpp, k=0.0, c=0.0, Ia=0.0):
    """(a1, a2, a3, a4) with arms. A and Lpp may have trailing grid axes."""
    Mp, Cp, Kp = Lpp
    d = A[0] + Ia * Mp
    a1 = A[1] + c * Mp + Ia * Cp
    a2 = A[2] + k * Mp + c * Cp + Ia * Kp
    a3 = A[3] + k * Cp + c * Kp
    a4 = A[4] + k * Kp
    return np.array([a1 / d, a2 / d, a3 / d, a4 / d])


def hurwitz3(a):
    return a[0] * a[1] * a[2] - a[2] ** 2 - a[0] ** 2 * a[3]


def k_fold(A, Lpp):
    """Spring stiffness at which the turn is on the capsize (fold) boundary: -det Ke / Ke_pp."""
    return -A[4] / Lpp[2]


# ---- independent check: Jacobian of the nonlinear field with the arm terms ----------------------
def jacobian_with_arms(bike: Whipple, phi, delta, plant="free", k=0.0, c=0.0, Ia=0.0):
    """Jacobian at the turn with the arm terms. State (phi, delta, dphi, ddelta[, s]).

    At a steady turn the force vanishes, so the derivative of the inverse mass matrix multiplies
    zero and the arm terms change only the acceleration rows.
    """
    J0, turn = st.turn_jacobian(bike, phi, delta, plant)
    Mr = bike.mass_matrix(phi, delta)
    if plant == "constant":
        Mr = Mr[:2, :2]
    e2 = np.zeros(Mr.shape[0])
    e2[1] = 1.0
    acc = Mr @ J0[2:]
    acc[:, 1] -= k * e2
    acc[:, 3] -= c * e2
    J = J0.copy()
    J[2:] = np.linalg.solve(Mr + Ia * np.outer(e2, e2), acc)
    return J, turn


def quartic_from_jacobian(J):
    lam = np.linalg.eigvals(J)
    if len(lam) == 5:
        lam = np.delete(lam, np.argmin(np.abs(lam)))
    return np.real(np.poly(lam))[1:5]


def steer_stiffness_quasi_static(bike: Whipple, phi, delta, plant="free"):
    """dT/d(delta) along the quasi-static path of the plant through the turn.

    Constant speed: the path of constant wheel spin s. Free rolling: the path with
    dE = T d(delta), equal to det d(T, H)/d(phi, delta) / (dE/dphi at constant delta).
    The capsize boundary with spring stiffness k is k = -dT/d(delta) on this path.
    """
    if plant == "constant":

        def TS(q):
            S, T = bike.turn_spin_torque(q[0], q[1])
            return np.array([T, S])

        q0 = np.array([phi, delta], complex)
        Jm = np.column_stack([np.imag(TS(q0 + 1j * H * np.eye(2)[j])) / H for j in range(2)])
        return Jm[0, 1] - Jm[0, 0] * Jm[1, 1] / Jm[1, 0]
    d_phi = st.torque_and_first_integral(bike, phi + 1j * H, delta).imag / H
    d_delta = st.torque_and_first_integral(bike, phi, delta + 1j * H).imag / H
    E_phi = d_phi[1] + delta * d_phi[0]  # E = H + T delta at constant delta
    return -(d_phi[0] * d_delta[1] - d_delta[0] * d_phi[1]) / E_phi


# ---- upright bicycle with arms: closed forms from the linear model ------------------------------
class Upright:
    """Closed forms for the upright bicycle with spring k, damper c and added inertia Ia.

    chi(s) det(M + Ia e e^T) = det(s^2 M + s V C1 + K) + (Ia s^2 + c s + k)(M_pp s^2 + K_pp),
    because C1_pp = 0. With mu = M_pp / det M and kap = g m_T h / det M (Ia = 0):
        a1 = a10 + c mu,  a2 = a20 + k mu,  a3 = a30 - c kap,  a4 = a40 - k kap.
    """

    def __init__(self, params: dict):
        self.p = dict(params)
        self.lin = lin = Linear(self.p)
        self.g = lin.g
        self.M, self.C1, self.K0, self.K2 = lin.M, lin.C1, lin.K0, lin.K2
        self.detM = np.linalg.det(self.M)
        self.Mpp = self.M[0, 0]
        self.Kpp = self.g * self.K0[0, 0]  # -g m_T h, the same at all speeds
        self.mu = self.Mpp / self.detM
        self.kap = -self.Kpp / self.detM
        dK0 = self.g**2 * np.linalg.det(self.K0)  # det K(V) = dK0 + dK2 V^2
        dK2 = self.g * bracket(self.K0, self.K2)
        self.cK0 = self.g * bracket(self.C1, self.K0)  # <C1, K> = cK0 + cK2 V^2
        self.cK2 = bracket(self.C1, self.K2)
        self.k0 = dK0 / (-self.Kpp)  # k_c(V) = k0 - kappa_c V^2
        self.kappa_c = -dK2 / (-self.Kpp)
        vc2 = -dK0 / dK2
        self.v_c = math.sqrt(vc2) if vc2 > 0 else math.nan
        va2 = -self.cK0 / self.cK2
        self.V_a3 = math.sqrt(va2) if va2 > 0 else math.nan

    def a(self, V, k=0.0, c=0.0, Ia=0.0):
        M, C, K = self.M, V * self.C1, self.lin.K(V)
        A = np.array(
            [
                self.detM,
                bracket(M, C),
                np.linalg.det(C) + bracket(M, K),
                bracket(C, K),
                np.linalg.det(K),
            ]
        )
        return quartic(A, np.array([self.Mpp, 0.0, self.Kpp]), k, c, Ia)

    def k_c(self, V):
        """Largest spring stiffness without capsize at speed V: det K / (g m_T h)."""
        return self.k0 - self.kappa_c * V * V

    def v_c_k(self, k):
        """Capsize speed with spring stiffness k: v_c(k)^2 = v_c^2 - k / kappa_c."""
        x = (self.k0 - k) / self.kappa_c
        return math.sqrt(x) if x > 0 else math.nan

    def c_3(self, V):
        """Damping rate at which a3 = 0 at speed V: V <C1, K(V)> / (g m_T h)."""
        return V * (self.cK0 + V * V * self.cK2) / (-self.Kpp)

    def V_3(self, c):
        """Lowest speed with a3 > 0 for the damping rate c."""
        if c == 0:
            return self.V_a3
        return brentq(lambda V: self.c_3(V) - c, self.V_a3, 60.0, xtol=1e-13)

    @property
    def k_star(self):
        """Closing stiffness of the upright stable speed range (c = 0): k_c(V_a3)."""
        return self.k_c(self.V_a3)

    @property
    def c_star(self):
        """Closing damping rate of the upright stable speed range (k = 0): c_3(v_c)."""
        return self.c_3(self.v_c)

    def Ia_star(self, k=0.0):
        """Added steer inertia (c = 0) at which the weave speed reaches v_c(k)."""
        V = self.v_c_k(k)
        if not V == V:
            return math.nan
        M, C, K = self.M, V * self.C1, self.lin.K(V)
        A1, A2, A3 = bracket(M, C), np.linalg.det(C) + bracket(M, K) + k * self.Mpp, bracket(C, K)
        return (A1 * A2 - A3 * self.detM) / (A3 * self.Mpp - A1 * self.Kpp)

    def corner(self, k=None, c=0.0):
        """a1 and a2 at the corner a3 = a4 = 0. The range closes there only if both are > 0."""
        k = self.k_star if k is None else k
        a = self.a(self.V_3(c), k, c)
        return a[0], a[1]

    def v_w(self, k=0.0, c=0.0, Ia=0.0):
        """Weave speed: lower end of the upright stable range (nan if the range is closed)."""
        lo, hi = self.V_3(c), self.v_c_k(k)
        if not (hi == hi and lo < hi):
            return math.nan

        def f(V):
            return hurwitz3(self.a(V, k, c, Ia))

        e = 1e-9 * (hi - lo)
        if f(lo + e) > 0:
            return lo
        if f(hi - e) < 0:
            return math.nan
        return brentq(f, lo + e, hi - e, xtol=1e-13)

    def band(self, k=0.0, c=0.0, Ia=0.0):
        """Upright stable speed range (v_w, v_c(k)) or None."""
        vw = self.v_w(k, c, Ia)
        return (vw, self.v_c_k(k)) if vw == vw else None

    def band_numeric(self, k=0.0, c=0.0, Ia=0.0, Vs=None):
        """Upright stable speed range from the eigenvalues on the speeds Vs (independent check)."""
        Vs = np.linspace(0.3, 1.3 * self.v_c, 2601) if Vs is None else Vs
        Minv = np.linalg.inv(self.M + np.diag([0.0, Ia]))
        A = np.zeros((len(Vs), 4, 4))
        A[:, :2, 2:] = np.eye(2)
        K = np.array([self.lin.K(V) for V in Vs]) + np.diag([0.0, k])
        C = Vs[:, None, None] * self.C1 + np.diag([0.0, c])
        A[:, 2:, :2] = -Minv @ K
        A[:, 2:, 2:] = -Minv @ C
        ok = np.linalg.eigvals(A).real.max(axis=1) < 0
        if not ok.any():
            return None
        i = np.flatnonzero(ok)
        return float(Vs[i[0]]), float(Vs[i[-1]])

    def meijaard_form(self):
        """(k0, kappa_c) of eq. (rider-kc) in the symbols of Meijaard et al. (2007)."""
        p = self.p
        w, c, lam = p["w"], p["c"], p["lam"]
        mT = p["mR"] + p["mB"] + p["mH"] + p["mF"]
        zT = (-p["rR"] * p["mR"] + p["zB"] * p["mB"] + p["zH"] * p["mH"] - p["rF"] * p["mF"]) / mT
        xT = (p["xB"] * p["mB"] + p["xH"] * p["mH"] + w * p["mF"]) / mT
        mA = p["mH"] + p["mF"]
        xA = (p["xH"] * p["mH"] + w * p["mF"]) / mA
        zA = (p["zH"] * p["mH"] - p["rF"] * p["mF"]) / mA
        uA = (xA - w - c) * math.cos(lam) - zA * math.sin(lam)
        SA = mA * uA + c / w * math.cos(lam) * mT * xT
        SF, SR = p["IFyy"] / p["rF"], p["IRyy"] / p["rR"]
        mTh = -mT * zT
        k0 = self.g * SA * (math.sin(lam) - SA / mTh)
        kappa_c = math.cos(lam) / w * (SF * math.sin(lam) - SA * (SF + SR) / mTh)
        return k0, kappa_c, dict(S_A=SA, mT=mT, h=-zT)


# ---- turns at a given speed ----------------------------------------------------------------------
def delta_at(bike, phi, V, guess):
    """Steer of the turn with lean phi and rear-contact speed V (secant)."""

    def res(d):
        return bike.p["rR"] * np.sqrt(bike.turn_spin_torque(phi, d)[0]) - V

    return newton(res, guess, x1=guess * (1 + 1e-4) + 1e-7, tol=1e-13, maxiter=60)


def family(bike, V, phis):
    """Steer of the turns at contact speed V for ascending leans; stops at the first failure."""
    r = Linear(bike.p).steer_ratio(V)
    ds = []
    for i, phi in enumerate(phis):
        if i == 0:
            guess = r * phi
        elif i == 1:
            guess = ds[0] * phi / phis[0]
        else:
            guess = ds[-1] + (ds[-1] - ds[-2]) * (phi - phis[i - 1]) / (phis[i - 1] - phis[i - 2])
        try:
            with np.errstate(invalid="ignore"):
                d = delta_at(bike, phi, V, guess)
            if not np.isfinite(d) or abs(d - guess) > 0.05 + 0.5 * abs(guess):
                break
        except (RuntimeError, ValueError, FloatingPointError):
            break
        ds.append(d)
    return np.array(ds)


def tip_exact(bike, plant, k=0.0, c=0.0, Ia=0.0, guess=None):
    """Newton solve of a3 = a4 = 0 in (phi, delta) with the exact matrices. Dict or None."""

    def F(x):
        pen, _ = pencils(bike, x[0], x[1])
        a = quartic(*numerators(*pen[plant]), k, c, Ia)
        return np.array([a[2], a[3]])

    try:
        with np.errstate(all="ignore"):
            x = st.newton_system(F, np.asarray(guess, float), h=1e-7, tol=1e-11, maxiter=40)
        pen, turn = pencils(bike, x[0], x[1])
    except (RuntimeError, ValueError, np.linalg.LinAlgError):
        return None
    if not (x[0] > 0 and np.all(np.isfinite(x))):
        return None
    a = quartic(*numerators(*pen[plant]), k, c, Ia)
    return dict(
        phi_deg=math.degrees(x[0]),
        delta=float(x[1]),
        V=turn.V,
        T=turn.T,
        a1=float(a[0]),
        a2=float(a[1]),
    )


# ---- quartic to order phi^2 with arms (tip start point) ------------------------------------------
def _arm(m, d, q, k, c, Ia):
    return np.array([Ia * m, c * m + Ia * d, k * m + c * d + Ia * q, k * d + c * q, k * q])


class ArmQuartic(Quartic):
    """Coefficients a_i0 + a_i2 phi^2 of the quartic with arms (whipple.quartic plus arm terms)."""

    def a02(self, V, plant="free", k=0.0, c=0.0, Ia=0.0):
        lin = self.lin
        M, C, K = lin.M, V * lin.C1, lin.K(V)
        Mh, Dh, Kh = self.L2(V, plant)
        A0 = np.array(
            [
                np.linalg.det(M),
                bracket(M, C),
                np.linalg.det(C) + bracket(M, K),
                bracket(C, K),
                np.linalg.det(K),
            ]
        )
        A2 = np.array(
            [
                bracket(M, Mh),
                bracket(M, Dh) + bracket(C, Mh),
                bracket(M, Kh) + bracket(C, Dh) + bracket(K, Mh),
                bracket(C, Kh) + bracket(K, Dh),
                bracket(K, Kh),
            ]
        )
        A0 = A0 + _arm(M[0, 0], 0.0, K[0, 0], k, c, Ia)
        A2 = A2 + _arm(Mh[0, 0], Dh[0, 0], Kh[0, 0], k, c, Ia)
        a0 = A0[1:] / A0[0]
        return a0, (A2[1:] - a0 * A2[0]) / A0[0]

    def tip_arms(self, plant="free", k=0.0, c=0.0, Ia=0.0):
        """Tip a3 = a4 = 0 of the truncated coefficients: (V, phi) or None."""

        def h(V):
            a0, a2 = self.a02(V, plant, k, c, Ia)
            return a0[2] * a2[3] - a2[2] * a0[3]

        v3 = self.v_a3()
        vs = np.linspace(0.6 * v3, 1.6 * v3, 101)
        hv = np.array([h(v) for v in vs])
        best = None
        for i in np.flatnonzero(np.sign(hv[:-1]) != np.sign(hv[1:])):
            V = brentq(h, vs[i], vs[i + 1], xtol=1e-12)
            a0, a2 = self.a02(V, plant, k, c, Ia)
            x = -a0[3] / a2[3]
            if x > 0 and (best is None or abs(V - v3) < abs(best[0] - v3)):
                best = (V, math.sqrt(x))
        return best


# ---- capsize rule with a spring ------------------------------------------------------------------
def series_k(s: cr.Series, V, k):
    """Series coefficients of u = T + k delta and of the speed of H_k = H - k delta^2 / 2."""
    d, T, Hh, _ = s.turn_series(V)
    Tk = T + k * d
    Hk = Hh - 0.5 * k * s._mul(d, d)
    h2, h4, _ = s._speed_series(V, Hk)
    return dict(V=V, r1=d[1], r3=d[3], tau1=Tk[1], tau3=Tk[3], tau5=Tk[5], h2=h2, h4=h4)


def series_k_d(s, V, k, rel=1e-4):
    """series_k with the first V-derivatives ("d...") by central differences."""
    h = rel * V
    c0, cp, cm = series_k(s, V, k), series_k(s, V + h, k), series_k(s, V - h, k)
    out = dict(c0)
    for key in c0:
        if key != "V":
            out["d" + key] = (cp[key] - cm[key]) / (2 * h)
    return out


def _turnover(t1, t3, t5, order):
    if order == 1:
        x = -t1 / (3 * t3) if t3 != 0 else math.nan
        return math.sqrt(x) if x > 0 else math.nan
    roots = np.roots([5 * t5, 3 * t3, t1])
    roots = roots[(np.abs(roots.imag) < 1e-12) & (roots.real > 0)].real
    return math.sqrt(roots.min()) if len(roots) else math.nan


def phi_c(params, V, k=0.0, plant="free", order=1):
    """Capsize lean (rad) from the rule with spring stiffness k: tau_i -> tau_i + k r_i, and the
    speed of H_k in place of H. ``order`` 1: first-order rule; 2: second-order rule."""
    s = cr.series(params)
    if plant == "constant":
        c = series_k(s, V, k)
        return _turnover(c["tau1"], c["tau3"], c["tau5"], order)
    roots = cr.free_polynomial(series_k_d(s, V, k), order).roots()
    roots = roots[(np.abs(roots.imag) < 1e-9) & (roots.real > 0)].real
    return math.sqrt(roots.min()) if len(roots) else math.nan


def largest_lean_rule(params, k=0.0, plant="free", order=1):
    """Rule for the largest stable lean with spring stiffness k: the first maximum of
    phi_c(V; k) below v_c(k), down to V_a3. Returns dict(phi_deg, V, at_tip)."""
    u = Upright(params)
    lo, hi = u.V_a3, u.v_c_k(k)
    if not (hi == hi and hi > lo):
        return dict(phi_deg=0.0, V=math.nan, at_tip=True)
    vs = np.linspace(hi, lo, 61)[1:]
    ph = np.array([phi_c(params, v, k, plant, order) for v in vs])
    n = len(vs)
    for i in range(n):
        if not np.isfinite(ph[i]):
            n = i
            break
    if n == 0:
        return dict(phi_deg=math.nan, V=math.nan, at_tip=False)
    for j in range(1, n - 1):
        if ph[j] >= ph[j - 1] and ph[j] > ph[j + 1]:
            i = j
            break
    else:
        i = n - 1
    if 0 < i < n - 1:
        r = minimize_scalar(
            lambda v: -phi_c(params, v, k, plant, order),
            bounds=(vs[i + 1], vs[i - 1]),
            method="bounded",
            options=dict(xatol=1e-7),
        )
        return dict(phi_deg=math.degrees(-r.fun), V=float(r.x), at_tip=False)
    return dict(phi_deg=math.degrees(ph[i]), V=float(vs[i]), at_tip=(i == n - 1 == len(vs) - 1))
