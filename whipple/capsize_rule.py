"""Capsize rules: fold lean of turns from Taylor coefficients (Sect. 6, "Capsize limit")."""

import json
import math

import numpy as np
from numpy.polynomial import Polynomial
from scipy.optimize import brentq, minimize_scalar

from whipple.linear import Linear
from whipple.model import Whipple

ORDER = 7  # highest total degree of the Taylor terms that are kept
PLANTS = ("free", "constant")
_CACHE: dict = {}


def _key(params: dict) -> str:
    return json.dumps({k: float(v) for k, v in sorted(params.items())})


def _check_plant(plant):
    if plant not in PLANTS:
        raise ValueError(f"plant must be one of {PLANTS}, not {plant!r}")


class Series:
    """Taylor terms of the model at upright and the small-lean series of the steady turn."""

    def __init__(self, params: dict, rho=0.2, n=24):
        self.bike = Whipple(params)
        self.lin = Linear(self.bike.p)
        self.rR = self.bike.p["rR"]
        # Taylor coefficients from samples on a polycircle of radius rho (2D FFT).
        w = rho * np.exp(2j * np.pi * np.arange(n) / n)
        F = np.empty((n, n, 6), complex)
        done = np.zeros((n, n), bool)
        zero, spin = np.zeros(3, complex), np.array([0, 0, 1], complex)
        for a in range(n):
            for b in range(n):
                if done[a, b]:
                    continue
                _, Mr, Fg, _ = self.bike.parts(w[a], w[b], zero)
                Fc = self.bike.parts(w[a], w[b], spin)[2] - Fg
                U = self.bike.potential(w[a], w[b])
                F[a, b] = [Mr[2, 2], U, Fg[0], Fg[1], Fc[0], Fc[1]]
                F[-a % n, -b % n] = F[a, b].conj()  # the model is real
                done[a, b] = done[-a % n, -b % n] = True
        c = np.fft.fft2(F, axes=(0, 1)) / n**2
        k = np.arange(ORDER + 1)
        scale = rho ** (k[:, None] + k[None, :])
        C = c[: ORDER + 1, : ORDER + 1].real / scale[:, :, None]
        C[k[:, None] + k[None, :] > ORDER] = 0.0
        # C[i, j, f] is the coefficient of phi^i delta^j of the function f.
        self.ms, self.U = C[:, :, 0], C[:, :, 1]
        self.Fg, self.Fc = C[:, :, 2:4], C[:, :, 4:6]
        self.m_tot = self.ms[0, 0] / self.rR**2
        self.v_c = self._capsize_speed()

    def _capsize_speed(self):
        K0, K2 = self.lin.K0, self.lin.K2
        x = -self.lin.g * np.linalg.det(K0) / (K0[0, 0] * K2[1, 1] - K0[1, 0] * K2[0, 1])
        return math.sqrt(x) if x > 0 else float("nan")

    def weave_speed(self):
        """Upright weave speed: the oscillatory pair crosses the imaginary axis (nan if none)."""
        if not hasattr(self, "_v_w"):

            def growth(v):
                lam = self.lin.eigenvalues(v)
                osc = lam[np.abs(lam.imag) > 1e-9]
                return osc.real.max() if len(osc) else math.nan

            top = 1.5 * self.v_c if self.v_c == self.v_c else 12.0
            vs = np.linspace(0.5, top, 300)
            g = np.array([growth(v) for v in vs])
            k = np.flatnonzero((g[:-1] > 0) & (g[1:] <= 0))
            self._v_w = brentq(growth, vs[k[0]], vs[k[0] + 1], xtol=1e-12) if len(k) else math.nan
        return self._v_w

    # ---- truncated power series in phi (index = power) -------------------------------------

    @staticmethod
    def _mul(a, b):
        return np.convolve(a, b)[: ORDER + 1]

    def _compose(self, C, d):
        """sum_ij C[i, j] phi^i d(phi)^j, truncated at phi^ORDER."""
        out = np.zeros(ORDER + 1)
        dj = np.zeros(ORDER + 1)
        dj[0] = 1.0
        for j in range(ORDER + 1):
            for i in range(ORDER + 1 - j):
                if C[i, j] != 0.0:
                    out[i:] += C[i, j] * dj[: ORDER + 1 - i]
            dj = self._mul(dj, d)
        return out

    def turn_series(self, V):
        """Series in phi of steer d, torque T, first integral H and energy E at contact speed V."""
        s2 = (V / self.rR) ** 2
        A = self.Fg[:, :, 0] + s2 * self.Fc[:, :, 0]  # lean equation
        B = self.Fg[:, :, 1] + s2 * self.Fc[:, :, 1]  # steer equation
        a01 = A[0, 1]
        A0 = A.copy()
        A0[0, 1] = 0.0
        d = np.zeros(ORDER + 1)
        for _ in range(ORDER + 1):
            d = -self._compose(A0, d) / a01
        T = -self._compose(B, d)
        E = 0.5 * s2 * self._compose(self.ms, d) + self._compose(self.U, d)
        H = E - self._mul(T, d)
        return d, T, H, E

    def _speed_series(self, V, H):
        """v = sqrt(2 (H - U0) / m_tot) = V + c2 phi^2 + c4 phi^4 + c6 phi^6."""
        a = 2.0 * H / self.m_tot
        a2, a4, a6 = a[2], a[4], a[6]
        return (
            a2 / (2 * V),
            a4 / (2 * V) - a2**2 / (8 * V**3),
            a6 / (2 * V) - a2 * a4 / (4 * V**3) + a2**3 / (16 * V**5),
        )

    def coefficients(self, V):
        """tau1, tau3, tau5, tau7, steer ratios r1, r3, r5, and h2, h4, h6 (H-speed) and
        e2, e4, e6 (energy speed) of the steady turn at contact speed V."""
        d, T, H, E = self.turn_series(V)
        h2, h4, h6 = self._speed_series(V, H)
        e2, e4, e6 = self._speed_series(V, E)
        return dict(
            V=V, r1=d[1], r3=d[3], r5=d[5], tau1=T[1], tau3=T[3], tau5=T[5], tau7=T[7],
            h2=h2, h4=h4, h6=h6, e2=e2, e4=e4, e6=e6,
        )  # fmt: skip

    def with_derivatives(self, V, rel=1e-4):
        """Coefficients at V with first ("d...") and second ("dd...") V-derivatives."""
        h = rel * V
        c0, cp, cm = self.coefficients(V), self.coefficients(V + h), self.coefficients(V - h)
        out = dict(c0)
        for k in c0:
            if k != "V":
                out["d" + k] = (cp[k] - cm[k]) / (2 * h)
                out["dd" + k] = (cp[k] - 2 * c0[k] + cm[k]) / h**2
        return out

    def fixed_H(self, v):
        """Torque series at fixed first integral: T = tau1 phi + tau3H phi^3 + tau5H phi^5,
        coefficients at v_H = v. Also the contact speed V = v - h2 phi^2 + g4 phi^4."""
        c = self.with_derivatives(v)
        g4 = c["h2"] * c["dh2"] - c["h4"]
        tau3H = c["tau3"] - c["h2"] * c["dtau1"]
        tau5H = (c["tau5"] - c["h2"] * c["dtau3"] + c["dtau1"] * g4
                 + 0.5 * c["ddtau1"] * c["h2"] ** 2)  # fmt: skip
        return dict(v=v, tau1=c["tau1"], tau3H=tau3H, tau5H=tau5H, h2=c["h2"], g4=g4,
                    tau3=c["tau3"], dtau1=c["dtau1"])  # fmt: skip


def series(params: dict) -> Series:
    """Cached :class:`Series` of a parameter set (about 1 s to build)."""
    k = _key(params)
    if k not in _CACHE:
        _CACHE[k] = Series(params)
    return _CACHE[k]


def coefficients(params: dict, V: float) -> dict:
    """tau1, tau3, tau5, h2, h4, ... of the steady turn at contact speed V."""
    return series(params).coefficients(V)


def _turnover(t1, t3, t5=None, order=1):
    """Smallest positive lean where d/dphi of t1 phi + t3 phi^3 (+ t5 phi^5) is zero, or nan."""
    if order == 1 or t5 is None:
        x = -t1 / (3 * t3) if t3 != 0 else math.nan
        return math.sqrt(x) if x > 0 else math.nan
    roots = np.roots([5 * t5, 3 * t3, t1])
    roots = roots[(np.abs(roots.imag) < 1e-12) & (roots.real > 0)].real
    return math.sqrt(roots.min()) if len(roots) else math.nan


def phi_c_constant(params: dict, V: float, order=1) -> float:
    """Capsize lean (rad) at constant rear-wheel speed V: turnover of T(phi) at fixed V.

    ``order=1``: sqrt(-tau1 / (3 tau3)). ``order=2``: smallest root of
    tau1 + 3 tau3 x + 5 tau5 x^2 = 0. nan where the rule has no real value.
    """
    c = series(params).coefficients(V)
    return _turnover(c["tau1"], c["tau3"], c["tau5"], order)


def free_polynomial(c: dict, order: int) -> Polynomial:
    """T_phi (v_H)_V - T_V (v_H)_phi as a polynomial in x = phi^2 (products not expanded).

    ``order=1``: T to phi^3 and v_H to phi^2 (first-order rule). ``order=2``: T to phi^5 and v_H to
    phi^4 (second-order rule). ``c`` is the output of :meth:`Series.with_derivatives`.
    """
    if order == 1:
        T_phi, T_V = Polynomial([c["tau1"], 3 * c["tau3"]]), Polynomial([c["dtau1"], c["dtau3"]])
        v_V, v_phi = Polynomial([1.0, c["dh2"]]), Polynomial([2 * c["h2"]])
    elif order == 2:
        T_phi = Polynomial([c["tau1"], 3 * c["tau3"], 5 * c["tau5"]])
        T_V = Polynomial([c["dtau1"], c["dtau3"], c["dtau5"]])
        v_V, v_phi = Polynomial([1.0, c["dh2"], c["dh4"]]), Polynomial([2 * c["h2"], 4 * c["h4"]])
    else:
        raise ValueError(f"order must be 1 or 2, not {order!r}")
    return T_phi * v_V - Polynomial([0.0, 1.0]) * T_V * v_phi


def phi_c_free(params: dict, V: float, order=2) -> float:
    """Capsize lean (rad) of the freely rolling bicycle at contact speed V.

    ``order=2``: second-order rule. ``order=1``: first-order rule.
    The rules come from Theorem 1 in the chart (phi, V): T_phi (v_H)_V = T_V (v_H)_phi.
    """
    c = series(params).with_derivatives(V)
    roots = free_polynomial(c, order).roots()
    roots = roots[(np.abs(roots.imag) < 1e-9) & (roots.real > 0)].real
    return math.sqrt(roots.min()) if len(roots) else math.nan


def phi_c(params: dict, V: float, plant="free", order=None) -> float:
    """Capsize lean (rad) of "free" (default order 2) or "constant" rolling (default order 1)."""
    _check_plant(plant)
    if plant == "free":
        return phi_c_free(params, V, 2 if order is None else order)
    return phi_c_constant(params, V, 1 if order is None else order)


def speed_range(params: dict):
    """Speed interval next to the capsize speed on which the constant-speed rule is real
    (tau1 and tau3 of opposite sign). Returns (v_far, v_c, direction).

    direction "down": the fold curve leaves v_c toward lower speed (tau3(v_c) > 0, the ordinary
    case). "up": toward higher speed (tau3(v_c) < 0); the fold curve then does not close the
    stable region. "none": no capsize speed.
    """
    s = series(params)
    if not s.v_c == s.v_c:
        return math.nan, math.nan, "none"
    hi = s.v_c
    down = s.coefficients(hi)["tau3"] > 0
    vs = np.linspace(hi, (0.25 if down else 2.0) * hi, 300)[1:]
    far = hi
    for v in vs:
        c = s.coefficients(v)
        if not c["tau1"] * c["tau3"] < 0:
            break
        far = v
    return far, hi, "down" if down else "up"


def largest_lean_rule(params: dict, plant="free", order=None) -> dict:
    """Top of the fold curve from a rule: the largest capsize lean over speed below v_c.

    Returns phi (rad), phi_deg, V (contact speed at the top), v_c, v_w and flags. The top of the
    fold curve is the largest stable lean if the three conditions of the paper hold (a capsize
    speed exists, the fold curve leaves v_c toward lower speed, and the weave Hopf curve does not
    cut the fold curve above its top). The flag ``above_weave`` (V > v_w) is only a hint.
    """
    _check_plant(plant)
    s = series(params)
    lo, hi, direction = speed_range(params)
    nan = dict(phi=math.nan, phi_deg=math.nan, V=math.nan, v_c=s.v_c, valid=False,
               direction=direction)  # fmt: skip
    if direction != "down":
        return nan

    def f(v):
        return phi_c(params, v, plant, order)

    # Scan downward from the capsize speed and take the first local maximum. Below it the
    # truncated polynomials can have other roots that do not belong to the fold curve.
    vs = np.linspace(hi, lo, 81)[1:]
    ph = np.array([f(v) for v in vs])
    ok = np.isfinite(ph)
    if not ok[0]:
        return nan
    i, interior = len(vs) - 1, False
    for k in range(1, len(vs) - 1):
        if not ok[k + 1]:
            i = k
            break
        if ph[k] >= ph[k - 1] and ph[k] > ph[k + 1]:
            i, interior = k, True
            break
    v_best = float(vs[i])
    if interior:
        r = minimize_scalar(lambda v: -f(v), bounds=(vs[i + 1], vs[i - 1]), method="bounded",
                            options=dict(xatol=1e-9))  # fmt: skip
        v_best = float(r.x)
    phi = f(v_best)
    v_w = s.weave_speed()
    return dict(phi=phi, phi_deg=math.degrees(phi), V=v_best, v_c=s.v_c, v_w=v_w,
                speed_range=(lo, hi), interior_maximum=interior, direction=direction,
                above_weave=bool(v_best > v_w) if v_w == v_w else False,
                valid=bool(interior))  # fmt: skip


def largest_lean_closed(params: dict, plant="free") -> dict:
    """Top of the fold curve from quantities at the capsize speed only (closed form).

    With tau1 linear in the steer ratio r of the linear turn and the model tau3 = b r (r - r0),
    fixed by the value and the slope of tau3 at v_c:

        phi_top = sqrt(l tau1' / (3 tau3)) / (1 + sqrt(n - 1)),   r_top = r_c (1 + 1/sqrt(n - 1)),
        l = -1 / (dln r/dV),   n = dln(tau3)/dln(r) = -l tau3'/tau3.

    For free rolling, 3 tau3 is replaced by D = 3 tau3 - 2 h2 tau1' + h2' tau1.
    Returns phi (rad), phi_deg, V, n, l (nan where the form does not apply).
    """
    _check_plant(plant)
    s = series(params)
    vc = s.v_c
    nan = dict(phi=math.nan, phi_deg=math.nan, V=math.nan, n=math.nan, l=math.nan)
    if not vc == vc:
        return nan

    def D(v):
        c = s.with_derivatives(v)
        if plant == "constant":
            return 3 * c["tau3"]
        return 3 * c["tau3"] - 2 * c["h2"] * c["dtau1"] + c["dh2"] * c["tau1"]

    dv = 1e-4 * vc

    def lnr(v):
        return math.log(s.lin.steer_ratio(v))

    ell = -2 * dv / (lnr(vc + dv) - lnr(vc - dv))
    D0 = D(vc)
    n = -ell * (D(vc + dv) - D(vc - dv)) / (2 * dv) / D0
    tau1p = s.with_derivatives(vc)["dtau1"]
    if not (D0 > 0 and n > 1 and tau1p > 0):
        return dict(nan, n=n, l=ell)
    phi = math.sqrt(ell * tau1p / D0) / (1 + math.sqrt(n - 1))
    r_top = s.lin.steer_ratio(vc) * (1 + 1 / math.sqrt(n - 1))
    V = brentq(lambda v: s.lin.steer_ratio(v) - r_top, 0.2 * vc, vc)
    return dict(phi=phi, phi_deg=math.degrees(phi), V=V, n=n, l=ell)
