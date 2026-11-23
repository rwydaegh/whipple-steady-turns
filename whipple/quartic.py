"""Quartic of a turn to second order in the lean, and the tip speed V_a3 (Sect. 8)."""

import numpy as np
from scipy.optimize import brentq

from whipple.lean_law import LeanLaw

PLANTS = ("free", "constant")


def bracket(A, B):
    """<A, B> = a11 b22 + a22 b11 - a12 b21 - a21 b12 for 2x2 matrices."""
    return A[0, 0] * B[1, 1] + A[1, 1] * B[0, 0] - A[0, 1] * B[1, 0] - A[1, 0] * B[0, 1]


def hurwitz_d3(a):
    """D3 = a1 a2 a3 - a3^2 - a1^2 a4 of the coefficients a = (a1, a2, a3, a4)."""
    a1, a2, a3, a4 = a
    return a1 * a2 * a3 - a3**2 - a1**2 * a4


class Quartic:
    """Coefficients a_i0(V) and a_i2(V) of the quartic of the turns of one parameter set."""

    def __init__(self, params, law=None):
        self.law = law or LeanLaw(params)
        self.lin = self.law.lin
        self.g = self.law.g

    # ---- upright -------------------------------------------------------------------------
    def a0(self, V):
        """Upright coefficients (a1, a2, a3, a4) at speed V."""
        lin = self.lin
        M, C, K = lin.M, V * lin.C1, lin.K(V)
        dM = np.linalg.det(M)
        return np.array([bracket(M, C), np.linalg.det(C) + bracket(M, K), bracket(C, K),
                         np.linalg.det(K)]) / dM  # fmt: skip

    def v_a3(self):
        """Upright speed where a3 = 0 (eq. Va3): V^2 = -g <C1,K0>/<C1,K2>; nan if none."""
        lin = self.lin
        q = -self.g * bracket(lin.C1, lin.K0) / bracket(lin.C1, lin.K2)
        return np.sqrt(q) if q > 0 else np.nan

    # ---- order phi^2 ---------------------------------------------------------------------
    def L2(self, V, plant="free"):
        """Second-order pencil matrices (Mh, Dh, Kh) at speed V."""
        if plant not in PLANTS:
            raise ValueError(f"plant must be one of {PLANTS}, not {plant!r}")
        law, lin = self.law, self.lin
        c = law.contractions(V)
        e = c["e"]
        s = 1.0 if plant == "free" else 0.0
        Pe, K2e, NKe = law.P @ e, lin.K2 @ e, (law.N - lin.K2) @ e
        mt = law.m_tot
        Mh = c["M2e"] - s * np.outer(Pe, Pe) / mt
        Dh = V * c["C2e"] - s * V / mt * (np.outer(Pe, NKe) + 2 * np.outer(K2e, Pe))
        Kh = c["U4h"] + V * V * c["c3j"] - s * 2 * V * V / mt * np.outer(K2e, NKe)
        return Mh, Dh, Kh

    def a2(self, V, plant="free"):
        """Second-order coefficients (a12, a22, a32, a42) at speed V."""
        lin = self.lin
        M, C, K = lin.M, V * lin.C1, lin.K(V)
        Mh, Dh, Kh = self.L2(V, plant)
        n = np.array([bracket(M, Dh) + bracket(C, Mh),
                      bracket(M, Kh) + bracket(C, Dh) + bracket(K, Mh),
                      bracket(C, Kh) + bracket(K, Dh), bracket(K, Kh)])  # fmt: skip
        return (n - self.a0(V) * bracket(M, Mh)) / np.linalg.det(M)

    def a(self, V, phi, plant="free"):
        """Coefficients a_i0 + a_i2 phi^2."""
        return self.a0(V) + self.a2(V, plant) * phi * phi

    # ---- boundaries from the truncated coefficients ---------------------------------------
    def fold_x(self, V, plant="free"):
        """phi^2 of the fold a4 = 0 (truncated a4)."""
        return -self.a0(V)[3] / self.a2(V, plant)[3]

    def a3_x(self, V, plant="free"):
        """phi^2 where the truncated a3 is zero."""
        return -self.a0(V)[2] / self.a2(V, plant)[2]

    def d3_poly(self, V, plant="free"):
        """Coefficients (ascending, in x = phi^2) of D3 with truncated a_i (a cubic in x)."""
        P = np.polynomial.polynomial
        a0, a2 = self.a0(V), self.a2(V, plant)
        p = [np.array([a0[i], a2[i]]) for i in range(4)]
        d = P.polymul(P.polymul(p[0], p[1]), p[2])
        d = P.polysub(d, P.polymul(p[2], p[2]))
        d = P.polysub(d, P.polymul(P.polymul(p[0], p[0]), p[3]))
        return d

    def hopf_x(self, V, plant="free"):
        """Smallest positive root x = phi^2 of D3 = 0 with all a_i to order phi^2 (nan if none)."""
        d = self.d3_poly(V, plant)
        r = np.roots(d[::-1])
        r = r[(np.abs(r.imag) < 1e-9) & (r.real > 0)].real
        return r.min() if len(r) else np.nan

    def bt(self, plant="free", bracket_V=None):
        """Tip of the truncated coefficients: a3 = a4 = 0. Returns (V, phi) or None."""

        def h(V):
            a0, a2 = self.a0(V), self.a2(V, plant)
            return a0[2] * a2[3] - a2[2] * a0[3]

        if bracket_V is not None:
            V = brentq(h, *bracket_V, xtol=1e-12)
            return V, np.sqrt(self.fold_x(V, plant))
        v3 = self.v_a3()
        vs = np.linspace(0.6 * v3, 1.4 * v3, 81)
        hv = np.array([h(v) for v in vs])
        best = None
        for i in np.flatnonzero(np.sign(hv[:-1]) != np.sign(hv[1:])):
            V = brentq(h, vs[i], vs[i + 1], xtol=1e-12)
            x = self.fold_x(V, plant)
            if x > 0 and (best is None or abs(V - v3) < abs(best[0] - v3)):
                best = (V, np.sqrt(x))
        return best
