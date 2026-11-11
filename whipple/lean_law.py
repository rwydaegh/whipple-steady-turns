"""Small-lean expansion: lambda_k(v_E, phi0) = lambda_k(v_E) + Lambda_k(v_E) phi0^2, Eq. (13).

Lambda_k depends only on the upright linear model and the next Taylor terms of the nonlinear model
at zero lean and steer. The Taylor terms are Cauchy integrals of the model on the polydisc
|phi| = |delta| = rho, computed by a 2D FFT.
"""

import numpy as np
from scipy.optimize import brentq, newton

from .linear import Linear
from .model import Whipple

MONO2 = ((2, 0), (1, 1), (0, 2))
MONO3 = ((3, 0), (2, 1), (1, 2), (0, 3))
MONO4 = ((4, 0), (3, 1), (2, 2), (1, 3), (0, 4))
TERMS = ("inertia", "gyroscopic", "gravity", "centrifugal", "energy_speed", "spin_coupling")


def _power(r, k):
    return r**k if k >= 0 else 0.0


def _gradient(i, j, r):
    """Gradient of phi^i delta^j at (1, r)."""
    return np.array([i * _power(r, j), j * _power(r, j - 1)])


def _hessian(i, j, r):
    """Hessian of phi^i delta^j at (1, r)."""
    off = i * j * _power(r, j - 1)
    return np.array([[i * (i - 1) * _power(r, j), off], [off, j * (j - 1) * _power(r, j - 2)]])


def _sample(bike: Whipple, phi, delta):
    """Mass matrix, potential, static forces and gyroscopic columns at one complex shape."""
    _, Mr, Fg, _ = bike.parts(phi, delta, np.zeros(3, complex))
    Fc = bike.parts(phi, delta, np.array([0, 0, 1], complex))[2] - Fg
    G = np.empty((3, 2), complex)
    for j in range(2):
        up, um = np.array([0, 0, 1], complex), np.array([0, 0, 1], complex)
        up[j], um[j] = 1, -1
        G[:, j] = (bike.parts(phi, delta, up)[2] - bike.parts(phi, delta, um)[2]) / 2
    U = bike.potential(phi, delta)
    return np.r_[Mr.ravel(), U, Fg, Fc, G.ravel()]


class LeanLaw:
    """Upright data and lean coefficients Lambda_k of one bicycle."""

    def __init__(self, params: dict, rho=0.2, n=24):
        self.bike = Whipple(params)
        self.lin = Linear(self.bike.p)
        self.g, self.rR, self.m_tot = self.bike.p["g"], self.bike.p["rR"], self.bike.m_tot
        self._taylor(rho, n)

    def _taylor(self, rho, n):
        w = rho * np.exp(2j * np.pi * np.arange(n) / n)
        F = np.empty((n, n, 22), complex)
        done = np.zeros((n, n), bool)
        for a in range(n):
            for b in range(n):
                if done[a, b]:
                    continue
                F[a, b] = _sample(self.bike, w[a], w[b])
                # The model is real, so f(conj z) = conj f(z).
                F[-a % n, -b % n] = F[a, b].conj()
                done[a, b] = done[-a % n, -b % n] = True
        c = np.fft.fft2(F, axes=(0, 1)) / n**2
        blocks = dict(M=slice(0, 9), U=slice(9, 10), Fg=slice(10, 13), Fc=slice(13, 16),
                      G=slice(16, 22))  # fmt: skip

        def coef(name, i, j):
            return (c[i, j][blocks[name]] / rho ** (i + j)).real

        rR = self.rR
        self.M2 = {m: coef("M", *m).reshape(3, 3)[:2, :2] for m in MONO2}
        m1 = [coef("M", 1, 0).reshape(3, 3)[:2, 2], coef("M", 0, 1).reshape(3, 3)[:2, 2]]
        self.P = -np.column_stack(m1) / rR
        n2 = {m: coef("M", *m).reshape(3, 3)[2, 2] / rR**2 for m in MONO2}
        self.N = np.array([[n2[2, 0], n2[1, 1] / 2], [n2[1, 1] / 2, n2[0, 2]]])
        self.U4 = {m: coef("U", *m)[0] for m in MONO4}
        self.c3 = {m: -coef("Fc", *m)[:2] / rR**2 for m in MONO3}
        self.C2 = {m: coef("G", *m).reshape(3, 2)[:2] / rR for m in MONO2}

    def contractions(self, V):
        """Taylor terms evaluated along the direction e = (1, r) of the linear turn at speed V."""
        r = self.lin.steer_ratio(V)
        e = np.array([1.0, r])
        M2e = sum(a * r**j for (i, j), a in self.M2.items())
        C2e = sum(a * r**j for (i, j), a in self.C2.items())
        U4h = sum(a * _hessian(i, j, r) for (i, j), a in self.U4.items())
        grad_U4 = sum(a * _gradient(i, j, r) for (i, j), a in self.U4.items())
        c3j = sum(np.outer(a, _gradient(i, j, r)) for (i, j), a in self.c3.items())
        c3e = sum(a * r**j for (i, j), a in self.c3.items())
        K = self.lin.K(V)
        tau1 = self.lin.tau1(V)
        e2 = (V * V * e @ (self.N - self.lin.K2) @ e + r * tau1) / (2 * self.m_tot * V)
        y_turn = np.array([-K[1, 1] / K[0, 1], 1.0])
        tau3 = y_turn @ (grad_U4 + V * V * c3e)
        return dict(r=r, e=e, M2e=M2e, C2e=C2e, U4h=U4h, c3j=c3j, tau1=tau1, tau3=tau3, e2=e2)

    def modes(self, V):
        """Upright modes (weave, capsize, castor) as (lambda, right vector x, left vector y)."""
        lin = self.lin
        lam, X = np.linalg.eig(lin.state_matrix(V))
        Y = np.linalg.inv(X)
        complex_ = np.flatnonzero(np.abs(lam.imag) > 1e-9)
        real = np.flatnonzero(np.abs(lam.imag) <= 1e-9)
        if len(complex_) == 2:
            order = [complex_[np.argmax(lam[complex_].imag)],
                     real[np.argmax(lam[real].real)], real[np.argmin(lam[real].real)]]  # fmt: skip
        else:
            order = list(np.argsort(-lam.real))[:3]
        Minv = np.linalg.inv(lin.M)
        # Rows of X^-1 times M^-1 give y with y^T (2 lambda M + V C1) x = 1.
        return [(lam[k], X[:2, k], Y[k, 2:] @ Minv) for k in order]

    def terms(self, V):
        """Lambda_k and its six terms, Eqs. (17) and (18), for each upright mode at v_E = V."""
        c = self.contractions(V)
        e, K2, C1 = c["e"], self.lin.K2, self.lin.C1
        Pe, NK2e = self.P @ e, (self.N - K2) @ e
        out = []
        for lam, x, y in self.modes(V):
            dlam_dV = -(y @ (lam * C1 + 2 * V * K2) @ x)
            t = dict(
                inertia=-(lam**2) * (y @ c["M2e"] @ x),
                gyroscopic=-lam * V * (y @ c["C2e"] @ x),
                gravity=-(y @ c["U4h"] @ x),
                centrifugal=-V * V * (y @ c["c3j"] @ x),
                energy_speed=-c["e2"] * dlam_dV,
                spin_coupling=(y @ (lam * Pe + 2 * V * K2 @ e))
                * ((lam * Pe + V * NK2e) @ x)
                / self.m_tot,
            )
            t.update(lam=lam, dlam_dV=dlam_dV, Lambda=sum(t[k] for k in TERMS))
            out.append(t)
        return out

    def Lambda(self, V):
        """Upright eigenvalues and lean coefficients (weave, capsize, castor) at v_E = V."""
        t = self.terms(V)
        return np.array([m["lam"] for m in t]), np.array([m["Lambda"] for m in t])

    def weave_speed(self):
        return brentq(lambda v: self.modes(v)[0][0].real, 3.0, 6.0, xtol=1e-13)

    def cusp(self):
        """Torque cusp at the capsize speed, Eq. (21): tau1', tau3, e2 and beta_E."""
        vc = self.lin.capsize_speed()
        c = self.contractions(vc)
        K = self.lin.K(vc)
        y_turn = np.array([-K[1, 1] / K[0, 1], 1.0])
        tau1p = 2 * vc * (y_turn @ self.lin.K2 @ c["e"])
        return dict(v_c=vc, tau1p=tau1p, tau3=c["tau3"], e2=c["e2"],
                    beta_E=-c["tau3"] / tau1p + c["e2"])  # fmt: skip


def energy_speed(bike: Whipple, turn):
    """Energy-equivalent speed v_E = sqrt(2 E / m_tot) (m/s), with E = 0 upright at rest."""
    E = bike.energy(bike.turn_state(turn)) - bike.potential(0.0, 0.0)
    return np.sqrt(2 * E / bike.m_tot)


def turn_at_energy_speed(bike: Whipple, phi, vE, guess):
    """Steady turn with lean phi and energy-equivalent speed vE."""
    d = newton(lambda d: energy_speed(bike, bike.turn(phi, d)) - vE, guess, tol=1e-14, maxiter=60)
    return bike.turn(phi, d)
