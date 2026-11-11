"""Whipple bicycle with tyre side slip: steady turns at constant steer torque and their eigenvalues.

The frames and the wheels are rigid, the vertical contact is holonomic and the wheels do not slip
along their ground line. Each tyre gives a side force N f(alpha, gamma) and a vertical moment
N m(alpha, gamma), with N the normal load, gamma the camber angle and alpha a slip angle with
relaxation length sigma (Magic Formula tyre of a 47-622 city-bicycle tyre). The state is
y = (lean, steer, vy, wy, wp, wd, wr, aR, aF): lateral velocity of the rear-wheel centre, yaw rate,
lean rate, steer rate, spin of the rear wheel and the two slip angles. The kinematic kernels are
those of whipple.model. Axes as in Meijaard et al. (2007).
"""

import numpy as np

from .bicycles import BENCHMARK
from .model import Whipple, _call, complex_step

EZ = np.array([0.0, 0.0, 1.0])
KNOWN = [1, 3, 4, 6, 7]  # velocities given by the state: vy, wy, wp, wd, wr
UNKNOWN = [0, 2, 5, 8]  # velocities fixed by the contact constraints
HELD = [0, 1, 2, 3, 4, 5, 7, 8]  # states that remain when the rear-wheel spin is held


def _rot(axis, a):
    c, s = np.cos(a), np.sin(a)
    if axis == 0:
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == 1:
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


class Tyre:
    """Side force f = N-normalised force and vertical moment m = N-normalised moment (m).

    Magic Formula: f = D sin(C atan(Ba - E (Ba - atan Ba)) + Cc atan(Bc gamma)),
    m = -tp D sin(C atan(...)) + mg gamma. With stiff=True the tyre is linear, f = Ca alpha +
    Cg gamma and m = -tp Ca alpha + mg gamma.
    """

    def __init__(
        self,
        B=10.39,
        C=1.5625,
        D=0.892,
        E=-0.442,
        Cc=0.6,
        Cg=0.86,
        tp=0.015,
        mg=0.01,
        sigma=0.075,
        stiff=None,
    ):
        self.B, self.C, self.D, self.E, self.Cc, self.Cg = B, C, D, E, Cc, Cg
        self.tp, self.mg, self.sigma, self.stiff = tp, mg, sigma, stiff
        self.Bc = Cg / (D * Cc)

    def __call__(self, a, g):
        if self.stiff:
            fa = self.stiff * a
            return fa + self.Cg * g, -self.tp * fa + self.mg * g
        Ba = self.B * a
        phase = self.C * np.arctan(Ba - self.E * (Ba - np.arctan(Ba)))
        f = self.D * np.sin(phase + self.Cc * np.arctan(self.Bc * g))
        return f, -self.tp * self.D * np.sin(phase) + self.mg * g


class TyreBike(Whipple):
    """Whipple bicycle on two tyres. The speed is the rear-contact path speed V."""

    def __init__(self, params=BENCHMARK, tyre_rear=None, tyre_front=None):
        super().__init__(params)
        self.tR = tyre_rear or Tyre()
        self.tF = tyre_front or self.tR

    @staticmethod
    def _ground(n):
        s = np.sqrt(n[0] * n[0] + n[1] * n[1])
        return np.array([n[1], -n[0], 0.0 * n[0]]) / s, np.array([n[0], n[1], 0.0 * n[0]]) / s, s

    def _setup(self, phi, delta):
        theta, A, jv, jw, inertia, _, _, _ = self._geometry(phi, delta)
        R = _rot(0, phi) @ _rot(1, theta)
        lam = self.p["lam"]
        H = R @ (_rot(1, lam) @ _rot(2, delta) @ _rot(1, -lam))
        nR, nF = R[:, 1], H[:, 1]
        tR, lR, sR = self._ground(nR)
        tF, lF, sF = self._ground(nF)
        Ac = np.vstack((tR @ A[:3], EZ @ A[:3], EZ @ A[3:], tF @ A[3:]))
        return dict(
            theta=theta,
            A=A,
            jv=jv,
            jw=jw,
            inertia=inertia,
            Ac=Ac,
            nR=nR,
            nF=nF,
            tR=tR,
            lR=lR,
            sR=sR,
            tF=tF,
            lF=lF,
            sF=sF,
        )

    def rhs(self, y, Td=0.0, Ts=0.0, full=False, hold=False):
        """Time derivative of the state. With hold=True the rear-wheel spin is held by the drive
        torque, which is then an output."""
        phi, delta, aR, aF = y[0], y[1], y[7], y[8]
        k = self._setup(phi, delta)
        jv, jw, inertia, Ac = k["jv"], k["jw"], k["inertia"], k["Ac"]
        qd = np.zeros(9, dtype=np.result_type(Ac, y))
        qd[KNOWN] = y[2:7]
        qd[UNKNOWN] = -np.linalg.solve(Ac[:, UNKNOWN], Ac[:, KNOWN] @ y[2:7])
        b = _call("bias", phi, k["theta"], delta, qd, self.par)
        av, aw, abias = b[:12].reshape(4, 3), b[12:24].reshape(4, 3), b[24:]
        omega = np.einsum("bki,i->bk", jw, qd)
        M = np.einsum("bki,b,bkj->ij", jv, self.mass, jv) + np.einsum(
            "bki,bkl,blj->ij", jw, inertia, jw
        )
        h = np.einsum("bki,b,bk->i", jv, self.mass, av) + np.einsum(
            "bki,bk->i",
            jw,
            np.einsum("bij,bj->bi", inertia, aw)
            + np.cross(omega, np.einsum("bij,bj->bi", inertia, omega)),
        )
        F = self.p["g"] * np.einsum("bi,b->i", jv[:, 2, :], self.mass)
        F = F.astype(np.result_type(F, Ts, Td))
        F[6] += Ts
        F[7] += Td
        A = k["A"]
        vcR, vcF = A[:3] @ qd, A[3:] @ qd
        vsR, vsF = k["lR"] @ vcR, k["lF"] @ vcF
        VR, VF = k["tR"] @ (jv[0] @ qd), k["tF"] @ (jv[3] @ qd)
        gR, gF = np.arcsin(k["nR"][2]), np.arcsin(k["nF"][2])
        fR, mR = self.tR(aR, gR)
        fF, mF = self.tF(aF, gF)
        GR = A[:3].T @ k["lR"] * fR + jw[0].T @ EZ * mR
        GF = A[3:].T @ k["lF"] * fF + jw[3].T @ EZ * mF

        def tdot(n, rate, s, t):
            nd = np.cross(rate, n)
            ud = np.array([nd[1], -nd[0], 0.0 * nd[0]])
            return (ud - t * (t @ ud)) / s

        tdR = tdot(k["nR"], omega[1], k["sR"], k["tR"])
        tdF = tdot(k["nF"], omega[2], k["sF"], k["tF"])
        c = np.array([k["tR"] @ abias[0:3] + tdR @ vcR, abias[2], abias[5],
                      k["tF"] @ abias[3:6] + tdF @ vcF])  # fmt: skip
        Gm = np.zeros((9, 4), dtype=np.result_type(GR, Ac))
        Gm[:, 1], Gm[:, 2] = GR, GF
        n = 14 if hold else 13
        K = np.zeros((n, n), dtype=np.result_type(M, Gm, Ac))
        K[:9, :9] = M
        K[:9, 9:13] = -(Ac.T - Gm)
        K[9:13, :9] = Ac
        r = np.r_[F - h, -c]
        if hold:
            r[7] -= Td
            K[7, 13] = -1.0
            K[13, 7] = 1.0
            r = np.r_[r, 0.0]
        sol = np.linalg.solve(K, r)
        qdd, mu = sol[:9], sol[9:13]
        wy = qd[3]
        dy = np.array([qd[4], qd[6], qdd[1] - wy * qd[0], qdd[3], qdd[4], qdd[6], qdd[7],
                       VR / self.tR.sigma * (np.arctan(-vsR / VR) - aR),
                       VF / self.tF.sigma * (np.arctan(-vsF / VF) - aF)])  # fmt: skip
        if not full:
            return dy
        return dict(dy=dy, qd=qd, NR=-mu[1], NF=-mu[2], Td=sol[13] if hold else Td, k=k)

    def jacobian(self, y, Td=0.0, Ts=0.0, hold=False):
        """Jacobian of the state equations, 9x9, or 8x8 without the spin if hold=True."""
        J = complex_step(lambda x: self.rhs(x, Td, Ts, hold=hold), y)
        return J[np.ix_(HELD, HELD)] if hold else J

    def upright(self, V):
        """Rear-wheel spin of upright running at the rear-centre speed V."""
        k = self._setup(0.0, 0.0)
        u = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
        return V / -np.linalg.solve(k["Ac"][:, UNKNOWN], k["Ac"][:, KNOWN] @ u)[0]

    # ------------------------------------------------------------------ steady turns
    @staticmethod
    def _state(phi, wr, x):
        return np.array([phi, x[0], x[1], x[2], 0.0 * x[0], 0.0 * x[0], wr, x[3], x[4]])

    def _residual(self, phi, wr, x):
        return self.rhs(self._state(phi, wr, x), Td=x[6], Ts=x[5])[2:]

    def solve_turn(self, phi, wr, x0, tol=1e-12, maxiter=40):
        """Steady turn at lean phi and rear spin wr. Unknowns x = (steer, vy, wy, aR, aF, T, Td).
        Returns x and the residual."""
        x = np.array(x0, float)
        for _ in range(maxiter):
            J = complex_step(lambda v: self._residual(phi, wr, v), x)
            dx = -np.linalg.solve(J, self._residual(phi, wr, x))
            s = min(1.0, 0.05 / max(1e-16, np.max(np.abs(dx[:5]))))
            x = x + s * dx
            if np.max(np.abs(dx)) < tol and s == 1.0:
                break
        return x, float(np.max(np.abs(self._residual(phi, wr, x))))

    def turn_speed(self, phi, wr, x):
        """Rear-contact path speed V, radius R and lateral acceleration of the steady turn."""
        d = self.rhs(self._state(phi, wr, x), Td=x[6], Ts=x[5], full=True)
        k, qd = d["k"], d["qd"]
        dn = np.cross(k["tR"], k["nR"])
        dn = dn / np.linalg.norm(dn) * (1 if dn[2] > 0 else -1)
        v = np.array([qd[0], qd[1], 0.0]) + np.cross([0.0, 0.0, qd[3]], self.p["rR"] * dn)
        V = float(np.hypot(v[0], v[1]))
        return dict(V=V, R=V / abs(qd[3]), ay=V * abs(qd[3]), NR=float(d["NR"]), NF=float(d["NF"]))

    def turn_at_speed(self, phi, V, x0, wr0):
        """Steady turn with rear-contact speed V at lean phi. Returns wr, x, residual."""
        wr, x = wr0, x0
        for _ in range(30):
            x, _ = self.solve_turn(phi, wr, x)
            new = wr * V / self.turn_speed(phi, wr, x)["V"]
            done = abs(new - wr) < 1e-13 * abs(wr)
            wr = new
            if done:
                break
        x, res = self.solve_turn(phi, wr, x)
        return wr, x, res

    def turn_eigenvalues(self, phi, wr, x):
        """Eigenvalues of the steady turn at held rear-wheel speed (8x8)."""
        return np.linalg.eigvals(self.jacobian(self._state(phi, wr, x), x[6], x[5], hold=True))
