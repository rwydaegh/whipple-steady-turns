"""Linear upright model of Meijaard et al. (2007): M q'' + v C1 q' + (g K0 + v^2 K2) q = f."""

import math

import numpy as np
from scipy.optimize import brentq


def matrices(p: dict):
    """M, C1, K0, K2 of the linear model (Appendix A of Meijaard et al. 2007)."""
    w, c, lam = p["w"], p["c"], p["lam"]
    sl, cl = math.sin(lam), math.cos(lam)
    mR, mB, mH, mF = p["mR"], p["mB"], p["mH"], p["mF"]
    rR, rF, xB, zB, xH, zH = p["rR"], p["rF"], p["xB"], p["zB"], p["xH"], p["zH"]

    mT = mR + mB + mH + mF
    xT = (xB * mB + xH * mH + w * mF) / mT
    zT = (-rR * mR + zB * mB + zH * mH - rF * mF) / mT
    ITxx = (p["IRxx"] + p["IBxx"] + p["IHxx"] + p["IFxx"]
            + mR * rR**2 + mB * zB**2 + mH * zH**2 + mF * rF**2)  # fmt: skip
    ITxz = p["IBxz"] + p["IHxz"] - mB * xB * zB - mH * xH * zH + mF * w * rF
    ITzz = (p["IRxx"] + p["IBzz"] + p["IHzz"] + p["IFxx"]
            + mB * xB**2 + mH * xH**2 + mF * w**2)  # fmt: skip
    mA = mH + mF
    xA = (xH * mH + w * mF) / mA
    zA = (zH * mH - rF * mF) / mA
    IAxx = p["IHxx"] + p["IFxx"] + mH * (zH - zA) ** 2 + mF * (rF + zA) ** 2
    IAxz = p["IHxz"] - mH * (xH - xA) * (zH - zA) + mF * (w - xA) * (rF + zA)
    IAzz = p["IHzz"] + p["IFxx"] + mH * (xH - xA) ** 2 + mF * (w - xA) ** 2
    uA = (xA - w - c) * cl - zA * sl
    IAll = mA * uA**2 + IAxx * sl**2 + 2 * IAxz * sl * cl + IAzz * cl**2
    IAlx = -mA * uA * zA + IAxx * sl + IAxz * cl
    IAlz = mA * uA * xA + IAxz * sl + IAzz * cl
    mu = c / w * cl
    SR, SF = p["IRyy"] / rR, p["IFyy"] / rF
    ST = SR + SF
    SA = mA * uA + mu * mT * xT

    M = np.array([[ITxx, IAlx + mu * ITxz],
                  [IAlx + mu * ITxz, IAll + 2 * mu * IAlz + mu**2 * ITzz]])  # fmt: skip
    C1 = np.array([[0.0, mu * ST + SF * cl + ITxz / w * cl - mu * mT * zT],
                   [-(mu * ST + SF * cl), IAlz / w * cl + mu * (SA + ITzz / w * cl)]])  # fmt: skip
    K0 = np.array([[mT * zT, -SA], [-SA, -SA * sl]])
    K2 = np.array([[0.0, (ST - mT * zT) / w * cl], [0.0, (SA + SF * sl) / w * cl]])
    return M, C1, K0, K2


class Linear:
    """Linear upright model of one bicycle."""

    def __init__(self, p: dict):
        self.g = p.get("g", 9.81)
        self.M, self.C1, self.K0, self.K2 = matrices(p)

    def K(self, v):
        return self.g * self.K0 + v * v * self.K2

    def state_matrix(self, v):
        A = np.zeros((4, 4))
        A[:2, 2:] = np.eye(2)
        A[2:, :2] = -np.linalg.solve(self.M, self.K(v))
        A[2:, 2:] = -v * np.linalg.solve(self.M, self.C1)
        return A

    def eigenvalues(self, v):
        return np.linalg.eigvals(self.state_matrix(v))

    def weave_speed(self, bracket=(3.0, 6.0)):
        """Speed (m/s) where the weave pair crosses the imaginary axis."""

        def growth(v):
            lam = self.eigenvalues(v)
            return lam[np.abs(lam.imag) > 1e-9].real.max()

        return brentq(growth, *bracket, xtol=1e-14)

    def capsize_speed(self):
        """Speed (m/s) where det(g K0 + v^2 K2) = 0. The first column of K2 is zero."""
        K0, K2 = self.K0, self.K2
        return math.sqrt(-self.g * np.linalg.det(K0) / (K0[0, 0] * K2[1, 1] - K0[1, 0] * K2[0, 1]))

    def steer_ratio(self, v):
        """Steer per unit lean of the linear steady turn, r = -K_phiphi / K_phidelta."""
        K = self.K(v)
        return -K[0, 0] / K[0, 1]

    def tau1(self, v):
        """Linear steady torque per unit lean, Eq. (7): tau1 = -det K / K_phidelta."""
        K = self.K(v)
        return -np.linalg.det(K) / K[0, 1]
