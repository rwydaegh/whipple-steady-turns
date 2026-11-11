"""Virtual rider: LQR steer torque with yaw-rate integral action and a PI drive torque on speed.

The gains come from the upright linearisation at the target speed only. Integral action drives the
rear-contact speed to V and the yaw rate to V / R, so a rest point of the closed loop is a steady
turn of the bicycle with that speed and radius, whatever the gains.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_are

from .model import COMPLEX_STEP, Whipple

# Bryson weights: lean 0.2 rad, steer 0.1 rad, rates 1 rad/s, yaw-rate integral 0.05 rad,
# torque 5 N m.
WEIGHTS = dict(phi=0.2, delta=0.1, rate=1.0, zr=0.05, T=5.0)


def lateral_linearisation(bike: Whipple, s):
    """A (4x4), torque column b and yaw-rate row c of the upright lateral dynamics at spin s."""
    A, c = np.empty((4, 4)), np.empty(4)
    for j in range(4):
        x = np.zeros(4, complex)
        x[j] = 1j * COMPLEX_STEP
        qd, ud = bike.accelerations(x[0], x[1], np.r_[x[2], x[3], s])
        A[:, j] = np.r_[x[2:4], ud[:2]].imag / COMPLEX_STEP
        c[j] = qd[3].imag / COMPLEX_STEP
    b = np.r_[0.0, 0.0, np.linalg.solve(bike.mass_matrix(0.0, 0.0), [0.0, 1.0, 0.0])[:2]]
    return A, b, c


class Rider:
    """Closed loop of the bicycle and the virtual rider for a turn entry at speed V (m/s) into
    radius R (m), with the yaw-rate reference ramped over t_ramp (s)."""

    def __init__(self, bike: Whipple, V, R, t_ramp=1.0, speed_bandwidth=1.5):
        self.bike, self.V, self.t_ramp = bike, V, t_ramp
        self.yaw_rate = V / R
        p = bike.p
        A, b, c = lateral_linearisation(bike, -V / p["rR"])
        Aa = np.zeros((5, 5))
        Aa[:4, :4], Aa[4, :4] = A, c
        Ba = np.r_[b, 0.0][:, None]
        w = WEIGHTS
        Q = np.diag(np.array([w["phi"], w["delta"], w["rate"], w["rate"], w["zr"]]) ** -2.0)
        Rw = np.array([[w["T"] ** -2]])
        self.K = np.linalg.solve(Rw, Ba.T @ solve_continuous_are(Aa, Ba, Q, Rw))[0]
        # PI on speed with natural frequency speed_bandwidth and damping ratio 1.
        self.kp = 2 * speed_bandwidth * bike.m_tot * p["rR"]
        self.ki = speed_bandwidth**2 * bike.m_tot * p["rR"]

    def reference(self, t):
        """Yaw-rate reference (rad/s): a quintic ramp from 0 to V / R."""
        x = min(max(t / self.t_ramp, 0.0), 1.0)
        return self.yaw_rate * x**3 * (10 - 15 * x + 6 * x * x)

    def control(self, y, qd):
        """Steer torque T, drive torque tau, rear-contact speed and yaw rate at state y."""
        T = -self.K @ np.r_[y[:4], y[5]]
        speed = qd[0] + qd[3] * self.bike.p["rR"] * np.sin(y[0])
        tau = -self.kp * (speed - self.V) - self.ki * y[6]
        return T, tau, speed, qd[3]

    def rhs(self, t, y):
        """dy/dt for y = (lean, steer, lean rate, steer rate, s, yaw-rate and speed integrals)."""
        qd, Mr, f, _ = self.bike.parts(y[0], y[1], y[2:5])
        T, tau, speed, yaw_rate = self.control(y, qd)
        ud = np.linalg.solve(Mr, f + np.array([0.0, T, -tau]))
        return np.r_[y[2:4], ud, yaw_rate - self.reference(t), speed - self.V]

    def run(self, t_end, n_out):
        """Time t (s), states Y and steer torque T (N m) from straight running at speed V."""
        y0 = np.zeros(7)
        y0[4] = -self.V / self.bike.p["rR"]
        sol = solve_ivp(self.rhs, (0, t_end), y0, method="DOP853", rtol=1e-12, atol=1e-13,
                        dense_output=True)  # fmt: skip
        t = np.linspace(0, t_end, n_out)
        Y = sol.sol(t).T
        T = np.array([self.control(y, self.bike.parts(y[0], y[1], y[2:5])[0])[0] for y in Y])
        return t, Y, T
