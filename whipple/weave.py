"""Weave beyond the Hopf curve at constant steer torque: the 4-state system on a level set of the
first integral H = E - T delta, the first Lyapunov coefficient, and the period doubling of the weave
cycle."""

import warnings

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, newton

from .model import COMPLEX_STEP, Whipple
from .stability import characteristic, hurwitz


class Leaf:
    """Lean and steer dynamics on the level set H = E - T delta, with the spin eliminated."""

    def __init__(self, bike: Whipple, T, H, forward=True):
        self.bike, self.T, self.H = bike, T, H
        self.sign = -1.0 if forward else 1.0

    def spin(self, y):
        """Rear-wheel spin s on the level set at y = (lean, steer, lean rate, steer rate)."""
        b = self.bike
        M = b.mass_matrix(y[0], y[1])
        rates = y[2:4]
        a = 0.5 * M[2, 2]
        bb = M[2, :2] @ rates
        c = 0.5 * rates @ M[:2, :2] @ rates + b.potential(y[0], y[1]) - self.T * y[1] - self.H
        return (-bb + self.sign * np.sqrt(bb * bb - 4 * a * c)) / (2 * a)

    def state(self, y):
        return np.r_[y, self.spin(y)]

    def field(self, y):
        z = self.state(y)
        return self.bike.field(z, self.T)[:4]

    def derivative(self, y, u):
        """Directional derivative Df(y) u by complex step."""
        y = np.asarray(y, complex) + 1j * COMPLEX_STEP * np.asarray(u, float)
        return self.field(y).imag / COMPLEX_STEP

    def jacobian(self, y):
        return np.column_stack([self.derivative(y, e) for e in np.eye(4)])

    def rhs(self, t, z):
        """Right side of the full 5-state system, for time integration."""
        return self.bike.field(z, self.T)


def leaf_at_turn(bike: Whipple, phi, delta):
    """Level set through the steady turn (lean, steer) and the turn state y0."""
    turn = bike.turn(phi, delta)
    H = bike.energy(bike.turn_state(turn)) - turn.T * delta
    return Leaf(bike, turn.T, H), np.array([phi, delta, 0.0, 0.0])


def upright_leaf(bike: Whipple, V):
    """Level set through upright running at rear-contact speed V (m/s), and the state y0 = 0."""
    z = np.array([0.0, 0.0, 0.0, 0.0, -V / bike.p["rR"]])
    return Leaf(bike, 0.0, bike.energy(z)), np.zeros(4)


def first_lyapunov(leaf: Leaf, y0, steps=(0.004, 0.002, 0.001)):
    """First Lyapunov coefficient at a Hopf point (Kuznetsov, Eq. 5.62), with |q| = 1, p^H q = 1."""
    A = leaf.jacobian(y0)
    w, V = np.linalg.eig(A)
    c = np.flatnonzero(w.imag > 1e-8)
    j = c[np.argmin(np.abs(w[c].real))]
    omega = w[j].imag
    q = V[:, j] / np.linalg.norm(V[:, j])
    wl, Vl = np.linalg.eig(A.T)
    p = Vl[:, np.argmin(np.abs(wl - np.conj(w[j])))]
    p = p / np.conj(np.vdot(p, q))

    def B2(u, h):
        return (leaf.derivative(y0 + h * u, u) - leaf.derivative(y0 - h * u, u)) / (2 * h)

    def C3(u, h):
        return (leaf.derivative(y0 + h * u, u) + leaf.derivative(y0 - h * u, u) - 2 * A @ u) / h**2

    def richardson(form, u, degree):
        n = np.linalg.norm(u)
        if n == 0:
            return np.zeros(4)
        v = [form(u / n, h) for h in steps]
        r1 = [(4 * v[k + 1] - v[k]) / 3 for k in range(2)]
        return (16 * r1[1] - r1[0]) / 15 * n**degree

    def B(u, v):
        return (richardson(B2, u + v, 2) - richardson(B2, u - v, 2)) / 4

    def Bc(u, v):
        return B(u.real, v.real) - B(u.imag, v.imag) + 1j * (B(u.real, v.imag) + B(u.imag, v.real))

    x, y = q.real, q.imag
    xxx, yyy = richardson(C3, x, 3), richardson(C3, y, 3)
    plus, minus = richardson(C3, x + y, 3), richardson(C3, x - y, 3)
    xyy, xxy = (plus + minus - 2 * xxx) / 6, (plus - minus - 2 * yyy) / 6
    Cqqq = xxx + xyy + 1j * (xxy + yyy)
    h11 = np.linalg.solve(A, Bc(q, np.conj(q)).real)
    h20 = np.linalg.solve(2j * omega * np.eye(4) - A, Bc(q, q))
    g = (
        np.vdot(p, Cqqq)
        - 2 * np.vdot(p, Bc(q, h11.astype(complex)))
        + np.vdot(p, Bc(np.conj(q), h20))
    )
    return g.real / (2 * omega), omega


class TorqueFamily:
    """Steady turns with one steer torque T, parametrised by lean."""

    def __init__(self, bike: Whipple, T, phi, delta):
        self.bike, self.T = bike, T
        self._last = (phi, delta)

    def steer(self, phi):
        p0, d0 = self._last

        def residual(d):
            return self.bike.turn_spin_torque(phi, d)[1] - self.T

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # tolerance reached on a flat residual
            d, info = newton(residual, d0 * phi / p0, tol=1e-14, maxiter=60, full_output=True,
                             disp=False)  # fmt: skip
        if not info.converged and abs(residual(d)) > 1e-10:
            raise RuntimeError(f"steady turn at lean {phi} not found")
        self._last = (phi, d)
        return d

    def turn(self, phi):
        return self.bike.turn(phi, self.steer(phi))

    def H(self, phi):
        t = self.turn(phi)
        return self.bike.energy(self.bike.turn_state(t)) - self.T * t.delta

    def hopf(self, lo, hi):
        """Lean of the Hopf point on the family, between lo and hi."""
        return brentq(lambda p: hurwitz(characteristic(self.bike, p, self.steer(p)))[0], lo, hi,
                      xtol=1e-13)  # fmt: skip

    def fold(self, lo, hi):
        """Lean and level of the maximum of H on the family (the fold)."""
        h = 1e-5
        p = brentq(lambda p: self.H(p + h) - self.H(p - h), lo, hi, xtol=1e-12)
        return p, self.H(p)

    def lean_on_level(self, H, lo, hi):
        return brentq(lambda p: self.H(p) - H, lo, hi, xtol=1e-13)


def poincare_map(leaf: Leaf, phi_s, x, t_max=60.0):
    """Return to the section lean = phi_s (lean decreasing) from (steer, lean rate, steer rate)."""
    z0 = leaf.state(np.r_[phi_s, x])

    def section(t, z):
        return z[0] - phi_s if t > 0.2 else -1.0

    section.terminal, section.direction = True, -1
    sol = solve_ivp(leaf.rhs, (0, t_max), z0, method="DOP853", rtol=1e-11, atol=1e-13,
                    events=section, max_step=0.05)  # fmt: skip
    if not len(sol.t_events[0]):
        raise RuntimeError("orbit did not return to the section")
    return sol.y_events[0][0][1:4], sol.t_events[0][0]


def periodic_orbit(leaf: Leaf, phi_s, x, tol=1e-10, maxiter=12):
    """Fixed point of the Poincare map, its period and Floquet multipliers."""
    for _ in range(maxiter):
        fx, period = poincare_map(leaf, phi_s, x)
        J = np.empty((3, 3))
        for j in range(3):
            h = 1e-7 * max(1.0, abs(x[j]))
            xp = x.copy()
            xp[j] += h
            J[:, j] = (poincare_map(leaf, phi_s, xp)[0] - fx) / h
        if np.max(np.abs(fx - x)) < tol:
            return x, period, np.linalg.eigvals(J)
        x = x - np.linalg.solve(J - np.eye(3), fx - x)
    raise RuntimeError("periodic orbit not found")


def settle(leaf: Leaf, phi_s, y0, t_end=400.0):
    """Point on the section after a long simulation from y0 (lands on the stable weave cycle)."""

    def section(t, z):
        return z[0] - phi_s

    section.direction = -1
    sol = solve_ivp(leaf.rhs, (0, t_end), leaf.state(y0), method="DOP853", rtol=1e-10,
                    atol=1e-12, events=section, max_step=0.05)  # fmt: skip
    return sol.y_events[0][-1][1:4]


def _min_real(mu):
    real = mu[np.abs(mu.imag) < 1e-9].real
    return real.min() if len(real) else 0.0


def period_doubling(bike: Whipple, T, hopf_guess, start=0.25, step=0.05, settle_time=200.0):
    """Level H where the weave cycle at steer torque T has a Floquet multiplier of -1.

    The level is lowered from the Hopf level in units of W = H_fold - H_Hopf. The Poincare
    section is the lean of the steady turn inside the weave cycle.
    """
    fam = TorqueFamily(bike, T, *hopf_guess)
    phi_h = fam.hopf(hopf_guess[0] - 0.02, hopf_guess[0] + 0.02)
    H_h = fam.H(phi_h)
    _, H_f = fam.fold(phi_h, phi_h + 0.1)
    W = H_f - H_h

    def multiplier(H, x):
        phi_s = fam.lean_on_level(H, phi_h - 0.1, phi_h)
        x, period, mu = periodic_orbit(Leaf(bike, T, H), phi_s, x)
        return x, period, mu, _min_real(mu) + 1

    H = H_h - start * W
    phi_s = fam.lean_on_level(H, phi_h - 0.1, phi_h)
    y0 = np.array([phi_s, fam.steer(phi_s), 0.01, 0.0])
    x = settle(Leaf(bike, T, H), phi_s, y0, settle_time)
    prev = None
    while True:
        x, period, mu, m = multiplier(H, x)
        if m < 0:
            break
        prev = (H, m, x)
        H -= step * W
    (Ha, ma, _), Hb, mb = prev, H, m
    for _ in range(20):
        Hc = Hb - mb * (Hb - Ha) / (mb - ma)
        x, period, mu, mc = multiplier(Hc, x)
        Ha, ma, Hb, mb = Hb, mb, Hc, mc
        if abs(mb) < 1e-8:
            break
    phi_s = fam.lean_on_level(Hb, phi_h - 0.1, phi_h)
    return dict(H=Hb, H_hopf=H_h, H_fold=H_f, turn=fam.turn(phi_s), hopf_turn=fam.turn(phi_h),
                period=period, multipliers=mu)  # fmt: skip


def weave_start(fam: TorqueFamily, H, phi_hopf, phi_fold):
    """Start of a weave run on level H: the weaving (focus) turn with a roll rate of 0.01 rad/s.

    Returns the state and the lean of the saddle turn on the level. The weave falls when the lean
    passes the saddle lean by more than 0.3 rad.
    """
    phi = fam.lean_on_level(H, phi_hopf - 0.1, phi_hopf)
    saddle = fam.lean_on_level(H, phi_fold, phi_fold + 0.075)
    return np.array([phi, fam.steer(phi), 0.01, 0.0]), saddle


def weave_run(leaf: Leaf, y0, lean_limit, t_trans=100.0, t_run=100.0, dt=0.5):
    """Lean maxima and the largest Lyapunov exponent chi of the weave on one level.

    Transient t_trans, then t_run with one tangent vector (complex step, Benettin), renormalised
    every dt. Returns the status ('bounded' or 'fall'), the lean maxima (rad) and chi (1/s).
    """

    def field(t, x):
        return np.r_[leaf.field(x[:4]), leaf.derivative(x[:4], x[4:])]

    def fall(t, x):
        return min(lean_limit - x[0], x[0] + 0.6)

    def peak(t, x):
        return x[2]

    fall.terminal, peak.direction = True, -1
    kw = dict(method="DOP853", rtol=1e-9, atol=1e-11, max_step=0.1, events=(fall, peak))
    x, S, maxima = np.r_[y0, 0.5, -0.5, 0.5, 0.5], 0.0, []
    for k in range(int(round((t_trans + t_run) / dt))):
        sol = solve_ivp(field, (0, dt), x, **kw)
        if len(sol.t_events[0]):
            return dict(status="fall", maxima=np.array(maxima), chi=np.nan)
        x = sol.y[:, -1]
        n = np.linalg.norm(x[4:])
        x[4:] /= n
        if k * dt >= t_trans:
            S += np.log(n)
            maxima += [e[0] for e in sol.y_events[1]]
    return dict(status="bounded", maxima=np.array(maxima), chi=S / t_run)
