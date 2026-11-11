"""Stability of steady turns: eigenvalues, Hopf and fold curves and the region of stable turns.

Three trims hold a turn:
  "free"    constant steer torque, free rolling (5 states, one zero eigenvalue)
  "held"    constant steer torque, rear-wheel speed held by an ideal servo (4 states)
  "offset"  zero steer torque, rear-frame mass centre offset sideways (Franke et al. 1990)
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq, newton

from .linear import Linear
from .model import COMPLEX_STEP, Turn, Whipple, complex_step

G = 9.81


def offset_trim(bike: Whipple, phi, delta) -> float:
    """Lateral offset W (m) of the rear-frame mass centre that holds the turn at T = 0."""
    return newton(
        lambda W: bike.with_params(yB=W).turn_spin_torque(phi, delta)[1], 0.0, x1=1e-3, tol=1e-13
    )


def trimmed_bike(bike: Whipple, phi, delta, plant="free") -> Whipple:
    """The bicycle with the mass-centre offset of the trim (unchanged unless plant is offset)."""
    if plant == "offset":
        return bike.with_params(yB=offset_trim(bike, phi, delta))
    return bike


def trimmed_turn(bike: Whipple, phi, delta, plant="free") -> Turn | None:
    return trimmed_bike(bike, phi, delta, plant).turn(phi, delta)


def turn_jacobian(bike: Whipple, phi, delta, plant="free"):
    """Jacobian at the steady turn through (lean, steer), and the turn."""
    bike = trimmed_bike(bike, phi, delta, plant)
    turn = bike.turn(phi, delta)
    if turn is None:
        raise ValueError("no forward steady turn through this lean and steer")
    if plant == "held":

        def lateral(x):
            _, Mr, f, _ = bike.parts(x[0], x[1], np.r_[x[2], x[3], turn.s])
            return np.r_[x[2:4], np.linalg.solve(Mr[:2, :2], f[:2] + np.array([0.0, turn.T]))]

        return complex_step(lateral, [phi, delta, 0.0, 0.0]), turn
    T = 0.0 if plant == "offset" else turn.T
    return bike.jacobian(bike.turn_state(turn), T), turn


def characteristic(bike, phi, delta, plant="free"):
    """Coefficients (a1, a2, a3, a4) of the quartic whose roots are the four nonzero eigenvalues."""
    J, _ = turn_jacobian(bike, phi, delta, plant)
    return np.real(np.poly(np.linalg.eigvals(J)))[1:5]


def eigenvalues(bike, phi, delta, plant="free"):
    """The four eigenvalues of the turn other than the zero eigenvalue of the first integral."""
    return np.roots(np.r_[1.0, characteristic(bike, phi, delta, plant)])


def torque_and_first_integral(bike: Whipple, phi, delta):
    """T(q) and H(q) = m_s S / 2 + U - T delta of the turn through q, Eq. (9)."""
    S, T = bike.turn_spin_torque(phi, delta)
    m_s = bike.mass_matrix(phi, delta)[2, 2]
    return np.array([T, 0.5 * m_s * S + bike.potential(phi, delta) - T * delta])


def capsize_formula(bike: Whipple, phi, delta):
    """Right side of Eq. (10): -2 F_c,phi / (m_s det M_eff) det d(T, H)/d(lean, steer)."""
    h = COMPLEX_STEP
    d_phi = torque_and_first_integral(bike, phi + 1j * h, delta).imag / h
    d_delta = torque_and_first_integral(bike, phi, delta + 1j * h).imag / h
    M = bike.mass_matrix(phi, delta)
    m_s = M[2, 2]
    M_eff = M[:2, :2] - np.outer(M[:2, 2], M[:2, 2]) / m_s
    F_c_phi = bike.static_forces(phi, delta)[1][0]
    return (
        -2
        * F_c_phi
        / (m_s * np.linalg.det(M_eff))
        * np.linalg.det(np.column_stack([d_phi, d_delta]))
    )


def hurwitz(a):
    """Hopf test D3 (Hurwitz determinant), fold test a4, and the Hopf frequency squared a3/a1."""
    a1, a2, a3, a4 = a
    return a1 * a2 * a3 - a3**2 - a1**2 * a4, a4, a3 / a1


def delta_for_speed(bike: Whipple, phi, V, guess, plant="free"):
    """Steer of the turn with lean phi and rear-contact speed V (m/s)."""

    def residual(d):
        s2 = trimmed_bike(bike, phi, d, plant).turn_spin_torque(phi, d)[0]
        return bike.p["rR"] * np.sqrt(s2) - V

    return newton(residual, guess, tol=1e-15, maxiter=100)


def gradient(f, x, h=1e-7):
    return np.array([(f(x + h * e) - f(x - h * e)) / (2 * h) for e in np.eye(len(x))])


def newton_system(F, x, h=1e-7, tol=1e-13, maxiter=40):
    """Newton iteration with a central-difference Jacobian."""
    for _ in range(maxiter):
        J = np.column_stack([(F(x + h * e) - F(x - h * e)) / (2 * h) for e in np.eye(len(x))])
        dx = -np.linalg.solve(J, F(x))
        x = x + dx
        if np.max(np.abs(dx)) < tol:
            return x
    raise RuntimeError("Newton iteration did not converge")


def continue_curve(f, x0, direction, stop, ds=1e-3, ds_max=2e-3, ds_min=1e-7, n_max=4000):
    """Points of the curve f(phi, delta) = 0 from x0, by pseudo-arclength continuation."""

    def tangent(x, ref):
        g = gradient(f, x)
        t = np.array([-g[1], g[0]]) / np.hypot(*g)
        return t if t @ ref > 0 else -t

    x, t = x0, tangent(x0, direction)
    points = [x]
    for _ in range(n_max):
        guess = x + ds * t

        def corrector(y, t=t, guess=guess):
            return np.array([f(y), t @ (y - guess)])

        try:
            with np.errstate(invalid="ignore"):
                y = newton_system(corrector, guess, tol=1e-12, maxiter=8)
            ok = np.linalg.norm(y - x) < 2.5 * ds
        except (RuntimeError, ValueError, np.linalg.LinAlgError):
            ok = False
        if not ok:
            ds /= 2
            if ds < ds_min:
                break
            continue
        x, t = y, tangent(y, t)
        points.append(x)
        ds = min(1.3 * ds, ds_max)
        if stop(x):
            break
    return np.array(points)


@dataclass
class Tongue:
    """Boundary of the region of stable turns in the (lean, steer) plane."""

    hopf: np.ndarray  # Hopf curve from upright to the Bogdanov-Takens point
    fold: np.ndarray  # fold curve from upright to the Bogdanov-Takens point
    fold_beyond: np.ndarray  # fold curve beyond the Bogdanov-Takens point
    bt: np.ndarray  # Bogdanov-Takens point (lean, steer)
    v_weave: float
    v_capsize: float


def turn_grid(bike: Whipple, phis, deltas):
    """Speed V, lateral acceleration a_y and steer torque T of turns on a (lean, steer) grid."""
    out = np.full((3, len(phis), len(deltas)), np.nan)
    for i, phi in enumerate(phis):
        for j, delta in enumerate(deltas):
            turn = bike.turn(phi, delta)
            if turn is not None:
                out[:, i, j] = turn.V, turn.ay, turn.T
    return out


def hands_free(bike: Whipple, phi_max=0.6, ds_max=5e-3):
    """Curve of hands-free forward turns (T = 0) from upright running at the capsize speed."""
    lin = Linear(bike.p)
    vc, phi0 = lin.capsize_speed(), 2e-3

    def torque(x):
        return bike.turn_spin_torque(*x)[1]

    def steer(V):
        return delta_for_speed(bike, phi0, V, phi0 * lin.steer_ratio(V))

    V0 = brentq(lambda V: torque([phi0, steer(V)]), vc - 0.05, vc + 0.05, xtol=1e-13)
    x0 = np.array([phi0, steer(V0)])
    curve = continue_curve(
        torque,
        x0,
        np.array([1.0, 0.0]),
        stop=lambda x: x[0] > phi_max or bike.turn(*x) is None,
        ds_max=ds_max,
    )
    return curve[:-1] if bike.turn(*curve[-1]) is None else curve


def bogdanov_takens(bike, guess, plant="free"):
    """Point (lean, steer) where a3 = a4 = 0, the tip of the region of stable turns."""
    return newton_system(lambda x: characteristic(bike, *x, plant)[2:], np.asarray(guess, float))


def tongue(bike: Whipple, plant="free", phi_max=0.36, ds_max=2e-3) -> Tongue:
    """Hopf and fold curves of turns held by the given trim, from upright to the tip and beyond."""
    lin = Linear(bike.p)
    vw, vc = lin.weave_speed(), lin.capsize_speed()

    def hopf_test(x):
        return hurwitz(characteristic(bike, *x, plant))[0]

    def fold_test(x):
        return characteristic(bike, *x, plant)[3]

    phi0 = 2e-3

    def start(test, v):
        def point(V):
            return np.array(
                [phi0, delta_for_speed(bike, phi0, V, phi0 * lin.steer_ratio(V), plant)]
            )

        return point(brentq(lambda V: test(point(V)), v - 0.1, v + 0.1, xtol=1e-12))

    up = np.array([1.0, 0.0])
    hopf = continue_curve(
        hopf_test, start(hopf_test, vw), up,
        stop=lambda x: characteristic(bike, *x, plant)[2] < 0 or x[0] > phi_max, ds_max=ds_max,
    )  # fmt: skip
    a3_up = characteristic(bike, *start(fold_test, vc), plant)[2]
    fold = continue_curve(fold_test, start(fold_test, vc), up, stop=lambda x: x[0] > phi_max,
                          ds_max=ds_max)  # fmt: skip
    a3 = np.array([characteristic(bike, *x, plant)[2] for x in fold])
    k = int(np.flatnonzero(np.sign(a3) != np.sign(a3_up))[0])
    bt = bogdanov_takens(bike, fold[k], plant)
    return Tongue(
        hopf=np.vstack([hopf[:-1], bt]),
        fold=np.vstack([fold[:k], bt]),
        fold_beyond=np.vstack([bt, fold[k:]]),
        bt=bt,
        v_weave=vw,
        v_capsize=vc,
    )


def boundary_extremum(bike, test, quantity, guess, plant="free", h=1e-6):
    """Point on the curve test = 0 where quantity(turn) is stationary along the curve."""

    def q(x):
        return quantity(trimmed_turn(bike, *x, plant))

    def F(x):
        g = gradient(test, x, h)
        t = np.array([-g[1], g[0]])
        return np.array([test(x) / np.hypot(*g), t @ gradient(q, x, h) / np.hypot(*t)])

    return newton_system(F, np.asarray(guess, float), h=h, tol=1e-8)


def largest_stable(bike, t: Tongue, quantity, plant="free"):
    """Largest value of quantity(turn) on the boundary of the region of stable turns."""
    tests = {
        "hopf": lambda x: hurwitz(characteristic(bike, *x, plant))[0],
        "fold": lambda x: characteristic(bike, *x, plant)[3],
    }
    best = None
    for name, curve in (("hopf", t.hopf), ("fold", t.fold)):
        values = [quantity(trimmed_turn(bike, *x, plant)) for x in curve]
        i = int(np.argmax(values))
        x = curve[i]
        if 0 < i < len(curve) - 1:
            x = boundary_extremum(bike, tests[name], quantity, x, plant)
        turn = trimmed_turn(bike, *x, plant)
        if best is None or quantity(turn) > quantity(best):
            best = turn
    return best
