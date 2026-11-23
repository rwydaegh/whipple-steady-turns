"""Independent trim, Jacobian, polynomial and boundary solvers that reuse only the model kernels."""

import math

import numpy as np
from scipy.optimize import brentq, minimize_scalar, root

from whipple.linear import matrices
from whipple.model import Whipple

H_CS = 1e-30  # complex step


def complex_jacobian(f, x, h=H_CS):
    """Jacobian of f at the real point x by complex-step differentiation."""
    x = np.asarray(x, dtype=float)
    cols = []
    for j in range(len(x)):
        y = x.astype(complex)
        y[j] += 1j * h
        cols.append(np.imag(f(y)) / h)
    return np.column_stack(cols)


def bracket(A, B):
    """<A, B> = a11 b22 + a22 b11 - a12 b21 - a21 b12, so det(A + B) = det A + <A, B> + det B."""
    return A[0, 0] * B[1, 1] + A[1, 1] * B[0, 0] - A[0, 1] * B[1, 0] - A[1, 0] * B[0, 1]


def pencil_coefficients(M, C, K):
    """(a1, a2, a3, a4) of det(lambda^2 M + lambda C + K) / det M for 2x2 matrices."""
    return np.array(
        [bracket(M, C), np.linalg.det(C) + bracket(M, K), bracket(C, K), np.linalg.det(K)]
    ) / np.linalg.det(M)


def hurwitz_d3(a):
    a1, a2, a3, a4 = a[:4]
    return a1 * a2 * a3 - a3**2 - a1**2 * a4


# ---- upright linear model ----------------------------------------------------------------------


def upright_coefficients(p, V, k=0.0, c=0.0):
    """Quartic of the upright bicycle with a steer spring k and damper c, from its state matrix."""
    M, C1, K0, K2 = matrices(p)
    g = p.get("g", 9.81)
    K = g * K0 + V * V * K2 + np.diag([0.0, k])
    D = V * C1 + np.diag([0.0, c])
    A = np.block([[np.zeros((2, 2)), np.eye(2)], [-np.linalg.solve(M, K), -np.linalg.solve(M, D)]])
    return np.real(np.poly(A))[1:]


def upright_speeds(p, k=0.0, c=0.0, v_max=40.0):
    """Upright weave speed v_w, capsize speed v_c and the zero V_a3 of a3, by scans and brentq."""
    vs = np.linspace(0.05, v_max, 800)

    def zeros(f):
        vals = [f(v) for v in vs]
        return [
            brentq(f, a, b, xtol=1e-14)
            for a, b, fa, fb in zip(vs[:-1], vs[1:], vals[:-1], vals[1:])
            if fa * fb < 0
        ]

    v_c = zeros(lambda v: upright_coefficients(p, v, k, c)[3])[0]
    weave = [
        v
        for v in zeros(lambda v: hurwitz_d3(upright_coefficients(p, v, k, c)))
        if v < v_c and upright_coefficients(p, v, k, c)[2] > 0
    ]
    v_a3 = zeros(lambda v: upright_coefficients(p, v, k, c)[2])
    return {"v_w": weave[-1] if weave else None, "v_c": v_c, "V_a3": v_a3[0] if v_a3 else None}


def zero_speed_rates(p):
    """Positive real eigenvalues of the linear model at zero speed (falling motions)."""
    M, _, K0, _ = matrices(p)
    lam2 = np.linalg.eigvals(-np.linalg.solve(M, p.get("g", 9.81) * K0))
    return np.sort(np.sqrt(np.abs(lam2.real)))


def composites(p):
    """Composite parameters of Meijaard et al. (2007), App. A, written out again."""
    w, c, lam, g = p["w"], p["c"], p["lam"], p.get("g", 9.81)
    mR, mB, mH, mF = p["mR"], p["mB"], p["mH"], p["mF"]
    rR, rF = p["rR"], p["rF"]
    mT = mR + mB + mH + mF
    xT = (p["xB"] * mB + p["xH"] * mH + w * mF) / mT
    zT = (-rR * mR + p["zB"] * mB + p["zH"] * mH - rF * mF) / mT
    ITxz = p["IBxz"] + p["IHxz"] - mB * p["xB"] * p["zB"] - mH * p["xH"] * p["zH"] + mF * w * rF
    ITzz = p["IRxx"] + p["IBzz"] + p["IHzz"] + p["IFxx"]
    ITzz += mB * p["xB"] ** 2 + mH * p["xH"] ** 2 + mF * w**2
    mA = mH + mF
    xA = (p["xH"] * mH + w * mF) / mA
    zA = (p["zH"] * mH - rF * mF) / mA
    IAxz = p["IHxz"] - mH * (p["xH"] - xA) * (p["zH"] - zA) + mF * (w - xA) * (rF + zA)
    IAzz = p["IHzz"] + p["IFxx"] + mH * (p["xH"] - xA) ** 2 + mF * (w - xA) ** 2
    uA = (xA - w - c) * math.cos(lam) - zA * math.sin(lam)
    IAlz = mA * uA * xA + IAxz * math.sin(lam) + IAzz * math.cos(lam)
    nu = c / w * math.cos(lam)
    SR, SF = p["IRyy"] / rR, p["IFyy"] / rF
    return dict(
        g=g,
        w=w,
        lam=lam,
        mT=mT,
        h=-zT,
        ITxz=ITxz,
        ITzz=ITzz,
        IAlz=IAlz,
        nu=nu,
        SF=SF,
        ST=SR + SF,
        SA=mA * uA + nu * mT * xT,
    )


def spring_constants_closed(p):
    """kappa_c, k0 (eq. rider-kc) and V_a3 (eq. Va3) from the composite parameters."""
    q = composites(p)
    sl, cl, mh = math.sin(q["lam"]), math.cos(q["lam"]), q["mT"] * q["h"]
    kappa_c = cl / q["w"] * (q["SF"] * sl - q["SA"] * q["ST"] / mh)
    k0 = q["g"] * q["SA"] * (sl - q["SA"] / mh)
    num = q["g"] * ((q["IAlz"] + q["nu"] * q["ITzz"]) * mh - q["ITxz"] * q["SA"])
    den = (q["nu"] * q["ST"] + q["SF"] * cl) * (q["ST"] + mh)
    return {"kappa_c": kappa_c, "k0": k0, "V_a3": math.sqrt(num / den)}


def spring_constants_elimination(p):
    """kappa_c and k0 by elimination of the lean from the upright static equations.

    With a spring k on the steer, det(K + k e2 e2^T) = det K + k K_phiphi = 0 gives the stiffness
    k_c(V) = -det K / K_phiphi = k0 - kappa_c V^2, because the first column of K2 is zero.
    """
    _, _, K0, K2 = matrices(p)
    g = p.get("g", 9.81)
    return {"kappa_c": bracket(K0, K2) / K0[0, 0], "k0": -g * np.linalg.det(K0) / K0[0, 0]}


# ---- nonlinear model ---------------------------------------------------------------------------


class Bike:
    """Trim, vector field and quartic of the steady turns of one parameter set."""

    def __init__(self, params):
        self.model = Whipple(params)
        self.p = self.model.p
        self.rR = self.p["rR"]
        _, _, self.K0, self.K2 = matrices(self.p)

    def with_params(self, **changes):
        return Bike({**self.p, **changes})

    # -- steady turns
    def lean_force(self, phi, delta, s):
        return self.model.parts(phi, delta, np.array([0.0, 0.0, s]))[2][0]

    def _newton_steer(self, V, phi, d):
        s = -V / self.rR
        for _ in range(40):
            f = self.lean_force(phi, d, s)
            step = f / (np.imag(self.lean_force(phi, d + 1j * H_CS, s)) / H_CS)
            d -= step
            if not np.isfinite(d):
                return None
            if abs(step) < 1e-15 * max(1.0, abs(d)):
                return float(d)
        return float(d) if abs(self.lean_force(phi, d, s)) < 1e-9 else None

    def steer(self, V, phi, guess=None):
        """Steer of the turn with lean phi and rear-wheel spin speed V: Newton on the lean force,
        started from the linear steer ratio, or by a march in lean if that start fails."""
        K = self.p["g"] * self.K0 + V * V * self.K2
        ratio = -K[0, 0] / K[0, 1]
        d = self._newton_steer(V, phi, ratio * phi if guess is None else guess)
        if d is not None:
            return d
        d, prev = ratio * phi / 12, phi / 12
        for x in np.linspace(1, 12, 12) * phi / 12:
            d = self._newton_steer(V, x, d * x / prev)
            if d is None:
                break
            prev = x
        if d is None:
            raise RuntimeError(f"no steady turn at V={V}, phi={phi}")
        return d

    def static(self, phi, delta):
        """Squared spin S and steer torque T of the turn through (lean, steer)."""
        f0 = self.model.parts(phi, delta, np.zeros(3))[2]
        fc = self.model.parts(phi, delta, np.array([0.0, 0.0, 1.0]))[2] - f0
        S = -f0[0] / fc[0]
        return S, -f0[1] - S * fc[1]

    def trim(self, V, phi, guess=None):
        """State z = (lean, steer, 0, 0, s) and steer torque T of the turn (V, phi)."""
        d = self.steer(V, phi, guess)
        s = -V / self.rR
        T = -self.model.parts(phi, d, np.array([0.0, 0.0, s]))[2][1]
        return np.array([phi, d, 0.0, 0.0, s]), float(T)

    def turn(self, V, phi):
        z, _ = self.trim(V, phi)
        return self.model.turn(z[0], z[1])

    # -- dynamics
    def field(self, z, T, plant="free", k=0.0, c=0.0, delta0=0.0, gain=0.0, s0=0.0):
        """State derivative with steer torque T - k (delta - delta0) - c ddelta and the drive
        force -gain (s - s0) on the spin. Plant "constant" holds the spin (4 states)."""
        _, M, f, _ = self.model.parts(z[0], z[1], z[2:5])
        f = f + np.array([0.0, T - k * (z[1] - delta0) - c * z[3], -gain * (z[4] - s0)])
        if plant == "constant":
            return np.r_[z[2:4], np.linalg.solve(M[:2, :2], f[:2])]
        return np.r_[z[2:4], np.linalg.solve(M, f)]

    def jacobian(self, z, T, plant="free", **control):
        kw = dict(control, delta0=z[1], s0=z[4])
        if plant == "constant":
            return complex_jacobian(lambda x: self.field(np.r_[x, z[4]], T, plant, **kw), z[:4])
        return complex_jacobian(lambda x: self.field(x, T, plant, **kw), z)

    def polynomial(self, V, phi, plant="free", **control):
        """Coefficients after the leading 1 of det(lambda I - J) at the turn (V, phi)."""
        z, T = self.trim(V, phi)
        return np.real(np.poly(self.jacobian(z, T, plant, **control)))[1:]

    def quartic(self, V, phi, plant="free", **control):
        """(a1, a2, a3, a4): for free rolling the zero root of the first integral is removed."""
        return self.polynomial(V, phi, plant, **control)[:4]

    def energy(self, z):
        M = self.model.parts(z[0], z[1], np.zeros(3))[1]
        return 0.5 * z[2:5] @ M @ z[2:5] + self.model.potential(z[0], z[1])

    # -- trim functions of q = (lean, steer) for the fold conditions
    def torque_and_first_integral(self, q):
        S, T = self.static(*q)
        m_s = self.model.parts(q[0], q[1], np.zeros(3))[1][2, 2]
        return np.array([T, 0.5 * m_s * S + self.model.potential(*q) - T * q[1], S])


# ---- identities of the paper -------------------------------------------------------------------


def fold_identity(bike, V, phi):
    """a4 of free rolling and of constant speed against Theorem 1 and its constant-speed form.

    Free rolling:   a4 = -2 F_c,phi / det M * det d(T, H)/d(lean, steer)    (eq. fold)
    Constant speed: a4 = -F_c,phi / det M_qq * det d(T, S)/d(lean, steer)   (Sect. 6.4)
    """
    z, _ = bike.trim(V, phi)
    q = z[:2]
    D = complex_jacobian(bike.torque_and_first_integral, q)  # rows T, H, S
    M = bike.model.parts(q[0], q[1], np.zeros(3))[1]
    f0 = bike.model.parts(q[0], q[1], np.zeros(3))[2]
    fc = bike.model.parts(q[0], q[1], np.array([0.0, 0.0, 1.0]))[2] - f0
    free = -2 * fc[0] / np.linalg.det(M) * np.linalg.det(D[[0, 1]])
    constant = -fc[0] / np.linalg.det(M[:2, :2]) * np.linalg.det(D[[0, 2]])
    return {
        "free": (bike.quartic(V, phi, "free")[3], free),
        "constant": (bike.quartic(V, phi, "constant")[3], constant),
    }


def pencil_matrices(bike, V, phi, plant):
    """M_e, C_e, K_e of the 2x2 pencil of a turn (eq. pencil), from the force kernel."""
    z, _ = bike.trim(V, phi)
    q, s = z[:2], z[4]
    m = bike.model
    _, M, f, _ = m.parts(q[0], q[1], np.array([0.0, 0.0, s]))
    R = complex_jacobian(lambda x: m.parts(x[0], x[1], np.array([0.0, 0.0, s]))[2][:2], q)
    Du = complex_jacobian(lambda u: m.parts(q[0], q[1], u)[2], np.array([0.0, 0.0, s]))
    Cq = Du[:2, :2] / s
    fc = (f - m.parts(q[0], q[1], np.zeros(3))[2])[:2] / s**2
    grad_ms = complex_jacobian(
        lambda x: np.atleast_1d(m.parts(x[0], x[1], np.zeros(3))[1][2, 2]), q
    )
    mv, ms = M[:2, 2], M[2, 2]
    cs = -fc - 0.5 * grad_ms[0]
    if plant == "constant":
        return M[:2, :2], -s * Cq, -R
    Me = M[:2, :2] - np.outer(mv, mv) / ms
    Ce = -s * Cq + s / ms * (np.outer(mv, cs) + 2 * np.outer(fc, mv))
    Ke = -R - 2 * s * s / ms * np.outer(fc, cs)
    return Me, Ce, Ke


def feedback_identity(bike, V, phi, G):
    """Polynomial with the speed feedback Q = -G (s - s0) against lambda chi_H + gamma chi_S."""
    z, T = bike.trim(V, phi)
    p = np.real(np.poly(bike.jacobian(z, T, "free", gain=G)))[1:]
    aH, aS = bike.quartic(V, phi, "free"), bike.quartic(V, phi, "constant")
    M = bike.model.parts(z[0], z[1], np.zeros(3))[1]
    gamma = G * np.linalg.det(M[:2, :2]) / np.linalg.det(M)
    family = np.r_[aH, 0.0] + gamma * np.r_[1.0, aS]
    return p, family


def power_residual(bike, z, T):
    """dE/dt along the free-rolling field minus the steer power T ddelta (eq. power)."""
    grad = complex_jacobian(lambda x: np.atleast_1d(bike.energy(x)), z)[0]
    return grad @ bike.field(z, T) - T * z[3]


# ---- boundary solvers --------------------------------------------------------------------------


def fold_lean(bike, V, plant="free", phi_max=0.7, n=36, **control):
    """Smallest positive lean of the fold a4 = 0 at the speed V, by a scan from upright."""

    def a4(phi):
        return bike.quartic(V, phi, plant, **control)[3]

    phis = np.linspace(0.002, phi_max, n)
    prev = a4(phis[0])
    if prev <= 0:  # not on the stable side of the fold at this speed
        return None
    for lo, hi in zip(phis[:-1], phis[1:]):
        val = a4(hi)
        if prev * val < 0:
            return brentq(a4, lo, hi, xtol=1e-13)
        prev = val
    return None


def bt_point(bike, guess, plant="free", **control):
    """Point (V, phi) with a3 = a4 = 0, the tip of the region, by a Newton-type solve."""
    sol = root(lambda x: bike.quartic(x[0], x[1], plant, **control)[2:4], guess, tol=1e-13)
    a = bike.quartic(*sol.x, plant, **control)
    if np.max(np.abs(a[2:])) > 1e-7:
        raise RuntimeError(f"BT point did not converge: {sol.x}, {a}")
    return sol.x, a


def tip_on_fold(bike, V_lo, V_hi, plant="free", n=12, **control):
    """Tip of the region: the first zero of a3 on the fold, scanned down from V_hi, then refined."""
    vs = np.linspace(V_hi, V_lo, n)
    prev = None
    for v in vs:
        phi = fold_lean(bike, v, plant, **control)
        a3 = None if phi is None else bike.quartic(v, phi, plant, **control)[2]
        if prev is not None and a3 is not None and prev[2] * a3 < 0:
            V0 = brentq(
                lambda x: bike.quartic(x, fold_lean(bike, x, plant, **control), plant, **control)[
                    2
                ],
                v,
                prev[0],
                xtol=1e-6,
            )
            return bt_point(bike, [V0, fold_lean(bike, V0, plant, **control)], plant, **control)
        prev = None if a3 is None else (v, phi, a3)
    raise RuntimeError("no zero of a3 on the fold")


def largest_on_fold(bike, V_lo, V_hi, plant="free", quantity="lean", n=10, **control):
    """Largest lean (or lateral acceleration) on the fold for V_lo <= V <= V_hi."""

    def lean(V):
        return fold_lean(bike, V, plant, **control)

    def value(V):
        phi = lean(V)
        if phi is None:
            return -np.inf
        if quantity == "lean":
            return phi
        return bike.turn(V, phi).ay

    vs = np.linspace(V_lo, V_hi, n)
    vals = [value(v) for v in vs]
    i = int(np.argmax(vals))
    lo, hi = vs[max(i - 1, 0)], vs[min(i + 1, n - 1)]
    opt = minimize_scalar(
        lambda v: -value(v), bounds=(lo, hi), method="bounded", options={"xatol": 1e-8}
    )
    V = float(opt.x)
    return V, lean(V), -float(opt.fun)


def fold_saddle(params, plant, guess, hv=2e-4, hp=2e-5):
    """Trail c and turn (V, phi) where a4 = da4/dV = da4/dphi = 0: the fold curve loses its top."""

    def F(x):
        V, phi, c = x
        b = Bike({**params, "c": c})

        def a4(v, p):
            return b.quartic(v, p, plant)[3]

        return [
            a4(V, phi),
            (a4(V + hv, phi) - a4(V - hv, phi)) / (2 * hv),
            (a4(V, phi + hp) - a4(V, phi - hp)) / (2 * hp),
        ]

    sol = root(F, guess, tol=1e-11)
    if np.max(np.abs(F(sol.x))) > 1e-6:
        raise RuntimeError(f"fold saddle did not converge: {sol.x}")
    return sol.x


def tau3(bike, V, h=0.006):
    """Third-order coefficient of the steady torque T = tau1 phi + tau3 phi^3 + ... at speed V."""
    ph = h * np.arange(1, 5)
    vals = np.array([bike.trim(V, x)[1] / x for x in ph])
    co = np.polynomial.polynomial.polyfit((ph / h) ** 2, vals, 3)
    return co[1] / h**2
