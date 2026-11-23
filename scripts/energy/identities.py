"""Power balance, drive force and speed-feedback polynomial (Sect. 3; supplementary Eq. family)."""

import math

import numpy as np
from common import (
    bench,
    drive_constant,
    field_constant,
    field_feedback,
    quartics,
    quintic,
    save,
    simulate,
    turn_at,
    turn_state,
)

from whipple.model import complex_step

H = 1e-30


def w_vector(bike, z0, T):
    """w = grad_q (m_s S/2 + U - T delta) at fixed S and T, by complex step."""
    S = z0[4] ** 2

    def H_static(p, d):
        return 0.5 * bike.mass_matrix(p, d)[2, 2] * S + bike.potential(p, d) - T * d

    return np.array(
        [H_static(z0[0] + 1j * H, z0[1]).imag / H, H_static(z0[0], z0[1] + 1j * H).imag / H]
    )


def power_identity(bike, rng):
    err = []
    for _ in range(10):
        z = np.r_[
            rng.uniform(-0.4, 0.4), rng.uniform(-0.3, 0.3), rng.normal(0, 1, 2), -rng.uniform(5, 25)
        ]
        _, _, f, _ = bike.parts(z[0], z[1], z[2:5])
        Mdot = bike.mass_matrix(z[0] + 1j * H * z[2], z[1] + 1j * H * z[3]).imag / H
        Uq = (
            np.array(
                [bike.potential(z[0] + 1j * H, z[1]).imag, bike.potential(z[0], z[1] + 1j * H).imag]
            )
            / H
        )
        u = z[2:5]
        lhs, rhs = u @ f, -0.5 * u @ Mdot @ u - Uq @ u[:2]
        err.append(abs(lhs - rhs) / (abs(lhs) + abs(rhs)))
    f3 = []
    for _ in range(10):
        q, s = rng.uniform(-0.4, 0.4, 2), -rng.uniform(5, 25)
        f = bike.parts(q[0], q[1], np.array([0.0, 0.0, s]))[2]
        f3.append(abs(f[2]) / np.abs(f).max())
    return dict(power_identity_relerr_max=max(err), f3_at_zero_rates_rel_max=max(f3))


def drive_order(bike, lin):
    """Largest |Q| over 3 s after lean-rate kicks eps, and the first-order formula
    Q1 = M_3q qdd + (1/s) w . qd (supplementary material, Sect. Constant speed)."""
    out = {}
    for name, V, phi in (
        ("upright", 5.0, 0.0),
        ("turn 8 deg", 5.0, 8.0),
        ("turn 12 deg", 4.6, 12.0),
    ):
        if phi == 0:
            z0, T = np.array([0.0, 0.0, 0.0, 0.0, -V / bike.p["rR"]]), 0.0
        else:
            t = turn_at(bike, lin, math.radians(phi), V)
            z0, T = turn_state(t), t.T
        M = bike.mass_matrix(z0[0], z0[1])
        w = w_vector(bike, z0, T)
        runs = []
        for eps in (0.04, 0.02, 0.01, 0.005):
            sol = simulate(field_constant(bike, T), np.r_[z0 + [0, 0, eps, 0, 0], 0.0], 3.0, 601)
            Q, Q1 = [], []
            for z in sol.y.T:
                q, qdd = drive_constant(bike, z, T)
                Q.append(q)
                Q1.append(M[2, :2] @ qdd + w @ z[2:4] / z0[4])
            Q, Q1 = np.array(Q), np.array(Q1)
            runs.append(
                dict(
                    eps=eps,
                    Q_max=np.abs(Q).max(),
                    first_order_relerr=np.abs(Q - Q1).max() / np.abs(Q).max(),
                )
            )
        ratios = [runs[i]["Q_max"] / runs[i + 1]["Q_max"] for i in range(3)]
        out[name] = dict(V=V, lean_deg=phi, M3q=M[2, :2], w=w, runs=runs, halving_ratios=ratios)
        print(name, "halving ratios", np.round(ratios, 4), "w", np.round(w, 2), flush=True)
    return out


def main():
    bike, lin = bench()
    rng = np.random.default_rng(1)
    out = power_identity(bike, rng)

    t = turn_at(bike, lin, math.radians(10), 5.0)
    z = turn_state(t)

    def f3(x):
        return bike.parts(x[0], x[1], x[2:5])[2][2]

    df3 = complex_step(lambda x: np.atleast_1d(f3(x)), z)[0]
    out["turn 5 m/s 10 deg"] = dict(
        df3_dqd=df3[2:4], minus_w_over_s=-w_vector(bike, z, t.T) / t.s, df3_dq_ds=df3[[0, 1, 4]]
    )
    z2 = z + np.r_[0.01, -0.003, 0.05, -0.1, 0.0]
    Q, _ = drive_constant(bike, z2, t.T)
    sdot = bike.accelerations(z2[0], z2[1], z2[2:5], t.T)[1][2]
    Q2 = -sdot / np.linalg.inv(bike.mass_matrix(z2[0], z2[1]))[2, 2]
    out["drive force two forms"] = [Q, Q2]
    out["drive force at the steady turn (N m)"] = drive_constant(bike, z, t.T)[0]

    a_free, a_const, gpg = quartics(bike, t)
    checks = []
    for G in (0.0, 0.01, 0.3, 3.0, 100.0):
        J = complex_step(lambda x, G=G: field_feedback(bike, t.T, G, t.s)(0, x), z)
        cJ = np.real(np.poly(np.linalg.eigvals(J)))[1:]
        cF = quintic(a_free, a_const, G * gpg)[1:]
        checks.append(
            dict(
                G=G,
                max_abs_diff=np.max(np.abs(cJ - cF)),
                max_rel_diff=np.max(np.abs(cJ - cF) / np.abs(cF[:4]).min()),
            )
        )
    out["quintic against Jacobian"] = checks
    out["m_eff = det M / det M_qq (kg m^2)"] = 1 / gpg
    out["m_tot (kg)"] = bike.m_tot
    out["m_tot rR^2 (kg m^2)"] = bike.m_tot * bike.p["rR"] ** 2
    out["drive order"] = drive_order(bike, lin)
    t8 = turn_at(bike, lin, math.radians(8), 5.0)
    out["w at 5 m/s, 8 deg (J/rad)"] = w_vector(bike, turn_state(t8), t8.T)
    for k, v in out.items():
        if k != "drive order":
            print(k, v)
    save("identities", out)


if __name__ == "__main__":
    main()
