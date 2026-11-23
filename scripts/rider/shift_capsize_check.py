"""Capsize condition with a mass shift tested at six turns (supplement, "Rider models")."""

import json
import math
from pathlib import Path

import numpy as np
from arms_model import pencils
from scipy.optimize import newton
from shift_model import Whipple, delta_for_speed, params_of, quartic

from whipple import stability as st

REPO = Path(__file__).resolve().parents[2]
CASES = [
    (4.5, 0.20, 0.0),
    (4.5, 0.20, 0.02),
    (4.5, 0.18, -0.03),
    (3.0, 0.25, 0.06),
    (5.8, 0.10, 0.015),
    (4.7, 0.24, 0.01),
]  # (V, phi, yB)


def main():
    p = params_of("benchmark")
    b = Whipple(p)

    def T_at(phi, V, yB, guess):
        bb = b.with_params(yB=yB)
        d = delta_for_speed(bb, phi, V, guess)
        return bb.turn(phi, d).T

    def yB_at(phi, delta, V, guess):
        def r(W):
            s2 = b.with_params(yB=W).turn_spin_torque(phi, delta)[0]
            return p["rR"] * math.sqrt(max(s2, 0)) - V

        return newton(r, guess, x1=guess + 1e-4, tol=1e-14, maxiter=80)

    def froll(phi, delta, s, yB):
        return b.with_params(yB=yB).parts(phi, delta, np.array([0.0, 0.0, s]))[2][0]

    rows = []
    for V, phi, yB in CASES:
        bb = b.with_params(yB=yB)
        delta = delta_for_speed(bb, phi, V, 0.981 * phi / V**2)
        turn = bb.turn(phi, delta)
        Me, _, Ke = pencils(bb, phi, delta)[0]["constant"]
        a_c = quartic(bb, phi, delta, "constant")
        a_f = quartic(bb, phi, delta, "free")
        # first form: -Ke_pd dT/dphi at fixed V and yB (central differences, Richardson)
        est = []
        for h in (1e-3, 5e-4, 2.5e-4):
            est.append((T_at(phi + h, V, yB, delta) - T_at(phi - h, V, yB, delta)) / (2 * h))
        dTdphi = est[-1] + (est[-1] - est[-2]) / 3
        form1 = -Ke[0, 1] * dTdphi / np.linalg.det(Me)
        # second form: -(dF_phi/dyB) det d(T, yB)/d(phi, delta) at fixed V
        J = np.zeros((2, 2))
        hh = 2e-4
        for j, e in enumerate(np.eye(2)):
            vals = []
            for sgn in (1, -1):
                q = np.array([phi, delta]) + sgn * hh * e
                W = yB_at(q[0], q[1], V, yB)
                vals.append([b.with_params(yB=W).turn_spin_torque(*q)[1], W])
            J[:, j] = (np.array(vals[0]) - np.array(vals[1])) / (2 * hh)
        dF = (froll(phi, delta, turn.s, yB + 1e-6) - froll(phi, delta, turn.s, yB - 1e-6)) / 2e-6
        form2 = -dF * np.linalg.det(J) / np.linalg.det(Me)
        free = st.capsize_formula(bb, phi, delta)  # Theorem 1 with the shifted bicycle
        rows.append(
            dict(
                V=V,
                phi=phi,
                yB=yB,
                delta=delta,
                T=turn.T,
                a4_constant=a_c[3],
                form1=form1,
                form2=form2,
                a4_free=a_f[3],
                theorem=free,
            )
        )

    def rel(key, ref):
        return max(abs(r[key] - r[ref]) / abs(r[ref]) for r in rows)

    out = dict(
        rows=rows,
        max_rel=dict(
            theorem_free=rel("theorem", "a4_free"),
            form1_constant=rel("form1", "a4_constant"),
            form2_constant=rel("form2", "a4_constant"),
        ),
    )
    path = REPO / "data" / "rider" / "shift_capsize_check.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out["max_rel"], indent=1))


if __name__ == "__main__":
    main()
