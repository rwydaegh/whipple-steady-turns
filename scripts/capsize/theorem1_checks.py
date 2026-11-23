"""prop:fold at small lean and the point Q (Sect. 6, "Capsize condition in a turn")."""

import numpy as np
from common import BICYCLES, Whipple, clean, delta_at, upright, write_json
from scipy.optimize import brentq

from whipple import stability as st


def main():
    p = clean(BICYCLES["benchmark"])
    bike = Whipple(p)
    lin = upright(p)["lin"]
    V = 5.0
    limit = np.linalg.det(lin.K(V)) / np.linalg.det(lin.M)
    rows = []
    for phi in (1e-2, 1e-3, 1e-4):
        d = delta_at(bike, lin, phi, V)
        value = st.capsize_formula(bike, phi, d)
        rows.append(dict(lean=phi, steer=d, formula=value, relative=value / limit - 1))
        print(
            f"lean {phi:.0e} rad: formula {value:.6f}, relative difference {value / limit - 1:.2e}"
        )

    def on_gamma(phi):
        """Steer near 1.6 rad with F_c,phi = 0, and U_phi there."""
        d = brentq(lambda x: bike.static_forces(phi, x)[1][0], 1.4, 1.8, xtol=1e-15)
        return -bike.static_forces(phi, d)[0][0], d

    lean = brentq(lambda x: on_gamma(x)[0], 0.005, 0.02, xtol=1e-15)
    q = (lean, on_gamma(lean)[1])
    print(
        f"straight-running value {limit:.4f} s^-4; point Q at lean {q[0]:.6f}, steer {q[1]:.6f} rad"
    )
    write_json(
        "theorem1_checks.json",
        dict(V=V, straight_running=limit, leans=rows, Q=dict(lean=q[0], steer=q[1])),
    )


if __name__ == "__main__":
    main()
