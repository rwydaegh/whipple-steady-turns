"""Nine time simulations of the full benchmark model with the steer spring and damper (supplementary material, "Centring spring: further results")."""  # noqa: E501

import json
import math
from pathlib import Path

import numpy as np
from arms_model import Upright, all_sets, delta_at, numerators, pencils, quartic
from scipy.integrate import solve_ivp

from whipple.model import Whipple

REPO = Path(__file__).resolve().parents[2]
CASES = [  # (V m/s, lean deg, k N m/rad, c N m s/rad)
    (4.3, 6.0, 0.0, 0.0),
    (4.3, 6.0, 2.0, 0.0),
    (4.3, 6.0, 4.0, 0.0),
    (3.6, 4.0, 3.0, 0.0),
    (3.6, 4.0, 0.0, 0.0),
    (5.0, 10.0, 0.0, 4.0),
    (4.0, 10.0, 0.0, 4.0),
    (4.5, 0.5, 59.4, 1.81),
    (4.5, 0.5, 1053.4, 19.28),
]


def main():
    p = all_sets()["benchmark"]
    bike, u = Whipple(p), Upright(p)
    out = []
    for V, lean, k, c in CASES:
        phi = math.radians(lean)
        d0 = delta_at(bike, phi, V, u.lin.steer_ratio(V) * phi)
        pen, turn = pencils(bike, phi, d0)
        lam = np.roots(np.r_[1.0, quartic(*numerators(*pen["free"]), k, c)])
        z0 = bike.turn_state(turn) + np.array([0, 0, 0.01, 0, 0])  # lean-rate kick

        def f(t, z, k=k, c=c, turn=turn, d0=d0):
            return bike.field(z, turn.T - k * (z[1] - d0) - c * z[3])

        def fall(t, z):
            return abs(z[0]) - 0.8

        fall.terminal = True
        sol = solve_ivp(
            f,
            (0, 60),
            z0,
            method="LSODA",
            rtol=1e-9,
            atol=1e-11,
            events=fall,
            t_eval=np.linspace(0, 60, 1201),
        )
        fell = sol.status == 1
        stable = bool(lam.real.max() < 0)
        row = dict(
            V=V,
            lean_deg=lean,
            k=k,
            c=c,
            max_real_eigenvalue=float(lam.real.max()),
            predicted_stable=stable,
            fell=bool(fell),
            t_end=float(sol.t[-1]),
            lean_end_deg=math.degrees(sol.y[0][-1]),
            agrees=bool(fell != stable),
        )
        out.append(row)
        print(row, flush=True)
    path = REPO / "data" / "rider" / "arms_simulations.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
