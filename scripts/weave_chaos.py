"""Weave below the Hopf curve at T = -0.418 N m: lean maxima and the largest Lyapunov exponent chi
at a few speed levels, from period doubling to chaos (Fig. 6b and 6c).

The Python integrator is slow (about 1 s per simulated second), so the default run is short. It
gives accurate lean maxima but only a rough chi. The full figure (4220 levels, 3000 s each) comes
from the compiled batch program in c/ (see the README).
"""

import numpy as np

from whipple.bicycles import BENCHMARK
from whipple.model import Whipple
from whipple.output import options, save_json
from whipple.weave import Leaf, TorqueFamily, weave_run, weave_start

TORQUE = -0.4179855714497167
HOPF_GUESS = (0.159, 0.119)  # lean and steer (rad) near the Hopf point at this torque
# Levels H - 794.1195 J of the first integral H = E - T delta: two lean maxima per period, then
# chaos.
OFFSET = 794.1195
LEVELS = (610.75, 613.328, 619.75)


def main():
    args = options(__doc__)
    t_trans, t_run = (30.0, 30.0) if args.quick else (100.0, 200.0)
    bike = Whipple(BENCHMARK)
    fam = TorqueFamily(bike, TORQUE, *HOPF_GUESS)
    phi_h = fam.hopf(HOPF_GUESS[0] - 0.02, HOPF_GUESS[0] + 0.02)
    phi_f, _ = fam.fold(phi_h, phi_h + 0.1)
    out = {"steer torque (N m)": TORQUE, "transient, record (s)": [t_trans, t_run]}
    for level in LEVELS:
        H = level + OFFSET
        y0, saddle = weave_start(fam, H, phi_h, phi_f)
        r = weave_run(Leaf(bike, TORQUE, H), y0, saddle + 0.3, t_trans, t_run)
        V = fam.turn(y0[0]).V
        maxima = np.degrees(r["maxima"][-8:])
        print(
            f"V = {V:.4f} m/s  {r['status']}  chi = {r['chi']:.3f} 1/s  last maxima (deg):",
            np.round(maxima, 3),
        )
        out[f"H = {level} J"] = dict(V=V, status=r["status"], chi=r["chi"],
                                     last_maxima_deg=maxima)  # fmt: skip
    save_json("weave_chaos", out)


if __name__ == "__main__":
    main()
