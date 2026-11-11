"""Period doubling of the weave cycle below the Hopf curve of the benchmark bicycle at
T = -0.418 N m (Sect. 6)."""

import numpy as np

from whipple.bicycles import BENCHMARK
from whipple.model import Whipple
from whipple.output import options, save_json
from whipple.weave import period_doubling

TORQUE = -0.418
HOPF_GUESS = (0.159, 0.119)  # lean and steer (rad) near the Hopf point at this torque


def main():
    args = options(__doc__)
    bike = Whipple(BENCHMARK)
    kw = dict(start=0.8, step=0.15, settle_time=100.0) if args.quick else {}
    r = period_doubling(bike, TORQUE, HOPF_GUESS, **kw)
    turn, hopf = r["turn"], r["hopf_turn"]
    save_json("period_doubling", {
        "steer torque (N m)": TORQUE,
        "turn inside the cycle: speed V (m/s)": turn.V,
        "turn inside the cycle: lean (deg)": np.degrees(turn.phi),
        "cycle period (s)": r["period"],
        "Floquet multipliers": list(r["multipliers"]),
        "Hopf turn: speed V (m/s)": hopf.V,
        "Hopf turn: lean (deg)": np.degrees(hopf.phi),
        "(H_hopf - H) / (H_fold - H_hopf)": (r["H_hopf"] - r["H"]) / (r["H_fold"] - r["H_hopf"]),
    })  # fmt: skip


if __name__ == "__main__":
    main()
