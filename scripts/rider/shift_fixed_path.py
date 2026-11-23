"""Quartic coefficients of a turn on a fixed path (V, phi_G) against the mass shift (supplementary material, "Mass shift: further results")."""  # noqa: E501

import json
from pathlib import Path

import numpy as np
from shift_model import Whipple, delta_for_speed, hurwitz, params_of, quartic, yB_from_Delta

from whipple.linear import Linear

REPO = Path(__file__).resolve().parents[2]


def main():
    p = params_of("benchmark")
    b, lin = Whipple(p), Linear(p)
    rows = []
    for V, phiG in ((4.5, 0.20), (3.5, 0.15), (5.5, 0.15)):
        for D in np.linspace(-0.06, 0.06, 7):
            phi = phiG - D
            bb = b.with_params(yB=yB_from_Delta(p, D))
            delta = delta_for_speed(bb, phi, V, lin.steer_ratio(V) * phiG)
            row = dict(V=V, phiG=phiG, Delta=float(D), delta=delta, T=bb.turn(phi, delta).T)
            for plant in ("constant", "free"):
                a = quartic(bb, phi, delta, plant)
                row[plant] = dict(a=list(a), D3=hurwitz(a))
            rows.append(row)
            print(
                f"V {V} phiG {phiG} Delta {D:+.3f}: a4 constant {row['constant']['a'][3]:.2f}"
                f" free {row['free']['a'][3]:.2f}"
            )
    path = REPO / "data" / "rider" / "shift_fixed_path.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=1) + "\n")


if __name__ == "__main__":
    main()
