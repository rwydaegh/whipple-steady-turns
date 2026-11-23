"""Equivalent steer torque of a mass shift, eqs. rider-T and rider-Teq (Sect. 13.3, "Mass shift and steer torque"; supplementary material, "Mass shift: further results")."""  # noqa: E501

import json
import math
from pathlib import Path

from shift_model import Whipple, delta_for_speed, params_of, totals, yB_from_Delta

from whipple.linear import Linear

REPO = Path(__file__).resolve().parents[2]
SETS = ("benchmark", "Browser+Jason", "Rigid+Jason", "Crescendo+Jason")
M_UPPER, L_UPPER = 42.0, 0.375  # upper body: mass (kg), mass-centre height above the hinge (m)
CAIN = {
    "Cain and Perkins, subject 1": (-85.050, -2.028),  # printed K0_pp, K0_pd
    "Cain and Perkins, subject 2": (-94.156, -1.969),
}


def main():
    out = {}
    for name in SETS:
        p = params_of(name)
        mT, h, SA = totals(p)
        g = p["g"]
        out[name] = dict(
            mT=mT,
            h=h,
            S_A=SA,
            gSA_per_deg=g * SA * math.pi / 180,
            Teq_per_cm_75kg=SA / (mT * h) * 75 * g * 0.01,
            Teq_per_deg_upper_body=g * SA * M_UPPER * L_UPPER / (mT * h) * math.pi / 180,
        )
    for name, (k0pp, k0pd) in CAIN.items():
        out[name] = dict(
            gSA_per_deg=9.81 * -k0pd * math.pi / 180,
            Teq_per_deg_upper_body=9.81 * -k0pd * M_UPPER * L_UPPER / -k0pp * math.pi / 180,
        )

    # Exact small-lean turns against T = tau1 phi_G + g S_A Delta.
    p = params_of("benchmark")
    mT, h, SA = totals(p)
    lin, b = Linear(p), Whipple(p)
    rows, worst = [], 0.0
    eps = 1e-4
    for V in (3.0, 4.5, 6.5):
        for phiG, D in ((eps, 0.0), (eps, eps), (2 * eps, -eps)):
            bb = b.with_params(yB=yB_from_Delta(p, D))
            d = delta_for_speed(bb, phiG - D, V, lin.steer_ratio(V) * phiG)
            T = bb.turn(phiG - D, d).T
            pred = lin.tau1(V) * phiG + p["g"] * SA * D
            worst = max(worst, abs(T - pred) / abs(pred))
            rows.append(
                dict(
                    V=V,
                    phiG=phiG,
                    Delta=D,
                    T=T,
                    T_linear=pred,
                    delta=d,
                    delta_linear=lin.steer_ratio(V) * phiG,
                )
            )
    out["linear_check"] = dict(rows=rows, max_rel_error=worst)
    path = REPO / "data" / "rider" / "shift_equivalent_torque.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "linear_check"}, indent=1))
    print("largest relative error of eq. rider-T:", worst)


if __name__ == "__main__":
    main()
