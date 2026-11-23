"""Largest hands-free shift at the energy of straight running at 4.5 m/s, and time simulations from straight running with a shift (Sect. 13.3, Proposition prop:rider-fold(ii); supplementary material, "Mass shift: further results")."""  # noqa: E501

import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from shift_model import Delta_from_yB, Whipple, hurwitz, params_of, quartic, yB_from_Delta

from whipple import stability as st

REPO = Path(__file__).resolve().parents[2]
V0 = 4.5
SHIFTS = (0.40, 0.55, 0.60, 0.62, 0.65, 1.20)  # deg


def largest_shift(b, p, E0):
    """Hands-free turns (T = 0) at the energy E0 against the frame lean: shift and a4."""
    rows, y = [], np.array([0.055, 0.006])
    for phi in np.arange(0.10, 0.27, 0.005):

        def F(y, phi=phi):
            bb = b.with_params(yB=y[1])
            t = bb.turn(phi, y[0])
            return np.array([t.T, bb.energy(bb.turn_state(t)) - E0])

        y = st.newton_system(F, y, h=1e-8, tol=1e-12)
        bb = b.with_params(yB=y[1])
        rows.append(
            dict(
                phi=float(phi),
                delta=y[0],
                yB=y[1],
                V=bb.turn(phi, y[0]).V,
                Delta_deg=math.degrees(Delta_from_yB(p, y[1])),
                a4=quartic(bb, phi, y[0], "free")[3],
            )
        )
    return rows


def simulate(bike, z0, t_end=40.0):
    def fall(t, z):
        return min(1.0 - abs(z[0]), 1.0 - abs(z[1]))

    fall.terminal = True
    sol = solve_ivp(
        lambda t, z: bike.field(z, 0.0),
        (0, t_end),
        z0,
        method="DOP853",
        rtol=1e-10,
        atol=1e-12,
        events=fall,
    )
    return sol.y[:, -1], bool(sol.t[-1] < t_end - 1e-9), float(sol.t[-1])


def main():
    p = params_of("benchmark")
    b = Whipple(p)
    z0 = np.array([0.0, 0.0, 0.0, 0.0, -V0 / p["rR"]])
    E0 = b.energy(z0)
    fam = largest_shift(b, p, E0)
    i = int(np.argmax([r["Delta_deg"] for r in fam]))
    print(
        f"largest shift {fam[i]['Delta_deg']:.4f} deg at frame lean "
        f"{math.degrees(fam[i]['phi']):.2f} deg"
    )
    sims = []
    for Dd in SHIFTS:
        bike = b.with_params(yB=yB_from_Delta(p, math.radians(Dd)))
        zf, fell, t_end = simulate(bike, z0)
        row = dict(
            Delta_deg=Dd,
            fell=fell,
            t_end=t_end,
            phi_end_deg=math.degrees(zf[0]),
            V_end=p["rR"] * abs(zf[4]),
        )
        if not fell:  # hands-free turn of this bicycle with the same energy

            def F(x, bike=bike):
                t = bike.turn(*x)
                return np.array([t.T, bike.energy(bike.turn_state(t)) - E0])

            x = st.newton_system(F, zf[:2], h=1e-7, tol=1e-12)
            a = quartic(bike, x[0], x[1], "free")
            row["predicted_turn"] = dict(
                phi_deg=math.degrees(x[0]),
                V=bike.turn(*x).V,
                stable=bool(min(a[0], a[2], a[3], hurwitz(a)) > 0),
            )
        sims.append(row)
        print(row, flush=True)
    out = dict(V0=V0, hands_free_family=fam, largest_shift=fam[i], simulations=sims)
    path = REPO / "data" / "rider" / "shift_simulations.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
