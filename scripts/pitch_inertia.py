"""Frame pitch inertias in a corner (Sect. 6): derivatives of tau3, Eq. (22), the slope of the weave
coefficient Lambda_w in I_Hyy, and the steer-torque difference of bicycles A and B."""

import math

import numpy as np

from whipple import stability as st
from whipple.bicycles import BICYCLE_A, BICYCLE_B
from whipple.lean_law import LeanLaw
from whipple.linear import Linear
from whipple.model import Whipple
from whipple.output import options, save_json

V_PAIR = 4.805
DI = BICYCLE_B["IHyy"] - BICYCLE_A["IHyy"]


def closed_derivatives(p, V):
    """d tau3 / d I_Byy and d tau3 / d I_Hyy (N m per kg m^2) from Eq. (22)."""
    lin = Linear(p)
    K, r, sl = lin.K(V), lin.steer_ratio(V), math.sin(p["lam"])
    k2r2 = (V * math.cos(p["lam"]) / p["w"]) ** 2 * r**2
    return k2r2 * K[1, 1] / K[0, 1], k2r2 * (1 + r * sl) * (K[1, 1] / K[0, 1] - sl)


def torque_at(bike, phi, V):
    """Steer torque (N m) and lateral acceleration (m/s^2) of the turn with lean phi and speed V."""
    r = Linear(bike.p).steer_ratio(V)
    turn = bike.turn(phi, st.delta_for_speed(bike, phi, V, r * phi))
    return turn.T, turn.ay


def main():
    options(__doc__)
    a, b = LeanLaw(BICYCLE_A), LeanLaw(BICYCLE_B)
    a_byy = LeanLaw(dict(BICYCLE_A, IByy=BICYCLE_A["IByy"] + DI))
    tau3 = {
        name: law.contractions(V_PAIR)["tau3"] for name, law in (("A", a), ("B", b), ("Byy", a_byy))
    }
    dB, dH = closed_derivatives(BICYCLE_A, V_PAIR)
    lam_a, lam_b = a.Lambda(V_PAIR)[1][0], b.Lambda(V_PAIR)[1][0]

    bike_a, bike_b = Whipple(BICYCLE_A), Whipple(BICYCLE_B)
    V = 4.0
    T_a, ay = torque_at(bike_a, math.radians(20), V)
    T_b, _ = torque_at(bike_b, math.radians(20), V)
    small = [(torque_at(bike_b, x, V)[0] - torque_at(bike_a, x, V)[0], torque_at(bike_a, x, V)[1])
             for x in np.radians([0.5, 1.0, 2.0])]  # fmt: skip
    dT, ays = np.array(small).T
    slope = np.polyfit(np.log(ays), np.log(np.abs(dT)), 1)[0]

    save_json("pitch_inertia", {
        "V (m/s)": V_PAIR,
        "d tau3/d I_Byy, Eq. (22)": dB,
        "d tau3/d I_Byy, small-lean expansion": (tau3["Byy"] - tau3["A"]) / DI,
        "d tau3/d I_Hyy, Eq. (22)": dH,
        "d tau3/d I_Hyy, small-lean expansion": (tau3["B"] - tau3["A"]) / DI,
        "d Lambda_w/d I_Hyy at v_E = V (1/(s kg m^2))": (lam_b - lam_a) / DI,
        "torque at 4 m/s and 20 deg": {
            "a_y/g": ay / 9.81,
            "T_A (N m)": T_a,
            "T_B - T_A (mN m)": 1e3 * (T_b - T_a),
        },
        "log-log slope of |T_B - T_A| over a_y at 4 m/s, leans 0.5 to 2 deg": slope,
        "c in T_B - T_A = c dI_Hyy a_y^3 / V^2": float(np.mean(dT * V**2 / (DI * ays**3))),
    })  # fmt: skip


if __name__ == "__main__":
    main()
