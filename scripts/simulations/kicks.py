"""Kicks from the turn at 4.5 m/s and 10 deg (Sect. 15.1, "Time simulations", Fig. sim)."""

import math

import numpy as np
from common import contacts, fit_circle, lean_speed, run, save, turn_at
from scipy.optimize import brentq

from whipple.bicycles import BENCHMARK
from whipple.model import Whipple
from whipple.stability import eigenvalues, torque_and_first_integral

KICK_TIME = 2.0


def end_turn(bike, tr, z):
    """Turn of the same torque T on the level H = E - T delta of the state z."""
    H1 = bike.energy(z) - tr.T * z[1]

    def steer(phi):
        return brentq(
            lambda d: bike.turn_spin_torque(phi, d)[1] - tr.T, tr.delta * 0.5, tr.delta * 2
        )

    phi = brentq(
        lambda p: torque_and_first_integral(bike, p, steer(p))[1] - H1, tr.phi, tr.phi + 0.05
    )
    return bike.turn(phi, steer(phi))


def main():
    bike = Whipple(BENCHMARK)
    tr = turn_at(bike, 4.5, 10.0)
    z0 = bike.turn_state(tr)
    out = dict(
        turn=dict(
            V=tr.V,
            lean_deg=10.0,
            steer_deg=math.degrees(tr.delta),
            T=tr.T,
            R=tr.R,
            eigenvalues=sorted(eigenvalues(bike, tr.phi, tr.delta), key=lambda x: x.real),
        )
    )

    r = run(bike, z0, tr.T, 8.0)
    pts = np.array([contacts(bike, y)[0] for y in r["Y"]])
    _, R_fit, res = fit_circle(pts[:, :2])
    out["check"] = dict(R_fit=R_fit, R_model=tr.R, max_radial_residual=res)

    gaps, drift = [0.0], [0.0]
    for name, kick, t_end in (("stable_turn", 0.4, 18.0), ("capsize", 0.8, 20.0)):
        zk = z0 + np.r_[0, 0, kick, 0, 0]
        end = end_turn(bike, tr, zk) if kick < 0.6 else None
        r = run(bike, z0, tr.T, t_end, [(KICK_TIME, "kick", kick)])
        lean, V = lean_speed(bike, r)
        rear = np.array([contacts(bike, y)[0] for y in r["Y"]])
        front = np.array([contacts(bike, y)[1] for y in r["Y"]])
        gaps.append(max(np.abs(rear[:, 2]).max(), np.abs(front[:, 2]).max()))
        drift.append(max(s["H_drift"] for s in r["segs"]))
        out[name] = dict(
            kick=kick,
            t=r["t"],
            lean_deg=lean,
            steer_deg=np.degrees(r["Y"][:, 1]),
            speed=V,
            track_rear_xy=np.round(rear[:, :2], 5),
            fell=r["fell"],
            end_turn=dict(lean_deg=math.degrees(end.phi), V=end.V) if end else None,
        )
        print(name, "end t", r["t"][-1], "lean", lean[-1], "speed", V[-1], flush=True)
    out["check"].update(max_contact_height=max(gaps), max_H_drift=max(drift))

    sweep = {}
    for kick in (0.2, 0.3, 0.4, 0.6, 0.8, 1.0):
        r = run(bike, z0, tr.T, 20.0, [(KICK_TIME, "kick", kick)], fps=5)
        sweep[str(kick)] = dict(
            capsize=r["fell"] is not None, lean_end_deg=math.degrees(r["Y"][-1, 0])
        )
    out["kick_sweep"] = sweep

    # a kick of 0.4 rad/s that keeps the energy: the spin is lowered to the energy of the turn
    z = z0.copy()
    z[2] = 0.4
    E0 = bike.energy(z0)
    z[4] = brentq(lambda s: bike.energy(np.r_[z[:4], s]) - E0, z0[4] * 0.9, z0[4])
    r = run(bike, z, tr.T, 60.0, fps=2)
    out["energy_keeping_kick"] = dict(
        lean_end_deg=math.degrees(r["Y"][-1, 0]), V_start=bike.p["rR"] * abs(z[4])
    )
    print({k: v for k, v in out.items() if k not in ("stable_turn", "capsize")})
    save("kicks", out)


if __name__ == "__main__":
    main()
