"""Upright case of Fig. 5.4 of Limebeer and Massaro (2018) (supplementary Sect. front-recovery)."""

import math

import numpy as np
from common import (
    bench,
    energy_parts,
    field_constant,
    field_free,
    jacobian_free,
    moving_mean,
    save,
    simulate,
    speeds,
)

DT = 0.005


def main():
    bike, _ = bench()
    rR = bike.p["rR"]
    s0 = -4.5 / rR
    z0 = np.array([0.0, 0.0, 0.5, 0.0, s0])
    sol = simulate(field_free(bike, 0.0), z0, 12.0, int(12.0 / DT) + 1)
    rows = []
    for z in sol.y.T:
        sp = speeds(bike, z)
        rows.append([z[0], sp["Vs"], sp["Vc"], sp["Vg"], sp["Va"], energy_parts(bike, z)["E"]])
    a = np.array(rows)
    phi, Vs, Vc, Vg, Va, E = a.T
    lam = [
        x
        for x in np.linalg.eigvals(jacobian_free(bike, np.r_[0, 0, 0, 0, s0], 0.0))
        if abs(x.imag) > 0.5
    ][0]
    period = 2 * math.pi / abs(lam.imag)
    n = int(round(period / DT))
    e0 = energy_parts(bike, np.r_[0, 0, 0, 0, s0])
    kick = E[0] - e0["E"]
    cm = {k: moving_mean(v, n) for k, v in (("Vs", Vs), ("Vc", Vc), ("Vg", Vg))}
    a2 = 2 * (moving_mean(phi**2, n) - moving_mean(phi, n) ** 2)
    end = dict(Vs=Vs[-1], Vc=Vc[-1], Vg=Vg[-1], Va=Va[-1])
    fits = {}
    for k in ("Vs", "Vc", "Vg"):
        x, y = a2, cm[k] - end[k]
        fits[k] = dict(k=np.sum(x * y) / np.sum(x * x), correlation=np.corrcoef(x, y)[0, 1])
    const = simulate(field_constant(bike, 0.0), np.r_[z0, 0.0], 12.0, 241)
    out = dict(
        kick_energy=kick,
        weave_eigenvalue=lam,
        V_end=end,
        V_end_spread=np.ptp([Vs[-1], Vc[-1], Vg[-1]]),
        V_upright_same_energy=math.sqrt(4.5**2 + 2 * kick / bike.m_tot),
        peak_Vs=dict(dV=Vs.max() - 4.5, t=sol.t[np.argmax(Vs)]),
        peak_Va=dict(dV=Va.max() - 4.5, t=sol.t[np.argmax(Va)]),
        cycle_fits=fits,
        H_drift=np.ptp(E),
        constant_speed_drive_work_12s=const.y[5, -1],
    )
    print(out)
    save("upright_recovery", out)


if __name__ == "__main__":
    main()
