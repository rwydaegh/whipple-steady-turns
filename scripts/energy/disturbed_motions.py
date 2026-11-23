"""Recovery after a kick and free-rolling capsize (Sect. 3.2, Fig. energy (a), (c), (d))."""

import argparse
import math

import numpy as np
from common import (
    bench,
    energy_parts,
    field_constant,
    field_free,
    jacobian_constant,
    jacobian_free,
    rounded,
    save,
    simulate,
    speeds,
    trim,
    turn_at,
    turn_state,
    turn_with,
)

SUBSAMPLE = 5


def series(bike, sol, T, constant=False):
    """Energy parts, the work of the steer torque and of the drive, H and speeds along a run."""
    names = "t phi delta phid deltad V U Kf Kl E WT Wd H Vcm".split()
    d0 = sol.y[1, 0]
    rows = []
    for t, z in zip(sol.t, sol.y.T):
        e = energy_parts(bike, z[:5])
        Wd = z[5] if constant else 0.0
        Vcm = speeds(bike, z[:5])["Vg"] if abs(z[1]) < 0.6 else math.nan
        rows.append(
            [
                t,
                *z[:4],
                bike.p["rR"] * abs(z[4]),
                e["U"],
                e["Kf"],
                e["Kl"],
                e["E"],
                T * (z[1] - d0),
                Wd,
                e["E"] - T * z[1],
                Vcm,
            ]
        )
    a = np.array(rows)
    return {n: a[:, i] for i, n in enumerate(names)}


def recovery(bike, lin):
    t = turn_at(bike, lin, math.radians(8.0), 5.0)
    z0 = turn_state(t) + np.array([0.0, 0.0, 0.3, 0.0, 0.0])
    kick = energy_parts(bike, z0)["E"] - energy_parts(bike, turn_state(t))["E"]
    out, keep = {}, {}
    for plant in ("free", "constant"):
        if plant == "free":
            sol = simulate(field_free(bike, t.T), z0, 25.0, 5001)
        else:
            sol = simulate(field_constant(bike, t.T), np.r_[z0, 0.0], 25.0, 5001)
        s = series(bike, sol, t.T, plant == "constant")
        balance = (s["E"] - s["E"][0]) - (s["WT"] + s["Wd"])
        info = dict(
            V=t.V,
            lean_deg=8.0,
            delta=t.delta,
            T=t.T,
            kick_energy=kick,
            balance_error_max=np.abs(balance).max(),
        )
        if plant == "free":
            J = jacobian_free(bike, turn_state(t), t.T)
            info["eigenvalues"] = sorted(np.linalg.eigvals(J), key=lambda x: x.real)
            info["H_drift_max"] = np.abs(s["H"] - s["H"][0]).max()
            end = turn_with(bike, t.T, s["H"][0], [t.phi, t.delta])
            info["predicted_end"] = dict(V=end.V, lean_deg=math.degrees(end.phi))
            info["simulated_25s"] = dict(V=s["V"][-1], lean_deg=math.degrees(s["phi"][-1]))
            S0, _, H0, E0 = trim(bike, t.phi, t.delta)
            S1, _, H1, E1 = trim(bike, end.phi, end.delta)
            e0 = energy_parts(bike, turn_state(t))
            e1 = energy_parts(bike, turn_state(end))
            info["end turn minus start turn"] = dict(
                Kf=e1["Kf"] - e0["Kf"],
                U=e1["U"] - e0["U"],
                minus_T_ddelta=-t.T * (end.delta - t.delta),
            )
            keep["recovery"] = {k: v for k, v in s.items() if k != "Vcm"}
        else:
            J = jacobian_constant(bike, turn_state(t), t.T)
            info["eigenvalues"] = sorted(np.linalg.eigvals(J), key=lambda x: x.real)
            info["drive_work_25s"] = s["Wd"][-1]
            info["drive_work_min"] = s["Wd"].min()
        out[plant] = info
        print("recovery", plant, {k: v for k, v in info.items() if k != "eigenvalues"}, flush=True)
    return out, keep


def capsize(bike, lin, quick):
    t = turn_at(bike, lin, math.radians(12.6), 4.54)
    z = turn_state(t)
    lam, vec = np.linalg.eig(jacobian_free(bike, z, t.T))
    k = int(np.argmax(lam.real))
    v = np.real(vec[:, k])
    v = v / np.max(np.abs(v[:4]))
    v = -v if v[0] < 0 else v
    z0 = z + 1e-3 * v
    t_end = 600.0
    if quick:  # same level of H, started near 18 deg of lean
        t_end = 12.0
        sol = simulate(
            field_free(bike, t.T),
            z0,
            150.0,
            2,
            max_step=0.05,
            events=_lean_event(math.radians(18.0)),
        )
        z0 = sol.y_events[0][0]

    def runaway(tt, y):
        return min(math.radians(62.0) - abs(y[0]), math.radians(60.0) - abs(y[1]))

    runaway.terminal = True
    leans = (20, 30, 45)
    crossings = [_lean_event(math.radians(lean), terminal=False) for lean in leans]
    sol = simulate(
        field_free(bike, t.T), z0, t_end, int(t_end / 0.01) + 1, events=[runaway, *crossings]
    )
    s = series(bike, sol, t.T)
    m_mass, m_tot, g = bike.mass.sum(), bike.m_tot, bike.p["g"]
    U0, V0 = s["U"][0], s["V"][0]
    drop = U0 - s["U"]
    s["Vsimple"] = np.sqrt(V0**2 + 2 * drop / m_mass)
    s["Veff"] = np.sqrt(V0**2 + 2 * (drop - (s["Kl"] - s["Kl"][0]) + s["WT"]) / m_tot)
    i45 = int(np.flatnonzero(np.degrees(s["phi"]) >= 45.0)[0])
    # table rows at the exact lean crossings (event states), not at the next output sample
    e0 = energy_parts(bike, sol.y[:5, 0])
    table = []
    for lean, te, ye in zip(leans, sol.t_events[1:], sol.y_events[1:]):
        z = ye[0][:5]
        e, sp = energy_parts(bike, z), speeds(bike, z)
        d_u = U0 - e["U"]
        torque_work = t.T * (z[1] - sol.y[1, 0])
        d_kl = e["Kl"] - e0["Kl"]
        table.append(
            dict(
                lean=lean,
                t=te[0],
                V_rear_contact=sp["Vc"],
                V_mass_centre=sp["Vg"],
                V_estimate=math.sqrt(V0**2 + 2 * d_u / m_mass),
                V_effective_mass=math.sqrt(V0**2 + 2 * (d_u - d_kl + torque_work) / m_tot),
                height_drop=d_u / (m_mass * g),
                U_drop=d_u,
                dKf=e["Kf"] - e0["Kf"],
                dKl=d_kl,
                torque_work=torque_work,
                share_Kf=(e["Kf"] - e0["Kf"]) / d_u,
                share_Kl=d_kl / d_u,
            )
        )
    # path radius of the rear contact and of the mass centre at 45 deg: speed over yaw rate
    sp = speeds(bike, sol.y_events[3][0][:5])
    radius = dict(
        rear_contact=sp["Vc"] / abs(sp["yaw_rate"]), mass_centre=sp["Vg"] / abs(sp["yaw_rate"])
    )
    info = dict(
        V=t.V,
        lean_deg=12.6,
        delta=t.delta,
        T=t.T,
        unstable_eigenvalue=lam[k].real,
        start=dict(V_rear_contact=V0, V_mass_centre=s["Vcm"][0]),
        t_end=s["t"][-1],
        lean_end_deg=math.degrees(s["phi"][-1]),
        steer_end_deg=math.degrees(s["delta"][-1]),
        H_drift_to_45deg=np.abs(s["H"][: i45 + 1] - s["H"][0]).max(),
        path_radius_at_45deg=radius,
        m_mass=m_mass,
        m_tot=m_tot,
        table=table,
        quick=quick,
    )
    print("capsize", {k: v for k, v in info.items() if k != "table"})
    for r in table:
        print("  ", {k: round(float(v), 4) for k, v in r.items()})
    return info, s


def _lean_event(limit, terminal=True):
    def ev(t, z):
        return z[0] - limit

    ev.terminal = terminal
    return ev


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--quick", action="store_true", help="short capsize run (numbers differ)")
    quick = p.parse_args().quick
    bike, lin = bench()
    rec, keep = recovery(bike, lin)
    cap, s = capsize(bike, lin, quick)
    save("disturbed_motions" + ("_quick" if quick else ""), dict(recovery=rec, capsize=cap))
    if not quick:
        out = {k: rounded({kk: vv[::SUBSAMPLE] for kk, vv in v.items()}) for k, v in keep.items()}
        cap_keys = "t phi delta V U Kf Kl WT H Vcm Vsimple Veff".split()
        out["capsize"] = rounded({k: s[k][::SUBSAMPLE] for k in cap_keys})
        save("series_disturbed", out, gz=True)


if __name__ == "__main__":
    main()
