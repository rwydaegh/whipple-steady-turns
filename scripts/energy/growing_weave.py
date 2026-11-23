"""Energy of a growing weave (Sect. 3.2, Fig. energy (b); supplementary Sect. front-weave)."""

import math

import numpy as np
from common import (
    bench,
    energy_parts,
    field_constant,
    field_free,
    jacobian_constant,
    jacobian_free,
    moving_mean,
    rounded,
    save,
    simulate,
    speeds,
    trim,
    turn_at,
    turn_state,
)

KEYS = ("V", "U", "Kf", "Kl", "E", "WT", "Wd", "H", "phi")


def run_slow(bike, lin):
    """Turn at 4.10 m/s and 6 deg, start on the weave eigenvector (lean 0.002 rad), cycle means."""
    t = turn_at(bike, lin, math.radians(6.0), 4.10)
    z_star = turn_state(t)
    H_star = trim(bike, t.phi, t.delta)[2]
    out, keep = {}, None
    for plant in ("free", "constant"):
        J = (
            jacobian_free(bike, z_star, t.T)
            if plant == "free"
            else jacobian_constant(bike, z_star, t.T)
        )
        lam, vec = np.linalg.eig(J)
        k = int(np.argmax(np.where(np.abs(lam.imag) > 0.5, lam.real, -np.inf)))
        sig, om = lam[k].real, abs(lam[k].imag)
        period = 2 * math.pi / om
        v = np.real(vec[:, k] / vec[0, k]) * 0.002
        z0 = z_star.copy()
        z0[: len(v)] += v
        t_end = min(250.0, math.log(30.0) / sig)
        n_per = 200
        n = int(t_end / (period / n_per)) + 1
        if plant == "free":
            sol = simulate(field_free(bike, t.T), z0, t_end, n, max_step=0.05)
        else:
            sol = simulate(field_constant(bike, t.T), np.r_[z0, 0.0], t_end, n, max_step=0.05)
        rows = []
        for z in sol.y.T:
            e = energy_parts(bike, z[:5])
            rows.append(
                [
                    bike.p["rR"] * abs(z[4]),
                    e["U"],
                    e["Kf"],
                    e["Kl"],
                    e["E"],
                    t.T * (z[1] - z0[1]),
                    z[5] if plant == "constant" else 0.0,
                    e["E"] - t.T * z[1],
                    z[0],
                ]
            )
        a = np.array(rows)
        s = {k_: a[:, i] for i, k_ in enumerate(KEYS)}
        cm = {k_: moving_mean(s[k_], n_per) for k_ in KEYS}
        cm["t"] = moving_mean(sol.t, n_per)
        r = dict(
            sigma=sig,
            omega=om,
            period=period,
            t_end=sol.t[-1],
            lean_amplitude_end_deg=math.degrees(np.ptp(s["phi"][-n_per:]) / 2),
            lean_amplitude_start_deg=math.degrees(np.ptp(s["phi"][:n_per]) / 2),
            mean_lean_deg=[math.degrees(cm["phi"][0]), math.degrees(cm["phi"][-1])],
            change_of_cycle_means=dict(
                U=cm["U"][-1] - cm["U"][0],
                Kl=cm["Kl"][-1] - cm["Kl"][0],
                Kf=cm["Kf"][-1] - cm["Kf"][0],
            ),
            balance_error=np.abs(s["E"] - s["E"][0] - s["WT"] - s["Wd"]).max(),
        )
        if plant == "free":
            r["H_drift"] = np.ptp(s["H"])
            r["mean_V"] = [cm["V"][0], cm["V"][-1]]
            keep = {k_: cm[k_][::5] for k_ in ("U", "Kf", "Kl", "WT", "H")}
            keep["t"] = cm["t"][::5]
        else:
            r["drive_work_cycle_mean_end"] = cm["Wd"][-1]
            r["H_cycle_mean_minus_H_star_end"] = cm["H"][-1] - H_star
        out[plant] = r
        print("weave at 6 deg", plant, r, flush=True)
    out["T"] = t.T
    return out, keep


def cycles(s, n):
    """Cycle means of the speeds, energies and height, and the lean amplitude squared."""
    out = {k: moving_mean(s[k], n) for k in ("Vs", "Vc", "Vg", "hg", "U", "Kf", "Kl")}
    pm, p2 = moving_mean(s["phi"], n), moving_mean(s["phi"] ** 2, n)
    out["a2"] = 2 * (p2 - pm**2)
    return out


def run_kick(bike, z0, T, t_end, dt=0.005):
    sol = simulate(field_free(bike, T), z0, t_end, int(t_end / dt) + 1)
    rows = []
    for z in sol.y.T:
        e, sp = energy_parts(bike, z), speeds(bike, z)
        rows.append([z[0], sp["Vs"], sp["Vc"], sp["Vg"], sp["hg"], e["U"], e["Kf"], e["Kl"]])
    a = np.array(rows)
    return {k: a[:, i] for i, k in enumerate("phi Vs Vc Vg hg U Kf Kl".split())}


def run_fits(bike, lin, m_tot):
    """Growing weaves from a lean-rate kick of 0.002 rad/s: upright and at 6 deg, 4.1 m/s."""
    out = {}
    for label, phi in (("upright", 0.0), ("turn 6 deg", 6.0)):
        if phi == 0:
            z0, T = np.array([0.0, 0.0, 0.002, 0.0, -4.1 / bike.p["rR"]]), 0.0
        else:
            t = turn_at(bike, lin, math.radians(phi), 4.1)
            z0, T = turn_state(t) + np.r_[0, 0, 0.002, 0, 0], t.T
        o = [
            x
            for x in np.linalg.eigvals(jacobian_free(bike, z0 * np.r_[1, 1, 0, 0, 1], T))
            if abs(x.imag) > 0.5
        ]
        o = max(o, key=lambda x: x.real)
        t_end = min(120.0, math.log(500.0) / o.real)
        s = run_kick(bike, z0, T, t_end)
        c = cycles(s, int(round(2 * math.pi / abs(o.imag) / 0.005)))
        d = {k: c[k][-1] - c[k][0] for k in ("Vs", "Vc", "Vg", "hg", "U", "Kf", "Kl")}
        k_fit = {k: np.sum(c["a2"] * (c[k] - c[k][0])) / np.sum(c["a2"] ** 2) for k in ("Vs", "Vg")}
        out[label] = dict(
            sigma=o.real,
            t_end=t_end,
            amplitude_end_deg=math.degrees(math.sqrt(c["a2"][-1])),
            change_of_cycle_means=d,
            k_fit=k_fit,
            m_tot_V_dVs=m_tot * 4.1 * d["Vs"],
        )
        print("weave from kick", label, out[label], flush=True)
    return out


def main():
    bike, lin = bench()
    slow, keep = run_slow(bike, lin)
    fits = run_fits(bike, lin, bike.m_tot)
    save("growing_weave", dict(turn_6deg_4p10=slow, kick_4p1=fits))
    save("series_weave", rounded(keep), gz=True)


if __name__ == "__main__":
    main()
