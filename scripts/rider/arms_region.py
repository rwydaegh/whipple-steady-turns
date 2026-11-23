"""Boundaries of the stable turns of the benchmark bicycle with a centring spring, k/k* = 0, 0.4, 0.8, for Fig. fig:rider(a) (Sect. 13.2, "Steer torque with a centring spring")."""  # noqa: E501

import json
from pathlib import Path

import numpy as np
from arms_analyse import load, tongue
from arms_model import Upright, all_sets, phi_c

REPO = Path(__file__).resolve().parents[2]
FRACS = (0.0, 0.4, 0.8)


def boundaries(g, plant, k, c=0.0):
    """Lower and upper lean (deg) of the stable region at each grid speed (nan outside)."""
    V, phi = g["V"], g["phi"]
    t, a, D3, tm = tongue(g, plant, k, c, full=True)
    tests = (a[3], D3, a[2], a[0])
    bot, top = np.full(len(V), np.nan), np.full(len(V), np.nan)
    if not t["n_tongue"]:
        return V, bot, top

    def cross(i, j0, j1):
        cr = [
            phi[j0] + (phi[j1] - phi[j0]) * f[i, j0] / (f[i, j0] - f[i, j1])
            for f in tests
            if np.isfinite(f[i, j1]) and f[i, j0] > 0 >= f[i, j1]
        ]
        return min(cr, key=lambda q: abs(q - phi[j0])) if cr else phi[j0]

    for i in range(len(V)):
        js = np.flatnonzero(tm[i])
        if not len(js):
            continue
        j = js.max()
        top[i] = cross(i, j, j + 1) if j + 1 < len(phi) else phi[j]
        j = js.min()
        bot[i] = cross(i, j, j - 1) if j > 0 else 0.0
    return V, np.degrees(bot), np.degrees(top)


def main():
    p = all_sets()["benchmark"]
    u = Upright(p)
    g = load("benchmark", fine=True)
    curves = []
    for f in FRACS:
        k = f * u.k_star
        V, bot, top = boundaries(g, "free", k)
        ok = np.isfinite(top)
        vw, vc = u.band(k)
        vr = np.linspace(u.V_a3, u.v_c_k(k), 80)[:-1]
        pr = np.degrees([phi_c(p, v, k, "free", 1) for v in vr])
        curves.append(
            dict(
                k_over_kstar=f,
                k=k,
                v_w=vw,
                v_c=vc,
                V=V[ok].round(4),
                lean_low_deg=bot[ok].round(5),
                lean_high_deg=top[ok].round(5),
                rule_V=vr.round(5),
                rule_lean_deg=pr.round(5),
                lean_star_grid=float(np.nanmax(top)),
            )
        )
        print(f"k/k* = {f:g}: largest lean on the grid {np.nanmax(top):.2f} deg")
    out = dict(k_star=u.k_star, V_a3=u.V_a3, curves=curves)
    path = REPO / "data" / "rider" / "arms_region.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, default=lambda x: x.tolist()) + "\n")


if __name__ == "__main__":
    main()
