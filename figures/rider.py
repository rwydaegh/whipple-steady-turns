"""Fig. fig:rider: stable turns of the benchmark bicycle with a centring spring (a) and with a mass shift (b) (Sect. 13, "Rider")."""  # noqa: E501

import json
import math

import numpy as np
from figstyle import (
    BLACK,
    CAPSIZE,
    CM,
    COL_W,
    DASH,
    DATA,
    STABLE_TEXT,
    WEAVE,
    XV,
    YLEAN,
    fill_region,
    panel,
    plt,
    save_paper,
    tip_marker,
)
from scipy.ndimage import binary_dilation, label

RIDER = DATA / "rider"
FILL_OPAQUE = "#D6F5D6"  # the stable fill on white, so that overlaps do not darken
DASHDOT = (0, (5, 1.5, 1, 1.5))
YG = r"Lean $\phi_G$ (deg)"


def spring_panel(ax):
    d = json.loads((RIDER / "arms_region.json").read_text())
    data = dict(k_star=d["k_star"], V_a3=d["V_a3"], curves=[])
    for c in d["curves"]:
        V, bot, top = (np.array(c[k]) for k in ("V", "lean_low_deg", "lean_high_deg"))
        w = bot > 0.05  # lower edge above upright: the weave curve
        hx, hy = np.r_[c["v_w"], V[w][::-1]], np.r_[0.0, bot[w][::-1]]
        fx, fy = np.r_[c["v_c"], V[::-1]], np.r_[0.0, top[::-1]]
        fill_region(ax, hx, hy, fx, fy, color=FILL_OPAQUE, alpha=1.0)
        ax.plot(hx, hy, color=WEAVE, lw=0.9, zorder=3)
        ax.plot(fx, fy, color=CAPSIZE, lw=0.9, zorder=3)
        ax.plot(c["rule_V"], c["rule_lean_deg"], color=BLACK, lw=0.7, ls=DASH, zorder=4)
        f = c["k_over_kstar"]
        i = int(np.nanargmax(top))
        txt = rf"$k/k^{{*}}={f:g}$" if f == 0 else rf"${f:g}$"
        if f == 0.4:
            ax.text(V[i] + 0.45, top[i] + 0.25, txt, fontsize=8, ha="left", va="bottom")
        else:
            ax.text(V[i], top[i] + 0.35, txt, fontsize=8, ha="center", va="bottom")
        data["curves"].append(dict(c, lean_star_grid=float(np.nanmax(top))))
    ax.set(
        xlim=(2.3, 6.3),
        ylim=(0, 15),
        xticks=[3, 4, 5, 6],
        yticks=[0, 5, 10, 15],
        xlabel=XV,
        ylabel=YLEAN,
    )
    ax.plot([2.4, 2.75], [14.3, 14.3], color=BLACK, lw=0.7, ls=DASH)
    ax.text(2.82, 14.3, "First-order rule", fontsize=8, va="center")
    return data


def tongue(f):
    r = json.loads((RIDER / "shift_tongues" / f"benchmark_f{f}_free.json").read_text())
    L = r["largest"]["phiG"]
    return (
        np.array(r["hopf"]["V"]),
        np.degrees(r["hopf"]["phiG"]),
        np.array(r["fold"]["V"]),
        np.degrees(r["fold"]["phiG"]),
        (L["V"], math.degrees(L["phiG"])),
    )


def fixed_shift_region():
    """Grid region joined to small lean at the fixed shift Delta = 2 deg, free rolling."""
    G = np.load(RIDER / "shift_grid_benchmark.npz")
    V, phiG = G["V"], np.degrees(G["phiG"])
    k = int(np.argmin(np.abs(G["Delta"] - G["fig_Delta"])))
    lab, _ = label(G["stable_free"][:, :, k])
    comp = np.isin(lab, list(set(lab[:, 0][lab[:, 0] > 0])))
    near = binary_dilation(comp, iterations=2)
    with np.errstate(invalid="ignore"):
        margin = np.minimum(G["fig_a4"] / 50.0, G["fig_d3"] / 5000.0)
    margin = np.where(near & np.isfinite(margin), margin, -1.0)
    return V, phiG, margin, comp, float(np.degrees(G["fig_Delta"]))


def shift_panel(ax):
    hv, hp, fv, fp, L0 = tongue("0")
    fill_region(ax, hv, hp, fv, fp)
    ax.plot(hv, hp, color=WEAVE, zorder=3)
    ax.plot(fv, fp, color=CAPSIZE, zorder=3)
    tip_marker(ax, hv[-1], hp[-1])
    sv, sp, gv, gp, L1 = tongue("1")
    ax.plot(np.r_[sv, gv[::-1]], np.r_[sp, gp[::-1]], color=BLACK, lw=0.9, zorder=4)
    V, phiG, margin, comp, dk = fixed_shift_region()
    ax.contour(
        V,
        phiG,
        margin.T,
        levels=[0.0],
        colors=[BLACK],
        linewidths=0.9,
        linestyles=[DASHDOT],
        zorder=4,
    )
    ax.text(5.15, 4.0, "Steer torque", color=STABLE_TEXT, fontsize=8, ha="center")
    ax.plot([2.62, 3.02], [19.0, 19.0], color=BLACK, lw=0.9)
    ax.text(3.1, 19.0, "Hands free", fontsize=8, va="center")
    ax.plot([2.62, 3.02], [17.6, 17.6], color=BLACK, lw=0.9, ls=DASHDOT)
    ax.text(3.1, 17.6, rf"Shift $\Delta = {dk:g}^\circ$ and torque", fontsize=8, va="center")
    ax.set(
        xlim=(2.5, 6.5),
        ylim=(0, 20),
        xticks=[3, 4, 5, 6],
        yticks=[0, 5, 10, 15, 20],
        xlabel=XV,
        ylabel=YG,
    )
    return dict(
        torque=dict(hopf_V=hv, hopf_phiG_deg=hp, fold_V=fv, fold_phiG_deg=fp, largest=L0),
        hands_free=dict(hopf_V=sv, hopf_phiG_deg=sp, fold_V=gv, fold_phiG_deg=gp, largest=L1),
        fixed_shift=dict(Delta_deg=dk, V=V, phiG_deg=phiG, stable=comp.astype(int)),
    )


def main():
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(COL_W, 5.6 * CM), gridspec_kw=dict(wspace=0.3))
    data = dict(panel_a=spring_panel(ax), panel_b=shift_panel(bx))
    panel(ax, "a", x=-0.2)
    panel(bx, "b", x=-0.2)
    save_paper(fig, "rider", data)
    for c in data["panel_a"]["curves"]:
        print(
            f"k/k* = {c['k_over_kstar']:g}: largest lean on the grid {c['lean_star_grid']:.2f} deg"
        )
    pb = data["panel_b"]
    print(
        "largest phi_G: torque", pb["torque"]["largest"], "hands free", pb["hands_free"]["largest"]
    )


if __name__ == "__main__":
    main()
