"""Region of stable turns of the benchmark bicycle and of Crescendo with Jason (fig:tongue, Sect.
5, "Region of stable turns at constant steer torque").
"""

import numpy as np
from figstyle import (
    CAPSIZE,
    CM,
    COL_W,
    DASH,
    DEG,
    EXACT_ALPHA,
    EXACT_LW,
    LW,
    STABLE_TEXT,
    WEAVE,
    XV,
    YLEAN,
    boundary,
    fill_region,
    label,
    load_result,
    params_of,
    plt,
    save_paper,
    tip_marker,
)

from whipple import capsize_rule
from whipple.quartic import Quartic

CASES = [("benchmark", "Benchmark"), ("Crescendo+Jason", "Crescendo, Jason")]


def formulas(p, v_w, v_c):
    """Weave side from the quartic to order phi^2, capsize side from the second-order rule."""
    q = Quartic(p)
    tip = q.bt("free")
    v0 = tip[0] if tip else 0.7 * v_w
    vh = np.linspace(v0, v_w, 150)
    lh = np.array([DEG(np.sqrt(q.hopf_x(v, "free"))) for v in vh])
    vf = np.linspace(v0, v_c, 220)
    lf = np.array([DEG(capsize_rule.phi_c_free(p, v, order=2)) for v in vf])
    lh[-1], lf[-1] = 0.0, 0.0
    return dict(
        weave_V=vh, weave_lean=lh, capsize_V=vf, capsize_lean=lf, V_a3=q.v_a3(), tip_closed=tip
    )


def draw(ax, name):
    r = load_result(name)
    Vh, lh, Vf, lf = boundary(r)
    fill_region(ax, Vh, lh, Vf, lf)
    kw = dict(lw=EXACT_LW, alpha=EXACT_ALPHA, zorder=3, solid_capstyle="butt")
    ax.plot(Vh, lh, color=WEAVE, **kw)
    ax.plot(Vf, lf, color=CAPSIZE, **kw)
    f = formulas(params_of(r), r["upright"]["v_w"], r["upright"]["v_c"])
    ax.plot(f["weave_V"], f["weave_lean"], color=WEAVE, ls=DASH, lw=LW, zorder=4)
    ax.plot(f["capsize_V"], f["capsize_lean"], color=CAPSIZE, ls=DASH, lw=LW, zorder=4)
    tip_marker(ax, Vh[-1], lh[-1])
    va3 = f["V_a3"]
    ax.plot([va3, va3], [0, 1.2], color="k", lw=1.0, zorder=5)
    ax.text(va3, 1.5, r"$V_{a3}$", ha="center", va="bottom", fontsize=8)
    fr = r["free"]
    top = capsize_rule.largest_lean_rule(params_of(r), "free", order=2)
    return dict(
        exact=dict(
            weave_V=Vh,
            weave_lean=lh,
            capsize_V=Vf,
            capsize_lean=lf,
            tip_V=fr["BT_V"],
            tip_lean=fr["BT_lean_deg"],
            largest_lean=fr["max_lean_deg"],
            V_at_largest=fr["V_at_max_lean"],
            v_w=r["upright"]["v_w"],
            v_c=r["upright"]["v_c"],
        ),
        closed_form=f,
        second_order_rule_top={k: v for k, v in top.items() if k != "speed_range"},
    )


def main():
    fig, axs = plt.subplots(1, 2, figsize=(COL_W, 5.0 * CM), gridspec_kw=dict(wspace=0.22))
    out = {}
    for ax, (name, title), letter in zip(axs, CASES, "ab"):
        out[name] = draw(ax, name)
        ax.set(xlabel=XV, ylim=(0, 21 if name != "benchmark" else 14))
        ax.text(
            -0.2,
            1.04,
            f"({letter})  {title}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
        )
    a, b = axs
    a.set(xlim=(2.2, 6.5), xticks=[3, 4, 5, 6], yticks=[0, 5, 10], ylabel=YLEAN)
    b.set(xlim=(3.5, 9.8), xticks=[4, 5, 6, 7, 8, 9], yticks=[0, 5, 10, 15, 20])
    label(a, 4.9, 4.0, "Stable", color=STABLE_TEXT, ha="center")
    label(a, 3.15, 4.5, "Weave", color=WEAVE, ha="center")
    label(a, 5.6, 11.3, "Capsize", color=CAPSIZE, ha="left")
    e = out["benchmark"]["exact"]
    a.annotate(
        "Tip",
        xy=(e["tip_V"], e["tip_lean"]),
        xytext=(e["tip_V"] + 0.05, e["tip_lean"] + 2.4),
        fontsize=8,
        ha="center",
        arrowprops=dict(arrowstyle="-", lw=0.5, shrinkA=0, shrinkB=3),
    )
    b.plot(
        [7.3, 7.85], [20.0, 20.0], color="k", lw=EXACT_LW, alpha=EXACT_ALPHA, solid_capstyle="butt"
    )
    b.plot([7.3, 7.85], [18.6, 18.6], color="k", lw=LW, ls=DASH)
    label(b, 7.98, 20.0, "Exact")
    label(b, 7.98, 18.6, "Formulas")
    for n, d in out.items():
        print(
            n,
            {k: round(v, 3) for k, v in d["exact"].items() if np.isscalar(v)},
            "rule top",
            round(d["second_order_rule_top"]["phi_deg"], 2),
            "V_a3",
            round(d["closed_form"]["V_a3"], 3),
            "closed tip",
            d["closed_form"]["tip_closed"],
        )
    save_paper(fig, "region", out)


if __name__ == "__main__":
    main()
