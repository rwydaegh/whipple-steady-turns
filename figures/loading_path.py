"""fig:loading, torque on the loading path (supplement, "Torque limit on a path")."""

import math

import numpy as np
from figstyle import (
    BLACK,
    CAPSIZE,
    CM,
    COL_W,
    DASH,
    DATA,
    DOT,
    GREY,
    LW,
    STABLE,
    STABLE_ALPHA,
    STABLE_TEXT,
    label,
    panel,
    plt,
    read_json,
    save_paper,
)

CACHE = DATA / "capsize" / "loading_path_figure.json"


def series_limits(s):
    """Torque limits of the series at fixed H to order phi^3 and phi^5."""
    xc = -s["tau1"] / (3 * s["tau3H"])
    r = np.roots([5 * s["tau5H"], 3 * s["tau3H"], s["tau1"]])
    r = r[(abs(r.imag) < 1e-12) & (r.real > 0)].real.min()
    return dict(
        cubic=dict(lean=math.degrees(math.sqrt(xc)), T=2 / 3 * s["tau1"] * math.sqrt(xc)),
        quintic=dict(
            lean=math.degrees(math.sqrt(r)),
            T=s["tau1"] * math.sqrt(r) + s["tau3H"] * r**1.5 + s["tau5H"] * r**2.5,
        ),
    )


def main():
    d = read_json(CACHE)
    P = np.array(d["loading_path"])
    lean, T = np.degrees(P[:, 0]), P[:, 1]
    fl, fT = math.degrees(d["fold"]["lean"]), d["fold"]["T"]
    s = d["series"]
    x = np.radians(np.linspace(0, 25, 300))
    cubic = s["tau1"] * x + s["tau3H"] * x**3
    quint = cubic + s["tau5H"] * x**5

    fig, (ax, bx) = plt.subplots(
        1, 2, figsize=(COL_W, 5.2 * CM), sharey=True, gridspec_kw=dict(wspace=0.1)
    )
    for a in (ax, bx):
        a.axvspan(0, fl, color=STABLE, alpha=STABLE_ALPHA, lw=0, zorder=0)
        a.axhline(0, color="k", lw=0.5, zorder=1)
        a.plot(lean, np.where(lean <= fl + 0.2, T, np.nan), color=BLACK, lw=1.6, zorder=3)
        a.plot(lean, np.where(lean >= fl - 0.2, T, np.nan), color=BLACK, lw=0.8, zorder=3)
        a.plot(fl, fT, "o", ms=4.5, mfc="white", mec=CAPSIZE, mew=1.1, zorder=6)
        a.set(
            xlim=(0, 25),
            ylim=(-0.25, 0.03),
            xticks=[0, 5, 10, 15, 20, 25],
            xlabel=r"Lean $\phi$ (deg)",
        )
    ax.set(ylabel=r"Steer torque $T$ (N m)", yticks=[-0.2, -0.1, 0])
    xd = np.degrees(x)
    ax.plot(xd, s["tau1"] * x, color=GREY, lw=0.6, zorder=2)
    ax.plot(xd, cubic, color=BLACK, ls=DASH, lw=LW, zorder=4)
    ax.plot(xd, quint, color=BLACK, ls=DOT, lw=1.1, zorder=4)
    label(ax, 17.2, -0.235, r"Linear $\tau_1\phi$", color=GREY, ha="left")
    keys = (
        ("Exact", dict(lw=1.0)),
        (r"Order $\phi^3$", dict(ls=DASH, lw=LW)),
        (r"Order $\phi^5$", dict(ls=DOT, lw=1.1)),
    )
    for i, (txt, kw) in enumerate(keys):
        y = -0.150 - 0.022 * i
        ax.plot([17.0, 19.6], [y, y], color=BLACK, **kw)
        label(ax, 20.2, y, txt)
    ax.text(
        fl,
        fT + 0.024,
        "Torque\nlimit",
        fontsize=8,
        color=CAPSIZE,
        ha="center",
        va="bottom",
        linespacing=1.0,
    )
    label(ax, fl / 2, 0.012, "Stable", color=STABLE_TEXT, ha="center")
    panel(ax, "a", x=-0.2)
    styles = {0.004: dict(color=GREY, lw=0.8, ls=DASH), 0.001: dict(color=GREY, lw=0.8)}
    at_fold = {}
    for sm in d["ramps"]:
        L = np.degrees(np.array(sm["lean"]))
        bx.plot(L, sm["T"], zorder=2, **styles[sm["rate"]])
        k = np.flatnonzero(L >= fl)[0]
        bx.plot(L[k], sm["T"][k], "o", ms=3, color=GREY, zorder=5)
        at_fold[sm["rate"]] = sm["T"][k]
    label(bx, 17.5, -0.214, r"$0.004$ N m/s", color=GREY)
    label(bx, 17.6, -0.136, r"$0.001$ N m/s", color=GREY)
    label(
        bx,
        1.0,
        -0.222,
        "Torque ramps\nfrom straight running",
        color=GREY,
        va="center",
        linespacing=1.1,
    )
    panel(bx, "b", x=-0.06)
    lim = series_limits(s)
    out = dict(
        exact_limit=dict(lean=fl, T=fT, V=d["fold"]["V"]),
        limits_from_series=lim,
        series=s,
        ramp_T_at_fold_lean=at_fold,
    )
    save_paper(fig, "loading_path", out)
    print(
        f"exact limit {fl:.3f} deg, {fT:.5f} N m; cubic {lim['cubic']}; quintic "
        f"{lim['quintic']}; ramps at the fold lean {at_fold}"
    )


if __name__ == "__main__":
    main()
