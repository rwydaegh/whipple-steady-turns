"""Energy in free-rolling simulations of the benchmark bicycle (Sect. 3, Fig. energy)."""

import numpy as np
from figstyle import (
    BLACK,
    CM,
    COL_W,
    DASH,
    DATA,
    DOT,
    GREY,
    LIGHT,
    ORANGE,
    YLEAN,
    panel,
    plt,
    read_json,
    save_paper,
)

D = read_json(DATA / "energy" / "series_disturbed.json.gz")
W = read_json(DATA / "energy" / "series_weave.json.gz")
COMP = [
    ("U", "Potential", BLACK, "-"),
    ("Kf", "Forward kinetic", "#8000C0", (0, (5, 1.5, 1, 1.5))),
    ("Kl", "Lateral kinetic", ORANGE, "-"),
    ("WT", "Steer-torque work", GREY, DASH),
]


def cut(s, lean_max, window):
    phi = np.degrees(s["phi"])
    k = np.flatnonzero(phi >= lean_max)
    n = k[0] + 1 if len(k) else len(phi)
    t = np.array(s["t"][:n])
    m = t > t[-1] - window
    out = {kk: np.array(v[:n])[m] for kk, v in s.items()}
    out["t"] = out["t"] - out["t"][-1]
    return out


def energy(ax, s, labels):
    t = np.asarray(s["t"])
    out = {"t": t}
    for k, lab, c, ls in COMP:
        y = np.asarray(s[k]) - s[k][0]
        ax.plot(t, y, color=c, ls=ls, lw=1.0)
        out[k] = y
        if k in labels:
            x, yy, ha = labels[k]
            ax.text(x, yy, lab, color=c, fontsize=8, ha=ha, va="center")
    y = np.asarray(s["H"]) - s["H"][0]
    ax.plot(t, y, color=BLACK, ls=DOT, lw=1.0)
    out["H"] = y
    ax.axhline(0, color=LIGHT, lw=0.4, zorder=0)
    return out


def main():
    fig, axs = plt.subplots(
        2, 2, figsize=(COL_W, 10.0 * CM), gridspec_kw=dict(hspace=0.55, wspace=0.38)
    )
    data = {}
    ax = axs[0, 0]
    s = {k: np.asarray(v) for k, v in D["recovery"].items()}
    m = s["t"] <= 12.0
    s = {k: v[m] for k, v in s.items()}
    data["a_recovery"] = energy(
        ax, s, {"Kf": (8.0, 7.2, "center"), "Kl": (8.0, -5.6, "center"), "U": (1.0, -9.6, "left")}
    )
    ax.text(11.8, 1.0, r"$\mathcal{H}-\mathcal{H}_0$", fontsize=8, ha="right", va="bottom")
    ax.set(
        xlim=(0, 12),
        ylim=(-12, 14),
        xticks=[0, 4, 8, 12],
        xlabel="Time (s)",
        ylabel="Energy change (J)",
    )
    panel(ax, "a", x=-0.25)

    ax = axs[0, 1]
    data["b_weave"] = energy(ax, W, {"Kf": (62, 0.27, "right"), "U": (50, -0.25, "right")})
    ax.annotate(
        "Lateral kinetic",
        xy=(66, 0.095),
        xytext=(30, 0.11),
        color=ORANGE,
        fontsize=8,
        ha="center",
        va="center",
        arrowprops=dict(arrowstyle="-", lw=0.5, color=ORANGE),
    )
    ax.set(
        xlim=(0, 88),
        ylim=(-0.55, 0.35),
        xlabel="Time (s), means over one period",
        ylabel="Energy change (J)",
    )
    panel(ax, "b", x=-0.25)

    ax = axs[1, 0]
    s = cut(D["capsize"], 40.0, 8.0)
    data["c_capsize"] = energy(ax, s, {"Kf": (-2.6, 90, "right"), "U": (-2.6, -90, "right")})
    ax.set(
        xlim=(-8, 0), ylim=(-180, 180), xlabel="Time to 40 deg lean (s)", ylabel="Energy change (J)"
    )
    panel(ax, "c", x=-0.25)

    ax = axs[1, 1]
    s = cut(D["capsize"], 45.0, 1e9)
    lean = np.degrees(s["phi"])
    ax.plot(lean, s["V"], color=BLACK, lw=1.0)
    ax.plot(lean, s["Vcm"], color=GREY, lw=1.0)
    ax.plot(lean, s["Vsimple"], color=BLACK, lw=0.8, ls=DASH)
    ax.text(39, 5.95, "Rear contact", fontsize=8, ha="right")
    ax.text(45.8, 5.06, r"$\sqrt{V_0^2+2g\,\Delta h}$", fontsize=7.5, ha="right", va="bottom")
    ax.text(45, 4.62, "Mass centre", fontsize=8, ha="right", va="top", color=GREY)
    ax.set(
        xlim=(12, 46), ylim=(4.3, 6.2), xticks=[15, 25, 35, 45], xlabel=YLEAN, ylabel="Speed (m/s)"
    )
    panel(ax, "d", x=-0.25)
    data["d_speed"] = dict(
        lean_deg=lean,
        V_rear=s["V"],
        V_cm=s["Vcm"],
        V_estimate=s["Vsimple"],
        V_effective_mass=s["Veff"],
    )
    save_paper(fig, "energy", data)
    i = int(np.argmin(abs(lean - 45)))
    print(
        f"45 deg: rear {s['V'][i]:.2f}, mass centre {s['Vcm'][i]:.2f}, "
        f"estimate {s['Vsimple'][i]:.2f}, effective mass {s['Veff'][i]:.2f}"
    )
    b = data["b_weave"]
    print(f"weave end: dU {b['U'][-1]:.3f} dKf {b['Kf'][-1]:.3f} dKl {b['Kl'][-1]:.3f}")


if __name__ == "__main__":
    main()
