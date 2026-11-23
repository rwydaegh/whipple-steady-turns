"""Capsize rules against the exact largest stable lean of the measured sets (fig:rule-measured,
Sect. 11.1, "Accuracy of the capsize rules").
"""

import json

import numpy as np
from figstyle import BARE, BLACK, CM, COL_W, DATA, GREY, MK, all_results, panel, plt, save_paper

# The five sets with an inadmissible front-frame tensor use the corrected tensor.
CORRECTED = json.loads((DATA / "measured" / "tensor_corrections.json").read_text())["corrected"]


def collect():
    rows = []
    for name, d in all_results().items():
        kind = {"rider": "rider", "riderless": "bare"}.get(d.get("group"))
        if kind is None or name == "A" or "free" not in d:
            continue
        ex = d["free"]["max_lean_deg"]
        f1 = d["rule"]["free_first_order"]["phi_deg"]
        f2 = d["rule"]["free_second_order"]["phi_deg"]
        if not np.isfinite(f1):
            continue
        fix = CORRECTED.get(f"{name}|projected")
        tensor = "stored"
        if fix:
            fx = fix["fixed"]
            ex, f1, f2, tensor = (
                fx["lean_free"],
                fx["rule_free"],
                fx["rule_free_second"],
                "corrected",
            )
        rows.append(
            dict(name=name, kind=kind, exact=ex, first_order=f1, second_order=f2, tensor=tensor)
        )
    return rows


def stats(rows, key, kind="rider", cap=20.0):
    e = np.array([r[key] - r["exact"] for r in rows if r["kind"] == kind and r["exact"] < cap])
    return dict(
        n=len(e),
        rms=float(np.sqrt(np.mean(e**2))),
        max=float(e[np.argmax(abs(e))]),
        mean=float(e.mean()),
    )


S_RIDER = dict(marker="o", ms=4.0, mec=BLACK, ls="none", **MK)
S_BARE = dict(marker="s", ms=3.6, mec=BARE, ls="none", **MK)
S_EXCL = dict(marker="^", ms=4.0, mec=BARE, ls="none", **MK)  # riderless Browser
S_F2 = dict(marker="+", ms=4.5, mec=BLACK, mew=0.8, ls="none")


def main():
    rows = collect()
    fig, (ax, bx) = plt.subplots(
        1, 2, figsize=(COL_W, 5.8 * CM), gridspec_kw=dict(width_ratios=[1, 1.45], wspace=0.32)
    )
    lim = (5, 33)
    ax.plot(lim, lim, color=GREY, lw=0.6, zorder=1)
    br = [r for r in rows if r["name"] == "Browser"][0]
    yr = [r for r in rows if r["name"] == "Yellowrev+Jason"][0]
    for kind, sty in (("bare", S_BARE), ("rider", S_RIDER)):
        sel = [r for r in rows if r["kind"] == kind and r["name"] != "Browser"]
        ax.plot([r["exact"] for r in sel], [r["first_order"] for r in sel], **sty, zorder=3)
        bx.plot(
            [r["exact"] for r in sel], [r["first_order"] - r["exact"] for r in sel], **sty, zorder=3
        )
    ax.plot(br["exact"], br["first_order"], **S_EXCL, zorder=3)
    ax.annotate(
        "Yellowrev, Jason",
        xy=(yr["exact"], yr["first_order"]),
        xytext=(yr["exact"] - 1.2, yr["first_order"] - 0.3),
        fontsize=7.5,
        ha="right",
        va="center",
        arrowprops=dict(arrowstyle="-", lw=0.5, shrinkA=0, shrinkB=2.5),
    )
    ax.set(
        xlim=lim,
        ylim=lim,
        xticks=[10, 20, 30],
        yticks=[10, 20, 30],
        aspect="equal",
        xlabel="Exact (deg)",
        ylabel="First-order rule (deg)",
    )
    bx.axhline(0, color=GREY, lw=0.6)
    bx.plot(br["exact"], br["first_order"] - br["exact"], **S_EXCL, zorder=3)
    rid = [r for r in rows if r["kind"] == "rider"]
    bx.plot(
        [r["exact"] for r in rid], [r["second_order"] - r["exact"] for r in rid], **S_F2, zorder=4
    )
    bx.annotate(
        "Yellowrev,\nJason",
        xy=(yr["exact"], yr["first_order"] - yr["exact"]),
        xytext=(yr["exact"] - 0.6, yr["first_order"] - yr["exact"] - 0.38),
        fontsize=7.5,
        ha="center",
        va="top",
        linespacing=1.0,
        arrowprops=dict(arrowstyle="-", lw=0.5, shrinkA=0, shrinkB=2.5),
    )
    bx.annotate(
        "Browser (excluded)",
        xy=(br["exact"], br["first_order"] - br["exact"]),
        xytext=(br["exact"] + 1.8, br["first_order"] - br["exact"] + 0.12),
        fontsize=7.5,
        ha="left",
        va="center",
        arrowprops=dict(arrowstyle="-", lw=0.5, shrinkA=0, shrinkB=2.5),
    )
    bx.set(
        xlim=(5, 33),
        ylim=(-0.8, 1.8),
        xticks=[10, 20, 30],
        yticks=[-0.5, 0, 0.5, 1, 1.5],
        xlabel="Exact largest stable lean (deg)",
        ylabel="Rule $-$ exact (deg)",
    )
    items = [
        (S_RIDER, "First-order rule, with rider"),
        (S_F2, "Second-order rule, with rider"),
        (S_BARE, "First-order rule, no rider"),
    ]
    for i, (sty, text) in enumerate(items):
        y = 0.95 - 0.085 * i
        bx.plot([0.05], [y], transform=bx.transAxes, clip_on=False, **sty)
        bx.text(0.09, y, text, transform=bx.transAxes, fontsize=7.5, va="center")
    panel(bx, "b", x=-0.2)
    fig.canvas.draw()
    pa = ax.get_position()
    yb = bx.texts[-1].get_window_extent().transformed(fig.transFigure.inverted()).y0
    fig.text(pa.x0 - 0.32 * pa.width, yb, "(a)", fontsize=bx.texts[-1].get_fontsize(), va="bottom")
    st = {k: stats(rows, k) for k in ("first_order", "second_order")}
    st["first_order_no_rider"] = stats(rows, "first_order", kind="bare")
    save_paper(fig, "rule_accuracy", dict(rows=rows, stats_below_20deg=st), width=0.87 * COL_W)
    print(json.dumps(st, indent=1))


if __name__ == "__main__":
    main()
