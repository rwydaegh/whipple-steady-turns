"""Elasticity of the largest stable lean for design moves, fig:design-chart (Sect. 12.2, "Design
moves")."""

from figstyle import BLACK, CM, COL_W, DATA, GREY, LIGHT, MK, plt, read_json, save_paper
from matplotlib.transforms import blended_transform_factory

GROUPS = [
    ("Robust", [
        ("trail", "Trail (steer axis slides forward)"),
        ("front_spin_inertia", "Front-wheel spin inertia"),
        ("rear_frame_forward", "Rider forward"),
        ("rear_frame_mass", "Rider mass (same build)"),
        ("rear_frame_higher", "Rider higher"),
        ("total_mass", "All masses and inertias"),
    ]),
    ("Wheelbase", [
        ("wheelbase_only", "Only $w$; bodies stay"),
        ("front_assembly_forward", "Front assembly forward"),
        ("rear_wheel_back", "Rear wheel back"),
        ("stretch", "Whole bicycle stretched"),
    ]),
    ("Front-wheel radius", [
        ("front_radius", "Trail and head angle fixed"),
        ("front_radius_same_fork", "Same fork; trail grows"),
        ("front_radius_same_frame", "Same frame; frame pitches back"),
    ]),
    ("Head angle (slacker)", [
        ("head_angle", "Front frame stays in place"),
        ("head_angle_front_frame_rotates", "Front frame turns with axis"),
    ]),
]  # fmt: skip
BASES = ("benchmark", "Crescendo+Jason")


def layout(exact, rule):
    """Rows (y, group, label, exact values, rule values), group headers, bottom y."""
    rows, heads, y = [], [], 0.0
    for g, items in GROUPS:
        heads.append((g, y))
        y -= 1.0
        for key, label in items:
            ex = [exact[b]["moves"][key]["lean_free"] for b in BASES]
            ru = [rule[b]["free_second_order"][key] for b in BASES]
            rows.append((y, g, label, ex, ru))
            y -= 1.0
        y -= 0.35
    return rows, heads, y


def main():
    exact = read_json(DATA / "design" / "elasticities.json")
    rule = read_json(DATA / "design" / "rule_elasticities.json")
    rows, heads, ybot = layout(exact, rule)
    fig, axs = plt.subplots(1, 2, figsize=(13.1 * CM, 8.6 * CM), sharey=True,
                            gridspec_kw=dict(wspace=0.08))  # fmt: skip
    fig.subplots_adjust(left=0.36, right=0.98, top=0.93, bottom=0.11)
    for k, (ax, title) in enumerate(zip(axs, ["(a)  Benchmark", "(b)  Crescendo, Jason"])):
        ax.axvspan(-0.1, 0.1, color=LIGHT, alpha=0.45, lw=0, zorder=0)
        ax.axvline(0, color="k", lw=0.5, zorder=3)
        for y, _, _, ex, ru in rows:
            ax.barh(y, ex[k], height=0.62, color=GREY, ec="none", zorder=2)
            ax.plot(ru[k], y, "D", ms=3.2, mec=BLACK, zorder=4, **MK)
        for _, y in heads:
            if y < 0:
                ax.axhline(y + 0.5, color="#D0D0D0", lw=0.5, zorder=0)
        ax.set(xlim=(-1.8, 1.6), xticks=[-1.5, -1, -0.5, 0, 0.5, 1, 1.5], ylim=(ybot + 0.6, 0.6))
        ax.set_xticklabels(["$-1.5$", "$-1$", "$-0.5$", "0", "0.5", "1", "1.5"])
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.text(0.5, 1.01, title, transform=ax.transAxes, ha="center", va="bottom", fontsize=8.5)
    ax0 = axs[0]
    ax0.set_yticks([r[0] for r in rows], [r[2] for r in rows])
    ax0.tick_params(axis="y", labelsize=8)
    tr = blended_transform_factory(fig.transFigure, ax0.transData)
    for g, y in heads:
        ax0.text(0.0, y, g, transform=tr, ha="left", va="center", fontsize=8.5)
    fig.supxlabel("Elasticity of largest stable lean (% per %)", fontsize=9, x=0.675, y=0.0)
    kx = fig.add_axes([0.0, 0.0, 0.3, 0.09])
    kx.set(xlim=(0, 1), ylim=(0, 1))
    kx.axis("off")
    kx.barh(0.72, 0.07, left=0.05, height=0.32, color=GREY, ec="none")
    kx.text(0.16, 0.72, "Exact", fontsize=8, va="center")
    kx.plot(0.085, 0.22, "D", ms=3.2, mec=BLACK, **MK)
    kx.text(0.16, 0.22, "Second-order rule", fontsize=8, va="center")
    data = [
        dict(
            group=g,
            label=lab,
            benchmark=ex[0],
            crescendo_jason=ex[1],
            rule_benchmark=ru[0],
            rule_crescendo_jason=ru[1],
        )
        for _, g, lab, ex, ru in rows
    ]
    save_paper(fig, "design_moves", dict(plant="free rolling", rows=data), width=0.985 * COL_W)
    worst = max(abs(d[f"rule_{b}"] - d[b]) for d in data for b in ("benchmark", "crescendo_jason"))
    print(f"largest |second-order rule - exact|: {worst:.4f}")


if __name__ == "__main__":
    main()
