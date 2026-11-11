"""Figure style: print sizes, one colour per concept, direct labels, PDF and PNG output."""

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

from .output import RESULTS  # noqa: E402
from .stability import trimmed_turn  # noqa: E402

CM = 1 / 2.54
FULL, HALF = 15.5 * CM, 7.5 * CM
G = 9.81

HOPF, FOLD, CASTOR = "#0000FF", "#FF0000", "#00A000"
STABLE, STABLE_ALPHA, STABLE_TEXT = "#00C000", 0.16, "#006000"
HANDS_FREE, CONTOUR, GREY, LIGHT = "#000000", "#B0B0B0", "#808080", "#C8C8C8"
BIKE = {"benchmark": "#000000", "A": "#FF7F00", "B": "#8000C0"}
LW, LW_THIN = 1.0, 0.6
DASH, DOT = (0, (4, 2)), (0, (1, 1.5))

XV = r"Speed $V$ (m/s)"
YLEAN = r"Lean $\phi$ (deg)"
YAY = r"Lateral acceleration $a_y/g$"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True, "axes.unicode_minus": False,
    "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.linewidth": 0.5, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "lines.linewidth": LW, "lines.markersize": 3.5, "contour.negative_linestyle": "solid",
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})  # fmt: skip


def figure(width=FULL, height=6.0 * CM, nrows=1, ncols=1, **kw):
    return plt.subplots(nrows, ncols, figsize=(width, height), **kw)


def save(fig, name):
    """Write results/<name>.pdf and results/<name>.png (300 dpi)."""
    RESULTS.mkdir(exist_ok=True)
    fig.savefig(RESULTS / f"{name}.pdf")
    fig.savefig(RESULTS / f"{name}.png", dpi=300)
    plt.close(fig)


def panel(ax, letter, x=-0.2):
    ax.text(x, 1.03, f"({letter})", transform=ax.transAxes, ha="left", va="bottom")


def label(ax, x, y, text, color="k", **kw):
    kw.setdefault("fontsize", 8)
    kw.setdefault("ha", "left")
    kw.setdefault("va", "center")
    ax.text(x, y, text, color=color, **kw)


def bt_marker(ax, x, y, color="k", filled=False):
    ax.plot(x, y, "o", ms=3.8, mfc=color if filled else "white", mec=color, mew=0.8, zorder=6)


def region(ax, x_hopf, y_hopf, x_fold, y_fold):
    """Fill the region between the Hopf curve and the fold curve, which meet at the tip."""
    points = list(zip(x_hopf, y_hopf)) + list(zip(x_fold[::-1], y_fold[::-1]))
    ax.add_patch(Polygon(points, closed=True, fc=STABLE, ec="none", alpha=STABLE_ALPHA, zorder=1))


def contour_labels(ax, cs, y_cap, x_min=-np.inf, x_max=np.inf, fmt="${:g}$"):
    """Label each contour once, at the top of an arc below y_cap, else where it crosses y_cap."""
    for level, segments in zip(cs.levels, cs.allsegs):
        parts = [s[(s[:, 0] > x_min) & (s[:, 0] < x_max)] for s in segments]
        parts = [p for p in parts if len(p) > 1]
        below = [p for p in parts if p[:, 1].max() <= y_cap]
        if below:
            seg = max(below, key=len)
            x, y = seg[np.argmax(seg[:, 1])]
        elif parts:
            seg = max(parts, key=len)
            x, y = seg[seg[:, 1] > y_cap][:, 0].max(), y_cap
        else:
            continue
        ax.text(
            x,
            y,
            fmt.format(level),
            fontsize=7,
            color=GREY,
            ha="center",
            va="center",
            zorder=1.5,
            bbox=dict(boxstyle="square,pad=0.08", fc="white", ec="none"),
        )


def tongue_curve(bike, points, kind="lean", plant="free"):
    """Speed and lean (deg) or a_y/g along a curve of (lean, steer) points."""
    turns = [trimmed_turn(bike, *x, plant) for x in points]
    y = [np.degrees(t.phi) if kind == "lean" else t.ay / G for t in turns]
    return np.array([t.V for t in turns]), np.array(y)


def draw_tongue(ax, bike, t, kind="lean", plant="free", color=None, ls="-", fill=True,
                beyond=True, filled_marker=False):  # fmt: skip
    """Hopf curve, fold curve and tip of a region of stable turns, blue and red by default."""
    xh, yh = tongue_curve(bike, t.hopf, kind, plant)
    xf, yf = tongue_curve(bike, t.fold, kind, plant)
    if fill:
        region(ax, xh, yh, xf, yf)
    ax.plot(xh, yh, color=color or HOPF, ls=ls, zorder=3)
    ax.plot(xf, yf, color=color or FOLD, ls=ls if color is None else DASH, zorder=3)
    if beyond:
        ax.plot(*tongue_curve(bike, t.fold_beyond, kind, plant), color=FOLD, ls=DOT, lw=LW_THIN)
    bt_marker(ax, xh[-1], yh[-1], color=color or "k", filled=filled_marker)
    return xh[-1], yh[-1]
