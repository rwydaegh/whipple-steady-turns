"""Style, sizes, colours, paths and data helpers of the paper figures."""

import gzip
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

from whipple import plotstyle as ps  # noqa: E402  (sets the rcParams of the paper)

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
OUT = REPO / "results" / "figures"
MEASURED_RESULTS = DATA / "measured" / "results"

# ---- sizes ---------------------------------------------------------------------------------------
CM = ps.CM
COL_W = 13.5 * CM  # text width of the paper
FULL_W, FULL_H = 14.0 * CM, 6.2 * CM  # figures of the supplementary material
HALF_W, HALF_H = 7.0 * CM, 4.6 * CM

# ---- one colour per concept, the same in every figure --------------------------------------------
WEAVE = ps.HOPF  # blue: weave (Hopf) curve
CAPSIZE = ps.FOLD  # red: capsize (fold) curve
STABLE = ps.STABLE  # green: stable region, light fill
STABLE_ALPHA = ps.STABLE_ALPHA
STABLE_TEXT = ps.STABLE_TEXT
BLACK, GREY, LIGHT = "#000000", ps.GREY, ps.LIGHT
BARE = "#808080"  # grey: bicycle without rider
ORANGE = ps.BIKE["A"]  # second bicycle (A) when two bicycles are compared
LW, LW_THIN = ps.LW, ps.LW_THIN
DASH, DOT = ps.DASH, ps.DOT  # dashed: formula; dotted: auxiliary
EXACT_LW, EXACT_ALPHA = 2.0, 0.5  # exact curve under a formula curve: wide and light
MK = dict(mfc="none", mew=0.9)  # unfilled markers

XV = r"Speed $V$ (m/s)"
YLEAN = r"Lean $\phi$ (deg)"
YSTAR = r"Largest stable lean (deg)"
DEG = np.degrees

plt.rcParams.update({
    "legend.frameon": True, "legend.fancybox": False, "legend.edgecolor": "black",
    "legend.framealpha": 0.92, "legend.borderpad": 0.25, "legend.handlelength": 1.6,
    "legend.fontsize": 8, "patch.linewidth": 0.5,
    "pdf.fonttype": 42,  # embed TrueType fonts, not Type 3
})  # fmt: skip


def fit_width(fig, target=COL_W, tol=0.002):
    """Change the figure width (fonts fixed) until the tight-bbox PDF is ``target`` inches wide.

    The paper includes most figures at the text width (13.5 cm). Without this step the tight
    bbox crops the PDF, LaTeX scales it up, and the 8 pt text prints at a different size in each
    figure.
    """
    pad = 2 * plt.rcParams["savefig.pad_inches"]
    prev = None
    for _ in range(8):
        fig.canvas.draw()
        w = fig.get_tightbbox(fig.canvas.get_renderer()).width + pad
        W, H = fig.get_size_inches()
        if abs(w - target) < tol:
            break
        slope = 0.85 if prev is None or abs(W - prev[0]) < 1e-6 else (w - prev[1]) / (W - prev[0])
        prev = (W, w)
        fig.set_size_inches(W + (target - w) / max(slope, 0.3), H)


def save_paper(fig, name, data=None, width=COL_W):
    """Write results/figures/<name>.pdf, .png (300 dpi) and, if given, the data as <name>.json.

    The PDF is made exactly ``width`` wide (default 13.5 cm, the text width); pass the width at
    which the paper includes the figure, or None to keep the set size.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    if width:
        fit_width(fig, width)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=300)
    plt.close(fig)
    if data is not None:
        (OUT / f"{name}.json").write_text(json.dumps(data, indent=1, default=_plain))
    print("saved", (OUT / f"{name}.pdf").relative_to(REPO))


def _plain(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return str(x)


def label(ax, x, y, text, color="k", **kw):
    kw.setdefault("fontsize", 8)
    kw.setdefault("ha", "left")
    kw.setdefault("va", "center")
    return ax.text(x, y, text, color=color, **kw)


def panel(ax, letter, x=-0.18, y=1.02):
    ax.text(x, y, f"({letter})", transform=ax.transAxes, ha="left", va="bottom", fontsize=9)


def tip_marker(ax, x, y, color="k"):
    ax.plot(x, y, "o", ms=3.8, mfc="white", mec=color, mew=0.8, zorder=6)


def fill_region(ax, xh, yh, xf, yf, color=STABLE, alpha=STABLE_ALPHA, **kw):
    """Fill between the weave curve and the capsize curve, which meet at the tip."""
    pts = list(zip(xh, yh)) + list(zip(xf[::-1], yf[::-1]))
    ax.add_patch(Polygon(pts, closed=True, fc=color, ec="none", alpha=alpha, zorder=1, **kw))


# ---- data ----------------------------------------------------------------------------------------
NICE = {
    "Jason72(Hess2012)": "Jason (Hess 2012)",
    "rider(Schwab2013)": "rider (Schwab 2013)",
    "Arend(Moore2012)": "Arend (Moore 2012)",
    "Jason(Moore2012)": "Jason (Moore 2012)",
}


def nice_name(name):
    """'Browser+Jason72(Hess2012)' -> 'Browser, Jason (Hess 2012)'."""
    if name == "benchmark":
        return "Benchmark"
    if "+" not in name:
        return name
    bike, rider = name.split("+", 1)
    bike = bike.replace("DelftBrowser", "Delft Browser")
    bike = bike.replace("Balanceassistv1", "Balance-assist e-bike")
    return f"{bike}, {NICE.get(rider, rider)}"


def read_json(path):
    """Read a .json or .json.gz file."""
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as f:
            return json.load(f)
    return json.loads(path.read_text())


def load_result(name):
    """Stored result of one parameter set (scripts/measured/analyse.py)."""
    return read_json(MEASURED_RESULTS / f"{name}.json.gz")


def all_results():
    """All stored results of scripts/measured/analyse.py, by name."""
    out = {}
    for f in sorted(MEASURED_RESULTS.glob("*.json.gz")):
        d = read_json(f)
        out[d["name"]] = d
    return out


def boundary(res, plant="free"):
    """Exact weave and capsize curves of a stored result: (V_h, lean_h, V_f, lean_f) in m/s, deg.

    The stored curves run from the upright speed (lean ~ 0) to the tip, where they meet.
    ``plant`` is "free" or "constant".
    """
    d = res[plant]
    h = np.array(d["curves"]["hopf"]), np.array(d["curves_V"]["hopf"])
    f = np.array(d["curves"]["fold"]), np.array(d["curves_V"]["fold"])
    return h[1], DEG(h[0][:, 0]), f[1], DEG(f[0][:, 0])


def params_of(res):
    return dict(res["params"])
