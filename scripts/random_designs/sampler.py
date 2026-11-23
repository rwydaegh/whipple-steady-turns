"""The 3000 random designs of the supplement, "Random designs" (Sect. "Measured bicycles")."""

import argparse
import gzip
import json
import math
from pathlib import Path

import numpy as np

from whipple.bicycles import BENCHMARK, measured_sets

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "data" / "random_designs" / "designs.json.gz"
SEED = 20261123
KEYS = ["w", "c", "lam", "g", "rR", "mR", "IRxx", "IRyy", "xB", "zB", "mB", "IBxx", "IByy",
        "IBzz", "IBxz", "xH", "zH", "mH", "IHxx", "IHyy", "IHzz", "IHxz", "rF", "mF", "IFxx",
        "IFyy"]  # fmt: skip


def anchors():
    """The 23 measured bicycle+rider sets."""
    return {k: {q: float(v[q]) for q in KEYS} for k, v in measured_sets("rider").items()}


def J_of(p, b):
    """Second-moment matrix J = tr(I)/2 - I of body b ("B" or "H")."""
    Ixx, Iyy, Izz, Ixz = (p[f"I{b}{s}"] for s in ("xx", "yy", "zz", "xz"))
    inertia = np.array([[Ixx, 0, Ixz], [0, Iyy, 0], [Ixz, 0, Izz]])
    return np.trace(inertia) / 2 * np.eye(3) - inertia


def I_of(J):
    inertia = np.trace(J) * np.eye(3) - J
    return dict(xx=inertia[0, 0], yy=inertia[1, 1], zz=inertia[2, 2], xz=inertia[0, 2])


def axis_coords(p):
    """Front-frame centre: height t up the steer axis from its ground point, offset d normal."""
    lam = p["lam"]
    u = np.array([-math.sin(lam), -math.cos(lam)])
    n = np.array([math.cos(lam), -math.sin(lam)])
    r = np.array([p["xH"] - (p["w"] + p["c"]), p["zH"]])
    return float(r @ u), float(r @ n)


def place_front(p, t, d):
    lam = p["lam"]
    u = np.array([-math.sin(lam), -math.cos(lam)])
    n = np.array([math.cos(lam), -math.sin(lam)])
    x = np.array([p["w"] + p["c"], 0.0]) + t * u + d * n
    p["xH"], p["zH"] = float(x[0]), float(x[1])


def rep(p):
    """Physical representation of a parameter set."""
    t, d = axis_coords(p)
    return dict(
        L={k: p[k] for k in ("w", "rR", "rF")},
        m={k: p[k] for k in ("mR", "mB", "mH", "mF")},
        xB=p["xB"], zB=p["zB"], t=t, d=d,
        JB=J_of(p, "B") / p["mB"], JH=J_of(p, "H") / p["mH"],
        kR=p["IRyy"] / (p["mR"] * p["rR"] ** 2), qR=p["IRxx"] / p["IRyy"],
        kF=p["IFyy"] / (p["mF"] * p["rF"] ** 2), qF=p["IFxx"] / p["IFyy"],
        c=p["c"], lam=p["lam"],
    )  # fmt: skip


def build(r):
    p = {"g": 9.81, "c": r["c"], "lam": r["lam"], **r["L"], **r["m"], "xB": r["xB"], "zB": r["zB"]}
    for b in ("B", "H"):
        for s, v in I_of(r["J" + b] * p["m" + b]).items():
            p[f"I{b}{s}"] = float(v)
    for wh in ("R", "F"):
        Iyy = r["k" + wh] * p["m" + wh] * p["r" + wh] ** 2
        p[f"I{wh}yy"], p[f"I{wh}xx"] = Iyy, r["q" + wh] * Iyy
    place_front(p, r["t"], r["d"])
    return {k: float(p[k]) for k in KEYS}


def admissible(p, r):
    for b in ("B", "H"):
        if np.linalg.eigvalsh(r["J" + b] * p["m" + b]).min() < 1e-6:
            return False
    for wh in ("R", "F"):
        if not (0.5 <= r["q" + wh] < 1.0 and 0 < r["k" + wh] <= 0.98):
            return False
    if not (p["zB"] < -0.2 and p["zH"] < -0.1 and 0 < p["xB"] < p["w"]):
        return False
    if not p["w"] > p["rR"] + p["rF"]:
        return False
    if not (0.02 <= p["c"] <= 0.18 and math.radians(13) <= p["lam"] <= math.radians(30)):
        return False
    return all(np.isfinite(list(p.values())))


def sample(rng, A):
    """Interpolate between two measured sets with a random weight, then perturb."""
    names = list(A)
    while True:
        a, b = rng.choice(len(names), 2, replace=False)
        ra, rb = rep(A[names[a]]), rep(A[names[b]])
        u = rng.uniform()
        r = dict(
            L={k: ra["L"][k] ** (1 - u) * rb["L"][k] ** u * math.exp(rng.normal(0, 0.04))
               for k in ra["L"]},
            m={k: ra["m"][k] ** (1 - u) * rb["m"][k] ** u * math.exp(rng.normal(0, 0.12))
               for k in ra["m"]},
            xB=(1 - u) * ra["xB"] + u * rb["xB"] + rng.normal(0, 0.03),
            zB=((1 - u) * ra["zB"] + u * rb["zB"]) * math.exp(rng.normal(0, 0.04)),
            t=((1 - u) * ra["t"] + u * rb["t"]) * math.exp(rng.normal(0, 0.10)),
            d=(1 - u) * ra["d"] + u * rb["d"] + rng.normal(0, 0.02),
        )  # fmt: skip
        for jb in ("JB", "JH"):
            J = (1 - u) * ra[jb] + u * rb[jb]
            th = rng.normal(0, 0.15)
            R = np.array([[math.cos(th), 0, -math.sin(th)], [0, 1, 0],
                          [math.sin(th), 0, math.cos(th)]])  # fmt: skip
            Am = R @ np.diag(np.exp(rng.normal(0, 0.08, 3)))
            r[jb] = Am @ J @ Am.T
        for k, s in (("kR", 0.06), ("kF", 0.06), ("qR", 0.04), ("qF", 0.04)):
            r[k] = ((1 - u) * ra[k] + u * rb[k]) * math.exp(rng.normal(0, s))
        r["c"] = rng.uniform(0.02, 0.18)
        r["lam"] = math.radians(rng.uniform(13.0, 30.0))
        p = build(r)
        if admissible(p, r):
            return p, dict(a=names[a], b=names[b], u=u)


def designs(n):
    rng = np.random.default_rng(SEED)
    A = anchors()
    out = [dict(id="anchor_" + k.replace("+", "_").replace("(", "").replace(")", ""),
                kind="anchor", name=k, params=p) for k, p in A.items()]  # fmt: skip
    out.append(dict(id="anchor_benchmark", kind="anchor", name="benchmark",
                    params={k: float(BENCHMARK.get(k, 9.81)) for k in KEYS}))  # fmt: skip
    for k in range(n):
        p, meta = sample(rng, A)
        out.append(dict(id=f"rand_{k:05d}", kind="rand", meta=meta, params=p))
    return out


def load():
    with gzip.open(OUT, "rt") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("n", type=int, nargs="?", default=3000)
    a = ap.parse_args()
    d = designs(a.n)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(OUT, "wt") as f:
        json.dump(d, f)
    print(f"wrote {OUT.relative_to(REPO)}: {len(d)} designs (24 anchors, {a.n} random)")


if __name__ == "__main__":
    main()
