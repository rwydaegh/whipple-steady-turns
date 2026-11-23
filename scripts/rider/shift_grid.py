"""Grid of steady turns (V, phi_G, Delta) with a fixed mass shift and a free steer torque (Sect. 13.3, Fig. fig:rider(b); supplementary material, "Mass shift: further results")."""  # noqa: E501

import argparse
import time
from pathlib import Path

import numpy as np
from shift_model import Whipple, delta_for_speed, hurwitz, params_of, quartic, yB_from_Delta

from whipple.linear import Linear

REPO = Path(__file__).resolve().parents[2]
FULL = REPO / "results" / "rider"
DATA = REPO / "data" / "rider"
FIG_DELTA = 2.0  # deg, the fixed shift drawn in Fig. fig:rider(b)


def axes(kind):
    if kind == "wide":
        return (
            np.arange(2.0, 9.0001, 0.2),
            np.radians(np.arange(1.0, 50.01, 1.0)),
            np.radians([0.0, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 14.0]),
        )
    if kind == "quick":
        return (
            np.arange(2.0, 8.0001, 0.2),
            np.radians(np.arange(0.5, 26.01, 1.0)),
            np.radians(np.arange(-3.0, 9.001, 1.0)),
        )
    return (
        np.arange(2.0, 8.0001, 0.1),
        np.radians(np.arange(0.5, 26.01, 0.5)),
        np.radians(np.arange(-3.0, 9.001, 0.5)),
    )


def compute(p, Vs, phiGs, Ds, log=True):
    b, lin = Whipple(p), Linear(p)
    shape = (len(Vs), len(phiGs), len(Ds))
    out = {k: np.full(shape, np.nan) for k in ("delta", "T", "ay", "phi", "yB")}
    for plant in ("constant", "free"):
        out[plant] = np.full(shape + (5,), np.nan)
    t0 = time.time()
    for i, V in enumerate(Vs):
        r = lin.steer_ratio(V)
        for k, D in enumerate(Ds):
            yB = yB_from_Delta(p, D)
            bb = b.with_params(yB=yB)
            guess = None
            for j, phiG in enumerate(phiGs):
                phi = phiG - D
                try:
                    d = delta_for_speed(bb, phi, V, r * phiG if guess is None else guess)
                    t = bb.turn(phi, d)
                    out["delta"][i, j, k], out["T"][i, j, k], out["ay"][i, j, k] = d, t.T, t.ay
                    out["phi"][i, j, k], out["yB"][i, j, k] = phi, yB
                    for plant in ("constant", "free"):
                        a = quartic(bb, phi, d, plant)
                        out[plant][i, j, k] = [*a, hurwitz(a)]
                    guess = d * phiGs[min(j + 1, len(phiGs) - 1)] / phiG
                except (RuntimeError, ValueError, ZeroDivisionError, np.linalg.LinAlgError):
                    guess = None
        if log:
            print(f"V {V:.2f} done, {time.time() - t0:.0f} s", flush=True)
    return out


def stable(q):
    a1, a2, a3, a4, d3 = np.moveaxis(q, -1, 0)
    with np.errstate(invalid="ignore"):
        return (a1 > 0) & (a2 > 0) & (a3 > 0) & (a4 > 0) & (d3 > 0)


def reduce(G, path):
    """Store the stable masks, T, a_y and, at Delta = 2 deg, a4 and D3 of free rolling."""
    D = np.degrees(G["Delta"])
    k = int(np.argmin(np.abs(D - FIG_DELTA)))
    red = dict(
        V=G["V"],
        phiG=G["phiG"],
        Delta=G["Delta"],
        stable_free=stable(G["free"]),
        stable_constant=stable(G["constant"]),
        T=G["T"].astype(np.float32),
        ay=G["ay"].astype(np.float32),
    )
    if abs(D[k] - FIG_DELTA) < 1e-9:
        red.update(
            fig_Delta=G["Delta"][k], fig_a4=G["free"][:, :, k, 3], fig_d3=G["free"][:, :, k, 4]
        )
    DATA.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **red)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="coarse grid (1 deg, 0.2 m/s)")
    ap.add_argument("--wide", action="store_true", help="large leans and shifts, coarse")
    ap.add_argument("--reduce", action="store_true", help="only rewrite data/ from results/")
    a = ap.parse_args()
    kind = "wide" if a.wide else ("quick" if a.quick else "fine")
    name = {"fine": "benchmark", "quick": "benchmark_quick", "wide": "benchmark_wide"}[kind]
    full = FULL / f"shift_grid_{name}.npz"
    if not a.reduce:
        out = compute(params_of("benchmark"), *axes(kind))
        FULL.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(full, V=axes(kind)[0], phiG=axes(kind)[1], Delta=axes(kind)[2], **out)
    reduce(np.load(full), DATA / f"shift_grid_{name}.npz")


if __name__ == "__main__":
    main()
