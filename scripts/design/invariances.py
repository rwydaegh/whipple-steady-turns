"""Exact invariances of the model (Sect. 12.1, "Exact invariances"; supplement, "Invariances")."""

import argparse
import math

import numpy as np
from common import BASES, DATA, Base, write_json
from moves import add_to_frame, total_mass

from whipple.model import Whipple

OUT = DATA / "invariances.json"
NAMES = ("w c lam rR mR IRxx IRyy xB zB mB IBxx IByy IBzz IBxz "
         "xH zH mH IHxx IHyy IHzz IHxz rF mF IFxx IFyy g").split()  # fmt: skip
LENGTHS = ("w", "c", "rR", "rF", "xB", "zB", "xH", "zH")
_rng = np.random.default_rng(7)
QS = list(zip(_rng.uniform(0.04, 0.45, 36), _rng.uniform(0.02, 0.32, 36)))  # (lean, steer)
US = list(zip(_rng.uniform(-0.8, 0.8, 16), _rng.uniform(-0.9, 0.9, 16), _rng.uniform(-22, -7, 16)))


def objects(p):
    """dyn: M(q) and F(q, u) at 16 states. trim: S = s^2 and T at 36 turns. trimH: S, T and
    H - U(0) at the same turns. *_sf: scale-free ratios of the same."""
    b = Whipple(p)
    dyn = []
    for (phi, d), u in zip(QS, US):
        _, M, F, _ = b.parts(phi, d, np.array(u))
        dyn += [M[0, 0], M[0, 1], M[0, 2], M[1, 1], M[1, 2], M[2, 2], *F]
    U0 = b.potential(0.0, 0.0)
    S, T, H = [], [], []
    for phi, d in QS:
        _, M, Fg, _ = b.parts(phi, d, np.zeros(3))
        Fc = b.parts(phi, d, np.array([0.0, 0.0, 1.0]))[2] - Fg
        s = -Fg[0] / Fc[0]
        t = -Fg[1] - s * Fc[1]
        S.append(s)
        T.append(t)
        H.append(0.5 * M[2, 2] * s + b.potential(phi, d) - U0 - t * d)
    S, T, H = map(np.array, (S, T, H))
    return {
        "dyn": np.array(dyn), "trim": np.r_[S, T], "trimH": np.r_[S, T, H],
        "trim_sf": np.r_[S[1:] / S[0], T[1:] / T[0]],
        "trimH_sf": np.r_[S[1:] / S[0], T[1:] / T[0], H[1:] / H[0], H[0] / T[0]],
    }  # fmt: skip


def absorb(p, wheel, frame, dm):
    """Move the point mass dm from a wheel (at its axle) into its frame."""
    x0, z0 = (0.0, -p["rR"]) if wheel == "R" else (p["w"], -p["rF"])
    q = add_to_frame(p, frame, dm, x0, z0)
    q["m" + wheel] = p["m" + wheel] - dm
    return q


def axis_point(p, h, dm, off=0.0):
    """Move the point mass dm at height h on the steer axis (shifted by off in x) from the
    front frame to the rear frame."""
    x0, z0 = p["w"] + p["c"] - h * math.tan(p["lam"]) + off, -h
    return add_to_frame(add_to_frame(p, "H", -dm, x0, z0), "B", dm, x0, z0)


def transformations(p):
    """Finite transformations; True: the statement says the object stays the same."""
    e, al = 0.03, 1.7
    out = {
        "1: all masses and inertias times 1.7": total_mass(p, al),
        "2: rear wheel mass 1 kg to the rear frame": absorb(p, "R", "B", 1.0),
        "2: front wheel mass 1 kg to the front frame": absorb(p, "F", "H", 1.0),
        "2: rear wheel diametral inertia 0.03 to the rear frame":
            dict(p, IRxx=p["IRxx"] - e, IBxx=p["IBxx"] + e, IBzz=p["IBzz"] + e),
        "2: front wheel diametral inertia 0.03 to the front frame":
            dict(p, IFxx=p["IFxx"] - e, IHxx=p["IHxx"] + e, IHzz=p["IHzz"] + e),
        "(c): rear frame inertia + 1.0 identity":
            dict(p, IBxx=p["IBxx"] + 1, IByy=p["IByy"] + 1, IBzz=p["IBzz"] + 1),
        "(c): front frame inertia + 0.05 identity":
            dict(p, IHxx=p["IHxx"] + 0.05, IHyy=p["IHyy"] + 0.05, IHzz=p["IHzz"] + 0.05),
        "(a): g times 1.7": dict(p, g=p["g"] * al),
        "(a): lengths times 1.7, inertias times 1.7^2":
            {k: v * al if k in LENGTHS else v * al**2 if k[0] == "I" else v for k, v in p.items()},
        "(b): lengths and g times 1.3, masses / 1.3^2":
            {k: v * 1.3 if k in LENGTHS + ("g",) else v / 1.69 if k[0] == "m" else v
             for k, v in p.items()},
    }  # fmt: skip
    for h in (0.2, 0.6, 1.0):
        out[f"3: 0.5 kg on the steer axis at {h} m, front to rear frame"] = axis_point(p, h, 0.5)
    out["control: the same 50 mm off the axis"] = axis_point(p, 0.6, 0.5, 0.05)
    return out


def relchange(a, b):
    return float(np.max(np.abs(b - a)) / np.max(np.abs(a)))


def rank(p, keys=("dyn", "trim_sf", "trimH_sf")):
    """Singular values of the log-derivative Jacobian of each object (fourth-order differences)."""
    base = objects(p)
    J = {k: np.empty((len(base[k]), len(NAMES))) for k in keys}
    for j, n in enumerate(NAMES):
        h = 1e-4 * max(abs(p[n]), 1e-2)
        f = {m: objects(dict(p, **{n: p[n] + m * h})) for m in (-2, -1, 1, 2)}
        for k in keys:
            J[k][:, j] = (8 * (f[1][k] - f[-1][k]) - (f[2][k] - f[-2][k])) / (12 * h)
    scale = np.array([max(abs(p[k]), 1e-3) for k in NAMES])
    out = {}
    for k in keys:
        rows = np.maximum(np.abs(base[k]), 1e-6 * np.abs(base[k]).max())
        sv = np.linalg.svd(J[k] * scale / rows[:, None], compute_uv=False)
        sv = sv / sv[0]
        r = int(np.sum(sv > 1e-8))
        out[k] = dict(rank=r, last_kept=float(sv[r - 1]), first_dropped=float(sv[r]),
                      sv=sv.tolist())  # fmt: skip
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="benchmark only, no rank")
    a = ap.parse_args()
    res = {}
    for name in ["benchmark"] + ([] if a.quick else ["Browser+Jason"]):
        p = dict(BASES[name][0], g=9.81)
        obj = objects(p)
        base = Base(name)
        r = res.setdefault(name, {"objects": {}, "lean": {}})
        print(f"\n{name}: largest relative change of each object")
        for label, q in transformations(p).items():
            o = objects(q)
            r["objects"][label] = {k: relchange(obj[k], o[k]) for k in obj}
            print(
                f"  {label:58s} "
                + "  ".join(f"{k} {v:.1e}" for k, v in r["objects"][label].items())
            )
        # largest stable lean (free rolling) under the moves that leave it unchanged
        for label in (
            "1: all masses and inertias times 1.7",
            "2: rear wheel mass 1 kg to the rear frame",
            "2: front wheel diametral inertia 0.03 to the front frame",
            "3: 0.5 kg on the steer axis at 0.6 m, front to rear frame",
            "(a): g times 1.7",
            "(a): lengths times 1.7, inertias times 1.7^2",
        ):
            q = transformations(p)[label]
            lean = base.evaluate(q, ("lean_free",))
            r["lean"][label] = dict(
                lean=lean["lean_free"],
                V=lean["V_free"],
                change=lean["lean_free"] - base.values["lean_free"],
            )
            print(
                f"  phi* {lean['lean_free']:.10f} (base {base.values['lean_free']:.10f}), "
                f"V {lean['V_free']:.6f}: {label}"
            )
        if not a.quick:
            r["rank"] = rank(p)
            for k, v in r["rank"].items():
                print(
                    f"  rank of {k}: {v['rank']} (last kept {v['last_kept']:.1e}, first "
                    f"dropped {v['first_dropped']:.1e})"
                )
    write_json(OUT, res)


if __name__ == "__main__":
    main()
