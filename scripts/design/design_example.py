"""Design example on Crescendo with Jason, tab:design-example (supplement, "A design example")."""

import math

import numpy as np
from common import DATA, Base, largest_ay, torque_per_g, write_json
from moves import fork_offset

OUT = DATA / "design_example.json"
NSUB = 10
LAM = math.radians(17.0)  # steer axis 17 deg from the vertical (head angle 73 deg)


def modified(p):
    """Steeper head, zero fork offset. The front-frame centre keeps its height along the axis
    and its distance ahead of the axis; its inertia rotates with the axis."""
    q = dict(p, lam=LAM, c=p["rF"] * math.tan(LAM))
    up = lambda r: np.array([-math.sin(r["lam"]), -math.cos(r["lam"])])  # noqa: E731
    fwd = lambda r: np.array([math.cos(r["lam"]), -math.sin(r["lam"])])  # noqa: E731
    d = np.array([p["xH"] - p["w"] - p["c"], p["zH"]])
    s, e = d @ up(p), d @ fwd(p)
    pos = np.array([q["w"] + q["c"], 0.0]) + s * up(q) + e * fwd(q)
    q["xH"], q["zH"] = float(pos[0]), float(pos[1])
    t = LAM - p["lam"]
    R = np.array([[math.cos(t), math.sin(t)], [-math.sin(t), math.cos(t)]])
    inertia = R @ np.array([[p["IHxx"], p["IHxz"]], [p["IHxz"], p["IHzz"]]]) @ R.T
    q["IHxx"], q["IHxz"], q["IHzz"] = inertia[0, 0], inertia[0, 1], inertia[1, 1]
    return q


def row(base, p, values):
    ay, _, _ = largest_ay(p, "free", values["x_free"])
    return dict(
        lean_free=values["lean_free"], ay_g=ay, v_w=values["v_w"], v_c=values["v_c"],
        tip_V=values["tip_V"], T_per_g_4=abs(torque_per_g(p, 4.0)),
        head_angle_deg=90 - math.degrees(p["lam"]), trail_mm=1e3 * p["c"],
        offset_mm=1e3 * fork_offset(p),
        flop_mm=1e3 * p["c"] * math.sin(p["lam"]) * math.cos(p["lam"]),
    )  # fmt: skip


def main():
    base = Base("Crescendo+Jason")
    p0, p1 = base.p, modified(base.p)
    for k in range(1, NSUB + 1):
        pk = {q: p0[q] + k / NSUB * (p1[q] - p0[q]) for q in p0}
        v = base.evaluate(pk, ("lean_free", "tip_V"))
        base.guess["free"], base.guess["tip"] = v["x_free"], v["x_tip"]
    v.update(base.evaluate(p1, ("v_w", "v_c")))
    res = {"Measured": row(base, p0, base.values), "Modified": row(base, p1, v)}
    print(
        "| | phi* (deg) | a_y*/g | v_w | v_c | V_tip | |T/a_y| at 4 m/s | head | trail | "
        "offset | flop |"
    )
    for name, r in res.items():
        print(f"| {name} | {r['lean_free']:.2f} | {r['ay_g']:.3f} | {r['v_w']:.2f} | "
              f"{r['v_c']:.2f} | {r['tip_V']:.2f} | {r['T_per_g_4']:.2f} | "
              f"{r['head_angle_deg']:.1f} | {r['trail_mm']:.1f} | {r['offset_mm']:.1f} | "
              f"{r['flop_mm']:.1f} |")  # fmt: skip
    write_json(OUT, res)


if __name__ == "__main__":
    main()
