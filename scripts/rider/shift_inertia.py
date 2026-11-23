"""Change of a4 when a rider translates sideways and the rear-frame products of inertia change (supplementary material, "Mass shift: further results")."""  # noqa: E501

import json
from pathlib import Path

from shift_model import (
    WhippleAsym,
    hurwitz,
    params_of,
    quartic,
    rider_parts,
    translated,
    with_rider,
)

from whipple.bicycles import BICYCLES, measured_sets

REPO = Path(__file__).resolve().parents[2]


def setups():
    pA = with_rider(BICYCLES["A"], 10.0)
    parts = rider_parts(10.0)
    m_r = sum(q[0] for q in parts)
    x_r = sum(q[0] * q[1] for q in parts) / m_r
    z_r = sum(q[0] * q[2] for q in parts) / m_r
    up = parts[1]
    full = params_of("Browser+Jason")
    bare = measured_sets("riderless")["Browser"]
    mr = full["mB"] - bare["mB"]
    xr = (full["mB"] * full["xB"] - bare["mB"] * bare["xB"]) / mr
    zr = (full["mB"] * full["zB"] - bare["mB"] * bare["zB"]) / mr
    return [
        ("A with rider, whole rider", pA, (m_r, x_r, z_r)),
        ("A with rider, upper body", pA, (up[0], up[1], up[2])),
        ("Browser+Jason, whole rider", full, (mr, xr, zr)),
    ]


def main():
    rows = []
    for name, p, rider in setups():
        for y_r in (0.03, 0.08):
            qT = translated(p, y_r, rider)
            qO = dict(p, yB=qT["yB"])
            for phi, delta in ((0.15, 0.08), (0.25, 0.10)):
                row = dict(set=name, y_r=y_r, yB=qT["yB"], phi=phi, delta=delta)
                for key, q in (("translate", qT), ("offset", qO)):
                    bb = WhippleAsym(q)
                    t = bb.turn(phi, delta)
                    row[key] = dict(V=t.V, T=t.T)
                    for plant in ("constant", "free"):
                        a = quartic(bb, phi, delta, plant)
                        row[key][plant] = dict(a4=a[3], D3=hurwitz(a))
                row["a4_rel_change"] = {
                    pl: abs(row["translate"][pl]["a4"] / row["offset"][pl]["a4"] - 1)
                    for pl in ("constant", "free")
                }
                rows.append(row)
                print(name, y_r, phi, row["a4_rel_change"])
    worst = {
        f"{y:g}": max(max(r["a4_rel_change"].values()) for r in rows if r["y_r"] == y)
        for y in (0.03, 0.08)
    }
    out = dict(rows=rows, max_a4_rel_change_by_y_r=worst)
    path = REPO / "data" / "rider" / "shift_inertia.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(worst)


if __name__ == "__main__":
    main()
