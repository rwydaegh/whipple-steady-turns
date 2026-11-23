"""Steer torque against hands-free riding with the shift alone, and the errors of eq. rider-rule (Sect. 13.3, "Mass shift and steer torque"; supplementary material, "Mass shift: further results")."""  # noqa: E501

import json
import math
from pathlib import Path

import numpy as np
from shift_model import tag
from shift_rule import OUT as RULES
from shift_tongue import DATA as TONGUES
from shift_tongue import SETS

REPO = Path(__file__).resolve().parents[2]


def deg(x):
    return math.degrees(x)


def main():
    out = {}
    for name in SETS:
        row = {}
        for pl in ("free", "constant"):
            r0 = json.loads((TONGUES / f"{tag(name)}_f0_{pl}.json").read_text())
            r1 = json.loads((TONGUES / f"{tag(name)}_f1_{pl}.json").read_text())
            L0, L1, F1 = r0["largest"]["phiG"], r1["largest"]["phiG"], r1["largest"]["phi"]
            row[pl] = dict(
                torque_deg=deg(L0["phiG"]),
                torque_V=L0["V"],
                hands_free_deg=deg(L1["phiG"]),
                hands_free_V=L1["V"],
                hands_free_frame_deg=deg(L1["phi"]),
                hands_free_Delta_deg=deg(L1["Delta"]),
                gain_deg=deg(L1["phiG"] - L0["phiG"]),
                speed_drop=L0["V"] - L1["V"],
                largest_frame_lean_deg=deg(F1["phi"]),
                largest_frame_lean_V=F1["V"],
                tip_torque_V=r0["bt"]["V"] if r0["bt"] else None,
                tip_hands_free_V=r1["bt"]["V"] if r1["bt"] else None,
            )
        rule = json.loads((RULES / f"{tag(name)}.json").read_text())
        row["rule_error_deg"] = {k: v["error_deg"] for k, v in rule["summary"].items()}
        rows = rule["rows"]
        row["beta21_range"] = [min(q["beta21"] for q in rows), max(q["beta21"] for q in rows)]
        row["b21"] = [dict(V=q["V"], b21=q["c21"]) for q in rows]
        out[name] = row

    def span(key, pl):
        v = [out[n][pl][key] for n in SETS]
        return [min(v), max(v)]

    tips = [abs(out[n]["free"]["tip_torque_V"] - out[n]["free"]["tip_hands_free_V"]) for n in SETS]
    err = {k: [out[n]["rule_error_deg"][k] for n in SETS] for k in out[SETS[0]]["rule_error_deg"]}
    b = out["benchmark"]["b21"]
    out["summary"] = dict(
        gain_free=span("gain_deg", "free"),
        speed_drop_free=span("speed_drop", "free"),
        gain_constant=span("gain_deg", "constant"),
        tip_speed_change_free_max=max(tips),
        rule_error_deg={
            k: dict(min=min(v), max=max(v), rms=float(np.sqrt(np.mean(np.square(v)))))
            for k, v in err.items()
        },
        benchmark_b21={f"{V:g}": min(b, key=lambda q: abs(q["V"] - V)) for V in (2.8, 4.4, 5.5)},
        benchmark_beta21_range=out["benchmark"]["beta21_range"],
    )
    path = REPO / "data" / "rider" / "shift_summary.json"
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out["summary"], indent=1))
    for n in SETS:
        print(
            n,
            {
                pl: {k: round(v, 3) for k, v in out[n][pl].items() if v is not None}
                for pl in ("free", "constant")
            },
        )


if __name__ == "__main__":
    main()
