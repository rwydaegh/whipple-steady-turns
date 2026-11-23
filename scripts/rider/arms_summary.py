"""Numbers of Sect. 13.2, "Steer torque with a centring spring", and of the supplementary section "Centring spring: further results" with Table tab:arms-lit, from the results of arms_analyse.py."""  # noqa: E501

import json
import math
from pathlib import Path

import numpy as np
from arms_analyse import LITERATURE, OUT
from arms_model import Upright, all_sets

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "rider"
B = 0.25  # m, distance of each hand from the steer axis
FR = (0.0, 0.2, 0.4, 0.6, 0.8, 0.9)


def load():
    out = {}
    for name in all_sets():
        safe = name.replace("+", "_").replace("(", "").replace(")", "")
        out[name] = json.loads((OUT / f"{safe}.json").read_text())
    return out


def per_set(d):
    row = {
        k: d[k]
        for k in (
            "k_star",
            "k_star_eigen_scan",
            "c_star",
            "c_star_eigen_scan",
            "Ia_star",
            "k0",
            "kappa_c",
            "V_a3",
            "v_c",
            "v_w",
        )
    }
    row["k_hand"] = d["k_star"] / (2 * B * B)
    for plant in ("free", "constant"):
        r = d[plant]
        rows = {round(q["frac"], 2): q for q in r["k_rows"]}
        l0 = rows[0.0]["lean_star"]
        xs = np.array([q["frac"] for q in r["k_rows"] if q["frac"] <= 0.61])
        ys = np.array([q["lean_star"] / l0 for q in r["k_rows"] if q["frac"] <= 0.61])
        row[plant] = dict(
            lean_star={f"{q['frac']:g}": q["lean_star"] for q in r["k_rows"]},
            tip_limited={f"{q['frac']:g}": q["tip_limited"] for q in r["k_rows"]},
            rule_order1={f"{f:g}": rows[f]["rule_order1"]["phi_deg"] for f in FR},
            rule_order2={f"{f:g}": rows[f]["rule_order2"]["phi_deg"] for f in FR},
            slope=float(-np.sum(xs * (ys - 1)) / np.sum(xs * xs)),
            beyond_k=[
                dict(frac=q["frac"], n_stable=q["n_stable"], min_lean=q.get("min_lean"))
                for q in r["beyond"]
            ],
            damper_lean_star={f"{q['frac']:g}": q["lean_star"] for q in r["c_rows"]},
            literature={
                k: dict(region=q["band"] is not None, lean=q["lean"], band=q["band"])
                for k, q in r["literature"].items()
            },
        )
    return row


def main():
    R = load()
    sets = {n: per_set(d) for n, d in R.items()}
    riders = [n for n in sets if n != "benchmark"]
    ks = np.array([sets[n]["k_star"] for n in riders])
    cs = np.array([sets[n]["c_star"] for n in riders])
    kh = {n: sets[n]["k_hand"] for n in riders}
    adults = [n for n in riders if n not in ("Gyro+Child", "Yellowrev+Jason")]
    browsers = [n for n in riders if "Browser" in n]
    S = dict(
        k_star_riders=dict(
            min=ks.min(),
            max=ks.max(),
            median=float(np.median(ks)),
            n_between_3p2_and_7p2=int(((ks >= 3.2) & (ks <= 7.2)).sum()),
            argmin=riders[int(ks.argmin())],
            argmax=riders[int(ks.argmax())],
        ),
        k_star_browser=dict(
            min=min(sets[n]["k_star"] for n in browsers),
            max=max(sets[n]["k_star"] for n in browsers),
        ),
        c_star_riders=dict(min=cs.min(), max=cs.max(), median=float(np.median(cs))),
        k_hand=dict(
            benchmark=sets["benchmark"]["k_hand"],
            adults_min=min(kh[n] for n in adults),
            adults_max=max(kh[n] for n in adults),
            child=kh["Gyro+Child"],
            yellowrev=kh["Yellowrev+Jason"],
        ),
        k_star_scan_max_abs=max(abs(s["k_star"] - s["k_star_eigen_scan"]) for s in sets.values()),
        c_star_scan_max_abs=max(abs(s["c_star"] - s["c_star_eigen_scan"]) for s in sets.values()),
    )
    for plant in ("free", "constant"):
        sl = [sets[n][plant]["slope"] for n in sets]
        S[f"slope_{plant}"] = dict(min=min(sl), max=max(sl))
        ok = [n for n in sets if sets[n][plant]["lean_star"]["0"] <= 20.0]
        for order in (1, 2):
            e = [
                sets[n][plant][f"rule_order{order}"][f"{f:g}"]
                - sets[n][plant]["lean_star"][f"{f:g}"]
                for n in ok
                for f in FR
            ]
            S[f"rule_order{order}_{plant}_max_abs_error_deg"] = float(np.nanmax(np.abs(e)))
        S[f"rule_sets_{plant}"] = len(ok)
    island = {
        n: [
            q["min_lean"]
            for pl in ("free", "constant")
            for q in sets[n][pl]["beyond_k"]
            if q["n_stable"]
        ]
        for n in sets
    }
    S["beyond_k_star"] = dict(
        sets_without_stable_turn=sum(not v for v in island.values()),
        sets_with_island=sorted(n for n, v in island.items() if v),
        island_min_lean_deg=min(min(v) for v in island.values() if v),
    )
    S["literature_sets_with_region_free"] = {
        k: sum(sets[n]["free"]["literature"][k]["region"] for n in sets) for k in LITERATURE
    }
    b = sets["benchmark"]
    S["benchmark"] = dict(
        k_star=b["k_star"],
        k0=b["k0"],
        kappa_c=b["kappa_c"],
        c_star=b["c_star"],
        Ia_star=b["Ia_star"],
        V_a3=b["V_a3"],
        v_c=b["v_c"],
        lean_star_free=b["free"]["lean_star"],
        damper_lean_star_free=b["free"]["damper_lean_star"],
        literature_free=b["free"]["literature"],
    )
    u = Upright(all_sets()["benchmark"])
    lam = np.roots(np.r_[1.0, u.a(5.0, 10.0)]).real.max()
    S["benchmark"]["growth_rate_5ms_k10"] = float(lam)
    S["benchmark"]["growth_rate_locked_steer"] = math.sqrt(-u.Kpp / u.Mpp)
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "arms_summary.json").write_text(
        json.dumps(dict(summary=S, sets=sets), indent=1, default=float) + "\n"
    )
    print(json.dumps(S, indent=1, default=float))


if __name__ == "__main__":
    main()
