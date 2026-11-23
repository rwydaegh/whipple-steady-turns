"""Capsize limit, capsize rules and countersteer counts (Sect. 6, 7 and supplement)."""

import gzip
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import brentq

from whipple import stability as st
from whipple.bicycles import BENCHMARK
from whipple.model import Whipple

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data" / "capsize"
_SPEC = importlib.util.spec_from_file_location("capsize_common", REPO / "scripts/capsize/common.py")
cc = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cc)


def data(name):
    return json.loads((DATA / name).read_text())


@pytest.fixture(scope="module")
def bench():
    p = cc.clean(BENCHMARK)
    return Whipple(p), cc.upright(p)["lin"]


def test_a4_from_statics_at_random_turns(bench):
    bike, lin = bench
    rng = np.random.default_rng(7)
    for _ in range(6):
        phi, V = rng.uniform(0.05, 0.5), rng.uniform(2.0, 8.0)
        d = cc.delta_at(bike, lin, phi, V)
        a = cc.a4_both(bike, phi, d)
        for plant in ("free", "constant"):
            a4 = st.characteristic(bike, phi, d, plant)[3]
            assert a["a4_" + plant] == pytest.approx(a4, rel=1e-7, abs=1e-8)


def test_a4_is_zero_at_the_stored_benchmark_fold(bench):
    bike, lin = bench
    with gzip.open(REPO / "data/measured/results/benchmark.json.gz", "rt") as fh:
        f = json.load(fh)["free"]
    phi = math.radians(f["max_lean_deg"])
    d = cc.delta_at(bike, lin, phi, f["V_at_max_lean"])
    a = cc.a4_both(bike, phi, d)
    scale = abs(cc.a4_both(bike, phi - 0.02, cc.delta_at(bike, lin, phi - 0.02, 4.5))["a4_free"])
    assert abs(a["a4_free"]) < 1e-5 * scale


def test_torque_limit_on_the_loading_path(bench):
    bike, lin = bench
    sol = cc.path(bike, lin, 5.0, "load", phi_end=0.25)

    def a4(x):
        return cc.a4_both(bike, x, float(sol.sol(x)[0]))["a4_free"]

    x = brentq(a4, 0.18, 0.23, xtol=1e-12)
    a = cc.a4_both(bike, x, float(sol.sol(x)[0]))
    assert math.degrees(x) == pytest.approx(11.83, abs=0.005)
    assert a["T"] == pytest.approx(-0.1238, abs=5e-5)
    assert a["V"] == pytest.approx(5.102, abs=5e-4)


def test_straight_running_limit(bench):
    bike, lin = bench
    limit = np.linalg.det(lin.K(5.0)) / np.linalg.det(lin.M)
    assert limit == pytest.approx(93.346, abs=5e-4)
    for phi, rel in ((1e-2, 2.2e-3), (1e-3, 2.2e-5)):
        value = st.capsize_formula(bike, phi, cc.delta_at(bike, lin, phi, 5.0))
        assert abs(value / limit - 1) == pytest.approx(rel, rel=0.1)


def test_paths_and_two_thirds_ratio():
    s = data("loading_paths.json")["summary"]
    assert (s["paths"], s["sets"]) == (68, 17)
    assert s["ratio_two_thirds"] == pytest.approx([0.968, 1.020], abs=5e-4)
    assert s["H_path_fold_vs_extremum_rad"] < 3.8e-8
    assert s["loading_fold_vs_extremum_rad"] < 2.4e-8
    assert s["loading_vs_H_fold_rad"] == pytest.approx(7.8e-7, abs=5e-8)
    assert s["free_over_constant_limit"] == pytest.approx([0.78, 0.92], abs=5e-3)
    assert s["constant_minus_free_lean_deg"] == pytest.approx([0.6, 3.6], abs=0.05)


def test_torque_ramps():
    r = {(x["plant"], x["rate"]): x for x in data("torque_ramp.json")["ramps"]}
    assert r["free", 0.004]["T_at_fold_lean"] == pytest.approx(-0.1592, abs=5e-5)
    assert r["free", 0.001]["T_at_fold_lean"] == pytest.approx(-0.1377, abs=5e-5)
    assert r["free", 0.004]["ratio_to_T_c"] == pytest.approx(1.285, abs=5e-4)
    assert r["free", 0.001]["ratio_to_T_c"] == pytest.approx(1.112, abs=5e-4)
    assert r["free", 0.001]["V_at_fold_lean"] == pytest.approx(5.102, abs=5e-4)


def test_rule_table():
    d = data("rule_accuracy.json")
    t = d["tab_rule"]
    row = t["measured bicycle and rider sets, lean below 20 deg"]
    assert row["n"] == 21
    assert row["exact"] == pytest.approx([9.0, 18.3], abs=0.05)
    assert abs(row["second_order"]["largest"]) < 0.02
    assert row["first_order"]["rms"] == pytest.approx(0.20, abs=0.005)
    assert row["first_order"]["largest"] == pytest.approx(0.39, abs=0.005)
    s = t["Schwab et al. rider"]
    assert s["second_order"] - s["exact"] == pytest.approx(0.08, abs=0.005)
    assert s["first_order"] - s["exact"] == pytest.approx(0.81, abs=0.005)
    y = t["Yellowrev with rider"]
    assert y["second_order"] - y["exact"] == pytest.approx(0.66, abs=0.005)
    assert y["first_order"] - y["exact"] == pytest.approx(1.66, abs=0.005)
    row = t["benchmark, A, B and nine riderless sets"]
    assert row["exact"] == pytest.approx([8.4, 17.1], abs=0.05)
    assert row["second_order"]["rms"] == pytest.approx(0.07, abs=0.005)
    assert row["second_order"]["largest"] == pytest.approx(-0.14, abs=0.005)
    assert row["first_order"]["rms"] == pytest.approx(0.28, abs=0.005)
    assert row["first_order"]["largest"] == pytest.approx(-0.59, abs=0.005)
    b = d["benchmark"]
    assert (b["second_order"], b["V"], b["exact"]) == pytest.approx((12.45, 4.54, 12.47), abs=5e-3)
    c = d["constant_rule_below_20deg"]
    assert (c["rms"], c["largest"]) == pytest.approx((0.22, 0.45), abs=5e-3)


def test_random_designs():
    s = data("random_designs.json")["summary"]["free"]
    assert s["n"] == 40
    assert s["exact"] == pytest.approx([5.7, 20.2], abs=0.05)
    assert s["second_order"]["rms"] == pytest.approx(0.04, abs=0.005)
    assert s["second_order"]["largest"] == pytest.approx(0.16, abs=0.005)
    assert s["first_order"]["rms"] == pytest.approx(0.49, abs=0.005)
    assert s["first_order"]["largest"] == pytest.approx(1.92, abs=0.005)


def test_countersteer_counts_on_the_grids():
    t = json.loads((REPO / "data/countersteer/grid_summary.json").read_text())["totals"]
    assert (t["sets"], t["turns"]) == (42, 42350)
    assert t["max_Rpd"] == pytest.approx(-5.7, abs=0.05) and t["max_Rpd_set"] == "Gyro"
    f, c = t["free"], t["constant"]
    assert (f["tested"], f["covered"], f["covered_T_neg"], f["monotone"]) == (
        4521,
        4165,
        4165,
        4165,
    )
    assert (f["below_vc"], f["below_vc_T_pos"], f["above_vc"], f["above_vc_T_pos"]) == (
        92,
        41,
        264,
        227,
    )
    assert (c["stable"], c["covered"], c["covered_T_neg"], c["slope_neg"]) == (
        7018,
        6041,
        6041,
        7018,
    )
    assert (t["first_hands_free"], t["first_hands_free_unstable"]) == (553, 553)
    assert (t["zeros_of_T"], t["zeros_without_fold"]) == (966, 1)


def test_order_of_the_two_folds():
    t = json.loads((REPO / "data/countersteer/grid_summary.json").read_text())["totals"]
    o, s = t["order"], t["signs"]
    assert (o["both"], o["free_first"]) == (638, 636)
    assert o["gap_max_deg_without_long_trail"] == pytest.approx(6.8, abs=0.05)
    assert (s["O1"]["n"], s["O1"]["sets"], s["O2"]["n"], s["O2"]["sets"]) == (343, 16, 890, 19)
    assert s["O3"]["n"] == 7284
    assert s["O1"]["lowest_lean_deg"] >= 20
    assert s["O1"]["stable"] == s["O2"]["stable"] == 0
    gaps = sorted(-g for e in o["exceptions"].values() for _, g in e)
    assert gaps == pytest.approx([0.013, 0.015], abs=1e-3)
