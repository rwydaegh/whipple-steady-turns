"""Torque coefficients and bicycle types (Sect. 10; supplement, "Bicycle types")."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from whipple.bicycles import BENCHMARK

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data" / "types"


def _load(name):
    sys.path.insert(0, str(REPO / "scripts" / "types"))
    saved = sys.modules.pop("common", None)
    try:
        spec = importlib.util.spec_from_file_location(
            f"types_{name}", REPO / "scripts/types" / f"{name}.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.path.pop(0)
        sys.modules.pop("common", None)
        if saved is not None:
            sys.modules["common"] = saved
    return mod


tc = _load("common")
fo = _load("first_order")


def data(name):
    return json.loads((DATA / name).read_text())


def test_quartic_polynomial_of_the_benchmark():
    o = tc.poly_data(tc.clean(BENCHMARK))
    assert o["coef"] == pytest.approx([-0.3362, -5.323, 40.851, 8.833, 0.2610], abs=5e-4)
    assert o["coef"][0] == pytest.approx(o["b0_check"], rel=1e-8)
    assert (o["r_c"], o["r3"], o["V3"]) == pytest.approx((0.288, 0.172, 7.79), abs=5e-3)
    assert (o["v_c"], o["tau3_vc"], o["V_r"]) == pytest.approx((6.024, 1.74, 0.58), abs=5e-3)
    assert o["err_tau3"] < 7e-9
    assert o["c30"] < 1e-12 and o["c21"] < 1e-12


def test_first_order_in_steer_and_v_min():
    w = fo.first_order(tc.clean(BENCHMARK))
    assert max(w[k] for k in "abcd") < 1e-11
    assert w["e"] < 1e-4
    v = fo.v_min(tc.clean(BENCHMARK), degs=(10, 20, 30, 45))
    assert [v[k] for k in (10, 20, 30, 45)] == pytest.approx([0.94, 2.00, 3.12, 4.98], abs=0.005)


def test_types_of_the_sets():
    d = data("upright_types.json")
    rows = d["rows"]
    assert d["types"]["open"] == ["Silver"]
    assert d["types"]["no capsize speed"] == ["Yellowrev"]
    s = rows["Silver"]
    assert (s["r_c"], s["r3"], s["tau3_vc"]) == pytest.approx((0.161, 0.238, -0.24), abs=5e-3)
    assert rows["Fisher"]["tau3_vc"] == pytest.approx(0.057, abs=5e-4)
    assert rows["Gyro"]["tau3_vc"] == pytest.approx(0.031, abs=5e-4)
    assert rows["Yellowrev+Jason"]["tau3_vc"] == pytest.approx(0.086, abs=5e-4)
    assert d["P4_largest_relative_error"] < 7e-9 and d["P4_sets"] == 40


def test_index_vc_over_v3():
    i = data("upright_types.json")["index"]
    assert i["n"] == 23
    assert i["spearman"] == pytest.approx(0.939, abs=5e-4)
    assert i["spearman_admissible"] == pytest.approx(0.917, abs=5e-4)
    assert len(i["inadmissible"]) == 5
    assert i["line"]["worst"] == "Yellowrev+Jason"
    assert i["line"]["largest"] == pytest.approx(5.42, abs=0.005)
    assert i["line"]["rms"] == pytest.approx(1.85, abs=0.005)
    assert i["line"]["leave_one_out_rms"] == pytest.approx(2.47, abs=0.005)
    assert i["line"]["leave_one_out_largest"] == pytest.approx(8.70, abs=0.005)
    c = i["closed"]["without Balanceassistv1"]
    assert c["n"] == 13
    assert (c["spearman_constant"], c["spearman_free"]) == pytest.approx((0.92, 0.88), abs=0.005)


@pytest.mark.parametrize(
    "name, plant, local, opening, lean, V",
    [
        ("benchmark trail", "free", 189.1, 198.9, 33.3, 9.30),
        ("benchmark trail", "constant", 146.2, 162.5, 33.0, 8.57),
        ("A trail", "free", 330.4, 354.6, 31.4, 9.18),
        ("A trail", "constant", 277.7, 320.4, 31.5, 8.81),
    ],
)
def test_critical_trails(name, plant, local, opening, lean, V):
    o = data("critical_trails.json")[name][plant]
    assert 1000 * o["local"] == pytest.approx(local, abs=0.05)
    assert 1000 * o["opening"] == pytest.approx(opening, abs=0.05)
    assert (o["lean_deg"], o["V"]) == pytest.approx((lean, V), abs=0.05)


def test_paths_between_measured_bicycles():
    d = data("critical_trails.json")
    y = d["Yellow to Yellowrev"]
    assert y["constant"]["opening"] == pytest.approx(0.32, abs=0.005)
    assert y["no capsize speed above"] == pytest.approx(0.953, abs=2e-3)
    t = d["tau3 at v_c"]
    assert t["Silver with the trail of Crescendo"] == pytest.approx(0.08, abs=0.005)
    assert t["Silver with the wheelbase of Crescendo"] == pytest.approx(0.11, abs=0.005)
    assert t["Crescendo with the trail of Silver"] == pytest.approx(0.25, abs=0.005)
    assert t["Silver"] == pytest.approx(-0.24, abs=0.005)


def test_trail_series_and_maps():
    s = data("trail_series.json")
    assert s["V3_range_trail_80_to_250mm"][0] > 7.75 and s["V3_range_trail_80_to_250mm"][1] < 8.2
    tops = [r["lean_deg"] for r in s["tops"]["benchmark"]]
    assert tops[0] == pytest.approx(12.47, abs=0.005) and np.all(np.diff(tops) > 0)
    for c, closed in ((0.080, True), (0.220, False)):
        m = np.load(DATA / f"benchmark_c{c:.3f}.npz")
        top = m["phis"][np.flatnonzero(m["stable"].any(axis=0)).max()]
        assert (top < 15) if closed else (top > 35)
