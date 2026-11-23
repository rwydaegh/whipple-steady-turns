"""Region of stable turns of the benchmark bicycle and the measured sets (Sects. 5, 8.3 and 11)."""

import gzip
import json
import math
from pathlib import Path

import numpy as np
import pytest

from whipple import stability as st
from whipple.bicycles import BENCHMARK, measured_sets
from whipple.linear import Linear
from whipple.model import Whipple
from whipple.quartic import bracket

DATA = Path(__file__).resolve().parent.parent / "data" / "measured"


def result(name):
    with gzip.open(DATA / "results" / f"{name}.json.gz", "rt") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def rider():
    return {n: result(n) for n in measured_sets("rider")}


@pytest.fixture(scope="module")
def bench():
    return result("benchmark"), Whipple(BENCHMARK)


def test_upright_speeds():
    lin = Linear(BENCHMARK)
    assert lin.weave_speed() == pytest.approx(4.292, abs=5e-4)
    assert lin.capsize_speed() == pytest.approx(6.024, abs=5e-4)
    V_a3 = math.sqrt(-lin.g * bracket(lin.C1, lin.K0) / bracket(lin.C1, lin.K2))
    assert V_a3 == pytest.approx(2.761, abs=5e-4)


def test_benchmark_tip(bench):
    """BT point at 9.26 deg and 2.79 m/s, recomputed by Newton's method from a stored guess."""
    r, bike = bench
    guess = np.array(r["free"]["curves"]["hopf"][-1]) * 1.01
    t = bike.turn(*st.bogdanov_takens(bike, guess))
    assert math.degrees(t.phi) == pytest.approx(9.26, abs=0.005)
    assert t.V == pytest.approx(2.79, abs=0.005)
    assert t.T == pytest.approx(-0.858, abs=5e-4)


def test_benchmark_largest_stable_lean(bench):
    """Top of the fold curve: 12.47 deg at 4.54 m/s."""
    r, bike = bench
    fold = np.array(r["free"]["curves"]["fold"])
    guess = fold[np.argmax(fold[:, 0])]
    x = st.boundary_extremum(bike, lambda x: st.characteristic(bike, *x)[3], lambda t: t.phi, guess)
    t = bike.turn(*x)
    assert math.degrees(t.phi) == pytest.approx(12.47, abs=0.005)
    assert t.V == pytest.approx(4.54, abs=0.005)


def test_all_rider_sets_have_bt_tip(rider):
    assert len(rider) == 23
    assert all(r["free"]["tip_type"] == "BT" for r in rider.values())
    tips = [r["free"]["BT_V"] for r in rider.values()]
    assert (round(min(tips), 2), round(max(tips), 2)) == (2.70, 4.36)


def rule_errors(rider, form):
    return {
        n: r["rule"][f"free_{form}"]["phi_deg"] - r["free"]["max_lean_deg"]
        for n, r in rider.items()
        if r["free"]["max_lean_deg"] < 20
    }


def test_first_order_rule(rider):
    """21 sets below 20 deg: rms 0.20 deg, largest +0.39 deg (Rigid with Jason), 19 too high."""
    e = rule_errors(rider, "first_order")
    assert len(e) == 21
    assert math.sqrt(np.mean(np.square(list(e.values())))) == pytest.approx(0.20, abs=0.005)
    worst = max(e, key=lambda n: abs(e[n]))
    assert worst == "Rigid+Jason" and e[worst] == pytest.approx(0.39, abs=0.005)
    assert sum(v > 0 for v in e.values()) == 19


def test_second_order_rule():
    """21 sets below 20 deg: rms 0.010 deg, largest error 0.02 deg."""
    s = json.loads((DATA / "tensor_corrections.json").read_text())["rule_stats_below_20deg"]
    s = s["projected"]["rule_free_second"]
    assert s["n"] == 21
    assert s["rms"] == pytest.approx(0.0101, abs=5e-4)
    assert abs(s["max_signed"]) == pytest.approx(0.02, abs=0.005)


def test_tip_speed_formula(rider):
    """V_a3 / V_BT is 0.98 to 1.05 for the 23 sets (free rolling)."""
    q = [r["upright"]["V_a3"] / r["free"]["BT_V"] for r in rider.values()]
    assert (round(min(q), 2), round(max(q), 2)) == (0.98, 1.05)


def test_inadmissible_front_frames():
    """Five rider sets break I_max <= I_mid + I_min for the front frame (tab:tensors)."""
    bad = {}
    for n, p in measured_sets("rider").items():
        inertia = np.array(
            [[p["IHxx"], 0, p["IHxz"]], [0, p["IHyy"], 0], [p["IHxz"], 0, p["IHzz"]]]
        )
        lam = np.sort(np.linalg.eigvalsh(inertia))
        if lam[2] > lam[1] + lam[0]:
            bad[n] = lam[2] - lam[1] - lam[0]
    assert len(bad) == 5
    assert bad["Yellow+Jason"] == pytest.approx(0.00227, abs=5e-6)
    assert bad["DelftBrowser+Arend(Moore2012)"] == pytest.approx(0.04750, abs=5e-6)
