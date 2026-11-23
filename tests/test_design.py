"""Design rules (Sect. 12) and the random designs (supplement, "Random designs")."""

import gzip
import importlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data" / "design"
_NAMES = ("common", "moves", "invariances", "analyse", "sampler")


def _load(folder, name):
    """Import scripts/<folder>/<name>.py without clashing with modules of other script folders."""
    saved = {n: sys.modules.pop(n) for n in _NAMES if n in sys.modules}
    sys.path.insert(0, str(REPO / "scripts" / folder))
    try:
        mod = importlib.import_module(name)
        deps = {n: sys.modules[n] for n in _NAMES if n in sys.modules}
    finally:
        sys.path.pop(0)
        for n in _NAMES:
            sys.modules.pop(n, None)
        sys.modules.update(saved)
    return mod, deps


invariances, _deps = _load("design", "invariances")
common = _deps["common"]
BASES, lean_star, stored = common.BASES, common.lean_star, common.stored
analyse, _ = _load("random_designs", "analyse")


def read(name):
    return json.loads((DATA / name).read_text())


@pytest.fixture(scope="module")
def bench():
    p = dict(BASES["benchmark"][0], g=9.81)
    fold = np.array(stored("benchmark")["free"]["curves"]["fold"])
    d, V, x, ok = lean_star(p, "free", fold[np.argmax(fold[:, 0])])
    assert ok and d == pytest.approx(12.47, abs=0.005)
    return p, d, V, x


@pytest.mark.parametrize("label", [
    "1: all masses and inertias times 1.7",
    "2: rear wheel mass 1 kg to the rear frame",
    "2: front wheel diametral inertia 0.03 to the front frame",
    "3: 0.5 kg on the steer axis at 0.6 m, front to rear frame",
    "(a): lengths times 1.7, inertias times 1.7^2",
    "(a): g times 1.7",
])  # fmt: skip
def test_largest_lean_invariant(bench, label):
    """Statements 1-3 of Sect. 12.1 and (a): the largest stable lean does not change."""
    p, d0, V0, x0 = bench
    q = invariances.transformations(p)[label]
    d, V, _, ok = lean_star(q, "free", x0)
    assert ok and d == pytest.approx(d0, abs=1e-7)
    factor = math.sqrt(1.7) if label.startswith("(a)") else 1.0
    assert V == pytest.approx(V0 * factor, rel=1e-7)


def test_equations_of_motion_invariant(bench):
    """Statements 2, 3 and (b) leave M(q) and F(q, u) unchanged; a mass off the axis does not."""
    p = bench[0]
    base = invariances.objects(p)["dyn"]
    for label, q in invariances.transformations(p).items():
        change = invariances.relchange(base, invariances.objects(q)["dyn"])
        if label[0] in "23" or label.startswith("(b)"):
            assert change < 1e-10, label
        if label.startswith("control"):
            assert change > 1e-5


@pytest.mark.parametrize("base, move, value", [
    ("benchmark", "trail", 0.69), ("Crescendo+Jason", "trail", 0.58),
    ("Browser+Jason", "trail", 0.69), ("benchmark", "front_spin_inertia", -0.55),
    ("Crescendo+Jason", "front_spin_inertia", -0.47), ("benchmark", "rear_frame_forward", 0.40),
    ("Browser+Jason", "rear_frame_forward", 0.45), ("benchmark", "wheelbase_only", -1.57),
    ("Crescendo+Jason", "wheelbase_only", -1.21), ("benchmark", "rear_wheel_back", 1.05),
    ("Crescendo+Jason", "front_radius_same_frame", 1.20), ("benchmark", "head_angle", 0.45),
    ("Crescendo+Jason", "head_angle_front_frame_rotates", 0.25),
    ("benchmark", "rear_frame_taller", 0.34), ("benchmark", "front_frame_forward_of_axis", 1.3),
])  # fmt: skip
def test_published_elasticities(base, move, value):
    e = read("elasticities.json")[base]["moves"][move]["lean_free"]
    nd = 1 if move.endswith("of_axis") else 2
    assert round(e, nd) == pytest.approx(value, abs=1e-9)


def test_total_mass_elasticity_is_zero():
    for b in BASES:
        assert abs(read("elasticities.json")[b]["moves"]["total_mass"]["lean_free"]) < 1e-6


def test_tradeoff_and_loads():
    a = read("elasticities.json")["A_rider"]["moves"]
    assert round(a["head_angle"]["v_w"], 2) == pytest.approx(0.48)
    assert round(a["front_assembly_forward"]["tip_V"], 2) == pytest.approx(0.43)
    loads = read("loads.json")["benchmark"]
    assert round(loads["Basket 5 kg on the handlebar (front frame)"]["lean_free"], 2) == 17.30
    assert round(loads["Child seat 20 kg"]["tip_V"], 2) == 3.21


def test_design_example():
    d = read("design_example.json")
    m = d["Modified"]
    assert (round(m["lean_free"], 2), round(m["v_w"], 2), round(m["v_c"], 2)) == (
        18.86,
        5.58,
        10.12,
    )
    assert round(d["Measured"]["lean_free"], 2) == 17.26


def test_second_order_rule_close_to_exact():
    r = json.loads((REPO / "data" / "design" / "rule_elasticities.json").read_text())
    e = read("elasticities.json")
    worst = max(abs(r[b]["free_second_order"][k] - e[b]["moves"][k]["lean_free"])
                for b in ("benchmark", "Crescendo+Jason") for k in r[b]["free_second_order"]
                if k != "base_deg")  # fmt: skip
    assert worst < 0.0185


@pytest.fixture(scope="module")
def designs():
    return analyse.load()


def test_random_design_types(designs):
    R = [r for r in designs if r["kind"] == "rand"]
    t = analyse.types(R)
    assert t["exact"] == {"closed_BT": 2711, "closed_zH": 93, "open": 189, "no_band": 7}
    assert [(b["n"], b["open"]) for b in t["D_free"]] == [
        (155, 150),
        (44, 23),
        (35, 10),
        (82, 6),
        (613, 0),
        (2064, 0),
    ]


def test_random_design_rule_errors(designs):
    e = analyse.rule_errors(designs)
    assert [b["n"] for b in e["bins"]] == [225, 540, 688, 598, 331, 98]
    assert [round(b["first_median"], 2) for b in e["bins"]] == [0.06, 0.12, 0.37, 0.85, 1.43, 1.88]
    assert [round(b["second_p90"], 2) for b in e["bins"]] == [0.01, 0.02, 0.07, 0.21, 0.57, 2.20]
    assert e["differ_gt_1_or_none"] == {"n": 354, "first_error_gt_1": 350}


def test_design_list_matches_results():
    with gzip.open(REPO / "data" / "random_designs" / "designs.json.gz", "rt") as f:
        ids = [d["id"] for d in json.load(f)]
    assert len(ids) == 3024 and ids[-1] == "rand_02999"
