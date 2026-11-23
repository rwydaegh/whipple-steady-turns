"""Tests of the comparison with Cain and Perkins (2012) (Sect. 15.2, Table cain)."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "cain_perkins", REPO / "scripts" / "experiments" / "cain_perkins.py"
)
cp = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cp)


@pytest.mark.parametrize("s", [1, 2])
def test_table_a1_stiffness(s):
    # K0 and K2 from the rounded Table A1 parameters agree with the printed matrices to 0.5 %.
    K0, K2 = cp.table_a1_K(s)
    assert K0 == pytest.approx(np.array(cp.A1_K[s]["K0"]), rel=6e-3)
    assert K2 == pytest.approx(np.array(cp.A1_K[s]["K2"]), rel=6e-3)


@pytest.mark.parametrize("s, m, b, vc", [(1, 0.0971, -5.962, 7.837), (2, 0.0991, -5.854, 7.685)])
def test_linear_ratio(s, m, b, vc):
    L = cp.linear_ratio(cp.parameter_set(s, "Rigid+Jason"))
    assert (L["m"], L["b"], L["v_c"]) == pytest.approx((m, b, vc), abs=1e-3)


def test_stored_zero_speeds():
    d = json.loads((REPO / "data" / "experiments" / "cain_perkins.json").read_text())
    v0 = {s: [v["fit"]["v_zero"] for k, v in d["nonlinear"].items() if k[0] == s] for s in "12"}
    assert min(v0["1"]) > 7.48 and max(v0["1"]) < 7.58
    assert min(v0["2"]) > 7.35 and max(v0["2"]) < 7.45
    assert d["bootstrap"]["2"]["v_zero_95"] == pytest.approx([6.95, 9.45], abs=0.01)
