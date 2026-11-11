import json
from pathlib import Path

import numpy as np
import pytest

from whipple import stability as st
from whipple.linear import Linear
from whipple.model import Whipple

DATA = Path(__file__).resolve().parent.parent / "whipple" / "data" / "measured_bicycles.json"
BIKES = json.loads(DATA.read_text())


@pytest.mark.parametrize("name, vw, vc", [("Gyro", 2.909, 4.439), ("Stratos", 3.605, 5.314),
                                          ("Rigid", 5.008, 6.429)])  # fmt: skip
def test_upright_speeds(name, vw, vc):
    lin = Linear(BIKES[name])
    assert lin.weave_speed((2.5, 6.5)) == pytest.approx(vw, abs=1e-3)
    assert lin.capsize_speed() == pytest.approx(vc, abs=1e-3)


def test_tongue_tip_of_pista():
    bike = Whipple(BIKES["Pista"])
    t = st.tongue(bike, phi_max=0.8, ds_max=1e-2, weave_bracket=(2.5, 6.5))
    bt = bike.turn(*t.bt)
    assert np.degrees(bt.phi) == pytest.approx(8.98, abs=0.01)
    assert bt.V == pytest.approx(2.464, abs=1e-3)
    assert bt.T == pytest.approx(-0.117, abs=1e-3)
