import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from whipple.bicycles import BENCHMARK
from whipple.model import Whipple
from whipple.weave import Leaf, TorqueFamily, weave_run, weave_start

C_DIR = Path(__file__).resolve().parent.parent / "c"
TORQUE, OFFSET = -0.4179855714497167, 794.1195


def test_weave_lean_maxima_and_chi():
    # Level H = 626.5 J (V = 3.6419 m/s) is periodic with two lean maxima per period, 11.561 and
    # 13.648 deg (compiled run: 0.201778 and 0.238206 rad), and a zero exponent.
    bike = Whipple(BENCHMARK)
    fam = TorqueFamily(bike, TORQUE, 0.159, 0.119)
    phi_h = fam.hopf(0.139, 0.179)
    phi_f, _ = fam.fold(phi_h, phi_h + 0.1)
    H = 626.5 + OFFSET
    y0, saddle = weave_start(fam, H, phi_h, phi_f)
    assert fam.turn(y0[0]).V == pytest.approx(3.6418870, abs=1e-6)
    r = weave_run(Leaf(bike, TORQUE, H), y0, saddle + 0.3, t_trans=30.0, t_run=15.0)
    assert r["status"] == "bounded"
    assert np.sort(np.degrees(r["maxima"][-2:])) == pytest.approx([11.561, 13.648], abs=0.01)
    assert abs(r["chi"]) < 0.1


@pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc with OpenMP")
def test_compiled_batch_program(tmp_path):
    # Level H = 610.75 J (V = 3.5907 m/s) is periodic with two lean maxima, 16.346 and 10.131 deg.
    exe = tmp_path / "hires"
    src = [str(C_DIR / f) for f in ("hires.c", "engine_model.c", "_generated.c")]
    subprocess.run(["gcc", "-O2", "-fopenmp", *src, "-lm", "-o", str(exe)], check=True)
    out = tmp_path / "out.txt"
    levels = str(C_DIR / "levels_example.txt")
    args = ["0.008", "300", "100", "0.5", "1"]
    subprocess.run([str(exe), levels, str(out), *args], check=True, capture_output=True)
    rows = {int(line.split()[0]): line.split() for line in out.read_text().splitlines()}
    row = rows[4]
    assert row[4] == "0" and abs(float(row[6])) < 0.05
    peaks = np.degrees([float(v) for v in row[13:]])
    assert np.allclose(np.sort(peaks[-2:]), [10.1310, 16.3459], atol=2e-3)
