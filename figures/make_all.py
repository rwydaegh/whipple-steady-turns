"""Regenerate all paper figures from the stored data in data/ (output in results/figures/)."""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SKIP = {"figstyle.py", "make_all.py"}

failed = []
for script in sorted(HERE.glob("*.py")):
    if script.name in SKIP:
        continue
    print("running", script.name, flush=True)
    if subprocess.run([sys.executable, str(script)], cwd=HERE.parent).returncode:
        failed.append(script.name)
if failed:
    sys.exit("failed: " + ", ".join(failed))
