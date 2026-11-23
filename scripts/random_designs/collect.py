"""Collect the per-design results into the stored file of the supplement, "Random designs"."""

import gzip
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "results" / "random_designs" / "out"
RESULTS = REPO / "data" / "random_designs" / "results.jsonl.gz"
DROP = {"top_x", "tip_x", "leans", "grid", "elasticity"}  # not used by the tables


def slim(row):
    """One design without the fields that the tables do not use; floats to 7 digits."""

    def clean(x):
        if isinstance(x, dict):
            return {k: clean(v) for k, v in x.items() if k not in DROP}
        if isinstance(x, float):
            return float(f"{x:.7g}")
        return x

    return {"id": row["id"], "kind": row["kind"], "seconds": round(row["seconds"], 1),
            "result": clean(row["result"])}  # fmt: skip


def main():
    rows = [slim(json.loads(f.read_text())) for f in sorted(OUT_DIR.glob("*.json"))]
    with gzip.open(RESULTS, "wt") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {RESULTS.relative_to(REPO)}: {len(rows)} designs")


if __name__ == "__main__":
    main()
