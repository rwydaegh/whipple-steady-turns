"""Command-line option and result files of the scripts."""

import argparse
import json
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parent.parent / "results"


def options(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--quick", action="store_true", help="coarser and faster computation")
    return parser.parse_args()


def _plain(x):
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [_plain(v) for v in x]
    if isinstance(x, (complex, np.complexfloating)):
        return [float(x.real), float(x.imag)]
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    return x


def save_json(name: str, data: dict):
    """Write results/<name>.json. Complex numbers become [real, imag]."""
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{name}.json").write_text(json.dumps(_plain(data), indent=1) + "\n")
