"""Largest stable phi_G joined to small lean for a fixed shift and a free steer torque (supplementary material, "Mass shift: further results")."""  # noqa: E501

import json
from pathlib import Path

import numpy as np
from scipy.ndimage import label

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "rider"


def connected(G, plant):
    V, P, D = G["V"], np.degrees(G["phiG"]), np.degrees(G["Delta"])
    S = G[f"stable_{plant}"]
    rows = []
    for k in range(len(D)):
        lab, _ = label(S[:, :, k])
        ids = set(lab[:, 0][lab[:, 0] > 0])  # regions that reach the smallest lean
        if not ids:
            continue
        m = np.isin(lab, list(ids))
        i, j = np.unravel_index(np.argmax(np.where(m, P[None, :], -1)), m.shape)
        rows.append(
            dict(
                Delta=float(D[k]),
                phiG=float(P[j]),
                V=float(V[i]),
                T=float(G["T"][i, j, k]),
                ay_g=float(G["ay"][i, j, k] / 9.81),
                top_of_grid=bool(j == len(P) - 1),
            )
        )
    return rows


def main():
    out = {}
    for name in ("benchmark", "benchmark_wide"):
        G = np.load(DATA / f"shift_grid_{name}.npz")
        for plant in ("free", "constant"):
            out[f"{name}_{plant}"] = rows = connected(G, plant)
            print(name, plant, " ".join(f"{r['Delta']:g}:{r['phiG']:g}" for r in rows))
    (DATA / "shift_connected.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
