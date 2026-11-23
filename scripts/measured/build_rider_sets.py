"""Parameter sets of the measured bicycles with a Yeadon rider (tab:measured-sets, Sect. "Parameter
sets").
"""

# Needs bicycleparameters 1.5.2 and yeadon 2.0.1 (not project dependencies) and a copy of the data
# folder of https://github.com/moorepants/BicycleParameters. Sets from published tables are copied
# from whipple/data/rider_bicycles.json.
#   uv run --with bicycleparameters==1.5.2 --with yeadon==2.0.1 \
#       python scripts/measured/build_rider_sets.py DATA SCRATCH OUTDIR

import json
import shutil
import sys
import warnings
from pathlib import Path

from whipple.bicycles import BENCHMARK

KEYS = list(BENCHMARK)
PACKAGE_DATA = Path(__file__).resolve().parents[2] / "whipple" / "data"
TORSO_PITCH = 0.175  # rad, Jason on the Browser; used where no configuration file exists
GH = (
    "BicycleParameters data (https://github.com/moorepants/BicycleParameters, data/ folder, "
    "BSD-2-Clause)"
)

# (bicycle, rider, True if the data hold a rider configuration file for this pair)
PAIRS = [
    ("Browser", "Jason", True),
    ("Browserins", "Jason", True),
    ("Stratos", "Jason", True),
    ("Rigid", "Jason", True),
    ("Balanceassistv1", "Jason", True),
    ("Rigidcl", "Luke", True),
    ("Rigidcl", "Charlie", True),
    ("Gyro", "Child", True),
    ("Crescendo", "Jason", False),
    ("Yellow", "Jason", False),
    ("Yellowrev", "Jason", False),
    ("Browser", "Mont", False),
    ("Browser", "Chris", False),
    ("Browser", "Aurelia", False),
    ("Browser", "Luke", False),
    ("Browser", "Charlie", False),
]
RIDERLESS = [
    "Browser",
    "Browserins",
    "Crescendo",
    "Fisher",
    "Pista",
    "Rigid",
    "Rigidcl",
    "Silver",
    "Stratos",
    "Yellow",
    "Yellowrev",
    "Gyro",
    "Balanceassistv1",
]
RIDER_NOTE = {
    "Jason": "Jason (J. K. Moore), 83.5 kg incl. clothes, 95 Yeadon measures",
    "Luke": "Luke (L. Peterson), 84.3 kg, 95 Yeadon measures",
    "Charlie": "Charlie, 78.7 kg, 95 Yeadon measures (measured 2011-09-28)",
    "Child": "Child: Charlie's measures SCALED to a 5-year-old (1.016 m, 18.96 kg); "
    "not a measured person",
    "Mont": "Mont (M. Hubbard), mass from Yeadon densities (no measured total mass)",
    "Chris": "Chris, mass from Yeadon densities (no measured total mass)",
    "Aurelia": "Aurelia, mass from Yeadon densities (totalmass -1 in file)",
}


def nominal(P):
    return {k: float(getattr(P[k], "nominal_value", P[k])) for k in KEYS}


def main(data, scratch, outdir):
    import bicycleparameters as bp

    warnings.filterwarnings("ignore")
    work = Path(scratch) / "data"
    if not work.exists():
        shutil.copytree(data, work)  # the package writes into its data folder
    rider, riderless = {}, {}
    for bike in RIDERLESS:
        try:
            b = bp.Bicycle(bike, pathToData=str(work), forceRawCalc=True)
        except (KeyError, OSError, ValueError) as e:  # Silver: no front-wheel measurement
            print(f"{bike}: {e!r}; kept from the package data", file=sys.stderr)
            continue
        riderless[bike] = {
            **nominal(b.parameters["Benchmark"]),
            "source": f"Computed with bicycleparameters 1.5.2 from the {GH}: "
            f"bicycles/{bike}/RawData/{bike}Measured.txt.",
            "status": "measured bicycle without rider",
        }
    for bike, name, measured in PAIRS:
        cfg = work / "riders" / name / "RawData" / f"{name}{bike}YeadonCFG.txt"
        if not measured:  # copy of Jason's Browser configuration with the torso pitch set
            src = work / "riders" / "Jason" / "RawData" / "JasonBrowserYeadonCFG.txt"
            cfg.write_text(
                "".join(
                    f"somersault: {TORSO_PITCH}\n" if ln.startswith("somersault") else ln
                    for ln in src.open()
                )
            )
        b = bp.Bicycle(bike, pathToData=str(work), forceRawCalc=True)
        b.add_rider(name, reCalc=True, draw=False)
        pose = (
            f"torso pitch from the configuration file riders/{name}/RawData/{cfg.name}"
            if measured
            else f"torso pitch assumed {TORSO_PITCH} rad"
        )
        rider[f"{bike}+{name}"] = {
            **nominal(b.parameters["Benchmark"]),
            "source": f"Computed with bicycleparameters 1.5.2 and yeadon 2.0.1 from the {GH}: "
            f"bicycles/{bike}/RawData/{bike}Measured.txt and "
            f"riders/{name}/RawData/{name}YeadonMeas.txt. Rider: {RIDER_NOTE[name]}. "
            f"Rider pose from the bicycle geometry; {pose}. "
            "Rider lumped rigidly into the rear frame.",
            "status": "measured bicycle, Yeadon rider model, "
            + ("measured" if measured else "assumed")
            + " rider configuration",
        }
    old = {
        "rider": json.loads((PACKAGE_DATA / "rider_bicycles.json").read_text()),
        "riderless": json.loads((PACKAGE_DATA / "riderless_bicycles.json").read_text()),
    }
    rider.update({k: v for k, v in old["rider"].items() if v["status"].startswith("published")})
    for k, v in old["riderless"].items():
        riderless.setdefault(k, v)
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "rider_bicycles.json").write_text(json.dumps(rider, indent=1))
    (out / "riderless_bicycles.json").write_text(json.dumps(riderless, indent=1))
    print(len(rider), "sets with rider,", len(riderless), "without")


if __name__ == "__main__":
    main(*sys.argv[1:4])
