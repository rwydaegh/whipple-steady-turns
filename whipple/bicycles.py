"""Parameters of the benchmark bicycle and of bicycles A and B (SI units, angles in rad)."""

import math

BENCHMARK = dict(
    w=1.02, c=0.08, lam=math.pi / 10, g=9.81,
    rR=0.3, mR=2.0, IRxx=0.0603, IRyy=0.12,
    xB=0.3, zB=-0.9, mB=85.0, IBxx=9.2, IByy=11.0, IBzz=2.8, IBxz=2.4,
    xH=0.9, zH=-0.7, mH=4.0, IHxx=0.05892, IHyy=0.06, IHzz=0.00708, IHxz=-0.00756,
    rF=0.35, mF=3.0, IFxx=0.1405, IFyy=0.28,
)  # fmt: skip

_CITY = dict(
    w=1.101, c=0.0832912572138, lam=0.366519142919, g=9.81,
    rR=0.339954958444, mR=3.96, IRxx=0.0966455697105, IRyy=0.143776919744,
    xB=0.312073833522, zB=-0.525854751698, mB=9.18,
    IBxx=0.499733285449, IByy=0.34, IBzz=0.73927170227, IBxz=-0.0148407505732,
    xH=0.906569561251, zH=-0.802835004591, mH=4.57,
    IHxx=0.3840330428, IHzz=0.166357442859, IHxz=-0.0777986089409,
    rF=0.342550408285, mF=3.545, IFxx=0.0954028662773, IFyy=0.165623459942,
)  # fmt: skip

# Bicycles A and B differ only in the front-frame pitch inertia IHyy.
BICYCLE_A = dict(_CITY, IHyy=0.30)
BICYCLE_B = dict(_CITY, IHyy=0.54)

BICYCLES = {"benchmark": BENCHMARK, "A": BICYCLE_A, "B": BICYCLE_B}
