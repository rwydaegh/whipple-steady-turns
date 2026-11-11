"""Small-lean expansion: coefficients Lambda_k at the upright weave and capsize speeds (Tables C1
and C2) and the comparison with eigenvalues of turns of the benchmark bicycle (Fig. 5)."""

import numpy as np

from whipple import plotstyle as ps
from whipple import stability as st
from whipple.bicycles import BICYCLES
from whipple.lean_law import TERMS, LeanLaw, turn_at_energy_speed
from whipple.output import options, save_json


def table(law: LeanLaw):
    vw = law.weave_speed()
    weave = law.terms(vw)[0]
    cusp = law.cusp()
    capsize = law.terms(cusp["v_c"])[1]
    alpha, alpha_c = weave["dlam_dV"].real, capsize["dlam_dV"].real
    return {
        "v_w (m/s)": vw,
        "alpha (1/m)": alpha,
        "Lambda_w (1/s)": weave["Lambda"],
        "kappa_w (m/s)": -weave["Lambda"].real / alpha,
        "Lambda_w terms (1/s)": {k: weave[k] for k in TERMS},
        "v_c (m/s)": cusp["v_c"],
        "alpha_c (1/m)": alpha_c,
        "Lambda_c (1/s)": capsize["Lambda"].real,
        "tau1' (N s)": cusp["tau1p"],
        "tau3 (N m)": cusp["tau3"],
        "e2 (m/s)": cusp["e2"],
        "beta_E (m/s)": cusp["beta_E"],
        "kappa_c = -Lambda_c / alpha_c (m/s)": -capsize["Lambda"].real / alpha_c,
        "3 beta_E (m/s)": 3 * cusp["beta_E"],
    }


def exact(law: LeanLaw, vE, leans):
    """Weave and capsize eigenvalues of turns at energy-equivalent speed vE, tracked in lean."""
    bike = law.bike
    lam0, _ = law.Lambda(vE)
    ref, ratio, rows = lam0[:2], law.lin.steer_ratio(vE), []
    for phi in leans:
        try:
            turn = turn_at_energy_speed(bike, phi, vE, ratio * phi)
        except RuntimeError:
            break
        lam = list(st.eigenvalues(bike, phi, turn.delta))
        ref = [lam.pop(int(np.argmin([abs(z - x) for x in lam]))) for z in ref]
        ratio = turn.delta / phi
        rows.append(ref)
    return np.array(rows)


def figure(law: LeanLaw, quick):
    vw = law.weave_speed()
    leans = np.radians(np.arange(0.5 if quick else 0.25, 16.001, 1.0 if quick else 0.25))
    fine = np.radians(np.linspace(0, 16, 161))
    speeds = (3.5, vw, 5.0, 6.0)
    names = ("$v_E=3.5$ m/s", f"${vw:.2f}$", "$5.0$", "$6.0$")
    fig, axs = ps.figure(ps.FULL, 6.2 * ps.CM, 1, 2)
    fig.subplots_adjust(wspace=0.32)
    for vE, name in zip(speeds, names):
        lam0, L = law.Lambda(vE)
        rows = exact(law, vE, leans)
        for k, ax, color in ((0, axs[0], ps.HOPF), (1, axs[1], ps.FOLD)):
            n = len(rows)
            if k == 0:
                n = int(np.argmin(np.abs(rows[:, 0].imag) > 1e-6)) or n
            x = np.r_[0.0, np.degrees(leans[:n])]
            y = np.r_[lam0[k].real, rows[:n, k].real]
            ax.plot(x, y, color=color)
            ax.plot(np.degrees(fine), (lam0[k] + L[k] * fine**2).real, color=color,
                    lw=ps.LW_THIN, ls=ps.DASH)  # fmt: skip
            below = k == 1 and vE == speeds[0]
            ps.label(ax, 0.25, y[0] + (-0.06 if below else 0.04), name, fontsize=7,
                     va="top" if below else "bottom")  # fmt: skip
    for ax in axs:
        ax.axhline(0, color=ps.LIGHT, lw=ps.LW_THIN, zorder=0)
        ax.set(xlim=(0, 16), xticks=[0, 4, 8, 12, 16], xlabel=r"Lean $\phi_0$ (deg)")
    axs[0].set(
        ylim=(-2.2, 1.5), yticks=[-2, -1, 0, 1], ylabel=r"Weave $\mathrm{Re}\,\lambda_w$ (1/s)"
    )
    axs[1].set(ylim=(-2.6, 1.2), yticks=[-2, -1, 0, 1], ylabel=r"Capsize $\lambda_c$ (1/s)")
    ps.panel(axs[0], "a", x=-0.25)
    ps.panel(axs[1], "b", x=-0.25)
    ps.save(fig, "lean_law")


def main():
    args = options(__doc__)
    laws = {name: LeanLaw(p) for name, p in BICYCLES.items()}
    save_json("lean_law", {name: table(law) for name, law in laws.items()})
    figure(laws["benchmark"], args.quick)


if __name__ == "__main__":
    main()
