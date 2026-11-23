"""Design moves and added loads (Sect. 12.2, "Design moves"; supplement, "Design moves")."""

import math

import numpy as np
from scipy.optimize import brentq


# ---- helpers -------------------------------------------------------------------------------------
def scale(keys):
    """Move that multiplies the parameters ``keys`` by the factor f."""

    def move(p, f):
        q = dict(p)
        for k in keys:
            q[k] = p[k] * f
        return q

    return move


def tensor(xx, yy, zz, xz):
    return np.array([[xx, 0.0, xz], [0.0, yy, 0.0], [xz, 0.0, zz]])


def combine(parts):
    """parts: (m, x, z, I about the own centre). Returns m, x, z, I about the common centre."""
    m = sum(q[0] for q in parts)
    x = sum(q[0] * q[1] for q in parts) / m
    z = sum(q[0] * q[2] for q in parts) / m
    inertia = np.zeros((3, 3))
    for mi, xi, zi, Ii in parts:
        d = np.array([xi - x, 0.0, zi - z])
        inertia += Ii + mi * (d @ d * np.eye(3) - np.outer(d, d))
    return m, x, z, inertia


def add_to_frame(p, body, m, x, z, inertia=None):
    """Add a mass m at (x, z) with own inertia to body "B" (rear frame) or "H" (front frame).
    The mass centre and the inertia tensor of the body follow from the parallel-axis theorem."""
    p = dict(p)
    inertia = np.zeros((3, 3)) if inertia is None else inertia
    own = tensor(p[f"I{body}xx"], p[f"I{body}yy"], p[f"I{body}zz"], p[f"I{body}xz"])
    mt, xt, zt, It = combine([(p[f"m{body}"], p[f"x{body}"], p[f"z{body}"], own),
                              (m, x, z, inertia)])  # fmt: skip
    p.update({f"m{body}": mt, f"x{body}": xt, f"z{body}": zt, f"I{body}xx": It[0, 0],
              f"I{body}yy": It[1, 1], f"I{body}zz": It[2, 2], f"I{body}xz": It[0, 2]})  # fmt: skip
    return p


def add_to_wheel(p, wheel, m, dIxx, dIyy):
    """Add a mass m at the axle of wheel "R" or "F" with extra (axisymmetric) inertia."""
    p = dict(p)
    p[f"m{wheel}"] += m
    p[f"I{wheel}xx"] += dIxx
    p[f"I{wheel}yy"] += dIyy
    return p


def steer_axis_x(p, height):
    """x of the steer axis at a height above the ground."""
    return p["w"] + p["c"] - height * math.tan(p["lam"])


def _rot_pt(x, z, cx, cz, th):
    """Rotate (x, z) about (cx, cz) by th. th > 0 lifts a point ahead of the centre (nose up);
    the axis direction (-sin l, -cos l) becomes (-sin(l + th), -cos(l + th))."""
    dx, dz = x - cx, z - cz
    return cx + dx * math.cos(th) + dz * math.sin(th), cz - dx * math.sin(th) + dz * math.cos(th)


def _rot_body(x, z, inertia, cx, cz, th):
    """Rotate a body (centre and in-plane inertia entries xx, zz, xz) about (cx, cz) by th."""
    R = np.array([[math.cos(th), 0, math.sin(th)], [0, 1, 0], [-math.sin(th), 0, math.cos(th)]])
    xx, zz, xz = inertia
    M = R @ np.array([[xx, 0, xz], [0, 0, 0], [xz, 0, zz]]) @ R.T
    return (*_rot_pt(x, z, cx, cz, th), (M[0, 0], M[2, 2], M[0, 2]))


def fork_offset(p):
    """Perpendicular distance from the front axle to the steer axis."""
    return p["rF"] * math.sin(p["lam"]) - p["c"] * math.cos(p["lam"])


# ---- rider model of the base "A + rider" ---------------------------------------------------------
def rider_parts(theta_deg=10.0):
    """Invented 75 kg rider: lower body 33 kg (legs, pelvis), upper body 42 kg as a cylinder
    (length 0.75 m, radius 0.17 m) hinged at the hip (0.25, -1.00) m and tilted forward by theta."""
    lower = (33.0, 0.36, -0.72, tensor(1.5, 1.2, 0.8, 0.0))
    th = math.radians(theta_deg)
    m, L, r = 42.0, 0.75, 0.17
    u = np.array([math.sin(th), 0.0, -math.cos(th)])
    Ia, Ip = m * r * r / 2, m * (3 * r * r + L * L) / 12
    inertia = Ip * np.eye(3) + (Ia - Ip) * np.outer(u, u)
    hip = np.array([0.25, -1.00])
    upper = (m, hip[0] + L / 2 * u[0], hip[1] + L / 2 * u[2], inertia)
    return [lower, upper]


def with_rider(p, theta_deg=10.0):
    q = dict(p)
    for m, x, z, inertia in rider_parts(theta_deg):
        q = add_to_frame(q, "B", m, x, z, inertia)
    return q


# ---- wheelbase ----------------------------------------------------------------------------------
def wheelbase_only(p, f):
    """W1: only w changes. xB, zB, xH, zH, c and lam stay fixed relative to the rear contact, so
    the front-frame centre falls dw cos(lam) behind the steer axis."""
    return scale(["w"])(p, f)


def front_assembly_forward(p, f):
    """W2: front wheel, steer axis and front-frame centre move forward by dw; rear frame, c and
    lam stay fixed."""
    q = dict(p)
    dw = p["w"] * (f - 1)
    q["w"] += dw
    q["xH"] += dw
    return q


def rear_wheel_back(p, f):
    """W3: the rear wheel moves back by dw; both frame centres stay fixed relative to the front,
    so xB and xH grow by dw."""
    q = front_assembly_forward(p, f)
    q["xB"] += p["w"] * (f - 1)
    return q


def stretch(p, f):
    """W4: w, xB and xH times f; c, lam, all heights and all inertias stay fixed."""
    return scale(["w", "xB", "xH"])(p, f)


# ---- trail and head angle -----------------------------------------------------------------------
def trail(p, f):
    """c only: w, lam, xH and zH stay fixed; the steer axis slides forward under a fixed
    front-frame centre."""
    return scale(["c"])(p, f)


def trail_with_front_frame(p, f):
    """c changes, and the front-frame centre moves with the steer axis (xH grows by dc)."""
    q = dict(p)
    dc = p["c"] * (f - 1)
    q["c"] += dc
    q["xH"] += dc
    return q


def head_angle(p, f):
    """lam only (larger is slacker): the axis pivots about its ground point (w, c fixed); the
    front-frame centre stays fixed in space."""
    return scale(["lam"])(p, f)


def head_angle_front_frame_rotates(p, f):
    """lam changes; the axis pivots about its ground point, and the front-frame centre and its
    inertia tensor rotate with the axis."""
    q = dict(p)
    dl = p["lam"] * (f - 1)
    q["lam"] += dl
    q["xH"], q["zH"], (q["IHxx"], q["IHzz"], q["IHxz"]) = _rot_body(
        p["xH"], p["zH"], (p["IHxx"], p["IHzz"], p["IHxz"]), p["w"] + p["c"], 0.0, dl
    )
    return q


# ---- rear frame (bicycle frame with the rider) --------------------------------------------------
def rear_frame_taller(p, f):
    """A taller rider of the same build: mB times f, the IB entries times f^(5/3) and the height
    -zB times f^(1/3); xB stays fixed."""
    q = dict(p)
    q["mB"] = p["mB"] * f
    for k in ("IBxx", "IByy", "IBzz", "IBxz"):
        q[k] = p[k] * f ** (5 / 3)
    q["zB"] = p["zB"] * f ** (1 / 3)
    return q


# ---- wheel radius -------------------------------------------------------------------------------
def front_radius(p, f):
    """rF times f with IFxx, IFyy times f^2 at fixed mF; c, lam, w, xH and zH stay fixed (the
    fork offset shrinks and the frame does not pitch)."""
    q = scale(["rF"])(p, f)
    return scale(["IFxx", "IFyy"])(q, f * f)


def front_radius_same_fork(p, f):
    """As front_radius, with a fixed fork offset: c grows by drF tan(lam), and zH rises with the
    axle; w, lam and xH stay fixed."""
    q = front_radius(p, f)
    q["c"] = (q["rF"] * math.sin(p["lam"]) - fork_offset(p)) / math.cos(p["lam"])
    q["zH"] = p["zH"] - (q["rF"] - p["rF"])
    return q


def front_radius_same_frame(p, f):
    """As front_radius; the whole frame and fork pitch back about the rear axle so that the
    front axle rises by drF. w, c, lam, xB, zB, xH, zH and the IB, IH tensors follow from the
    rotation; the fork offset stays fixed."""
    q = dict(p)
    rF1 = p["rF"] * f
    q["rF"], q["IFxx"], q["IFyy"] = rF1, p["IFxx"] * f * f, p["IFyy"] * f * f
    ax, az = 0.0, -p["rR"]  # rear axle (pivot)
    rel = np.array([p["w"], p["rR"] - p["rF"]])  # front axle relative to the rear axle
    th = brentq(lambda t: -rel[0] * math.sin(t) + rel[1] * math.cos(t) - (p["rR"] - rF1), -0.3, 0.3)
    q["w"] = _rot_pt(p["w"], -p["rF"], ax, az, th)[0]
    q["lam"] = p["lam"] + th
    g = _rot_pt(p["w"] + p["c"], 0.0, ax, az, th)  # old ground point of the axis, moved
    s = g[1] / math.cos(q["lam"])
    q["c"] = g[0] - s * math.sin(q["lam"]) - q["w"]
    for b in "BH":
        q[f"x{b}"], q[f"z{b}"], (q[f"I{b}xx"], q[f"I{b}zz"], q[f"I{b}xz"]) = _rot_body(
            p[f"x{b}"], p[f"z{b}"], (p[f"I{b}xx"], p[f"I{b}zz"], p[f"I{b}xz"]), ax, az, th
        )
    return q


def rear_radius(p, f):
    """rR times f with IRxx, IRyy times f^2 at fixed mR; everything else fixed (no frame pitch)."""
    q = scale(["rR"])(p, f)
    return scale(["IRxx", "IRyy"])(q, f * f)


def rear_radius_same_frame(p, f):
    """As rear_radius; the whole frame and fork pitch forward about the front axle so that the
    rear axle rises by drR, then everything shifts in x so that the rear contact stays at the
    origin."""
    q = dict(p)
    rR1 = p["rR"] * f
    q["rR"], q["IRxx"], q["IRyy"] = rR1, p["IRxx"] * f * f, p["IRyy"] * f * f
    ax, az = p["w"], -p["rF"]  # front axle (pivot)
    rel = np.array([-p["w"], p["rF"] - p["rR"]])  # rear axle relative to the front axle
    th = brentq(lambda t: -rel[0] * math.sin(t) + rel[1] * math.cos(t) - (p["rF"] - rR1), -0.3, 0.3)
    sx = -_rot_pt(0.0, -p["rR"], ax, az, th)[0]  # shift so that the rear axle is at x = 0
    q["w"] = p["w"] + sx
    q["lam"] = p["lam"] + th
    g = _rot_pt(p["w"] + p["c"], 0.0, ax, az, th)
    s = g[1] / math.cos(q["lam"])
    q["c"] = g[0] + sx - s * math.sin(q["lam"]) - q["w"]
    for b in "BH":
        inertia = (p[f"I{b}xx"], p[f"I{b}zz"], p[f"I{b}xz"])
        x, z, inertia = _rot_body(p[f"x{b}"], p[f"z{b}"], inertia, ax, az, th)
        q[f"x{b}"], q[f"z{b}"] = x + sx, z
        q[f"I{b}xx"], q[f"I{b}zz"], q[f"I{b}xz"] = inertia
    return q


# ---- position moves of the front-frame centre (step s in cm) ------------------------------------
def front_frame_forward_of_axis(p, s):
    """(xH, zH) moved by s cm along (cos lam, -sin lam), normal to the steer axis (+ forward)."""
    q = dict(p)
    q["xH"] += s / 100 * math.cos(p["lam"])
    q["zH"] -= s / 100 * math.sin(p["lam"])
    return q


def front_frame_up_axis(p, s):
    """(xH, zH) moved by s cm along (-sin lam, -cos lam), up the steer axis."""
    q = dict(p)
    q["xH"] -= s / 100 * math.sin(p["lam"])
    q["zH"] -= s / 100 * math.cos(p["lam"])
    return q


def total_mass(p, f):
    """Every mass and every moment of inertia times f."""
    return scale([k for k in p if k[0] in "mI"])(p, f)


# ---- the registry -------------------------------------------------------------------------------
# key: (move, kind, label). kind "pct": elasticity in % per %; "cm": change in % per cm.
MOVES = {
    "trail": (trail, "pct", "Trail c (steer axis slides forward)"),
    "trail_with_front_frame": (
        trail_with_front_frame,
        "pct",
        "Trail c, front-frame centre with the axis",
    ),
    "head_angle": (head_angle, "pct", "Head angle, slacker (front-frame centre fixed)"),
    "head_angle_front_frame_rotates": (
        head_angle_front_frame_rotates,
        "pct",
        "Head angle, slacker (front frame rotates)",
    ),
    "front_spin_inertia": (scale(["IFyy"]), "pct", "Front wheel spin inertia IFyy"),
    "front_wheel_mass": (scale(["mF", "IFxx", "IFyy"]), "pct", "Front wheel mass (mF, IF scaled)"),
    "front_radius": (front_radius, "pct", "Front wheel radius (trail and head angle fixed)"),
    "front_radius_geometry_only": (scale(["rF"]), "pct", "Front wheel radius, geometry only"),
    "front_radius_inertia_only": (
        lambda p, f: scale(["IFxx", "IFyy"])(p, f * f),
        "pct",
        "Front wheel radius, inertia only (IF times f^2)",
    ),
    "front_radius_same_fork": (front_radius_same_fork, "pct", "Front wheel radius, same fork"),
    "front_radius_same_frame": (
        front_radius_same_frame,
        "pct",
        "Front wheel radius, same frame pitched",
    ),
    "rear_radius": (rear_radius, "pct", "Rear wheel radius (no pitch)"),
    "rear_radius_same_frame": (
        rear_radius_same_frame,
        "pct",
        "Rear wheel radius, same frame pitched",
    ),
    "rear_wheel_mass": (scale(["mR", "IRxx", "IRyy"]), "pct", "Rear wheel mass (mR, IR scaled)"),
    "IRyy": (scale(["IRyy"]), "pct", "Rear wheel spin inertia IRyy"),
    "rear_frame_forward": (scale(["xB"]), "pct", "Rear-frame mass centre forward xB"),
    "rear_frame_higher": (scale(["zB"]), "pct", "Rear-frame mass centre higher -zB"),
    "rear_frame_mass": (
        scale(["mB", "IBxx", "IByy", "IBzz", "IBxz"]),
        "pct",
        "Rear-frame mass, same build (mB, IB scaled)",
    ),
    "rear_frame_mass_only": (scale(["mB"]), "pct", "Rear-frame mass only (IB fixed)"),
    "rear_frame_taller": (rear_frame_taller, "pct", "Taller rider"),
    "front_frame_mass": (
        scale(["mH", "IHxx", "IHyy", "IHzz", "IHxz"]),
        "pct",
        "Front-frame mass (mH, IH scaled)",
    ),
    "wheelbase_only": (wheelbase_only, "pct", "W1: w only"),
    "front_assembly_forward": (front_assembly_forward, "pct", "W2: front assembly forward"),
    "rear_wheel_back": (rear_wheel_back, "pct", "W3: rear wheel back"),
    "stretch": (stretch, "pct", "W4: stretch of w, xB, xH"),
    "total_mass": (total_mass, "pct", "All masses and inertias"),
    "front_frame_forward_of_axis": (
        front_frame_forward_of_axis,
        "cm",
        "Front-frame centre forward of the steer axis (% per cm)",
    ),
    "front_frame_up_axis": (
        front_frame_up_axis,
        "cm",
        "Front-frame centre up the steer axis (% per cm)",
    ),
}
for _k in ("IBxx", "IByy", "IBzz", "IBxz", "IHxx", "IHyy", "IHzz", "IHxz", "IFxx", "IRxx"):
    MOVES[_k] = (scale([_k]), "pct", f"{_k} only")


# ---- added loads (supplement, table tab:design-loads) -------------------------------------------
def hub_motor(p, wheel):
    """3 kg direct-drive hub motor: 1.5 kg stator on the frame at the axle, 1.5 kg rotor ring of
    radius 0.10 m that spins with the wheel (dIyy = 0.015, dIxx = 0.0075 kg m^2)."""
    body, x, z = ("H", p["w"], -p["rF"]) if wheel == "F" else ("B", 0.0, -p["rR"])
    q = add_to_frame(p, body, 1.5, x, z)
    return add_to_wheel(q, wheel, 1.5, 0.0075, 0.015)


def basket(p, body):
    """5 kg basket, centre 0.95 m above the ground and 0.18 m ahead of the steer axis."""
    x = steer_axis_x(p, 0.95) + 0.18
    return add_to_frame(p, body, 5.0, x, -0.95, tensor(0.05, 0.05, 0.05, 0.0))


LOADS = {
    "Battery 3 kg on the rear rack": lambda p: add_to_frame(p, "B", 3.0, -0.02, -0.80),
    "Battery 3 kg on the down tube": lambda p: add_to_frame(p, "B", 3.0, 0.52 * p["w"], -0.45),
    "Child seat 20 kg": lambda p: add_to_frame(p, "B", 20.0, -0.08, -1.10,
                                               tensor(0.6, 0.6, 0.2, 0.0)),
    "Basket 5 kg on the handlebar (front frame)": lambda p: basket(p, "H"),
    "Basket 5 kg at the same place on the rear frame": lambda p: basket(p, "B"),
    "Hub motor 3 kg, front": lambda p: hub_motor(p, "F"),
    "Hub motor 3 kg, rear": lambda p: hub_motor(p, "R"),
}  # fmt: skip
