"""
Orbital-frame figure using the transformed Itokawa shape model 

xy-plane = orbital plane
z        = orbit normal
"""

import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from scipy.spatial import ConvexHull
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch
import mpl_toolkits.mplot3d.proj3d as proj3d

# Shape file
DEFAULT_SHAPE = "./quad64q.tab"

# Orbital geometry
OBLIQUITY = np.radians(23.44) # Earth's axial tilt (ecliptic → equatorial)
OM        = np.radians(214.417)
INCL      = np.radians(10.687)

BG       = "#080818"
TEXT     = "#e8e8f8"
SPIN_COL = "#ffdd55"

TICK_SZ  = 14
AXIS_SZ  = 20
TITLE_SZ = 20


# Shape model processing

def parse_icq_tab(path):
    rows = []
    with open(path) as fh:
        first = True
        for line in fh:
            parts = line.split()
            if not parts:
                continue
            if first:
                first = False
                continue
            if len(parts) == 3:
                rows.append([float(p) for p in parts])
    return np.array(rows, dtype=np.float64)  # km


def apply_readme_transform(verts_km, r_sphere=75.0):
    v = verts_km * 1000.0  # km → m

    semi = (v.max(axis=0) - v.min(axis=0)) / 2.0
    target_x = 2.0 * semi.min()
    v[:, 0] *= target_x / semi[0]

    target_vol = (4.0 / 3.0) * np.pi * r_sphere**3
    scale = (target_vol / ConvexHull(v).volume) ** (1.0 / 3.0)
    v *= scale

    return v


def reorient_axes(verts):
    extents = verts.max(axis=0) - verts.min(axis=0)
    order = np.argsort(extents)
    return verts[:, [order[1], order[0], order[2]]]


def load_shape_body_frame(path):
    """
    Load transformed shape, then remap it into the body frame 
    """
    raw = parse_icq_tab(path)
    verts = apply_readme_transform(raw)
    verts = reorient_axes(verts)

    verts_body = np.column_stack([
        verts[:, 2],   # long axis
        verts[:, 0],   # medium axis
        verts[:, 1],   # short/spin axis
    ])

    verts_body -= verts_body.mean(axis=0)

    semi = (verts_body.max(axis=0) - verts_body.min(axis=0)) / 2.0
    print(
        f"Semi-axes after SHAPE processing [m]: "
        f"long/body-x = {semi[0]:.1f}, "
        f"body-y = {semi[1]:.1f}, "
        f"spin/body-z = {semi[2]:.1f}"
    )

    return verts_body, semi


# Rotations and vectors
def Rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s,  c]], float)


def rodrigues(target):
    z = np.array([0., 0., 1.])
    t = np.asarray(target, float)
    t /= np.linalg.norm(t)

    k = np.cross(z, t)
    sa = np.linalg.norm(k)
    ca = float(np.dot(z, t))

    if sa < 1e-10:
        return np.eye(3) if ca > 0 else np.diag([1., 1., -1.])

    k /= sa
    K = np.array([[0, -k[2], k[1]],
                  [k[2], 0, -k[0]],
                  [-k[1], k[0], 0]])

    return np.eye(3) * ca + sa * K + (1 - ca) * np.outer(k, k)


def ra_dec_xyz(ra_deg, dec_deg):
    ra, dec = np.radians(ra_deg), np.radians(dec_deg)
    return np.array([
        np.cos(dec) * np.cos(ra),
        np.cos(dec) * np.sin(ra),
        np.sin(dec)
    ])


def build_orbital_frame():
    an_ecl = np.array([np.cos(OM), np.sin(OM), 0.])
    x_orb = Rx(OBLIQUITY) @ an_ecl
    x_orb /= np.linalg.norm(x_orb)

    n_ecl = np.array([
        np.sin(INCL) * np.sin(OM),
        -np.sin(INCL) * np.cos(OM),
        np.cos(INCL)
    ])
    z_orb = Rx(OBLIQUITY) @ n_ecl
    z_orb /= np.linalg.norm(z_orb)

    y_orb = np.cross(z_orb, x_orb)
    y_orb /= np.linalg.norm(y_orb)

    return np.stack([x_orb, y_orb, z_orb])


# Plot helpers
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return min(zs)


def arrow3d(ax, p0, p1, color, lw=2.0, ms=16, zorder=5, style="->"):
    a = Arrow3D(
        [p0[0], p1[0]],
        [p0[1], p1[1]],
        [p0[2], p1[2]],
        mutation_scale=ms,
        lw=lw,
        arrowstyle=style,
        color=color,
        zorder=zorder,
    )
    ax.add_artist(a)


def grey_cmap():
    stops = [
        (0.0, (0.22, 0.22, 0.25)),
        (0.5, (0.52, 0.52, 0.55)),
        (1.0, (0.78, 0.78, 0.80)),
    ]
    cd = {ch: [] for ch in ("red", "green", "blue")}
    for pos, (r, g, b) in stops:
        cd["red"].append((pos, r, r))
        cd["green"].append((pos, g, g))
        cd["blue"].append((pos, b, b))
    return LinearSegmentedColormap("grey_rock", cd)


GREY = grey_cmap()


def plot_shape_surface(ax, verts_body, R_body, scale=1.0, alpha=1.0, color=None):
    """
    Plot shape point cloud in orbital frame.
    """
    vb = verts_body * scale
    vo = (R_body @ vb.T).T

    if color is None:
        cvals = vb[:, 0]
        ax.scatter(
            vo[:, 0], vo[:, 1], vo[:, 2],
            c=cvals,
            cmap=GREY,
            s=3.0,
            alpha=alpha,
            depthshade=True,
            zorder=5,
        )
    else:
        ax.scatter(
            vo[:, 0], vo[:, 1], vo[:, 2],
            color=color,
            s=2.5,
            alpha=alpha,
            depthshade=True,
            zorder=4,
        )

    return vo


def ray_to_shape_surface(direction, verts_orb):
    """
    Approximate distance from origin to shape surface along a direction.
    Used only for axis arrow lengths.
    """
    d = np.asarray(direction, float)
    d /= np.linalg.norm(d)

    proj = verts_orb @ d
    return np.nanmax(proj)


# Main figure
def make_figure(shape_path=DEFAULT_SHAPE, elev=28, azim=-60, save_path=None):
    matplotlib.rcParams.update({
        "figure.facecolor": BG,
        "text.color": TEXT,
        "axes.labelcolor": TEXT,
        "xtick.color": TEXT,
        "ytick.color": TEXT,
    })

    fig = plt.figure(figsize=(15, 8), facecolor=BG)
    ax = fig.add_subplot(111, projection="3d", facecolor=BG)

    # Orbital frame and spin axis
    R_eq_to_orb = build_orbital_frame()

    spin_eq = ra_dec_xyz(253, 74)
    spin_orb = R_eq_to_orb @ spin_eq

    angle_from_z = np.degrees(
        np.arccos(np.clip(spin_orb[2], -1, 1))
    )

    # Load actual asteroid shape
    verts_body_m, semi_m = load_shape_body_frame(shape_path)

    # Scale so long axis is roughly ±1 
    long_semi = semi_m[0]
    verts_body = verts_body_m / long_semi

    # Rotate body z/spin axis into orbital-frame spin axis
    R_body = rodrigues(spin_orb)

    # Diameter uncertainty shells
    sigma_D = 10.0
    D_mean = 150.0
    sigma_rel_vis = sigma_D / D_mean

    for n_sig, alpha, color in [
        (3.0, 0.12, "#cc44ff"),
        (2.0, 0.28, "#ff7733"),
    ]:
        scale = 1.0 + n_sig * sigma_rel_vis
        plot_shape_surface(
            ax,
            verts_body,
            R_body,
            scale=scale,
            alpha=alpha,
            color=color,
        )

    # Mean actual shape
    verts_orb = plot_shape_surface(
        ax,
        verts_body,
        R_body,
        scale=1.0,
        alpha=0.97,
        color=None,
    )

    # Orbital plane reference circle
    t = np.linspace(0, 2 * np.pi, 300)
    R_p = 1.5

    ax.plot(
        R_p * np.cos(t),
        R_p * np.sin(t),
        np.zeros_like(t),
        color="#4488cc",
        lw=3.5,
        ls="--",
        alpha=0.55,
        zorder=3,
    )

    lbl_ang = np.radians(210)
    ax.text(
        R_p * np.cos(lbl_ang) * 1.5,
        R_p * np.sin(lbl_ang) * 1.5,
        -0.10,
        "Orbital plane (z=0)",
        color="#4488cc",
        fontsize=AXIS_SZ - 4,
        ha="center",
        alpha=0.80,
        zorder=9,
        fontweight="bold",
    )

    # Spin axis arrow
    spin_len = 0.80
    spin_tip = spin_orb * spin_len

    arrow3d(
        ax,
        [0, 0, 0],
        spin_tip,
        color=SPIN_COL,
        lw=2.8,
        ms=20,
        zorder=8,
    )

    spin_neg = -spin_orb * 0.35
    ax.plot(
        [0, spin_neg[0]],
        [0, spin_neg[1]],
        [0, spin_neg[2]],
        color=SPIN_COL,
        lw=1.6,
        ls="--",
        alpha=0.45,
        zorder=4,
    )

    ax.text(
        spin_tip[0] - 0.2,
        spin_tip[1] + 1.5,
        spin_tip[2] + 0.5,
        f"Spin axis\nRA=253°, Dec=74°\n({angle_from_z:.1f}° from orbit normal)",
        color=SPIN_COL,
        fontsize=AXIS_SZ - 2,
        ha="left",
        fontweight="bold",
        zorder=10,
    )

    lx = ray_to_shape_surface([1, 0, 0], verts_orb)
    ly = ray_to_shape_surface([0, 1, 0], verts_orb)
    lz = ray_to_shape_surface([0, 0, 1], verts_orb)

    for vec, col in [
        ([lx, 0, 0], "#f9311b"),
        ([0, ly, 0], "#88cc44"),
        ([0, 0, lz], "#aaaaff"),
    ]:
        arrow3d(ax, [0, 0, 0], vec, color=col, lw=1.8, ms=13, zorder=6)

    PAD = 0.26
    ax.text(lx + PAD, 0, -0.2, "x\n(asc. node)",
            color="#f9311b", fontsize=AXIS_SZ - 3,
            ha="left", zorder=9, fontweight="bold")

    ax.text(0, ly + PAD, 0, "y",
            color="#88cc44", fontsize=AXIS_SZ - 3,
            ha="left", zorder=9, fontweight="bold")

    ax.text(-PAD * 2, 0, lz + PAD, "z\n(orbit normal)",
            color="#aaaaff", fontsize=AXIS_SZ - 3,
            ha="right", zorder=9, fontweight="bold")

    ax.set_xlabel("x", fontsize=AXIS_SZ - 3, color=TEXT, labelpad=6)
    ax.set_ylabel("y", fontsize=AXIS_SZ - 3, color=TEXT, labelpad=6)
    ax.set_zlabel("z", fontsize=AXIS_SZ - 3, color=TEXT, labelpad=6)

    lim = 1.6
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect([1, 1, 1])

    ticks = np.arange(-1.5, 2.0, 0.5)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_zticks(ticks)

    pane_col = (0.04, 0.04, 0.10, 0.55)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_facecolor(pane_col)
        pane.set_edgecolor("#1a1a3a")

    ax.tick_params(labelsize=TICK_SZ, colors=TEXT, length=3)
    ax.view_init(elev=elev, azim=azim)

    ax.set_title(
        "Asteroid Shape Model in Orbital Frame  (xy = orbital plane)\n"
        f"Spin axis is {angle_from_z:.1f}° from orbit normal  |  "
        r"$D_\mathrm{eff}$ = 150 m,  $\sigma \approx$ 10 m",
        fontsize=TITLE_SZ,
        color=TEXT,
        pad=16,
        linespacing=1.5,
        fontweight="bold",
    )


    if save_path:
        fig.savefig(
            save_path,
            dpi=150,
            facecolor=fig.get_facecolor(),
        )
        print(f"Saved -> {save_path}")

    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", type=str, default=DEFAULT_SHAPE)
    parser.add_argument("--elev", type=float, default=28)
    parser.add_argument("--azim", type=float, default=-60)
    parser.add_argument("--save", type=str, default=None)

    args, _ = parser.parse_known_args()

    make_figure(
        shape_path=args.shape,
        elev=args.elev,
        azim=args.azim,
        save_path=args.save or "./asteroid_celestial_sphere_shape.png",
    )

    plt.show()