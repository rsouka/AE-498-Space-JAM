"""
Three-panel ΔV efficiency figure for a kinetic impactor, using the
*actual* Gaskell Itokawa-derived shape model 

Shape transformations (per README.txt):
  1. X axis of the Itokawa ICQ model reduced to exactly 2x the shortest
     semi-axis.
  2. Entire model scaled so its volume equals that of a 75-m sphere.


Surface normals are estimated per-vertex via PCA of k-nearest neighbours,
which handles non-convex geometry 
"""

import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial import ConvexHull, cKDTree

DEFAULT_SHAPE = './quad64q.tab'
 
def parse_icq_tab(path):
    """
    Read a Gaskell ICQ .tab file.
    Returns (N, 3) float array of vertex coordinates in **km**.
    """
    rows = []
    with open(path) as fh:
        first = True
        for line in fh:
            parts = line.split()
            if not parts:
                continue
            if first:          # Q-value header — skip
                first = False
                continue
            if len(parts) == 3:
                rows.append([float(p) for p in parts])
    return np.array(rows, dtype=np.float64)    # km
 
 
def apply_readme_transform(verts_km, r_sphere=75.0):
    """
    Returns vertices in meters.
    """
    v = verts_km * 1000.0                      # km → m
 
    #  reduce X to 2 × shortest semi-axis 
    semi = (v.max(axis=0) - v.min(axis=0)) / 2.0    # [sx, sy, sz]
    target_x = 2.0 * semi.min()
    v[:, 0]  *= target_x / semi[0]
 
    # scale to target volume 
    target_vol = (4.0 / 3.0) * np.pi * r_sphere ** 3
    scale      = (target_vol / ConvexHull(v).volume) ** (1.0 / 3.0)
    v         *= scale
 
    return v
 
 
def reorient_axes(verts):
    extents = verts.max(axis=0) - verts.min(axis=0)
    order   = np.argsort(extents)
    return verts[:, [order[1], order[0], order[2]]]
 
 
def rotate_view(verts, theta_deg, phi_deg, yaw_deg=0.0):
    
    if yaw_deg != 0.0:
        psi = np.radians(yaw_deg)
        Rp  = np.array([[1, 0,            0           ],
                         [0, np.cos(psi), -np.sin(psi)],
                         [0, np.sin(psi),  np.cos(psi)]])
        verts = verts @ Rp.T
 
    
    if theta_deg == 0.0 and phi_deg == 0.0:
        return verts
 
    th = np.radians(theta_deg)
    ph = np.radians(phi_deg)
    d  = np.array([np.cos(ph)*np.cos(th),
                   np.cos(ph)*np.sin(th),
                   np.sin(ph)])
    d /= np.linalg.norm(d)
 
    a  = np.array([1.0, 0.0, 0.0])
    v  = np.cross(a, d)
    c  = float(np.dot(a, d))
    if abs(1.0 + c) < 1e-9:
        return verts * np.array([-1, 1, 1])
    vx = np.array([[ 0,    -v[2],  v[1]],
                   [ v[2],  0,    -v[0]],
                   [-v[1],  v[0],  0   ]])
    R  = np.eye(3) + vx + (vx @ vx) / (1.0 + c)
    return verts @ R.T
 
 
def pca_vertex_normals(verts, k=20):
    """
    Estimate outward unit normals for every vertex using PCA on its
    k-nearest neighbours.  
 
    Returns (N, 3) unit-normal array.
    """
    tree      = cKDTree(verts)
    centroid  = verts.mean(axis=0)
    _, idx    = tree.query(verts, k=k)           # (N, k) neighbour indices
 
    nb        = verts[idx]                        # (N, k, 3)
    centred   = nb - nb.mean(axis=1, keepdims=True)
    cov       = np.einsum('nki,nkj->nij', centred, centred)   # (N, 3, 3)
    _, evecs  = np.linalg.eigh(cov)              # eigenvalues ascending
    normals   = evecs[:, :, 0]                   
 
    # Flip so every normal points away from the body centroid
    outward   = verts - centroid
    flip      = (normals * outward).sum(axis=1) < 0
    normals[flip] *= -1
    return normals                               
 
 
def find_flattest_aim_point(fv, fn, nx_threshold=0.96):
    nx = fn[:, 0]
    flat_mask = nx > nx_threshold
    flat_verts = fv[flat_mask]          # (K, 3)
 
    if len(flat_verts) == 0:
        # Fallback: single flattest vertex
        best = np.argmax(nx)
        return fv[best, 1], fv[best, 2]
 
    # Typical spacing among all front-face vertices
    tree_all = cKDTree(fv[:, 1:])
    dists, _ = tree_all.query(fv[:, 1:], k=2)   # k=2: self + nearest
    median_spacing = np.median(dists[:, 1])
    link_radius = 2.5 * median_spacing
 
    tree_flat = cKDTree(flat_verts[:, 1:])
    pairs = tree_flat.query_pairs(link_radius)
 
    parent = list(range(len(flat_verts)))
 
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
 
    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
 
    labels = np.array([find(i) for i in range(len(flat_verts))])
    unique, counts = np.unique(labels, return_counts=True)
    largest_label  = unique[np.argmax(counts)]
 
    cluster = flat_verts[labels == largest_label]
    y_aim   = cluster[:, 1].mean()
    z_aim   = cluster[:, 2].mean()
 
    return y_aim, z_aim
 
 
def load_mesh(path=DEFAULT_SHAPE, view_theta=0.0, view_phi=0.0, view_yaw=0.0):
    """
    Returns a dict used by all drawing functions.
    """
    print(f'Loading shape model from  {path}  …')
    raw   = parse_icq_tab(path)
    verts = apply_readme_transform(raw)
    verts = reorient_axes(verts)
 
    if view_theta != 0.0 or view_phi != 0.0 or view_yaw != 0.0:
        print(f'  Rotating view: θ={view_theta}°  φ={view_phi}°  ψ={view_yaw}°')
        verts = rotate_view(verts, view_theta, view_phi, yaw_deg=view_yaw)
 
    print('  Computing PCA vertex normals …')
    vn    = pca_vertex_normals(verts, k=20)
 
    # Approach-facing vertices (x > 0)
    front_mask = verts[:, 0] > 0
    fv         = verts[front_mask]      # (M, 3)
    fn         = vn[front_mask]         # (M, 3)
 
    # KD-trees
    tree3d = cKDTree(verts)            # full 3-D
    tree2d = cKDTree(fv[:, 1:])       # 2-D (y, z) — front face only
 
    # 2-D silhouette hull for inside/outside test in Panel 1
    sil_hull = ConvexHull(fv[:, 1:])   
 
    # Effective semi-axes
    semi = (verts.max(axis=0) - verts.min(axis=0)) / 2.0
    print(f'  Semi-axes (m):  approach(x) = {semi[0]:.1f}   '
          f'cross-track(y) = {semi[1]:.1f}   long(z) = {semi[2]:.1f}')
 
    # Aim point: centroid of the largest flat surface patch
    y_aim, z_aim = find_flattest_aim_point(fv, fn)
 
    return dict(verts=verts, vn=vn,
                fv=fv, fn=fn,
                tree3d=tree3d, tree2d=tree2d,
                sil_hull=sil_hull, semi=semi,
                aim=(y_aim, z_aim))
 
 
 
def inside_silhouette(y_flat, z_flat, mesh):
    """
    True for query points (y, z) that lie inside the 2-D silhouette
    """
    eq  = mesh['sil_hull'].equations      # (nfacets, 3): [a, b, c]
    yz  = np.column_stack([y_flat, z_flat])
    # Point is inside iff all half-space inequalities are satisfied
    return (yz @ eq[:, :2].T + eq[:, 2]).max(axis=1) <= 1e-10
 
 
def surface_x_and_nx(y_flat, z_flat, mesh):
    """
    For each 2-D query (y, z) on the impact face:
      x_val  — x-coordinate of the nearest approach-facing surface vertex
      nx     — x-component of that vertex's outward normal
      inside —  True where (y, z) is within the asteroid silhouette
    Returns three flat arrays of length len(y_flat).
    """
    ins     = inside_silhouette(y_flat, z_flat, mesh)
    yz      = np.column_stack([y_flat, z_flat])
    _, idx  = mesh['tree2d'].query(yz)
 
    x_val      = np.where(ins, mesh['fv'][idx, 0], np.nan)
    nx         = np.where(ins, mesh['fn'][idx, 0], np.nan)
    return x_val, nx, ins
 
 
def nearest_3d_normal(x_flat, y_flat, z_flat, mesh):
    """Return the outward-normal (nx, ny, nz) at the nearest vertex."""
    pts    = np.column_stack([x_flat, y_flat, z_flat])
    _, idx = mesh['tree3d'].query(pts)
    return mesh['vn'][idx]   # (N, 3)
 
 
# Constants 
AS = BS = CS = 1.0  # overwritten by shape model 
 
SIGMA_VALUES  = (25,)
SIGMA_COLOURS = ('#20528e',)
SIGMA_LABELS  = ('σ = 25 m ',)
 
BG   = '#ffffff'
FIG  = '#ffffff'
TEXT = '#000000'
TICK = '#353535'
GRID = '#8A8A8A'
 
TICK_SZ  = 18
AXIS_SZ  = 18
TITLE_SZ = 22
LEG_SZ   = 16
ANN_SZ   = 16
 
def _make_cmap():
    stops = [
        (0.00, (0.03, 0.02, 0.55)),
        (0.22, (0.84, 0.21, 0.07)),
        (0.55, (0.98, 0.60, 0.00)),
        (1.00, (0.98, 0.90, 0.22)),
    ]
    cd = {ch: [] for ch in ('red', 'green', 'blue')}
    for pos, (r, g, b) in stops:
        cd['red'  ].append((pos, r, r))
        cd['green'].append((pos, g, g))
        cd['blue' ].append((pos, b, b))
    return LinearSegmentedColormap('hot_custom', cd)
 
HOT = _make_cmap()
 
def dv_efficiency(nx, beta):
    """ΔV efficiency relative to a head-on centre impact.
    eff = sqrt(1 + (β²−1) nx²) / β,   nx = cos α."""
    return np.sqrt(1.0 + (beta ** 2 - 1.0) * nx ** 2) / beta
 
 
def _style_ax(ax):
    ax.tick_params(axis='both', labelsize=TICK_SZ, colors=TICK,
                   length=5, width=1.0)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)
 
 
# Panel 1 : impact face 
def draw_face(ax, beta, mesh):
    """
    Colour-mapped colormesh of ΔV efficiency over the asteroid silhouette .
    """
    s  = mesh['semi']
    ry = s[1] * 1.20      
    rz = s[2] * 1.10      
    N  = 400
 
    y_arr = np.linspace(-ry, ry, N)
    z_arr = np.linspace(-rz, rz, N)
    Y, Z  = np.meshgrid(y_arr, z_arr)
 
    y_flat = Y.ravel()
    z_flat = Z.ravel()
    _, nx_flat, ins = surface_x_and_nx(y_flat, z_flat, mesh)
 
    eff = np.full(len(y_flat), np.nan)
    ok  = ins & np.isfinite(nx_flat)
    eff[ok] = dv_efficiency(nx_flat[ok], beta)
 
    eff_min = 1.0 / beta
    t = (eff - eff_min) / (1.0 - eff_min)
 
    im = ax.pcolormesh(y_arr, z_arr, t.reshape(N, N),
                       cmap=HOT, vmin=0, vmax=1,
                       rasterized=True, shading='auto')
 
    # surface-normal quivers 
    ny_q, nz_q = 7, 13
    yq = np.linspace(-s[1] * 0.85, s[1] * 0.85, ny_q)
    zq = np.linspace(-s[2] * 0.85, s[2] * 0.85, nz_q)
    YQ, ZQ = np.meshgrid(yq, zq)
    yqf, zqf = YQ.ravel(), ZQ.ravel()
    xqf, nxq, inq = surface_x_and_nx(yqf, zqf, mesh)
 
    valid_pts = np.column_stack([xqf[inq], yqf[inq], zqf[inq]])
    full_n    = nearest_3d_normal(valid_pts[:, 0], valid_pts[:, 1],
                                   valid_pts[:, 2], mesh)
    nyq_v, nzq_v = full_n[:, 1], full_n[:, 2]
    alpha_q   = np.arccos(np.clip(nxq[inq], -1, 1))
    show      = alpha_q > 0.02
    ax.quiver(yqf[inq][show], zqf[inq][show],
              nyq_v[show], nzq_v[show],
              color='white', alpha=0.50, scale=11, width=0.003,
              headwidth=4, headlength=5, zorder=4)
 
    #  aim-point sigma ring 
    y_aim, z_aim = mesh['aim']
    for sigma, col, lbl in zip(SIGMA_VALUES, SIGMA_COLOURS, SIGMA_LABELS):
        el = mpatches.Ellipse((y_aim, z_aim), width=2 * sigma, height=2 * sigma,
                              edgecolor=col, facecolor='none',
                              linestyle='--', linewidth=2.0,
                              label=lbl, zorder=5)
        ax.add_patch(el)
    ax.plot(y_aim, z_aim, 'w+', ms=14, mew=2.0, zorder=6)
    ax.plot(y_aim, z_aim, 'wo', ms=7,  fillstyle='none', mew=1.8, zorder=6)
 
    ax.set_xlim(-ry, ry)
    ax.set_ylim(-rz, rz)
    ax.set_aspect('equal')
    ax.set_xlabel('cross-track  b  [m]', color=TICK, fontsize=AXIS_SZ)
    ax.set_ylabel('normal (long axis)  a  [m]', color=TICK, fontsize=AXIS_SZ)
    _style_ax(ax)
    ax.legend(loc='lower right', fontsize=LEG_SZ, framealpha=0.40,
              labelcolor=TEXT, facecolor=BG)
 
    cbar = plt.colorbar(im, ax=ax, fraction=0.028, pad=0.03)
    cbar.set_label('ΔV efficiency  (|ΔV| / |ΔV|$_{center}$)',
                   fontsize=ANN_SZ, color=TICK)
    cbar.ax.tick_params(labelsize=TICK_SZ - 1, colors=TICK)
    cbar.outline.set_edgecolor(GRID)
 
    ax.set_title(
        'Impact face\n'
        f'aₛ = {s[2]:.1f} m   bₛ = {s[1]:.1f} m   cₛ = {s[0]:.1f} m',
        fontsize=TITLE_SZ, color=TEXT, pad=6, linespacing=1.5)
 
 
# Panel 2: cross-section at the aim-point z-slice
def draw_xsec(ax, beta, mesh):
    """
    Actual asteroid contour projected onto the b-c plane.
    x-axis = cross-track b
    y-axis = approach direction c
    """

    verts = mesh['verts']
    s = mesh['semi']
    sigma = SIGMA_VALUES[0]
    col_sigma = SIGMA_COLOURS[0]

    B = verts[:, 1]
    C = verts[:, 0]

    pts2d = np.column_stack([B, C])
    hull = ConvexHull(pts2d)
    hull_pts = pts2d[hull.vertices]
    hull_pts = np.vstack([hull_pts, hull_pts[0]])

    # Asteroid projected contour
    ax.fill(
        hull_pts[:, 0],
        hull_pts[:, 1],
        color='#d9d9d9',
        alpha=0.25,
        zorder=1
    )
    ax.plot(
        hull_pts[:, 0],
        hull_pts[:, 1],
        color='#444444',
        lw=2.5,
        zorder=4
    )

    y_aim, z_aim = mesh['aim']
    _, idx = mesh['tree2d'].query([[y_aim, z_aim]])
    surf_pt = mesh['fv'][idx[0]]

    b_aim = surf_pt[1]
    c_aim = surf_pt[0]

    # Blue σ region in cross-track b
    ax.axvspan(
        b_aim - sigma,
        b_aim + sigma,
        color=col_sigma,
        alpha=0.13,
        zorder=0
    )

    ax.axvline(b_aim + sigma, color=col_sigma, ls='--', lw=1.1, alpha=0.55)
    ax.axvline(b_aim - sigma, color=col_sigma, ls='--', lw=1.1, alpha=0.55)

    # Normal-vector arrows and angle labels
    offsets = [0, 10, 20, 25] 

    for y_off in offsets:
        for sign in ([1] if y_off == 0 else [-1, 1]):
            qy = y_aim + sign * y_off

            _, idx2 = mesh['tree2d'].query([[qy, z_aim]])
            sv = mesh['fv'][idx2[0]]
            sn = mesh['fn'][idx2[0]]

            b0 = sv[1]
            c0 = sv[0]

            nb = sn[1]
            nc = sn[0]

            alpha = np.degrees(np.arccos(np.clip(nc, -1, 1)))

            eff = dv_efficiency(nc, beta)
            t = (eff - 1.0 / beta) / (1.0 - 1.0 / beta)
            r, g, b_, a_ = HOT(float(np.clip(t, 0, 1)))
            col_dark = (r * 0.85, g * 0.85, b_ * 0.85, a_)

            scale = 28
            tb = b0 + scale * nb
            tc = c0 + scale * nc

            lw = 4.0 if y_off == 0 else 3.0

            ax.annotate(
                '',
                xy=(tb, tc),
                xytext=(b0, c0+1),
                arrowprops=dict(
                    arrowstyle='->',
                    color=col_dark,
                    lw=lw,
                    mutation_scale=18
                ),
                zorder=9
            )
            if abs(abs(sign * y_off) - 0) <= 1 or abs(abs(sign* y_off)- sigma) <= 1:
                ax.text(
                    tb + 2,
                    tc + 10,
                    f'{alpha:.1f}°',
                    fontsize=ANN_SZ,
                    color=TEXT,
                    va='bottom',
                    ha='left',
                    zorder=10
                )

    # σ label
    ax.text(
        b_aim + sigma + 3,
        c_aim - 50,
        rf'$\sigma = {sigma:.0f}\,\mathrm{{m}}$',
        fontsize=ANN_SZ,
        color=col_sigma,
        zorder=10
    )

    ax.set_xlim(-(s[1] + 25), s[1] + 25)
    ax.set_ylim(-20, s[0] + 60)

    ax.set_xlabel('cross-track  b  [m]', color=TICK, fontsize=AXIS_SZ)
    ax.set_ylabel('approach direction  c  [m]', color=TICK, fontsize=AXIS_SZ)

    ax.set_title(
        'b–c target-plane projection',
        fontsize=TITLE_SZ,
        color=TEXT,
        pad=6
    )

    _style_ax(ax)

   
# Panel 3:  delta V efficiency vs. cross-track offset
def draw_efficiency_plot(ax, beta, mesh):
    s      = mesh['semi']
    y_aim, z_aim = mesh['aim']
    r_arr  = np.linspace(0, s[1] * 0.98, 400)
    nx_arr = np.empty(len(r_arr))
    for i, y in enumerate(r_arr):
        _, idx    = mesh['tree2d'].query([[y, z_aim]])
        nx_arr[i] = mesh['fn'][idx[0], 0]
 
    for sigma, col, lbl in zip(SIGMA_VALUES, SIGMA_COLOURS, SIGMA_LABELS):
        ax.axvspan(0, sigma, alpha=0.09, color=col, label=lbl)
        ax.axvline(sigma, color=col, ls='--', lw=1.4, alpha=0.85)
 
    for ev in [0.75, 0.80, 0.85, 0.90, 0.95, 1.00]:
        ax.axhline(ev, color=GRID, lw=0.5)
 
    beta_cases = [
        (1.5, '#1a6fbb', (8, 3), 'β = 1.5  porous rubble-pile'),
        (4.5, '#cc3300', (2, 3), 'β = 4.5  dense rock'),
    ]
    for bt, col, dashes, lbl in beta_cases:
        eff_arr = dv_efficiency(nx_arr, bt)
        lw = 3.5 if bt == beta else 1.6
        ax.plot(r_arr, eff_arr, color=col, ls=(0, dashes), lw=lw, label=lbl)
 
    # Annotate efficiency loss at the σ boundary
    sigma = SIGMA_VALUES[0]
    _, idx25    = mesh['tree2d'].query([[float(sigma), z_aim]])
    nx25        = mesh['fn'][idx25[0], 0]
    y_offsets   = {1.5: 5, beta: 22, 4.5: -50}
    for bt, col, _, _ in beta_cases:
        e    = dv_efficiency(nx25, bt)
        loss = (1.0 - e) * 100
        ax.annotate(
            f'−{loss:.1f}%',
            xy=(sigma, e),
            xytext=(30, y_offsets.get(bt, -20)),
            textcoords='offset points',
            fontsize=ANN_SZ,
            color=col,
            ha='center',
            va='bottom',
            clip_on=False,
            zorder=20
        )
 
    ax.set_xlim(0, 75)
    ax.set_ylim(0.69, 1.02)
    ax.set_xlabel('cross-track offset from aim point  [m]',
                  color=TICK, fontsize=AXIS_SZ)
    ax.set_ylabel('ΔV efficiency', color=TICK, fontsize=AXIS_SZ)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f'{v:.2f}'))
    ax.legend(fontsize=LEG_SZ, loc='lower left', framealpha=0.40,
              labelcolor=TEXT, facecolor=BG)
    ax.set_title('ΔV efficiency vs. cross-track offset',
                 fontsize=TITLE_SZ, color=TEXT, pad=10)
    _style_ax(ax)
 
 
# Figure assembly
def make_figure(beta=2.5, shape_path=DEFAULT_SHAPE,
                view_theta=0.0, view_phi=0.0, view_yaw=0.0, save_path=None):
    mesh = load_mesh(shape_path, view_theta=view_theta,
                     view_phi=view_phi, view_yaw=view_yaw)
 
    matplotlib.rcParams.update({
        'axes.facecolor':   BG,
        'axes.edgecolor':   GRID,
        'axes.labelcolor':  TICK,
        'xtick.color':      TICK,
        'ytick.color':      TICK,
        'text.color':       TEXT,
        'figure.facecolor': FIG,
        'grid.color':       GRID,
    })
 
    fig = plt.figure(figsize=(24, 12.5), facecolor=FIG)
    gs  = fig.add_gridspec(
        2, 2,
        width_ratios=[1.1, 1],
        hspace=0.48, wspace=0.28,
        left=0.05, right=0.97,
        top=0.92, bottom=0.09,
    )
 
    ax_face = fig.add_subplot(gs[:, 0])
    ax_xsec = fig.add_subplot(gs[0, 1])
    ax_plot = fig.add_subplot(gs[1, 1])
 
    draw_face(ax_face, beta, mesh)
    draw_xsec(ax_xsec, beta, mesh)
    draw_efficiency_plot(ax_plot, beta, mesh)
 
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        print(f'Saved → {save_path}')
 
    return fig
 
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='ΔV efficiency figure using the Itokawa-derived shape model.')
    parser.add_argument('--beta',       type=float, default=2.5)
    parser.add_argument('--shape',      type=str,   default=DEFAULT_SHAPE)
    parser.add_argument('--view-theta', type=float, default=90.0,
                        help='Azimuth θ from viewer (degrees)')
    parser.add_argument('--view-phi',   type=float, default=-180.0,
                        help='Elevation φ from viewer (degrees)')
    parser.add_argument('--yaw',        type=float, default=180.0,
                        help='Roll ψ around approach axis from viewer (degrees)')
    parser.add_argument('--save',       type=str,   default=None, metavar='PATH')
    args, _ = parser.parse_known_args()
 
    fig = make_figure(
        beta        = args.beta,
        shape_path  = args.shape,
        view_theta  = args.view_theta,
        view_phi    = args.view_phi,
        view_yaw    = args.yaw,
        save_path   = args.save or 'asteroid_dv_efficiency_shape.png',
    )
    plt.show()