"""Rendering helpers shared by the figure scripts: one camera convention, one colour set, one lighting.

3D panels are PyVista off-screen renders with a transparent background, cropped to the part; matplotlib places
them and adds the labels. Sweep images (saved only as PNGs) are cropped and recoloured to the same material grey.
"""
import numpy as np

from _style import LOAD_ARROW, MATERIAL, MESH_EDGE, SUPPORT, crop

WINDOW = (1800, 1500)
LIGHT = dict(ambient=0.2, diffuse=0.85)              # close to PyVista's default, as in PyTO's own plots
EDGE = MESH_EDGE


def plotter(window=WINDOW):
    import pyvista as pv
    pv.OFF_SCREEN = True
    p = pv.Plotter(off_screen=True, window_size=window)
    p.enable_anti_aliasing("ssaa")
    return p


def camera(p, bounds_lo, bounds_hi, direction, up, parallel=False, distance=3.0):
    lo, hi = np.asarray(bounds_lo, float), np.asarray(bounds_hi, float)
    c = 0.5 * (lo + hi)
    d = np.asarray(direction, float)
    p.camera_position = [tuple(c + distance * np.linalg.norm(hi - lo) * d / np.linalg.norm(d)), tuple(c), tuple(up)]
    if parallel:
        p.enable_parallel_projection()
    p.reset_camera()


def shot(p, pad=8):
    img = p.screenshot(transparent_background=True, return_img=True)
    p.close()
    return crop(img, pad)


def hex_grid(node_xyz, elems):
    import pyvista as pv
    cells = np.hstack([np.full((len(elems), 1), 8), elems]).ravel()
    return pv.UnstructuredGrid(cells, np.full(len(elems), pv.CellType.HEXAHEDRON, dtype=np.uint8),
                               np.asarray(node_xyz, float))


def add_design(p, grid, density, elems, fixed=None, edges=True, threshold=0.5):
    """Solid elements (density > threshold) in the material grey; elements touching supports in blue."""
    from matplotlib.colors import ListedColormap
    keep = np.nonzero(density > threshold)[0]
    role = np.zeros(len(elems))
    if fixed is not None and len(fixed):
        role[np.isin(elems, fixed).any(1)] = 1
    g = grid.extract_cells(keep)
    g.cell_data["role"] = role[keep]
    p.add_mesh(g, scalars="role", cmap=ListedColormap([MATERIAL, SUPPORT]), clim=(0, 1), show_scalar_bar=False,
               show_edges=edges, edge_color=EDGE, line_width=1, **LIGHT)


def add_load_arrows(p, node_xyz, force, n_max=4, length=None, tail_at_node=False):
    """Up to n_max arrows along the load direction, spread over the loaded nodes, ending on them (or starting
    from them, tail_at_node, when an arrow ending on the node would cover the part)."""
    import pyvista as pv
    f = np.asarray(force, float).reshape(-1, 3)
    loaded = np.nonzero(np.abs(f).sum(1))[0]
    if not len(loaded):
        return
    direction = f[loaded].sum(0)
    direction /= np.linalg.norm(direction)
    xyz = np.asarray(node_xyz, float)
    span = np.linalg.norm(xyz.max(0) - xyz.min(0))
    length = length or 0.14 * span
    pts = xyz[loaded]
    if len(pts) > n_max:                                   # spread: extreme points along the main axis of the set
        axis = np.argmax(pts.max(0) - pts.min(0))
        pts = pts[np.argsort(pts[:, axis])][np.linspace(0, len(pts) - 1, n_max).astype(int)]
    for q in pts:
        p.add_mesh(pv.Arrow(start=q if tail_at_node else q - direction * length, direction=direction, scale=length, tip_length=0.3,
                            tip_radius=0.11, shaft_radius=0.04), color=LOAD_ARROW, ambient=0.4)


def sweep_part_box(path):
    """(y0, y1, x0, x1) of the part in a sweep PNG: its green faces, without the title and the axis triad."""
    from PIL import Image
    rgb = np.asarray(Image.open(path).convert("RGB"))
    r, g, b = (rgb[..., k].astype(int) for k in range(3))
    green = (g > r + 18) & (g > b + 18)                   # the part's faces
    h, w = green.shape
    green[int(0.78 * h):, :int(0.14 * w)] = False         # the axis triad, always in the bottom-left corner
    ys, xs = np.nonzero(green)
    return ys.min(), ys.max(), xs.min(), xs.max()


def crop_sweep_png(path, pad=10, box=None):
    """A sweep PNG (PyTO's own render: green elements on white, a title and an axis triad), cropped to the part, or
    to `box` (y0, y1, x0, x1): pass the union of two renders' boxes to crop a pair at one scale. Colours are kept
    as PyTO drew them. Returns RGBA, opaque, on white."""
    from PIL import Image
    rgb = np.asarray(Image.open(path).convert("RGB"))
    y0, y1, x0, x1 = box if box is not None else sweep_part_box(path)
    part = rgb[y0:y1 + 1, x0:x1 + 1]
    part = np.pad(part, ((pad, pad), (pad, pad), (0, 0)), constant_values=255)
    return np.dstack([part, np.full(part.shape[:2], 255, np.uint8)])


def panel_grid(images, labels, ncols, width, label_size=8, gap=0.12, label_h=0.17, max_h=None):
    """A figure with images in a grid (equal width per column), each with its part label, "(a)", "(b)", ...,
    underneath (descriptions go in the LaTeX caption, FIGURE_GUIDE.md). Returns the matplotlib figure."""
    import matplotlib.pyplot as plt
    nrows = int(np.ceil(len(images) / ncols))
    cw = (width - (ncols - 1) * gap) / ncols
    row_h = []
    for r in range(nrows):
        hs = [im.shape[0] / im.shape[1] * cw for im in images[r * ncols:(r + 1) * ncols]]
        row_h.append(min(max(hs), max_h) if max_h else max(hs))
    H = sum(row_h) + nrows * label_h + (nrows - 1) * gap * 0.5 + 0.04
    fig = plt.figure(figsize=(width, H))
    y = H
    for r in range(nrows):
        y -= row_h[r]
        for c in range(ncols):
            k = r * ncols + c
            if k >= len(images):
                break
            ax = fig.add_axes([c * (cw + gap) / width, y / H, cw / width, row_h[r] / H])
            ax.imshow(images[k], interpolation="antialiased")
            ax.set_xlim(-0.5, images[k].shape[1] - 0.5)
            ax.set_ylim(images[k].shape[0] - 0.5, -0.5)
            ax.set_anchor("S")
            ax.axis("off")
            fig.text((c * (cw + gap) + cw / 2) / width, (y - 0.03) / H, labels[k], ha="center", va="top",
                     fontsize=label_size)
        y -= label_h + gap * 0.5
    return fig


def recovered_part(d, stl_path, cache_path):
    """The smooth STL of a design, recovered as the GUI does it (pyto.io.topopt_stl_recovery: CNN-smoothed void
    iso-surface subtracted from the design-domain STL). d: a design npz (x, node_xyz, elems). Cached as .vtp."""
    import os
    import pyvista as pv
    if not os.path.exists(cache_path):
        from pyto.io.topopt_stl_recovery import extract_isosurface, subtract_voids_from_stl
        vtu = hex_grid(d["node_xyz"], d["elems"])
        vtu.cell_data["density"] = (np.asarray(d["x"]) > 0.5).astype(float)
        part = subtract_voids_from_stl(pv.read(stl_path), extract_isosurface(vtu, 0.5, method="cnn"))
        part.save(cache_path)
    return pv.read(cache_path)


def smoothed_material_surface(d, cache_path, sigma=None):
    """The material iso-surface of a design without the Boolean step, for a domain STL that is not a closed volume:
    the 0/1 density on the regular grid (void outside the mesh), smoothed as in the STL recovery (its CNN, or with
    sigma its Gaussian filter of width sigma grid spacings), contoured at 0.5. Cached as .vtp."""
    import os
    import pyvista as pv
    if not os.path.exists(cache_path):
        vtu = hex_grid(d["node_xyz"], d["elems"])
        vtu.cell_data["density"] = (np.asarray(d["x"]) > 0.5).astype(float)
        vtu = vtu.cell_data_to_point_data()
        axes = [np.unique(np.round(vtu.points[:, k], 9)) for k in range(3)]
        h = [(a[-1] - a[0]) / (len(a) - 1) for a in axes]
        grid = pv.ImageData(dimensions=[len(a) for a in axes], spacing=h, origin=[a[0] for a in axes]).sample(vtu)
        field = np.nan_to_num(grid.point_data["density"], nan=0.0).reshape([len(a) for a in axes], order="F")
        field = np.pad(field, 1)
        if sigma is not None:
            from scipy.ndimage import gaussian_filter
            smooth = gaussian_filter(field, sigma=sigma)
        else:
            smooth = _cnn_smooth(field)
        img = pv.ImageData(dimensions=smooth.shape, spacing=h, origin=[a[0] - s for a, s in zip(axes, h)])
        img.point_data["s"] = smooth.flatten(order="F")
        img.contour([0.5], scalars="s", method="marching_cubes").save(cache_path)
    return pv.read(cache_path)


def _cnn_smooth(field):
    import torch
    from pyto.io.topopt_stl_recovery import CNN3D
    x = torch.tensor(field).float()[None, None]
    model = CNN3D()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    for _ in range(400):                                   # the recovery's own fit: MSE 2e-4 or 400 epochs
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model(x), x.to(model.device))
        loss.backward()
        opt.step()
        if loss.item() < 2e-4:
            break
    model.eval()
    with torch.no_grad():
        return model(x).cpu().numpy()[0, 0]


def add_surface(p, surf, color=MATERIAL, edges=True, role=None):
    """A smooth surface (an STL or a recovered part) in `color` with its feature edges; with role (per face 0/1/2,
    from bc_faces or transfer_roles) the faces at the supports in blue and at the load in the load colour."""
    from matplotlib.colors import ListedColormap
    surf = surf.copy()
    surf.cell_data["role"] = np.zeros(surf.n_cells) if role is None else np.asarray(role, float)
    p.add_mesh(surf, scalars="role", cmap=ListedColormap([color, SUPPORT, LOAD_ARROW]), clim=(0, 2),
               show_scalar_bar=False, smooth_shading=True, split_sharp_edges=True, specular=0.25, **LIGHT)
    if edges:
        p.add_mesh(surf.extract_feature_edges(60, boundary_edges=False, non_manifold_edges=False),
                   color=EDGE, line_width=2)


def bc_faces(stl, node_xyz, fixed, force, tol, split=True, skip_axis=None):
    """The domain STL (its large triangles split if split) with per-face role 0/1/2 (plain/support/load): faces whose
    three corners lie within tol of a fixed or loaded node. skip_axis: leave out faces normal to that axis (the
    front and back faces of a plate whose supports and loads act on the hole walls only)."""
    from scipy.spatial import cKDTree
    surf = stl.triangulate().subdivide_adaptive(max_edge_len=0.5 * tol) if split else stl.triangulate()
    tri = surf.faces.reshape(-1, 4)[:, 1:]
    role = np.zeros(surf.n_cells)
    f = np.asarray(force, float).reshape(-1, 3)
    for k, nodes in ((1, fixed), (2, np.nonzero(np.abs(f).sum(1))[0])):
        if len(nodes):                                     # all three corners near such nodes
            dist = cKDTree(node_xyz[nodes]).query(surf.points)[0]
            role[dist[tri].max(1) < tol] = k
    if skip_axis is not None:
        role[np.abs(surf.compute_normals(cell_normals=True, point_normals=False).cell_data["Normals"][:, skip_axis]) > 0.9] = 0
    surf.cell_data["role"] = role
    return surf


def transfer_roles(part, ref, tol):
    """Per-face roles of `part` copied from `ref` (bc_faces) where the part's face lies on the domain STL (within tol):
    the recovered part keeps the domain's surfaces at supports and loads, so only those faces are coloured."""
    c = part.cell_centers().points
    cells, closest = ref.find_closest_cell(c, return_closest_point=True)
    return np.where(np.linalg.norm(c - closest, axis=1) < tol, ref.cell_data["role"][cells], 0.0)


def add_hole_arrow(p, node_xyz, force, length, z_front=None):
    """One load arrow from the centre of the loaded nodes along the total load (from the front face if z_front)."""
    import pyvista as pv
    f = np.asarray(force, float).reshape(-1, 3)
    loaded = np.nonzero(np.abs(f).sum(1))[0]
    direction = f[loaded].sum(0) / np.linalg.norm(f[loaded].sum(0))
    start = node_xyz[loaded].mean(0)
    if z_front is not None:
        start[2] = z_front
    p.add_mesh(pv.Arrow(start=start, direction=direction, scale=length, tip_length=0.3, tip_radius=0.11,
                        shaft_radius=0.04), color=LOAD_ARROW, ambient=0.4)
