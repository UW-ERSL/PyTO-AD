"""Shared style for the paper figures made in fig_gen/.

Conventions for every figure of this repository: FIGURE_GUIDE.md (next to this file).
Same look as pyto.examples_benchmarks.make_paper_figures (Helvetica-class sans, 8 pt, Okabe-Ito colours), sized for the
journal text width (sn-jnl: 372 pt = 5.15 in) so figures go in at \\includegraphics[width=\\linewidth] unscaled.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))          # repository root (this file is in <repo>/paper/fig_gen)
# Figures go to <repo>/paper/figures/generated (git-ignored); set PYTO_PAPER_FIGURES to write elsewhere.
OUT = os.environ.get("PYTO_PAPER_FIGURES") or os.path.join(REPO, "paper", "figures", "generated")
CACHE = os.path.join(HERE, "_cache")                  # cached optimization runs (git-ignored)

TEXT_WIDTH = 5.15                    # in
PAPER_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "Nimbus Sans", "DejaVu Sans"],
    "mathtext.fontset": "custom", "mathtext.rm": "sans", "mathtext.it": "sans:italic", "mathtext.bf": "sans:bold",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.linewidth": 0.7, "savefig.dpi": 600, "pdf.fonttype": 42, "ps.fonttype": 42,
}
plt.rcParams.update(PAPER_RC)

# one colour per role, the same in every figure.
# RULE: topology (designs, recovered parts, meshes of them) always in PyTO's classic look, as plot_mesh draws it
# (pyto/physics/structural/hex_structural_fea.py): 'lightgreen' elements with black edges, dark-red load arrows.
# Field plots (stress, temperature) keep their own colour maps.
MATERIAL = "#90EE90"                 # PyTO 'lightgreen': solid material / design / recovered part
MESH_EDGE = "black"                  # PyTO element edges
DOMAIN = "#D4F2D4"                   # design domain before optimization (pale PyTO green)
LOAD_ARROW = "#B00000"               # PyTO dark-red load arrows
SUPPORT = "#0072B2"                  # supports (Okabe-Ito blue)
LOAD = "#D55E00"                     # load highlight and the backward pass in diagrams (vermilion)
DENSITY_CMAP_COLORS = ("white", "#3E8E3E")   # continuous density 0 -> 1, white to PyTO green
ACCENT = "#009E73"                   # formulas (bluish green)
INK = "#222222"


def save(fig, name):
    """Write <name>.pdf (vector, for the paper) and <name>.png (preview) to OUT (paper/figures/generated/ by default)."""
    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=600 if ext == "png" else None)
    plt.close(fig)
    print("wrote", os.path.join(OUT, name + ".pdf"))


def crop(rgba, pad=6):
    """Crop an RGBA render to its non-transparent pixels (plus pad)."""
    import numpy as np
    alpha = rgba[..., 3] if rgba.shape[-1] == 4 else (rgba[..., :3].min(-1) < 250)
    ys, xs = np.nonzero(alpha)
    return rgba[max(ys.min() - pad, 0):ys.max() + pad, max(xs.min() - pad, 0):xs.max() + pad]
