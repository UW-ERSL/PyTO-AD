"""Fig. 4 (fig:gui_formulation; the former Figs. 4 and 5, merged): the recorded GUI session on ThreeHoleBracketThick.

Assembled from the screen captures of gui_session_threehole.py (Results/GUISession_ThreeHole/raw), cropped, never
edited: the GUI's own colours and text.
  In workflow order: (a) the input STL with supports and load (geometry drawn semi-transparent, Display Options, to
  show the load on the hole surface), (b) Objective tab, (c) Constraints tab, (d) gradient check, (e) the live
  objective and constraint plot up to iteration 15 (integer iteration ticks, commit ffa221d), (f) the density in the
  3D view at iteration 15 (the GUI's "Iteration 15" overlay cropped), (g) the recovered smooth STL. Crops are sized so
  the GUI's own text prints at >= 6 pt (checked in fig4).

Run from docs/paper/fig_gen:  python fig04_05_gui.py  ->  ../figures/generated/fig04_gui_formulation
"""
import os

import numpy as np
from PIL import Image

from _style import INK, REPO, TEXT_WIDTH, save

RAW = os.path.join(REPO, "Results", "GUISession_ThreeHole", "raw")


def load(name, box=None):
    im = Image.open(os.path.join(RAW, name + ".png")).convert("RGB")
    return np.asarray(im.crop(box) if box else im)


def content(img, top=0, pad=12, white=245):
    """Crop a 3D-view capture to its non-white content below row `top` (drops the model-info text)."""
    sub = img[top:]
    ys, xs = np.nonzero(sub.min(-1) < white)
    return sub[max(ys.min() - pad, 0):ys.max() + pad, max(xs.min() - pad, 0):xs.max() + pad]


def place(fig, W, H, img, x, y, w, label, frame=True):
    """Image at (x, y) inches (bottom-left), width w; its height follows the aspect. Label '(a)' underneath."""
    import matplotlib.pyplot as plt  # noqa: F401
    h = w * img.shape[0] / img.shape[1]
    ax = fig.add_axes([x / W, y / H, w / W, h / H])
    ax.imshow(img, interpolation="antialiased")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(frame)
        s.set_linewidth(0.5)
        s.set_color("0.6")
    fig.text((x + w / 2) / W, (y - 0.04) / H, label, ha="center", va="top", fontsize=8, color=INK)
    return h


# Printed text size of the GUI captures: the GUI draws its text at fixed pixel sizes (QT_SCALE_FACTOR=2), so the
# size in print is em_px * (placed width in pt) / (crop width in px). Measured on the captures: the widget font's
# capital height is 17 px (em = 17 / 0.72 = 23.6 px); the live plot's tick digits and legend (the legend at the default
# size since the GUI change of 2026-10-22) are 21 px high (em 21 / 0.72 = 29 px). Every panel must print at >= 6 pt
# (FIGURE_GUIDE.md), and the figure must fit a page with its caption: <= 5.2 in tall.
GUI_EM_PX, PLOT_EM_PX, MIN_PT, MAX_HEIGHT_IN = 17 / 0.72, 21 / 0.72, 6.0, 5.2


def printed_pt(em_px, crop_px, width_in):
    return em_px * width_in * 72 / crop_px


def fig4():
    """The session in workflow order, three rows: (a) input STL with supports and load | (b) Objective tab;
    (c) Constraints tab | (d) gradient check; (e) live objective/constraint plot | (f) density at iteration 15 |
    (g) recovered smooth STL."""
    import matplotlib.pyplot as plt
    a = content(load("1b_stl_supports_load_transparent_view"), top=140)   # below the model-info text
    b = load("3_topopt_objective", (0, 0, 915, 470))
    c = load("4_topopt_constraints", (40, 0, 862, 205))      # up to the "Bound" header (x 785-854)
    d = load("5_gradient_check", (58, 18, 626, 172))         # the four result lines (text x 64-619); OK button cropped
    e = load("7_running_topopt", (40, 800, 925, 1195))       # the plot, incl. the blue "objective" label
    f = content(load("6_running_main_view"), top=100)       # below the GUI's "Iteration 15" overlay (rows 5-49)
    g = content(load("10_recovered_stl_view"))
    W, gap, lab = TEXT_WIDTH, 0.12, 0.18
    wb, wc, wd, we = 3.30, 2.95, 2.08, 3.0
    sizes = {"(b)": printed_pt(GUI_EM_PX, b.shape[1], wb), "(c)": printed_pt(GUI_EM_PX, c.shape[1], wc),
             "(d)": printed_pt(GUI_EM_PX, d.shape[1], wd), "(e)": printed_pt(PLOT_EM_PX, e.shape[1], we)}
    print("printed GUI text (pt):", {k: round(v, 2) for k, v in sizes.items()})
    assert min(sizes.values()) >= MIN_PT, f"GUI text below {MIN_PT} pt: {sizes}"
    hb = wb * b.shape[0] / b.shape[1]
    wa = W - wb - gap
    ha = min(hb, wa * a.shape[0] / a.shape[1])
    wa = ha * a.shape[1] / a.shape[0]
    r1 = hb
    hc, hd = wc * c.shape[0] / c.shape[1], wd * d.shape[0] / d.shape[1]
    r2 = max(hc, hd)
    he = we * e.shape[0] / e.shape[1]
    r3 = he                                                  # (f), (g) at the plot's height, or narrower to fit
    wf, wg = r3 * f.shape[1] / f.shape[0], r3 * g.shape[1] / g.shape[0]
    room = W - we - 2 * gap
    if wf + wg > room:
        k = room / (wf + wg)
        wf, wg = wf * k, wg * k
    hf, hg = wf * f.shape[0] / f.shape[1], wg * g.shape[0] / g.shape[1]
    H = r1 + r2 + r3 + 3 * lab + 0.02
    assert H <= MAX_HEIGHT_IN, f"figure {H:.2f} in tall, over {MAX_HEIGHT_IN} in"
    fig = plt.figure(figsize=(W, H))
    y = H - r1
    place(fig, W, H, a, (W - wb - gap - wa) / 2, y + (r1 - ha) / 2, wa, "(a)", frame=False)
    place(fig, W, H, b, W - wb, y, wb, "(b)")
    y -= lab + r2
    x_c = (W - wc - wd - gap) / 2
    place(fig, W, H, c, x_c, y + r2 - hc, wc, "(c)")
    place(fig, W, H, d, x_c + wc + gap, y + r2 - hd, wd, "(d)")
    y -= lab + r3
    place(fig, W, H, e, 0, y, we, "(e)")
    place(fig, W, H, f, we + gap, y + (r3 - hf) / 2, wf, "(f)", frame=False)
    place(fig, W, H, g, we + 2 * gap + wf, y + (r3 - hg) / 2, wg, "(g)", frame=False)
    print(f"figure {W:.2f} x {H:.2f} in")
    save(fig, "fig04_gui_formulation")


if __name__ == "__main__":
    fig4()
