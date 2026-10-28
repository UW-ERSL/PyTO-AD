"""Publication figures from the recorded results (no optimization is rerun).

Conventions (font, colours, part labels, captions): paper/fig_gen/FIGURE_GUIDE.md.

Every figure is a single plot (no subplots), sized for one journal column (3.5 in) so it goes into LaTeX at
\\includegraphics[width=\\columnwidth] without rescaling; fonts are Helvetica-class sans serif (Arial, Helvetica or a
metric clone; Springer/SMO artwork guideline) at final print size and are
embedded as TrueType. Each figure is written as vector PDF (for the paper) and PNG (preview).

Encoding is fixed across figures: one colour per linear solver (Okabe-Ito, colour-blind safe); automatic
differentiation solid with filled markers, analytical sensitivities dashed with open markers (readable in greyscale);
a black x marks a run that failed, with its reason. Time and memory against problem size are log-log with the fitted
slope (points >= 100k dofs) in the legend. SPSOLVE and PCG are not plotted.

Data (all under Results/):
  SolverScaling_2026-10-21_noretain/  10-iteration scaling study, EdgeCantilever 29k-1.06M dofs, AD and analytical
                                      (after the retention fix; SolverScaling_2026-10-16b is the same study before it)
  SolverScaling_full_2026-10-17(b)/   full optimizations (AD): 85k check, 540k, 1.06M
  GradientComparison_2026-10-18b/     AD vs analytical per iteration, 3 repeats (MMA; OC once its run is complete)
  Old/GRADIENT_COMPARISON_2026-10-15.csv  AD vs analytical totals (used for OC until the new OC run is complete)
  Results_2026-10-15/                 benchmark sweep (MMA, AD)
  ../paper/data/benchmark_problems.csv           objective types of the sweep problems

Run from the repository root:
    python -m pyto.examples_benchmarks.make_paper_figures
Output: Results/PaperFigures/fig_*.pdf and .png, figures_manifest.csv (figure, data source, notes).
"""
import csv
import glob
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402
import matplotlib.patches  # noqa: E402,F401
import matplotlib.transforms  # noqa: E402,F401
from matplotlib.lines import Line2D                 # noqa: E402

RESULTS = "Results"
OUT = os.path.join(RESULTS, "PaperFigures")
# 10-iteration scaling study after the 2026-10-21 fix (FE solvers no longer keep the previous solve's graph and factors
# alive into the next solve); SolverScaling_2026-10-16b is the same study before the fix
SCALING = os.path.join(RESULTS, "SolverScaling_2026-10-21_noretain")
FULL = [os.path.join(RESULTS, "SolverScaling_full_2026-10-17"), os.path.join(RESULTS, "SolverScaling_full_2026-10-17b")]
GRADCOMP = os.path.join(RESULTS, "Old", "GRADIENT_COMPARISON_2026-10-15.csv")   # totals only; fallback for OC
# per-iteration comparison with phase timers, 1 repeat, on the frozen code snapshot of the C1 study
GRADCOMP_NEW = os.path.join(RESULTS, "GradientComparison_2026-10-22_instr")
# C1 timing: two uninstrumented sessions (--plain, 3 repeats each) on the same snapshot; the time-per-iteration figure
# pools their per-run medians (6 per problem and gradient method)
GRADCOMP_TIME = [os.path.join(RESULTS, "GradientComparison_2026-10-22_plain_s1"),
                 os.path.join(RESULTS, "GradientComparison_2026-10-22_plain_s2")]
# memory of the same pairs after the 2026-10-21 retention fix (1 repeat): the 2026-10-18b runs kept the previous
# solve's graph and factors alive into the next solve, which inflated their peaks; their times are unaffected.
# MMA: 10 iterations. OC: run to convergence, as in 2026-10-18b; capped at 10 iterations, the end-of-run evaluation of
# an unconverged design raised OC's run peak above its iteration peak (TorquePlate analytical: 3.25 vs 1.99 GB).
GRADCOMP_MEMORY = {"MMA": os.path.join(RESULTS, "GradientComparison_2026-10-21_memory"),
                   "OC": os.path.join(RESULTS, "GradientComparison_2026-10-22_memory_oc")}
MEMORY_FIELDS = ("peak_rss_ad_gb", "peak_rss_an_gb", "ad_graph_gb")
NEW_MIN_PROBLEMS = 20         # use the new comparison for a method once it has at least this many finished pairs
SWEEP = os.path.join(RESULTS, "Results_2026-10-15")
PROBLEM_TABLE = os.path.join("paper", "data", "benchmark_problems.csv")
FD_CHECK = os.path.join(RESULTS, "GradientFD", "fd_vs_step.csv")              # gradient_fd_check.py
PROBLEM = "EdgeCantilever"
MACHINE_RAM_GB = 78.3
GIB = 1024 ** 3

SINGLE_COL = 3.5                                     # inches
HEIGHT = 2.6
PAPER_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "Nimbus Sans", "DejaVu Sans"],
    "mathtext.fontset": "custom", "mathtext.rm": "sans", "mathtext.it": "sans:italic", "mathtext.bf": "sans:bold",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "legend.frameon": False, "legend.handlelength": 2.2,
    "axes.linewidth": 0.7, "lines.linewidth": 1.1, "lines.markersize": 3.8,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7, "xtick.minor.width": 0.5, "ytick.minor.width": 0.5,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "grid.linewidth": 0.4, "grid.color": "0.88", "axes.grid": True, "axes.axisbelow": True,
    "savefig.dpi": 600, "savefig.bbox": "standard",     # exact figure size (no tight bounding box)
    "pdf.fonttype": 42, "ps.fonttype": 42,
}

SOLVERS = ("PARDISO", "DPCG", "DPCG_GPU")
SOLVER_COLOR = {"PARDISO": "#0072B2", "DPCG": "#009E73", "DPCG_GPU": "#CC79A7"}
SOLVER_MARKER = {"PARDISO": "o", "DPCG": "s", "DPCG_GPU": "^"}
SOLVER_LABEL = {"PARDISO": "PARDISO (direct)", "DPCG": "DPCG (iterative, CPU)", "DPCG_GPU": "DPCG (iterative, GPU)"}
GRAD_LABEL = {"AUTODIFF": "AD", "ANALYTICAL": "analytical"}
CLASS_COLOR = {"structural compliance": "#56B4E9", "stress / volume / mass": "#D55E00", "compliant mechanism": "#000000",
               "thermal compliance": "#E69F00", "thermo-structural compliance": "#CC79A7"}
PHASE_COLOR = {"assembly": "#BBBBBB", "factorization / set-up": "#0072B2", "forward solve": "#56B4E9",
               "backward pass": "#E69F00", "adjoint solve": "#D55E00", "MMA update and other": "#777777"}

MANIFEST = []


# ------------------------------------------------------------------------------------------------- helpers

def _f(v):
    try:
        v = float(v)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _new(height=HEIGHT, width=SINGLE_COL):
    """One plot at its final printed size; constrained layout keeps labels and legends inside the figure, so the
    PDF is exactly `width` inches wide."""
    fig, ax = plt.subplots(figsize=(width, height), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02)
    return fig, ax


# Figures the paper includes at a width other than SINGLE_COL: drawn at that width (height scaled alike, except
# plots whose height is set by their rows, KEEP_HEIGHT), so LaTeX
# includes them unscaled and the 8/7 pt text stays 8/7 pt in print (FIGURE_GUIDE.md). sn-jnl \linewidth = 5.15 in.
FINAL_WIDTH = {
    "fig_ad_vs_analytical_time_per_iteration_mma": 0.85 * 5.15,     # 4.38 in
    "fig_scaling_ad_graph_memory_vs_elements": 0.85 * 5.15,
    "fig_scaling_time_per_iteration_vs_dofs": 0.85 * 5.15,
    "fig_gradient_vs_fd_step": 0.85 * 5.15,
    "fig_ad_vs_analytical_objective_difference": 0.80 * 5.15,       # 4.12 in
}
KEEP_HEIGHT = {"fig_ad_vs_analytical_time_per_iteration_mma"}         # one row per problem


def _save(fig, name, sources, note=""):
    os.makedirs(OUT, exist_ok=True)
    if name in FINAL_WIDTH:
        w, h = fig.get_size_inches()
        fig.set_size_inches(FINAL_WIDTH[name], h if name in KEEP_HEIGHT else h * FINAL_WIDTH[name] / w)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=600)
    plt.close(fig)
    MANIFEST.append({"figure": name, "sources": "; ".join(sources), "notes": note})
    print(f"  {name}")


def _slope(x, y, xmin=1e5):
    x, y = np.asarray(x, float), np.asarray(y, float)
    keep = (x >= xmin) & (y > 0)
    if keep.sum() < 2:
        return None
    return np.polyfit(np.log10(x[keep]), np.log10(y[keep]), 1)[0]


def _dofs_axis(ax, label="Degrees of freedom"):
    ax.set_xscale("log")
    ax.set_xlabel(label)
    ax.grid(True, which="major")


def _line(ax, x, y, solver, grad, label, **kw):
    filled = grad == "AUTODIFF"
    ax.plot(x, y, color=SOLVER_COLOR[solver], ls="-" if filled else "--", marker=SOLVER_MARKER[solver],
            mfc=SOLVER_COLOR[solver] if filled else "white", mec=SOLVER_COLOR[solver], label=label, **kw)


def _legend_below(ax, handles=None, ncol=2, **kw):
    """Legend under the plot, inside the figure's fixed size (the plot area shrinks to make room)."""
    if handles is None:
        handles, labels = ax.get_legend_handles_labels()
        if kw.pop("reverse", False):
            handles, labels = handles[::-1], labels[::-1]
        ax.figure.legend(handles, labels, loc="outside lower center", ncol=ncol, columnspacing=1.2, **kw)
    else:
        ax.figure.legend(handles=handles, loc="outside lower center", ncol=ncol, columnspacing=1.2, **kw)


def _dofs_ticks(ax):
    lo, hi = ax.get_xlim()
    ticks = [(3e4, "$3{\\times}10^4$"), (1e5, "$10^5$"), (3e5, "$3{\\times}10^5$"), (1e6, "$10^6$")]
    ticks = [(t, l) for t, l in ticks if lo <= t <= hi * 1.0001]
    ax.set_xticks([t for t, _ in ticks])
    ax.set_xticklabels([l for _, l in ticks])
    ax.set_xlim(lo, hi)


def _style_legend(ax, extra=None, **kw):
    """Legend: solvers by colour/marker, gradient method by line style (one entry each)."""
    handles = [Line2D([], [], color=SOLVER_COLOR[s], marker=SOLVER_MARKER[s], ls="-", label=SOLVER_LABEL[s])
               for s in SOLVERS]
    handles += [Line2D([], [], color="0.3", ls="-", label="AD"), Line2D([], [], color="0.3", ls="--", label="analytical")]
    ax.legend(handles=handles + (extra or []), **kw)


# --------------------------------------------------------------------------------------- scaling data (10 it.)

def scaling_rows():
    rows = [r for r in _rows(os.path.join(SCALING, "solver_scaling_summary.csv"))
            if r["problem"] == PROBLEM and r["solver"] in SOLVERS]
    for r in rows:
        r["dofs_f"] = _f(r["dofs"])
    return rows


def series(rows, solver, grad, key, ok_only=True):
    s = sorted((r for r in rows if r["solver"] == solver and r["gradient"] == grad
                and (not ok_only or r["status"] in ("OK", "LOW_ACCURACY", "TIME_LIMIT")) and _f(r[key]) is not None),
               key=lambda r: r["dofs_f"])
    return [r["dofs_f"] for r in s], [_f(r[key]) for r in s]


def failures(rows):
    return [r for r in rows if r["status"] in ("MEMORY_CAP", "OUT_OF_MEMORY", "TOO_SLOW", "INACCURATE_SOLVE", "ERROR",
                                                 "CRASHED")]


def _slope_column(ax, items, x_col, min_gap_pt=8.0, header=None):
    """Curve labels in one column at x = x_col, each joined to its curve's last point by a thin black line.
    items: (x_end, y_end, text, colour). Labels that would overlap are pushed apart vertically; `header` is written
    above the column."""
    fig = ax.figure
    fig.canvas.draw()
    to_pt = 72.0 / fig.dpi
    pos = sorted(([ax.transData.transform((x, y))[1] * to_pt, x, y, k, c] for x, y, k, c in items),
                 key=lambda p: p[0])
    for _ in range(100):                                 # simple repulsion until neighbours are min_gap apart
        moved = False
        for a, b in zip(pos, pos[1:]):
            gap = b[0] - a[0]
            if gap < min_gap_pt - 1e-6:
                shift = (min_gap_pt - gap) / 2
                a[0] -= shift
                b[0] += shift
                moved = True
        if not moved:
            break
    inv = ax.transData.inverted()
    x_col_disp = ax.transData.transform((x_col, 1.0))[0]
    for y_pt, x, y, text, colour in pos:
        y_lab = inv.transform((x_col_disp, y_pt / to_pt))[1]
        ax.annotate(text, (x, y), xytext=(x_col, y_lab), textcoords="data", ha="left", va="center",
                    fontsize=6.5, fontweight="bold", color=colour, annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color="k", lw=0.5, shrinkA=4, shrinkB=1.5,
                                    relpos=(0.0, 0.5)))             # line ends at the label's left edge
    if header:                                           # just inside the top of the plot, above the column
        ax.text(x_col, 0.965, header, transform=matplotlib.transforms.blended_transform_factory(ax.transData,
                ax.transAxes), ha="left", va="top", fontsize=6.3, color="0.3", style="italic")


def fig_time_per_iteration(rows):
    """Each curve is labelled at its end, in one column: solver, gradient method and the fitted log-log slope, e.g.
    "PARDISO-AD (1.75)" (colour = solver; solid/filled = AD, dashed/open = analytical). No separate legend."""
    short = {"PARDISO": "PARDISO", "DPCG": "DPCG", "DPCG_GPU": "DPCG-GPU"}
    fig, ax = _new(height=2.7)
    ends = []
    for g in ("AUTODIFF", "ANALYTICAL"):
        for s in SOLVERS:
            x, y = series(rows, s, g, "s_per_iter")
            if not x:
                continue
            _line(ax, x, y, s, g, None)
            k = _slope(x, y)
            ends.append((x[-1], y[-1], f"{short[s]}-{'AD' if g == 'AUTODIFF' else 'An'}"
                                       + (f" ({k:.2f})" if k is not None else ""), SOLVER_COLOR[s]))
    _dofs_axis(ax)
    ax.set_yscale("log")
    ax.set_xlim(2.3e4, 1.6e7)                            # room for the label column
    ax.set_xticks([1e5, 1e6])
    ax.set_ylim(0.5, 1500)                               # headroom for the column header
    ax.set_ylabel("Time per MMA iteration (s)")
    _slope_column(ax, ends, x_col=1.6e6, header="solver-gradient (slope)")
    _save(fig, "fig_scaling_time_per_iteration_vs_dofs", [SCALING],
          "Median of iterations 2-11 per run. Labels: solver-gradient method (AD = automatic differentiation, An = "
          "analytical) and, in brackets, the log-log slope of the curve fitted on points >= 1e5 dofs. Solid/filled: AD; "
          "dashed/open: analytical.")


def fig_peak_memory(rows):
    fig, ax = _new()
    for s in SOLVERS:
        for g in ("AUTODIFF", "ANALYTICAL"):
            x, y = series(rows, s, g, "peak_rss_gb")
            if x:
                _line(ax, x, y, s, g, None)
    ax.axhline(MACHINE_RAM_GB, color="k", lw=0.8)
    ax.text(3e4, MACHINE_RAM_GB * 1.08, f"machine memory ({MACHINE_RAM_GB:.0f} GB)", fontsize=6.5, va="bottom")
    for r in failures(rows):
        cap = _f(r["mem_cap_gb"]) or 56.0
        ax.plot(r["dofs_f"], cap, marker="x", color="k", ms=6, mew=1.2, ls="none")
        ax.annotate(f"PARDISO AD: cap\n({cap:.0f} GB) reached", (r["dofs_f"], cap), xytext=(-5, 0),
                    textcoords="offset points", ha="right", va="center", fontsize=6.5)
    _dofs_axis(ax)
    ax.set_yscale("log")
    ax.set_ylim(0.8, 160)
    ax.set_ylabel("Peak resident memory (GB)")
    _style_legend(ax, loc="lower right")
    _save(fig, "fig_scaling_peak_memory_vs_dofs", [SCALING],
          "Kernel peak resident memory of the whole process over the run.")


def fig_ad_graph(rows):
    """AD graph memory vs elements: measured points, linear fit, the element matrices alone (lower bound) and the
    machine memory; the size at which the graph alone would fill the machine is a marker, explained in the legend."""
    fig, ax = _new(height=3.0)
    pts = {}
    for s in SOLVERS:                                  # the graph does not depend on the solver
        for r in rows:
            if r["solver"] == s and r["gradient"] == "AUTODIFF" and _f(r["ad_graph_gb"]):
                pts[_f(r["elements"])] = _f(r["ad_graph_gb"])
    e = np.array(sorted(pts))
    gb = np.array([pts[k] for k in e])
    per_elem = np.polyfit(e, gb * GIB, 1)[0]
    lim_e = MACHINE_RAM_GB * GIB / per_elem
    ee = np.logspace(np.log10(e.min() * 0.8), np.log10(lim_e * 1.4), 50)
    ax.axhline(MACHINE_RAM_GB, color="0.55", lw=2.2, alpha=0.5, solid_capstyle="butt",
               label=f"machine memory ({MACHINE_RAM_GB:.0f} GB)")
    ax.plot(ee, 24 * 24 * 8 * ee / GIB, color="#56B4E9", lw=1.1, ls=(0, (1.2, 1.6)),
            label="element matrices only")
    ax.plot(ee, per_elem * ee / GIB, color="k", lw=1.0, label=f"fit: {per_elem / 1024:.1f} KB/element")
    ax.plot(e, gb, color="k", marker="o", ls="none", ms=4, label="measured AD graph")
    ax.plot(lim_e, MACHINE_RAM_GB, marker="D", ms=5.5, mfc="white", mec="#D55E00", mew=1.3, ls="none",
            label=f"machine full at {lim_e / 1e6:.1f}M elements")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(0.02, 300)
    ax.set_xlabel("Number of elements")
    ax.set_ylabel("AD graph memory per iteration (GB)")
    h, l = ax.get_legend_handles_labels()
    order = [3, 2, 1, 0, 4]                              # measured, fit, element matrices, machine, limit
    _legend_below(ax, handles=[h[i] for i in order], ncol=2)
    _save(fig, "fig_scaling_ad_graph_memory_vs_elements", [SCALING],
          f"Bytes of tensors saved by autograd in one iteration (compliance, EdgeCantilever); fit "
          f"{per_elem:.0f} B/element; the element matrices alone would be 4608 B/element.")


def fig_ad_time_ratio(rows):
    fig, ax = _new()
    for s in SOLVERS:
        xa, ya = series(rows, s, "AUTODIFF", "s_per_iter")
        xn, yn = series(rows, s, "ANALYTICAL", "s_per_iter")
        common = sorted(set(xa) & set(xn))
        if common:
            ax.plot(common, [ya[xa.index(x)] / yn[xn.index(x)] for x in common], color=SOLVER_COLOR[s],
                    marker=SOLVER_MARKER[s], label=SOLVER_LABEL[s])
    ax.axhline(1.0, color="k", lw=0.8)
    _dofs_axis(ax)
    ax.set_ylim(0.9, 2.0)
    ax.set_ylabel("Time per iteration, AD / analytical")
    ax.legend(loc="upper left")
    _save(fig, "fig_scaling_ad_to_analytical_time_ratio_vs_dofs", [SCALING],
          "Compliance: the analytical gradient needs no adjoint solve; AD runs the backward pass and one adjoint solve.")


def fig_ad_memory_overhead(rows):
    fig, ax = _new()
    for s in SOLVERS:
        xa, ya = series(rows, s, "AUTODIFF", "peak_rss_gb")
        xn, yn = series(rows, s, "ANALYTICAL", "peak_rss_gb")
        common = sorted(set(xa) & set(xn))
        if common:
            ax.plot(common, [ya[xa.index(x)] - yn[xn.index(x)] for x in common], color=SOLVER_COLOR[s],
                    marker=SOLVER_MARKER[s], label=SOLVER_LABEL[s])
    x, y = series(rows, "DPCG", "AUTODIFF", "ad_graph_gb")
    ax.plot(x, y, color="k", ls=":", marker="", label="AD graph (saved tensors)")
    _dofs_axis(ax)
    ax.set_yscale("log")
    ax.set_ylabel("Peak memory, AD − analytical (GB)")
    ax.legend(loc="upper left")
    _save(fig, "fig_scaling_ad_memory_overhead_vs_dofs", [SCALING],
          "Difference of peak memory between AD and analytical runs with the same solver and mesh.")


def _phases(r):
    fe, fac, fwd = _f(r["fe_solve_s"]) or 0, _f(r["factor_s"]) or 0, _f(r["fwd_solve_s"]) or 0
    grad, adj, mma = _f(r["grad_s"]) or 0, _f(r["adjoint_s"]) or 0, _f(r["mma_update_s"]) or 0
    total = _f(r["s_per_iter"]) or 0
    p = {"assembly": max(fe - fac - fwd, 0), "factorization / set-up": fac, "forward solve": fwd,
         "backward pass": max(grad - adj, 0), "adjoint solve": adj}
    p["MMA update and other"] = max(total - sum(p.values()), mma)
    return p


def fig_time_breakdown(rows, solver):
    s = sorted((r for r in rows if r["solver"] == solver and r["gradient"] == "AUTODIFF"
                and r["status"] in ("OK", "LOW_ACCURACY")), key=lambda r: r["dofs_f"])
    if not s:
        return
    x = [r["dofs_f"] for r in s]
    ph = [_phases(r) for r in s]
    names = list(PHASE_COLOR)
    frac = np.array([[p[n] / sum(p.values()) for p in ph] for n in names]) * 100
    fig, ax = _new(height=3.3)
    ax.stackplot(x, frac, colors=[PHASE_COLOR[n] for n in names], labels=names, lw=0.3, edgecolor="white")
    _dofs_axis(ax)
    ax.set_xlim(min(x), max(x))
    ax.set_ylim(0, 100)
    ax.set_ylabel("Share of iteration time, AD (%)")
    _dofs_ticks(ax)
    _legend_below(ax, ncol=2, reverse=True)
    ax.grid(False)
    _save(fig, f"fig_scaling_{solver.lower()}_ad_time_breakdown_vs_dofs", [SCALING],
          f"{SOLVER_LABEL[solver]}, automatic differentiation: medians of iterations 2-11. Backward pass = autograd "
          "through assembly and objective, without the adjoint solve.")


def _factor_memory(rows):
    """PARDISO factor memory per size. The 10-iteration study recorded values above 2 GiB modulo 4 GiB (int32
    overflow, fixed later): exact values are taken from the full runs where they exist; otherwise the value is
    reconstructed only when exactly one candidate (recorded + k*4 GiB) lies between the previous size's factors and
    the analytical run's peak memory minus its set-up memory. Ambiguous sizes are left out."""
    exact = {}
    for d in FULL:
        p = os.path.join(d, "solver_scaling_summary.csv")
        if os.path.exists(p):
            for r in _rows(p):
                if r["solver"] == "PARDISO" and _f(r["factor_mem_gb"]):
                    exact[_f(r["dofs"])] = _f(r["factor_mem_gb"])
    out, prev = {}, 0.0
    for r in sorted((r for r in rows if r["solver"] == "PARDISO" and r["gradient"] == "ANALYTICAL"),
                    key=lambda r: r["dofs_f"]):
        rec, d = _f(r["factor_mem_gb"]), r["dofs_f"]
        if d in exact:
            out[d] = (exact[d], "exact")
        elif rec is not None:
            upper = (_f(r["peak_rss_gb"]) or 0) - (_f(r["rss_setup_gb"]) or 0)
            cand = [rec + 4 * k for k in range(0, 64) if prev < rec + 4 * k <= upper]
            if len(cand) == 1:
                out[d] = (cand[0], "exact" if cand[0] == rec else "reconstructed")
        if d in out:
            prev = out[d][0]
    return out


def fig_pardiso_memory_composition(rows):
    fac = _factor_memory(rows)
    xs = sorted(fac)
    fig, ax = _new(height=3.3)
    x, y = series(rows, "PARDISO", "AUTODIFF", "peak_rss_gb")
    _line(ax, x, y, "PARDISO", "AUTODIFF", "peak memory, AD")
    x, y = series(rows, "PARDISO", "ANALYTICAL", "peak_rss_gb")
    _line(ax, x, y, "PARDISO", "ANALYTICAL", "peak memory, analytical")
    ax.plot(xs, [fac[k][0] for k in xs], color="k", marker="D", ms=3, label="PARDISO factors")
    recon = [k for k in xs if fac[k][1] == "reconstructed"]
    if recon:
        ax.plot(recon, [fac[k][0] for k in recon], ls="none", marker="D", ms=5, mfc="white", mec="k",
                label="factors (reconstructed)")
    x, y = series(rows, "PARDISO", "AUTODIFF", "ad_graph_gb")
    ax.plot(x, y, color="#E69F00", marker="v", label="AD graph")
    x, y = series(rows, "PARDISO", "AUTODIFF", "rss_setup_gb")
    ax.plot(x, y, color="0.55", marker=".", label="after set-up (mesh, solver)")
    ax.axhline(MACHINE_RAM_GB, color="k", lw=0.8, ls=":")
    _dofs_axis(ax)
    ax.set_yscale("log")
    ax.set_ylabel("Memory (GB)")
    _legend_below(ax)
    _save(fig, "fig_scaling_pardiso_memory_composition_vs_dofs", [SCALING] + FULL,
          "Factor memory from PARDISO iparm(15-17). Values recorded above 2 GiB in the 10-iteration study wrapped "
          "around (int32); 540k is exact from the full run, 217k/273k reconstructed as the unique consistent value, "
          "1.06M omitted (ambiguous).")


def fig_cg_iterations(rows):
    fig, ax = _new()
    for key, ls, lab in (("cg_iters_fwd", "-", "forward solve"), ("cg_iters_adj", "--", "adjoint solve (AD)")):
        x, y = series(rows, "DPCG", "AUTODIFF", key)
        ax.plot(x, y, color=SOLVER_COLOR["DPCG"], ls=ls, marker="s", mfc=SOLVER_COLOR["DPCG"] if ls == "-" else "white",
                label=lab)
    _dofs_axis(ax)
    ax.set_ylim(0, 140)
    ax.set_ylabel("DPCG iterations per solve")
    ax.legend(loc="lower right")
    _save(fig, "fig_scaling_dpcg_cg_iterations_vs_dofs", [SCALING],
          "Median over iterations 2-11, relative residual 1e-8; CPU and GPU DPCG run identical iterations.")


def fig_gradient_agreement():
    rows = [r for r in _rows(os.path.join(SCALING, "solver_scaling_checks.csv"))
            if r["problem"] == PROBLEM and r["solver"] in SOLVERS and r["reference"].startswith("PARDISO")]
    fig, ax = _new()
    for s in SOLVERS:
        for g in ("AUTODIFF", "ANALYTICAL"):
            sel = sorted((r for r in rows if r["solver"] == s and r["gradient"] == g
                          and _f(r["grad_probe_iter1_rel_diff"]) is not None), key=lambda r: _f(r["dofs"]))
            if sel:
                _line(ax, [_f(r["dofs"]) for r in sel],
                      [max(_f(r["grad_probe_iter1_rel_diff"]), 1e-16) for r in sel], s, g, None)
    ax.axhline(1e-8, color="0.5", lw=0.7, ls=":")
    ax.text(3e4, 1.3e-8, "linear-solver tolerance", fontsize=6.5, color="0.4")
    _dofs_axis(ax)
    ax.set_yscale("log")
    ax.set_ylim(1e-15, 1e-5)
    ax.set_ylabel("Gradient difference to PARDISO")
    _style_legend(ax, loc="upper left", ncol=2)
    _save(fig, "fig_scaling_gradient_agreement_vs_dofs", [SCALING],
          "Relative difference of the first-iteration gradient (projection on a fixed random vector) to the PARDISO "
          "AD run of the same mesh; all runs "
          "start from the same design. CPU and GPU DPCG coincide.")


# ------------------------------------------------------------------------------------- full optimizations

def full_runs():
    out = []
    for d in FULL:
        p = os.path.join(d, "solver_scaling_summary.csv")
        if not os.path.exists(p):
            continue
        for r in _rows(p):
            it = os.path.join(d, r["run_dir"], "iterations.csv")
            if r["status"] == "OK" and os.path.exists(it) and r["solver"] in SOLVERS and _f(r["dofs"]) > 2e5:
                out.append((r, _rows(it)))
    return out


SIZE_STYLE = {540792: ("-", "540k dofs"), 1058400: ("--", "1.06M dofs")}


def _full_label(r):
    ls, size = SIZE_STYLE.get(int(_f(r["dofs"])), (":", r["dofs"]))
    return ls, f"{SOLVER_LABEL[r['solver']]}, {size}"


def fig_full_objective(runs):
    fig, ax = _new()
    for r, its in runs:
        ls, lab = _full_label(r)
        k = [int(i["k"]) for i in its]
        ax.plot(k, [_f(i["objective"]) for i in its], color=SOLVER_COLOR[r["solver"]], ls=ls, lw=0.9, label=lab)
    ax.set_yscale("log")
    ax.set_xlabel("MMA iteration")
    ax.set_ylabel("Compliance (J)")
    ax.legend(loc="upper right")
    _save(fig, "fig_full_objective_history", FULL,
          "Full MMA optimizations with AD; the three solvers at 540k dofs follow the same path (final objectives agree "
          "to ~1e-10, see solver_scaling_checks.csv).")


def _place_line_labels(ax, lines, fontsize=6.2, gap_pt=3.0, clear_pt=2.5):
    """Label each curve along itself (rotated to its slope on screen, just above or below it) at a position where
    the label touches no other curve and stays inside the plot. Candidates run along the curve; positions further
    from the origin are preferred, and among clear candidates the one with the most room wins.
    lines: list of (x, y, text, colour)."""
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    to_pt = 72.0 / fig.dpi
    disp = [ax.transData.transform(np.column_stack([np.asarray(x, float), np.asarray(y, float)])) * to_pt
            for x, y, _t, _c in lines]
    box = ax.get_window_extent(renderer)
    x0, y0, x1, y1 = (v * to_pt for v in (box.x0, box.y0, box.x1, box.y1))

    def clearance(pts, others):
        best = np.inf
        for P in others:                                     # distance of the label's points to a polyline
            a, b = P[:-1], P[1:]
            ab = b - a
            L2 = np.maximum((ab ** 2).sum(1), 1e-12)
            for q in pts:
                t = np.clip(((q - a) * ab).sum(1) / L2, 0, 1)
                best = min(best, np.sqrt((((a + t[:, None] * ab) - q) ** 2).sum(1)).min())
        return best

    for n, (x, y, text, colour) in enumerate(lines):
        probe = ax.text(0, 0, text, fontsize=fontsize, fontweight="bold")
        ext = probe.get_window_extent(renderer)
        probe.remove()
        w, h = ext.width * to_pt, ext.height * to_pt
        P = disp[n]
        others = [d for m, d in enumerate(disp) if m != n]
        best = None
        for at in np.linspace(0.85, 0.25, 25):              # far from the origin first
            i = int(np.clip(at * (len(P) - 1), 1, len(P) - 2))
            j0, j1 = max(i - 8, 0), min(i + 8, len(P) - 1)
            d = P[j1] - P[j0]
            u = d / max(np.hypot(*d), 1e-9)                  # along the curve
            nrm = np.array([-u[1], u[0]])                    # normal, pointing "up"
            for side in (-1, +1):
                centre = P[i] + side * (gap_pt + h / 2) * nrm
                corners = [centre + sx * w / 2 * u + sy * h / 2 * nrm for sx in (-1, 1) for sy in (-1, 1)]
                if any(not (x0 + 2 <= c[0] <= x1 - 2 and y0 + 2 <= c[1] <= y1 - 2) for c in corners):
                    continue
                pts = [centre + sx * w / 2 * u + sy * h / 2 * nrm for sx in np.linspace(-1, 1, 9)
                       for sy in (-1, 0, 1)]
                room = clearance(pts, others)
                if room >= clear_pt:
                    score = room + 30 * at                   # room matters, but prefer far from the origin
                    if best is None or score > best[0]:
                        best = (score, i, side, np.degrees(np.arctan2(u[1], u[0])))
            if best is not None and best[0] - 30 * at > 12:  # clearly free spot found far out: take it
                break
        if best is None:                                     # no clear spot: below the curve near its end
            i = int(0.8 * (len(P) - 1))
            d = P[min(i + 8, len(P) - 1)] - P[i - 8]
            best = (0, i, -1, np.degrees(np.arctan2(d[1], d[0])))
        _score, i, side, angle = best
        nx, ny = -np.sin(np.radians(angle)), np.cos(np.radians(angle))
        ax.annotate(text, (np.asarray(x)[i], np.asarray(y)[i]), xytext=(side * gap_pt * nx, side * gap_pt * ny),
                    textcoords="offset points", rotation=angle, rotation_mode="anchor", ha="center",
                    va="top" if side < 0 else "bottom", fontsize=fontsize, color=colour, fontweight="bold")


def fig_full_cumulative_time(runs):
    """Cumulative wall time of the full optimizations; each line is labelled along itself (no legend), at a place
    where the label touches no other line."""
    fig, ax = _new()
    lines = []
    for r, its in runs:
        ls, lab = _full_label(r)
        k = [int(i["k"]) for i in its]
        t = np.cumsum([_f(i["iter_s"]) or 0 for i in its]) / 3600
        ax.plot(k, t, color=SOLVER_COLOR[r["solver"]], ls=ls)
        lines.append((k, t, lab, SOLVER_COLOR[r["solver"]]))
    ax.set_xlabel("MMA iteration")
    ax.set_ylabel("Cumulative wall time (h)")
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    _place_line_labels(ax, lines)
    _save(fig, "fig_full_cumulative_time", FULL,
          "Sum of iteration wall times (excludes set-up and final evaluation). Solid: 540k dofs; dashed: 1.06M dofs.")


def fig_full_iteration_time(runs):
    fig, ax = _new(height=3.3)
    for r, its in runs:
        ls, lab = _full_label(r)
        ax.plot([int(i["k"]) for i in its], [_f(i["iter_s"]) for i in its], color=SOLVER_COLOR[r["solver"]], ls=ls,
                lw=0.8, label=lab)
    ax.set_ylim(0, None)
    ax.set_xlabel("MMA iteration")
    ax.set_ylabel("Time per iteration (s)")
    _legend_below(ax)
    _save(fig, "fig_full_time_per_iteration_history", FULL, "Wall time of each MMA iteration, AD.")


def fig_full_cg_history(runs):
    fig, ax = _new()
    for r, its in runs:
        if r["solver"] != "DPCG":                      # GPU runs the same iterations
            continue
        ls, lab = _full_label(r)
        k = [int(i["k"]) for i in its]
        ax.plot(k, [_f(i["cg_fwd"]) for i in its], color=SOLVER_COLOR["DPCG"], ls=ls, lw=0.9,
                label=f"forward, {lab.split(', ')[1]}")
        ax.plot(k, [_f(i["cg_adj"]) for i in its], color="#E69F00", ls=ls, lw=0.9,
                label=f"adjoint (AD), {lab.split(', ')[1]}")
    ax.set_xlabel("MMA iteration")
    ax.set_ylabel("DPCG iterations per solve")
    ax.set_ylim(0, 160)
    ax.legend(loc="lower right")
    _save(fig, "fig_full_dpcg_iterations_history", FULL,
          "CG iterations of the forward and adjoint solve in each MMA iteration (DPCG, relative residual 1e-8).")


# ----------------------------------------------------------------------------------- AD vs analytical (sweep)

def _problem_classes():
    cls = {}
    for r in _rows(PROBLEM_TABLE):
        phys, obj = r["physics"], r["objective"].upper()
        if phys == "thermal":
            c = "thermal compliance"
        elif phys == "thermo-structural":
            c = "thermo-structural compliance"
        elif obj.startswith("GVECTOR"):
            c = "compliant mechanism"
        elif obj.startswith("COMPLIANCE"):
            c = "structural compliance"
        else:
            c = "stress / volume / mass"
        cls[(r["category"], r["problem"])] = c
    return cls


def _true_dofs(category, ndof):
    """The sweep CSVs record 3 x nodes for every physics: thermal has 1 dof per node, thermo-structural 3 + 1."""
    nd = _f(ndof)
    if nd is None:
        return None
    if category == "ThermoStructural":
        return nd * 4 / 3
    if category.startswith("Thermal"):
        return nd / 3
    return nd


def _class_of(cls, category, problem):
    return cls.get((category, problem)) or next((v for (c, p), v in cls.items() if p == problem), "structural compliance")


def _pooled_times(method, dirs, key="s_per_iter"):
    """{(problem, gradient): [key of every OK run, in session and repeat order]} over several uninstrumented
    sessions."""
    out = {}
    for d in dirs:
        rows = sorted((x for x in _rows(os.path.join(d, "runs.csv")) if x["method"] == method and x["status"] == "OK"
                       and _f(x[key]) is not None), key=lambda x: int(x["repeat"]))
        for x in rows:
            out.setdefault((x["problem"], x["gradient"]), []).append(_f(x[key]))
    return out


def gc_pairs(method, time_dirs=None):
    """AD vs analytical pairs of one method ("MMA" or "OC") as dicts with: problem, category, label, dofs, an, ad
    (s per iteration), an_lo/an_hi/ad_lo/ad_hi (min-max over repeats), extra (%), rel_diff, and, for the
    per-iteration comparison, the phase times and memory. Uses the per-iteration comparison when it has at least
    NEW_MIN_PROBLEMS pairs for the method, otherwise the old totals (time / FE solves). time_dirs: take the times
    (median and min-max of every run) from these uninstrumented sessions instead. Returns (pairs, source)."""
    new = os.path.join(GRADCOMP_NEW, "pairs.csv")
    pairs = [r for r in _rows(new) if r["method"] == method] if os.path.exists(new) else []
    if len(pairs) >= NEW_MIN_PROBLEMS:
        runs = _rows(os.path.join(GRADCOMP_NEW, "runs.csv"))
        out = []
        for r in pairs:
            per = {g: [_f(x["s_per_iter"]) for x in runs if x["method"] == method and x["problem"] == r["problem"]
                       and x["gradient"] == g and x["status"] == "OK" and _f(x["s_per_iter"])]
                   for g in ("AUTODIFF", "ANALYTICAL")}
            d = {k: _f(v) for k, v in r.items() if _f(v) is not None}
            d.update(problem=r["problem"], category=r["category"], dofs=_f(r["dofs"]), elements=_f(r["elements"]),
                     an=_f(r["s_per_iter_an"]), ad=_f(r["s_per_iter_ad"]), extra=_f(r["ad_extra_pct"]),
                     rel_diff=_f(r["objective_rel_diff"]),
                     ad_lo=min(per["AUTODIFF"]), ad_hi=max(per["AUTODIFF"]),
                     an_lo=min(per["ANALYTICAL"]), an_hi=max(per["ANALYTICAL"]), new=True)
            out.append(d)
        src = GRADCOMP_NEW
        mem_file = os.path.join(GRADCOMP_MEMORY[method], "pairs.csv")
        if os.path.exists(mem_file):
            mem = {r["problem"]: r for r in _rows(mem_file) if r["method"] == method}
            for d in out:
                m = mem.get(d["problem"])
                if m is None:
                    print(f"  warning: no post-fix memory for {method} {d['problem']}; pre-fix values kept")
                    continue
                d.update({k: _f(m[k]) for k in MEMORY_FIELDS if _f(m.get(k)) is not None})
            src = f"{GRADCOMP_NEW} (time); {GRADCOMP_MEMORY[method]} (memory)"
        if time_dirs:
            pooled = _pooled_times(method, time_dirs)
            for d in out:
                ad, an = pooled[(d["problem"], "AUTODIFF")], pooled[(d["problem"], "ANALYTICAL")]
                d.update(ad=float(np.median(ad)), an=float(np.median(an)), ad_lo=min(ad), ad_hi=max(ad),
                         an_lo=min(an), an_hi=max(an), n_runs=min(len(ad), len(an)))
                d["extra"] = 100 * (d["ad"] / d["an"] - 1)
            final = _pooled_times(method, time_dirs, "objective_final")    # AD and analytical run k: same session, repeat
            for d in out:
                ad, an = final[(d["problem"], "AUTODIFF")], final[(d["problem"], "ANALYTICAL")]
                d["rel_diff"] = float(np.median([abs(a - b) / abs(b) for a, b in zip(ad, an)]))
            src = "; ".join(time_dirs) + " (time, final objective)"
    else:
        old_method = METHODS_OLD[method]
        out = []
        for r in _rows(GRADCOMP):
            if r["method"] != old_method or not (_f(r["s_per_fea_ad"]) and _f(r["s_per_fea_an"])):
                continue
            ad, an = _f(r["s_per_fea_ad"]), _f(r["s_per_fea_an"])
            out.append(dict(problem=r["problem"], category=r["category"], dofs=_true_dofs(r["category"], r["ndof"]),
                            an=an, ad=ad, ad_lo=ad, ad_hi=ad, an_lo=an, an_hi=an, extra=100 * (ad / an - 1),
                            rel_diff=_f(r["objective_rel_diff"]), new=False))
        src = GRADCOMP
    for d in out:
        d["label"] = d["problem"] + (" (thermo-str.)" if d["category"] == "ThermoStructural" and d["problem"] == "MBBBeam"
                                     else "")
    return out, src


METHODS_OLD = {"MMA": "DENSITYMMA", "OC": "DENSITYOCM"}


def fig_ad_extra_time_per_iteration(method):
    """Dumbbell plot per problem: median time per iteration with analytical sensitivities (open circle) and with
    AD (filled), log axis; AD's extra time in % in a column on the right (repeat ranges are in the manifest note). Both modes
    run the same optimization path and the same FE solve, so the gap is the cost of the AD gradient."""
    cls = _problem_classes()
    rows, src = gc_pairs(method, time_dirs=GRADCOMP_TIME)
    if not rows:
        return
    rows.sort(key=lambda r: r["extra"])
    fig, ax = _new(height=0.15 * len(rows) + 1.2)
    for i, r in enumerate(rows):
        c = CLASS_COLOR[_class_of(cls, r["category"], r["problem"])]
        ax.plot([r["an"], r["ad"]], [i, i], color=c, lw=1.6, alpha=0.45, solid_capstyle="butt", zorder=1)
        ax.plot(r["an"], i, "o", ms=4, mfc="white", mec="k", mew=0.8, zorder=3)
        ax.plot(r["ad"], i, "o", ms=4, color=c, mec="k", mew=0.4, zorder=4)
        ax.text(1.02, i, f"{r['extra']:+.0f}%", transform=matplotlib.transforms.blended_transform_factory(
            ax.transAxes, ax.transData), ha="left", va="center", fontsize=6.5, fontweight="bold",
            color="k" if r["extra"] >= 0 else "0.45")
    ax.text(1.02, len(rows) - 0.1, "AD extra", transform=matplotlib.transforms.blended_transform_factory(
        ax.transAxes, ax.transData), ha="left", va="bottom", fontsize=6.3, color="0.25")
    ax.set_xscale("log")
    lo = min(r["an_lo"] for r in rows)
    hi = max(r["ad_hi"] for r in rows)
    ax.set_xticks([t for t in (0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50) if lo / 1.6 <= t <= hi * 1.6])
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels([r["label"] for r in rows], fontsize=6.5)
    ax.tick_params(axis="y", which="both", length=0, right=False)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlabel(f"Time per iteration (s), {method}")
    ax.grid(True, axis="x", which="major")
    ax.grid(False, axis="y")
    present = list(dict.fromkeys(_class_of(cls, r["category"], r["problem"]) for r in rows))
    handles = [Line2D([], [], marker="o", ls="none", mfc="white", mec="k", label="analytical sensitivities"),
               Line2D([], [], marker="o", ls="none", color="0.4", mec="k", mew=0.4, label="AD (colour: problem type)")]
    handles += [Line2D([], [], color=CLASS_COLOR[c], lw=1.6, alpha=0.6, label=c) for c in present]
    _legend_below(ax, handles=handles, ncol=2)
    wide = sorted(((r["label"], g, lo, hi, med) for r in rows
                   for g, lo, hi, med in (("AD", r["ad_lo"], r["ad_hi"], r["ad"]), ("analytical", r["an_lo"], r["an_hi"], r["an"]))
                   if med and (hi - lo) / med > 0.1), key=lambda t: -(t[3] - t[2]) / t[4])
    spread = "; ".join(f"{p} {g}: {lo:.3g}-{hi:.3g} s" for p, g, lo, hi, _m in wide) or "all within 10%"
    note = (f"Median time per iteration (iterations 2 and later) over {min(r.get('n_runs', 3) for r in rows)} runs "
            "(two uninstrumented sessions, 3 repeats each). Right column: extra time of AD in % of "
            f"the analytical time. Repeat ranges wider than 10% of the median: {spread}.") if rows[0]["new"] else \
        ("Time per iteration = optimizer time / FE solves (2026-10-15 totals; these timings overlapped with other "
         "jobs, values within about +-10% are within machine noise).")
    _save(fig, f"fig_ad_vs_analytical_time_per_iteration_{method.lower()}", [src, PROBLEM_TABLE], note)


def fig_ad_extra_time_breakdown(method):
    """Where AD's extra time per iteration goes, per problem (per-iteration comparison only): the AD backward pass
    (minus the analytical formula's time) and the adjoint solves (minus the analytical adjoint, for p-norm stress),
    as % of the analytical iteration time; the dot is the measured total extra (it also contains the small
    differences in FE solve and optimizer update between the two runs)."""
    cls = _problem_classes()
    rows, src = gc_pairs(method)
    if not rows or not rows[0]["new"]:
        return
    for r in rows:
        an = r["an"]
        r["bwd"] = 100 * max((r["grad_ad"] - r["adjoint_ad"]) - (r["grad_an"] - r.get("adjoint_an", 0.0)), 0) / an
        r["adj"] = 100 * max(r["adjoint_ad"] - r.get("adjoint_an", 0.0), 0) / an
    rows.sort(key=lambda r: r["extra"])
    fig, ax = _new(height=0.15 * len(rows) + 1.2)
    y = np.arange(len(rows))
    ax.barh(y, [r["bwd"] for r in rows], height=0.62, color="#E69F00", label="AD backward pass (excl. adjoint)")
    ax.barh(y, [r["adj"] for r in rows], left=[r["bwd"] for r in rows], height=0.62, color="#D55E00",
            label="adjoint solve (AD only)")
    ax.plot([r["extra"] for r in rows], y, "o", ms=3.5, mfc="white", mec="k", mew=0.8, ls="none",
            label="measured total extra time")
    ax.set_yticks(y)
    ax.set_yticklabels([r["label"] for r in rows], fontsize=6.5)
    ax.tick_params(axis="y", which="both", length=0, right=False)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlim(0, None)
    ax.set_xlabel(f"AD extra time per iteration (%), {method}")
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    _legend_below(ax, ncol=1)
    _save(fig, f"fig_ad_vs_analytical_extra_time_breakdown_{method.lower()}", [src],
          "Bars: AD's backward pass and adjoint solves beyond what the analytical gradient costs, in % of the "
          "analytical iteration time (medians over iterations and repeats). Dot: measured total extra time; the "
          "difference is noise in the FE solve and optimizer update, which are the same computation in both modes.")


def fig_ad_vs_analytical_peak_memory(method):
    """Peak memory of the whole run, AD (filled) vs analytical (open), against problem size, coloured by problem
    type (per-iteration comparison only)."""
    cls = _problem_classes()
    rows, src = gc_pairs(method)
    if not rows or not rows[0]["new"]:
        return
    fig, ax = _new(height=3.3)
    for r in rows:
        c = CLASS_COLOR[_class_of(cls, r["category"], r["problem"])]
        ax.plot([r["dofs"], r["dofs"]], [r["peak_rss_an_gb"], r["peak_rss_ad_gb"]], color=c, lw=0.8, alpha=0.6)
        ax.plot(r["dofs"], r["peak_rss_an_gb"], "o", ms=3.5, mfc="white", mec=c)
        ax.plot(r["dofs"], r["peak_rss_ad_gb"], "o", ms=3.5, color=c)
    _dofs_axis(ax)
    ax.set_yscale("log")
    ax.set_yticks([0.5, 1, 2, 5, 10, 20])
    ax.set_ylim(0.5, 12)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_ylabel(f"Peak memory of the run (GB), {method}")
    present = list(dict.fromkeys(_class_of(cls, r["category"], r["problem"]) for r in rows))
    handles = [Line2D([], [], marker="o", ls="none", color="0.3", label="AD"),
               Line2D([], [], marker="o", ls="none", mfc="white", mec="0.3", label="analytical")]
    handles += [Line2D([], [], marker="o", ls="none", color=CLASS_COLOR[c], label=c) for c in present]
    _legend_below(ax, handles=handles, ncol=2)
    _save(fig, f"fig_ad_vs_analytical_peak_memory_vs_dofs_{method.lower()}", [src, PROBLEM_TABLE],
          "Kernel peak resident memory of each run (median over repeats); the vertical segment is AD's extra memory.")


def fig_ad_graph_per_element(method):
    """AD graph memory per element for every problem, against problem size (per-iteration comparison only):
    structural problems store more per element (24x24 element matrices) than thermal ones (8x8)."""
    cls = _problem_classes()
    rows, src = gc_pairs(method)
    if not rows or not rows[0]["new"]:
        return
    fig, ax = _new(height=3.1)
    for r in rows:
        if not r.get("ad_graph_gb") or not r.get("elements"):
            continue
        c = CLASS_COLOR[_class_of(cls, r["category"], r["problem"])]
        ax.plot(r["dofs"], r["ad_graph_gb"] * GIB / r["elements"] / 1024, "o", ms=3.8, color=c)
    _dofs_axis(ax)
    ax.set_ylim(0, None)
    ax.set_ylabel("AD graph per element (KB)")
    present = list(dict.fromkeys(_class_of(cls, r["category"], r["problem"]) for r in rows))
    _legend_below(ax, handles=[Line2D([], [], marker="o", ls="none", color=CLASS_COLOR[c], label=c)
                               for c in present], ncol=2)
    _save(fig, f"fig_ad_graph_per_element_by_problem_{method.lower()}", [src, PROBLEM_TABLE],
          "Bytes of tensors autograd saves per iteration, divided by the number of elements (median over iterations "
          "and repeats).")


def fig_iteration_time_ad_vs_analytical(rows, solver):
    """For one solver, at every mesh size: an AD bar and an analytical bar, both split into the phases of an
    iteration and scaled to the analytical iteration time (100%). The FE solve is the same in both; what AD adds
    (backward pass and adjoint solve) sticks out beyond 100%. Absolute time and AD's extra % on the right."""
    sizes = []
    for r in sorted((r for r in rows if r["solver"] == solver and r["gradient"] == "AUTODIFF"
                     and r["status"] in ("OK", "LOW_ACCURACY")), key=lambda r: r["dofs_f"]):
        an = next((q for q in rows if q["solver"] == solver and q["gradient"] == "ANALYTICAL"
                   and q["ndof_desired"] == r["ndof_desired"] and q["status"] in ("OK", "LOW_ACCURACY")), None)
        if an:
            sizes.append((r, an))
    if not sizes:
        return
    seg = [("FE solve (assembly + linear solve)", "#A6A6A6"), ("gradient: AD backward pass / analytical formula",
                                                                 "#E69F00"),
           ("adjoint solve (AD only)", "#D55E00"), ("MMA update and other", "#DDDDDD")]

    def parts(r):
        fe, grad, adj, tot = (_f(r[k]) or 0 for k in ("fe_solve_s", "grad_s", "adjoint_s", "s_per_iter"))
        return [fe, max(grad - adj, 0), adj, max(tot - fe - grad, 0)], tot

    fig, ax = _new(height=0.32 * len(sizes) + 1.4)
    ylabels, yt, y = [], [], 0.0
    for r, an in sizes:
        p_an, t_an = parts(an)
        p_ad, t_ad = parts(r)
        for bar, (ps, tot, tag) in enumerate(((p_ad, t_ad, "AD"), (p_an, t_an, "analytical"))):
            yy = y + (0.0 if bar == 0 else 0.42)
            left = 0.0
            for (name, col), v in zip(seg, ps):
                w = 100 * v / t_an
                ax.barh(yy, w, left=left, height=0.36, color=col, edgecolor="white", lw=0.3)
                left += w
            note = f"{tot:.3g} s" + (f"  (+{100 * (t_ad / t_an - 1):.0f}%)" if tag == "AD" else "")
            ax.text(left + 2, yy, note, va="center", ha="left", fontsize=6.2,
                    fontweight="bold" if tag == "AD" else "normal")
            yt.append(yy)
            ylabels.append(f"{_short_dofs(r['dofs_f'])}  {tag}" if tag == "AD" else tag)
        y += 1.1
    ax.axvline(100, color="k", lw=0.7, ls=(0, (2, 2)))
    ax.set_yticks(yt)
    ax.set_yticklabels(ylabels, fontsize=6.3)
    ax.tick_params(axis="y", length=0, right=False)
    ax.invert_yaxis()
    xmax = max(100 * parts(r)[1] / parts(an)[1] for r, an in sizes)
    ax.set_xlim(0, xmax * 1.45)
    ax.set_xlabel("Time per iteration (% of the analytical iteration)")
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    _legend_below(ax, handles=[matplotlib.patches.Patch(color=c, label=n) for n, c in seg], ncol=1)
    _save(fig, f"fig_scaling_{solver.lower()}_iteration_time_ad_vs_analytical", [SCALING],
          f"{SOLVER_LABEL[solver]}, EdgeCantilever: medians of iterations 2-11. Bars are scaled to the analytical "
          "iteration time at the same mesh; labels give the absolute time per iteration and AD's extra time.")


def _short_dofs(d):
    return f"{d / 1e6:.2f}M" if d >= 1e6 else f"{d / 1e3:.0f}k"


def _new_methods():
    """Methods whose per-iteration comparison is complete; the combined figures use only these (mixing them with
    the noisier 2026-10-15 totals would put data of different quality on one axis). Falls back to all methods
    with the old data when none is complete."""
    got = [m for m in ("MMA",) if gc_pairs(m)[0] and gc_pairs(m)[0][0]["new"]]     # OC removed from PyTO-AD
    return got or ["MMA"]


def fig_ratio_vs_dofs():
    cls = _problem_classes()
    fig, ax = _new(height=3.3)
    srcs, methods, present, shown = [], [], {}, []
    for method, filled in (("MMA", True),):                 # OC removed from PyTO-AD
        rows, src = gc_pairs(method)
        if not rows or method not in _new_methods():
            continue
        srcs.append(src)
        shown.append((method, filled))
        methods.append(f"{method}: {'per iteration, 3 repeats' if rows[0]['new'] else '2026-10-15 totals'}")
        for r in rows:
            k = _class_of(cls, r["category"], r["problem"])
            present[k] = True
            c = CLASS_COLOR[k]
            ax.plot(r["dofs"], r["ad"] / r["an"], "o" if filled else "s", color=c, mfc=c if filled else "white", ms=3.5)
    ax.axhline(1.0, color="k", lw=0.8)
    _dofs_axis(ax)
    ax.set_ylabel("Time per iteration, AD / analytical")
    handles = [Line2D([], [], marker="o", ls="none", color=CLASS_COLOR[c], label=c) for c in CLASS_COLOR if c in present]
    if len(shown) > 1:
        handles += [Line2D([], [], marker="o" if f else "s", ls="none", color="0.3", mfc="0.3" if f else "white",
                           label=m) for m, f in shown]
    _legend_below(ax, handles=handles, ncol=2)
    _save(fig, "fig_ad_vs_analytical_time_ratio_vs_dofs", srcs + [PROBLEM_TABLE],
          "Data: " + "; ".join(methods) + ". Thermo-structural dofs = structural + thermal.")


def fig_objective_agreement():
    fig, ax = _new(height=3.3)
    floor = 1e-16
    cls = _problem_classes()
    srcs, methods, present, shown = [], [], {}, []
    for method, filled in (("MMA", True),):                 # the paper's figure: MMA, the 25 pairs
        rows, src = gc_pairs(method, time_dirs=GRADCOMP_TIME)
        if not rows:
            continue
        srcs.append(src)
        shown.append((method, filled))
        methods.append(f"{method}: two uninstrumented sessions, 3 repeats each")
        for r in rows:
            if r["rel_diff"] is None:
                continue
            k = _class_of(cls, r["category"], r["problem"])
            present[k] = True
            ax.plot(r["dofs"], max(r["rel_diff"], floor), "o" if filled else "s", color=CLASS_COLOR[k],
                    mfc=CLASS_COLOR[k] if filled else "white", ms=3.5)
    ax.axhline(floor, color="0.6", lw=0.6, ls=":")
    _dofs_axis(ax)
    ax.set_yscale("log")
    ax.set_ylim(floor / 10, None)
    ax.set_ylabel("Final objective: relative difference")
    handles = [Line2D([], [], marker="o", ls="none", color=CLASS_COLOR[c], label=c) for c in CLASS_COLOR if c in present]
    handles.append(Line2D([], [], color="0.6", lw=0.6, ls=":", label=f"identical (drawn at {floor:g})"))
    if len(shown) > 1:
        handles += [Line2D([], [], marker="o" if f else "s", ls="none", color="0.3", mfc="0.3" if f else "white",
                           label=m) for m, f in shown]
    _legend_below(ax, handles=handles, ncol=2)
    _save(fig, "fig_ad_vs_analytical_objective_difference", srcs,
          f"Data: {'; '.join(methods)}. Each point: median over the 6 runs of |J_AD - J_An| / |J_An| of the final "
          f"(thresholded) objective, AD and analytical run k paired by session and repeat; exact zeros are drawn at "
          f"{floor:g}. The p-norm stress problem runs to the iteration limit and its repeats differ among themselves by "
          "up to 5e-3 (multithreaded round-off accumulated over 150 iterations); ThreeHoleBracketThick has a "
          "threshold tie (two 0/1 designs, volume 0.250000 or 0.250043) in 3 of its 12 runs.")


def fig_gradient_vs_fd_step():
    """Relative error of central finite differences against the AD directional derivative, vs the step size, for
    quantities without a hand-written gradient (and compliance). V shape: h^2 truncation error on the right,
    round-off ~eps/h on the left; the minimum shows the AD gradient is exact to round-off."""
    if not os.path.exists(FD_CHECK):
        return
    rows = _rows(FD_CHECK)
    colors = {"compliance": "#0072B2", "displacement p-norm": "#009E73", "stress p-norm": "#D55E00",
              "inverter output": "#CC79A7"}
    markers = {"compliance": "o", "displacement p-norm": "s", "stress p-norm": "^", "inverter output": "D"}
    fig, ax = _new(height=3.2)
    for q in dict.fromkeys(r["quantity"] for r in rows):
        sel = [r for r in rows if r["quantity"] == q]
        ax.loglog([_f(r["step"]) for r in sel], [max(_f(r["rel_error"]), 1e-16) for r in sel],
                  color=colors.get(q, "0.3"), marker=markers.get(q, "o"), ms=3, lw=1.0,
                  label=f"{q}: {sel[0]['expression']}")
    h = np.logspace(-10, -1, 50)
    ax.loglog(h, 0.3 * h ** 2, color="0.55", lw=0.7, ls="--", label="$h^2$ (truncation)")
    ax.loglog(h, 2e-13 / h, color="0.55", lw=0.7, ls=":", label=r"$\varepsilon/h$ (round-off)")
    ax.set_ylim(1e-11, 1e-1)
    ax.set_xlabel("Finite-difference step $h$")
    ax.set_ylabel("Relative error, FD vs AD")
    ax.grid(True, which="major")
    _legend_below(ax, ncol=1, fontsize=6.2)
    _save(fig, "fig_gradient_vs_fd_step", [FD_CHECK],
          "Central differences of J along a fixed random direction at a fixed non-uniform density, against the AD "
          "directional derivative; small meshes (~6k dofs), PARDISO. Stress p-norm is the smooth stand-in for the "
          "maximum stress used by the optimizer.")


def fig_sweep_cost():
    cls = _problem_classes()
    pts = []
    for f in glob.glob(os.path.join(SWEEP, "*", "DENSITYMMA", "DENSITYMMA_summary.csv")):
        cat = f.split(os.sep)[-3]
        for r in _rows(f):
            t, n, nd = _f(r["time (s)"]), _f(r["#FEAs"]), _true_dofs(cat, r["ndof"])
            if t and n and nd:
                pts.append((nd, t / n, _class_of(cls, cat, r["name"])))
    fig, ax = _new()
    for c in CLASS_COLOR:
        sel = [(x, y) for x, y, k in pts if k == c]
        if sel:
            ax.plot(*zip(*sel), "o", color=CLASS_COLOR[c], ms=3.5, label=c)
    x, y = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
    k, b = np.polyfit(np.log10(x), np.log10(y), 1)
    xx = np.logspace(np.log10(x.min()), np.log10(x.max()), 20)
    ax.plot(xx, 10 ** b * xx ** k, color="k", lw=0.8, label=f"fit: slope {k:.2f}")
    _dofs_axis(ax)
    ax.set_yscale("log")
    ax.set_ylabel("Time per FE solve, AD (s)")
    ax.legend(loc="upper left", fontsize=6.3)
    _save(fig, "fig_sweep_time_per_fe_solve_vs_dofs", [SWEEP, PROBLEM_TABLE],
          "Benchmark sweep, MMA with AD, PARDISO; time per FE solve = optimizer time / number of FE solves.")


def main():
    plt.rcParams.update(PAPER_RC)
    print(f"Writing figures to {OUT}/")
    rows = scaling_rows()
    fig_time_per_iteration(rows)
    fig_peak_memory(rows)
    fig_ad_graph(rows)
    fig_ad_time_ratio(rows)
    fig_ad_memory_overhead(rows)
    for s in SOLVERS:
        fig_time_breakdown(rows, s)
    fig_pardiso_memory_composition(rows)
    fig_cg_iterations(rows)
    fig_gradient_agreement()
    runs = full_runs()
    if runs:
        fig_full_objective(runs)
        fig_full_cumulative_time(runs)
        fig_full_iteration_time(runs)
        fig_full_cg_history(runs)
    for s in SOLVERS:
        fig_iteration_time_ad_vs_analytical(rows, s)
    for method in ("MMA",):                                  # OC removed from PyTO-AD
        fig_ad_extra_time_per_iteration(method)
        fig_ad_extra_time_breakdown(method)
        fig_ad_vs_analytical_peak_memory(method)
        fig_ad_graph_per_element(method)
    fig_ratio_vs_dofs()
    fig_objective_agreement()
    fig_sweep_cost()
    fig_gradient_vs_fd_step()
    with open(os.path.join(OUT, "figures_manifest.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["figure", "sources", "notes"])
        w.writeheader()
        w.writerows(MANIFEST)
    print(f"{len(MANIFEST)} figures; manifest: {OUT}/figures_manifest.csv")


if __name__ == "__main__":
    main()
