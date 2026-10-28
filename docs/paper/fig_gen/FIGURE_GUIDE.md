# Figure guide (PyTO paper, Structural and Multidisciplinary Optimization)

These rules apply to every figure made in this repository: the paper (its LaTeX source `main.tex` is kept outside this repository and includes `figures/generated/…`), the slides and the README. The figure scripts read the shared values from `docs/paper/fig_gen/_style.py` (diagrams and renders) and from `PAPER_RC` in `src/pyto/examples_benchmarks/make_paper_figures.py` (data plots). Change a rule here, and in those two places, never in a single script.

## Rules

1. **Journal conventions (Springer / SMO artwork guidelines).**
   - Figure parts are labelled with lowercase letters in parentheses: (a), (b), (c), and so on. Fig. 1 included: no digits, even for steps in a sequence.
   - Inside a figure, a part carries only its label "(a)". What the part shows goes in the LaTeX caption, as "(a)~…, (b)~…".
   - No titles inside figures. Axis labels, colour-bar labels, legends, row and column headers of a parameter grid, and data values written on a tile are allowed, because they are labels, not titles.
   - Diagram content (box texts in a flowchart, the formula card in Fig. 1) is part of the illustration and stays.
2. **Font.**
   - Use Helvetica-class sans serif: `["Arial", "Helvetica", "Liberation Sans", "Nimbus Sans", "DejaVu Sans"]`. The first one installed is used; Liberation Sans has Arial's metrics.
   - Math text uses the same sans face (`mathtext.fontset = "custom"`, `rm/it/bf = sans`).
   - Code strings (formulas, function names) are set in monospace.
   - Use 8 pt for labels and 7 pt for tick labels, never below 6 pt at final size.
3. **Size.**
   - Draw figures at their final size, so LaTeX does not rescale them.
   - Full text width is 5.15 in (`\linewidth`, sn-jnl 372 pt), for the `fig_gen` figures.
   - Data plots the paper uses are drawn at the width they are included at (`FINAL_WIDTH` in `make_paper_figures.py`): 0.85\linewidth = 4.38 in, or 0.80\linewidth = 4.12 in. Include them at exactly that width. The other data plots stay 3.5 in.
4. **Colours.**
   - **Topology:** designs, voxel meshes, recovered STL parts and their tetrahedral meshes, and sweep galleries always use PyTO's classic look, as `plot_mesh` in `src/pyto/physics/structural/hex_structural_fea.py` draws it: `lightgreen` (#90EE90) elements, black edges and dark-red (#B00000) load arrows.
   - Never recolour PyTO's own renders.
   - Supports are blue (#0072B2).
   - An undesigned domain is pale green (#D4F2D4).
   - Continuous density runs from white (0) to PyTO green #3E8E3E (1).
   - Field plots (von Mises stress, temperature) use a perceptual colour map (viridis), with a colour bar that has a label and a unit.
   - Data plots use the Okabe-Ito colours of `make_paper_figures.py`: one colour per solver, AD solid with filled markers, analytical dashed with open markers.
5. **Output.**
   - Each script writes a vector PDF (for LaTeX) and a 600 dpi PNG (preview).
   - `fig_gen` scripts write to `docs/paper/figures/generated/` (git-ignored; set `PYTO_PAPER_FIGURES` to write elsewhere). The LaTeX tree links its `figures/generated` to this folder.
   - `make_paper_figures.py` writes to `Results/PaperFigures/`. After rerunning it, copy every `fig_*.pdf/png` over the same-named files in the paper's `figures/` folder (and `docs/images/readme/` for the README copy).
6. **Captions.**
   - The caption states what each part shows, using the texts in the table below, plus the numbers that make the figure checkable (DOF, iterations, objective values).
   - Keep the caption and this table in step.

## Figures and their part captions

| Paper | File (`figures/…` in the paper) | Script | Parts: caption text for each label |
|---|---|---|---|
| `fig:graphical_abstract` (Fig. 1, §3; also the graphical abstract for submission) | `generated/fig01_graphical_abstract` | `fig_gen/fig01_graphical_abstract.py` (run `run_round2_designs.py threehole` once; `_cache/round2_threehole.npz`) | 2×2, 1.0\linewidth, ThreeHoleBracketThick at 211 410 DOF, one camera and one scale: (a) STL with supports (blue hole walls) and load (red hole wall, arrow); (b) von Mises stress of the solid domain on the hex mesh, colorbar 0–36 MPa (99.5th percentile, max 47.3 MPa); (c) optimized design (density > 0.5); (d) smooth recovered STL (CNN, iso-surface, Boolean) |
| `fig:graph` | `generated/fig03_computational_graph` | dropped from the paper; its script is not part of `docs/paper/fig_gen` | single diagram, no parts |
| `fig:gui_formulation` (the former GUI and STL-recovery figures, merged) | `generated/fig04_gui_formulation` | `fig_gen/gui_session_threehole.py` (recorded GUI session; `--keep-recovered` re-records the captures only) → `fig_gen/fig04_05_gui.py` | (a) input STL with the supports (dark, two left hole surfaces) and the 10 kN load (arrows, right hole), geometry semi-transparent; (b) Objective tab, `Compliance()`; (c) Constraints tab, `VolumeFraction() <= 0.25`; (d) gradient check before the run; (e) live objective and constraint plot up to iteration 15; (f) density in the 3D view at iteration 15; (g) recovered smooth STL. Panels in workflow order, three rows, 5.15 × 4.33 in (fits a page with its caption; include at \linewidth, never scaled); GUI text at >= 6 pt in print and height <= 5.2 in (both checked by the script) |
| `fig:quickstart` (after Listing 2) | `generated/fig02_quickstart` | `fig_gen/fig02_quickstart.py` | ThreeHoleBracketThick (the two docs/examples scripts, PARDISO), one 3D view of the smooth recovered STL of each design: (a) `Compliance()`; (b) `Displacement(All, magnitude, pnorm)`. Supports and load on the hole walls; legend below |
| `fig:teaching` (§7) | `generated/fig06_teaching_grid` | `fig_gen/fig06_teaching_grid.py` | single grid: rows filter radius r = 1.1, 1.5, 3 elements; columns SIMP exponent p = 1, 3, 5; tile labels c₀/₁ and grey fraction |
| `fig:gradcheck` (appendix) | `ad_vs_analytical/fig_gradient_vs_fd_step` | `make_paper_figures.py` | single plot, 0.85\linewidth (4.38 in) |
| `fig:objdiff` | `ad_vs_analytical/fig_ad_vs_analytical_objective_difference` | `make_paper_figures.py` | single plot, 0.80\linewidth (4.12 in); MMA, 25 pairs, median of 6 uninstrumented runs (two sessions, C1 snapshot) |
| `fig:adtime` | `ad_vs_analytical/fig_ad_vs_analytical_time_per_iteration_mma` | `make_paper_figures.py` | single plot, 0.85\linewidth (4.38 in): median time per iteration, AD and analytical, and AD's extra time, per problem |
| `fig:admemory` | `solver_scaling/fig_scaling_ad_graph_memory_vs_elements` | `make_paper_figures.py` | single plot, 0.85\linewidth (4.38 in): bytes autograd saves per iteration against the number of elements |
| `fig:scaling` | `solver_scaling/fig_scaling_time_per_iteration_vs_dofs` | `make_paper_figures.py` | single plot, 0.85\linewidth (4.38 in) |
| `fig:formulations` | `generated/fig13_formulations` | `fig_gen/fig13_formulations.py` | (a) `Compliance()`; (b) `StressPNorm() / YieldStrength`; (c) `Displacement(FootEnd, magnitude, pnorm)`; (d) `Mass()` subject to `Compliance() <= 0.536` |
| `fig:intro_gegrabcad` (Introduction) | `generated/fig_intro_gegrabcad` | `fig_gen/fig_intro_gegrabcad.py` (run `run_round2_designs.py ge` once; `_cache/round2_ge.npz`) | 1×2, 1.0\linewidth, GEGrabCAD at 326 691 DOF: (a) input STL with the problem's supports (blue; bolt-hole bottoms only, STL triangles 3245–3532 since 2026-10-27) and load (red) triangles, load arrow; (b) material iso-surface of the design after Gaussian smoothing (σ = 1 grid spacing), no Boolean (the GE STL has 2 non-manifold edges) |
| `fig:gallery_nhg` | `generated/fig14_gallery_no_handwritten_gradient` | `fig_gen/fig_sweep_galleries.py` | 1×4, 1.0\linewidth (5.15 × 1.41 in): (a) Inverter; (b) CentrifugalPlate; (c) GravityPlate; (d) LBracketTopLoad_Vol_Stress, converged 400-iteration run (ReviewDesigns_2026-10-24) |
| `fig:thermostructural` (suggested label; replaces `fig:biclamp` and `fig:mbbheat`) | `generated/fig15_thermostructural` | `fig_gen/fig_sweep_galleries.py` | (a) BiClamp, temperature rise ΔT = 0 K; (b) BiClamp, ΔT = 1 K; (c) half MBB beam (Ooms et al. set-up) without heating; (d) the same with the hot lower edge. Each pair at one scale; compliance J on each tile (0.1089, 0.6761, 38.02, 276.7 J) |
| `fig:casestudy_stl` | `generated/fig17_casestudy` | `fig_gen/fig17_casestudy.py` | ThreeHoleBracketThick: (a) voxel design; (b) recovered STL; (c) tetrahedral mesh. Stress-designed L-bracket, von Mises on one scale capped at 4 MPa: (d) voxel design; (e) recovered part |

**Not in the paper** (files and scripts kept):
- `ad_vs_analytical/fig_ad_vs_analytical_time_ratio_vs_dofs`, `ad_vs_analytical/fig_ad_vs_analytical_extra_time_breakdown_mma`, `solver_scaling/fig_scaling_peak_memory_vs_dofs` (`make_paper_figures.py`).
- `generated/fig05_stl_recovery`, `generated/fig15_biclamp`, `generated/fig16_mbb_heated`: superseded by `fig04_gui_formulation` and `fig15_thermostructural`; no longer written by the scripts.

Rows are keyed by `\label`; figure numbers follow `main.tex`.

### Data behind the measurement figures (after the 2026-10-21 retention fix)

- **Scaling figures (`fig:admemory`, `fig:scaling` and the other `fig_scaling_*`):** `Results/SolverScaling_2026-10-21_noretain`, EdgeCantilever, PARDISO, DPCG and DPCG-GPU, AD and analytical, 29k–1.06M DOF, 10 iterations, one run per configuration. `SolverScaling_2026-10-16b` is the same study before the fix; it is not used.
- **AD against analytical (`fig:objdiff`, `fig:adtime`, and the `fig_ad_vs_analytical_*` plots not in the paper):**
  - Times: `Results/GradientComparison_2026-10-18b` (3 repeats, quiet machine). The fix does not change time per iteration materially.
  - Memory (`fig_ad_vs_analytical_peak_memory_*`, `fig_ad_graph_per_element_*`), overlaid on the 18b pairs in `make_paper_figures.gc_pairs`, 1 repeat each:
    - MMA: `Results/GradientComparison_2026-10-21_memory` (10 iterations).
    - OC: `Results/GradientComparison_2026-10-22_memory_oc`, run to convergence. With a 10-iteration cap, OC's end-of-run evaluation raised the run peak (TorquePlate analytical: 3.25 GB against 1.99 GB in its iterations).
  - Post-fix numbers for the cost table: `Results/GradientComparison_2026-10-21_memory/memory_summary.txt`.

### Figures not in the paper, with data from before the 2026-10-21 retention fix

- `fig_full_*` (`fig_full_cumulative_time`, `fig_full_objective_history`, `fig_full_time_per_iteration_history`, `fig_full_dpcg_iterations_history`) and `fig_scaling_pardiso_memory_composition_vs_dofs`. They are built from the full optimizations `Results/SolverScaling_full_2026-10-17` and `17b`, recorded before the fix. Then the FE solvers kept the previous solve's graph and factors alive into the next solve.
  - **Time:** these lines stay valid. The fix changes time per iteration by about 1 % (PARDISO-AD at 540k: 96.4 s before, 97.2 s after).
  - **Memory:** values from these runs are pre-fix and must not be quoted.
  - **The missing line:** PARDISO-AD at 1.06M hit the memory cap then. It fits after the fix but was not rerun (decision of 2026-10-21; a full run is about 15 h).
  - **Use in the paper:** before using one of these figures, rerun the full optimizations.

## Regenerating

From `docs/paper/fig_gen/`, with `PYTHONPATH=../../../src`, the PyToLib environment and `PYVISTA_OFF_SCREEN=true`:

```bash
for f in fig01_graphical_abstract fig02_quickstart fig06_teaching_grid \
         fig13_formulations fig_sweep_galleries fig_intro_gegrabcad fig17_casestudy fig04_05_gui; do
    ~/anaconda3/envs/PyToLib/bin/python $f.py; done
```

GUI figure (`fig:gui_formulation`): `QT_SCALE_FACTOR=2 python gui_session_threehole.py` first (`--keep-recovered` re-records the captures only). It drives the real GUI on the X display (a few minutes), writes the captures, project file and logs to `Results/GUISession_ThreeHole/`; `fig04_05_gui.py` only crops and places them. GUI captures keep the GUI's own colours and text (cropped, never edited); keep the TopOpt Execute window narrow (470 logical px) so its text stays legible at print size.

`fig02` and `fig06` cache their optimization runs in `docs/paper/fig_gen/_cache/` (git-ignored); delete a cache file to rerun it.

From the repository root, for the data plots:

```bash
PYTHONPATH=src MPLBACKEND=Agg python -m pyto.examples_benchmarks.make_paper_figures
```

Then copy each `Results/PaperFigures/fig_*` over its same-named copies in the paper's `figures/` folder.
