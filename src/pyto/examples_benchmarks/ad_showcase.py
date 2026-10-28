"""AD showcase: what automatic differentiation makes easy (GIFs and transparent PNGs).

Four cases, all built from the benchmark problems (src/pyto/examples_benchmarks):
  1 LBracket_compliance   3D thick L-bracket (LBracketThickTopLoad: top load 1.5e4 N, z-symmetry, ~75k dofs),
                          objective "Compliance()", volume fraction <= 0.25 (as the
                          benchmark), iteration limit 400 so that every L-bracket run can converge
  2 LBracket_stress       the same problem, only the objective text changed to "StressPNorm() / YieldStrength":
                          compiled by the expression system and differentiated by AD (no derived sensitivity)
  3 LBracket_blend        the same problem again, objective "0.4 * Compliance() / C0 + 0.6 * StressPNorm() /
                          YieldStrength / S0": a weighted blend written as one line; C0, S0 = the two terms at the
                          start design (so both weights act on terms of size ~1), inserted as numbers
  4 Inverter              compliant mechanism (benchmark Inverter): output displacement, plus a deformation GIF
  5 EdgeCantilever        3D edge-loaded cantilever (benchmark EdgeCantilever), objective "Compliance()"
The three L-bracket runs share one OptimizationSpec except for the objective string.

The designs feed the paper's formulations figure (paper/fig_gen/fig13_formulations.py) and two cases of the
body-fitted re-analysis (pyto.examples_benchmarks.body_fitted_study: LBracket_stress, EdgeCantilever).

Stage 1, run (optimizations; ~35 min; keep the machine free of other benchmarks):
    cd <repo>; export PYTHONPATH=src MPLBACKEND=Agg PYVISTA_OFF_SCREEN=true
    python -m pyto.examples_benchmarks.ad_showcase run            # all cases, or: run LBracket_stress Inverter
Stage 2, render (GIFs and PNGs from the saved runs; can be repeated to change the look):
    python -m pyto.examples_benchmarks.ad_showcase render
Objective-history GIFs only (also made by render): python -m pyto.examples_benchmarks.ad_showcase history
Code images for slides (white background, Python highlighting; the formulations exactly as run):
    python -m pyto.examples_benchmarks.ad_showcase code
Output: Results/AD_Showcase/ (another folder: environment variable PYTO_AD_SHOWCASE_OUT): <case>/history.npz,
<case>_optimization.gif, <case>_final.png (transparent),
Inverter_deformation.gif, README.md (settings and results of each case).
"""
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))     # repository root
OUT = os.environ.get("PYTO_AD_SHOWCASE_OUT") or os.path.join(ROOT, "Results", "AD_Showcase")
CASES = ("LBracket_compliance", "LBracket_stress", "LBracket_blend", "LBracket_displacement", "LBracket_mass",
         "Inverter", "EdgeCantilever")
OBJECTIVE_TEXT = {"LBracket_compliance": "Compliance()", "LBracket_stress": "StressPNorm() / YieldStrength",
                  "LBracket_blend": None,                     # set from the start design, see blend_objective()
                  "LBracket_displacement": "Displacement(FootEnd, magnitude, pnorm)",
                  "LBracket_mass": "Mass()",
                  "Inverter": "output displacement (compliant mechanism)", "EdgeCantilever": "Compliance()"}
TITLE = {"LBracket_compliance": "L-bracket: minimize compliance",
         "LBracket_stress": "L-bracket: minimize p-norm stress (one-line objective)",
         "LBracket_blend": None,                      # from the weights, see below
         "LBracket_displacement": "L-bracket: minimize displacement p-norm of the foot end",
         "LBracket_mass": "L-bracket: minimize mass, compliance limit",
         "Inverter": "Compliant inverter: maximize inward output motion",
         "EdgeCantilever": "Edge-loaded cantilever: minimize compliance"}


# ------------------------------------------------------------------------------------------------- stage 1: run

VOLUME_LIMIT = 0.25          # L-bracket runs: as the benchmark LBracketThickTopLoad
LBRACKET_MAX_ITERS = 400     # above the benchmark's 150 so the stress runs can converge (MMA stops when converged)
W_COMPLIANCE, W_STRESS = 0.4, 0.6
# LBracket_mass: compliance limit = the converged (continuous) compliance of LBracket_compliance at volume 0.25, so
# the mass-minimal design should come out near volume 0.25 (the same trade-off solved from the other side).
COMPLIANCE_LIMIT = 0.536
CONSTRAINTS = {"LBracket_mass": [("Compliance()", "<=", COMPLIANCE_LIMIT)]}   # other L-bracket cases: volume limit
TITLE["LBracket_blend"] = f"L-bracket: {W_COMPLIANCE:g} compliance + {W_STRESS:g} stress (one-line blend)"


def blend_objective(fe, volume_fraction):
    """'W_C * Compliance() / C0 + W_S * StressPNorm() / YieldStrength / S0' with C0, S0 = the two terms at the
    start design (uniform density = the volume limit, as MMA starts), so each term is ~1 at the start and the
    weights mean what they say. Cached in blend_reference.json so the run, the GIF and the code image agree."""
    cache = os.path.join(OUT, f"blend_reference_{fe.mesh.num_elems}el_vf{volume_fraction:g}.json")  # per mesh, start
    if os.path.exists(cache):
        ref = json.load(open(cache))
    else:
        import torch
        from pyto.autodiff.material_model import MaterialModel
        from pyto.topopt.expressions import compile_expression
        x = torch.full((fe.mesh.num_elems,), float(volume_fraction), dtype=torch.float64)
        with torch.no_grad():
            sol = fe.solve(x, MaterialModel.SIMP)
            c0 = float(compile_expression("Compliance()").fn(sol, x, fe))
            s0 = float(compile_expression("StressPNorm() / YieldStrength").fn(sol, x, fe))
        ref = {"C0": float(f"{c0:.4g}"), "S0": float(f"{s0:.4g}")}
        os.makedirs(OUT, exist_ok=True)
        with open(cache, "w") as f:
            json.dump(ref, f, indent=1)
    return (f"{W_COMPLIANCE} * Compliance() / {ref['C0']:g} + "
            f"{W_STRESS} * StressPNorm() / YieldStrength / {ref['S0']:g}")


def _build(case, ndof=None, max_iters=None):
    """(fe_solver, to_params) of a case, from the benchmark definitions."""
    import contextlib
    import io
    import pyto.examples_benchmarks.topopt_run_benchmarks as SB      # build_benchmark looks the builder up here
    from pyto.examples_benchmarks.topopt_run_benchmarks import build_benchmark, StructuralTOExamples
    from pyto.topopt.spec import (ConstraintSpec, ManufacturingSpec, MethodSpec, ObjectiveSpec, OptimizationSpec,
                                  RegularizationSpec, compile_spec)
    problem = {"LBracket_compliance": StructuralTOExamples.LBracketThickTopLoad,
               "LBracket_displacement": StructuralTOExamples.LBracketThickTopLoad,
               "LBracket_mass": StructuralTOExamples.LBracketThickTopLoad,
               "LBracket_stress": StructuralTOExamples.LBracketThickTopLoad,
               "LBracket_blend": StructuralTOExamples.LBracketThickTopLoad,
               "Inverter": StructuralTOExamples.Inverter,
               "EdgeCantilever": StructuralTOExamples.EdgeCantilever}[case]
    if ndof:                                            # smaller mesh for a quick check
        orig = SB.getStructuralTOProblem
        SB.getStructuralTOProblem = lambda p, nDOFDesired=None, **kw: orig(p, nDOFDesired=ndof, **kw)
    with contextlib.redirect_stdout(io.StringIO()):
        fe, to_params, _ = build_benchmark(problem)
    if ndof:
        SB.getStructuralTOProblem = orig
    if case == "LBracket_blend":
        OBJECTIVE_TEXT[case] = blend_objective(fe, VOLUME_LIMIT)
    selections = {}
    if case == "LBracket_displacement":           # foot end face (x = xMax), away from the loaded nodes
        xyz, h = fe.mesh.node_xyz, fe.mesh.elem_size[0] / 2
        labels = np.asarray(fe.mesh.node_indices[:, 3])
        selections["FootEnd"] = {"nodes": np.where((np.abs(xyz[:, 0] - xyz[:, 0].max()) < h) & (labels != 2))[0]}
    if case.startswith("LBracket"):
        # One formulation for all L-bracket runs; only the objective string (and for the mass case the
        # constraint) differs.
        spec = OptimizationSpec(
            objective=ObjectiveSpec(OBJECTIVE_TEXT[case], "minimize"),
            constraints=[ConstraintSpec(*c) for c in CONSTRAINTS.get(case, [("VolumeFraction()", "<=", VOLUME_LIMIT)])],
            method=MethodSpec("MMA", max_iterations=LBRACKET_MAX_ITERS, gradient="autodiff"),
            regularization=RegularizationSpec(filter_radius=to_params.RelativeFilterRadius, heaviside=False),
            manufacturing=ManufacturingSpec(symmetry=["Z"]))
        to_params = compile_spec(spec, selections, base=to_params)
    if max_iters:
        to_params.MaxIterations = int(max_iters)
    return fe, to_params


def run_case(case, ndof=None, max_iters=None):
    import torch
    from pyto.topopt.drivers.mma import topopt_mma
    fe, to_params = _build(case, ndof, max_iters)
    mesh = fe.mesh
    frames, objective = [], []
    t0 = time.time()

    def record(history):
        frames.append(np.asarray(mesh.elemPseudoDensity, dtype=np.float32).copy())
        objective.append(float(np.asarray(history["objective"][-1]).reshape(-1)[0]))
        if len(frames) % 10 == 0:
            print(f"  {case}: iteration {len(frames)}, objective {objective[-1]:.4g}, "
                  f"{time.time() - t0:.0f} s", flush=True)

    print(f"{case}: {3 * mesh.num_nodes:,} dofs, {mesh.num_elems:,} elements, objective {OBJECTIVE_TEXT[case]}",
          flush=True)
    u, history, success, msg, n_fe = topopt_mma(
        fe, to_params=to_params, maxMMAIterations=to_params.MaxIterations,
        binarize_topology=not to_params.HeavisideProjection, print_progress=False, iteration_callback=record)
    final = np.asarray(mesh.elemPseudoDensity, dtype=np.float32).copy()
    labels = np.asarray(mesh.node_indices[:, 3]).astype(int)
    extra = {}
    if case == "Inverter":                              # output nodes as the benchmark defines them
        xyz, h = mesh.node_xyz, mesh.elem_size[0] / 2
        y_mid = (xyz[:, 1].max() + xyz[:, 1].min()) / 2
        extra["output_nodes"] = np.where((abs(xyz[:, 0] - xyz[:, 0].max()) < h) & (abs(xyz[:, 1] - y_mid) < h))[0]
    out = os.path.join(OUT, case)
    os.makedirs(out, exist_ok=True)
    np.savez_compressed(
        os.path.join(out, "history.npz"), frames=np.array(frames), final=final, objective=np.array(objective),
        node_xyz=np.asarray(mesh.node_xyz), elems=np.asarray(mesh.elemArray)[:, :8].astype(np.int64),
        labels=labels, force=np.asarray(fe.bc.force, dtype=float), u=np.asarray(u, dtype=float).reshape(-1),
        extrude_z=bool(to_params.ExtrudeZ), **extra)
    summary = dict(case=case, objective_text=OBJECTIVE_TEXT[case], dofs=3 * mesh.num_nodes,
                   elements=int(mesh.num_elems), iterations=len(frames), fe_solves=int(n_fe),
                   objective_final=float(np.asarray(history["objective"][-1]).reshape(-1)[0]),
                   volume_fraction=float(history["volfrac"][-1]), time_s=time.time() - t0, message=msg,
                   max_iterations=int(to_params.MaxIterations),
                   converged=len(frames) - 1 < int(to_params.MaxIterations),
                   gradient="automatic differentiation", constraints=[str(c[:1] + c[2:]) for c in to_params.Constraints])
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=str)
    print(f"{case}: done in {summary['time_s']:.0f} s, {len(frames)} iterations, final objective "
          f"{summary['objective_final']:.4g}, volume {summary['volume_fraction']:.3f}", flush=True)
    del torch


# ---------------------------------------------------------------------------------------------- stage 2: render

# The benchmark images' style (HexStructuralFEA.plot_mesh): light-green elements with black edges, black support
# dots, red load arrows.
SOLID = "lightgreen"       # design elements
EDGE = "black"             # element edges
GHOST = "#B9C2CC"          # undeformed outline in the deformation GIF
LOAD = "red"               # loads
OUTPUT = "#1F5FBF"         # desired output motion (inverter)
SUPPORT = "black"


def _grid(d):
    import pyvista as pv
    elems = d["elems"]
    cells = np.hstack([np.full((len(elems), 1), 8), elems]).ravel()
    types = np.full(len(elems), pv.CellType.HEXAHEDRON, dtype=np.uint8)
    return pv.UnstructuredGrid(cells, types, d["node_xyz"].astype(float))


def _solid(grid, rho, threshold=None):
    """Surface of the elements shown as solid: the volume-preserving threshold (the same rule the drivers use to
    binarize) so every frame shows a design of the current volume, or a fixed threshold."""
    if threshold is None:
        s = np.sort(rho)
        threshold = s[min(int((1 - rho.mean()) * len(s)), len(s) - 1)]
        threshold = min(max(threshold, 1e-3), 0.5)
    ids = np.where(rho >= threshold)[0]
    if len(ids) == 0:
        return None
    return grid.extract_cells(ids).extract_surface()


def _arrows(d, kind="load", n_max=5, rel_len=0.15):
    """A few evenly spaced arrows along the loaded nodes, tip at the node, length a fixed share of the model size,
    pointing in the direction of the total load (output: the desired inward motion)."""
    import pyvista as pv
    xyz = d["node_xyz"]
    diag = np.linalg.norm(xyz.max(0) - xyz.min(0))
    if kind == "load":
        nodes = np.where(d["labels"] == 2)[0]
        f = d["force"].reshape(-1, 3)[nodes]
        direction = f.sum(0)
    else:
        nodes = np.asarray(d["output_nodes"])
        direction = np.array([-1.0, 0.0, 0.0])              # inverter: output moves against the input push
    if len(nodes) == 0 or np.linalg.norm(direction) == 0:
        return []
    direction = direction / np.linalg.norm(direction)
    pts = xyz[nodes]
    order = np.lexsort(pts.T[::-1])                          # spread the chosen arrows over the loaded region
    pick = pts[order][np.unique(np.linspace(0, len(nodes) - 1, min(n_max, len(nodes))).round().astype(int))]
    length = rel_len * diag
    return [pv.Arrow(start=p - direction * length, direction=direction, scale=length, tip_length=0.28,
                     tip_radius=0.09, shaft_radius=0.03) for p in pick]


def _camera(plotter, d, iso):
    """Fixed camera for a whole animation: isometric with +y up (loads in -y point down on screen), or a 2D view
    from +z for the extruded (2.5D) problems."""
    xyz = d["node_xyz"]
    c = (xyz.max(0) + xyz.min(0)) / 2
    diag = np.linalg.norm(xyz.max(0) - xyz.min(0))
    if iso:
        direction = np.array([1.0, 0.75, 1.35])
        direction /= np.linalg.norm(direction)
        plotter.camera_position = [tuple(c + 2.6 * diag * direction), tuple(c), (0.0, 1.0, 0.0)]
    else:
        plotter.camera_position = [tuple(c + np.array([0, 0, 2.6 * diag])), tuple(c), (0.0, 1.0, 0.0)]
        plotter.enable_parallel_projection()
    lo, hi = xyz.min(0), xyz.max(0)                          # fit the model, not whatever is drawn now
    plotter.reset_camera(bounds=(lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]))
    plotter.camera.zoom(1.0 if iso else 0.85)                # leave room for the title and the arrows
    return plotter.camera_position


def _scene(plotter, d, grid, rho, iso, solid_threshold=None, with_bc=True):
    import pyvista as pv
    surf = _solid(grid, rho, solid_threshold)                # same look in 3D and 2.5D, as in the benchmarks
    if surf is not None:
        plotter.add_mesh(surf, color=SOLID, show_edges=True, edge_color=EDGE, line_width=1)
    if with_bc:
        sup = d["node_xyz"][d["labels"] == 1]
        if len(sup):
            plotter.add_mesh(pv.PolyData(sup), color=SUPPORT, point_size=5 if iso else 4,
                             render_points_as_spheres=True)
        for a in _arrows(d, "load"):
            plotter.add_mesh(a, color=LOAD)
        if "output_nodes" in d:
            for a in _arrows(d, "output", n_max=1):
                plotter.add_mesh(a, color=OUTPUT)


def _formulation_text(case):
    """Objective text, plus the constraint when it is not the usual volume limit (mass case)."""
    text = _objective_text(case)
    if case in CONSTRAINTS:
        text += "  s.t. " + ", ".join(f"{e} {op} {b:g}" for e, op, b in CONSTRAINTS[case])
    return text


def _objective_text(case):
    p = os.path.join(OUT, case, "summary.json")
    return json.load(open(p))["objective_text"] if os.path.exists(p) else OBJECTIVE_TEXT[case]


def _text(plotter, case, it=None, n=None, size=(900, 700)):
    plotter.add_text(TITLE[case], position=(16, size[1] - 40), font_size=11, color="black", font="times")
    text = _formulation_text(case)
    plotter.add_text(f"objective:  {text}", position=(16, size[1] - 70), font_size=9 if len(text) < 60 else 7,
                     color="#333333", font="courier")
    if it is not None:
        plotter.add_text(f"iteration {it}/{n}", position=(16, 16), font_size=9, color="#333333", font="times")


def _save_gif(images, path, ms=90, hold_ms=2200):
    from PIL import Image
    pil = [Image.fromarray(im) for im in images]
    durations = [ms] * (len(pil) - 1) + [hold_ms]
    pil[0].save(path, save_all=True, append_images=pil[1:], duration=durations, loop=0, optimize=True, disposal=2)


def render_case(case, size=(900, 700), png_size=(3000, 2300)):
    import pyvista as pv
    path = os.path.join(OUT, case, "history.npz")
    if not os.path.exists(path):
        print(f"{case}: no saved run ({path}); run stage 1 first")
        return
    d = dict(np.load(path))
    iso = not bool(d["extrude_z"])
    grid = _grid(d)
    frames = list(d["frames"])
    step = 2 if len(frames) > 90 else 1
    picks = list(range(0, len(frames), step))
    if picks[-1] != len(frames) - 1:
        picks.append(len(frames) - 1)

    plotter = pv.Plotter(off_screen=True, window_size=size)
    plotter.set_background("white")
    plotter.enable_anti_aliasing("ssaa")
    _scene(plotter, d, grid, d["final"], iso, solid_threshold=0.5)
    cam = _camera(plotter, d, iso)
    images = []
    for i in picks:
        plotter.clear()
        _scene(plotter, d, grid, frames[i], iso)
        _text(plotter, case, i + 1, len(frames), size)
        plotter.camera_position = cam
        images.append(plotter.screenshot(return_img=True))
    plotter.clear()                                          # final binarized design, held
    _scene(plotter, d, grid, d["final"], iso, solid_threshold=0.5)
    _text(plotter, case, len(frames), len(frames), size)
    plotter.add_text("final design", position=(size[0] - 140, 16), font_size=9, color="#333333", font="times")
    plotter.camera_position = cam
    images.append(plotter.screenshot(return_img=True))
    plotter.close()
    _save_gif(images, os.path.join(OUT, f"{case}_optimization.gif"))

    # high-resolution, transparent static picture of the final design (no text)
    p = pv.Plotter(off_screen=True, window_size=png_size)
    p.enable_anti_aliasing("ssaa")
    _scene(p, d, grid, d["final"], iso, solid_threshold=0.5)
    _camera(p, d, iso)
    p.screenshot(os.path.join(OUT, f"{case}_final.png"), transparent_background=True)
    p.close()
    print(f"{case}: {len(images)} frames -> {case}_optimization.gif, {case}_final.png", flush=True)

    if case == "Inverter":
        render_deformation(case, d, grid, size)
    render_history(case)


def render_deformation(case, d, grid, size, n=36):
    """The final mechanism under its input load: displacement scaled up from 0 to a visible amount and back."""
    import pyvista as pv
    solid_ids = np.where(d["final"] >= 0.5)[0]
    body = grid.extract_cells(solid_ids)
    u = d["u"].reshape(-1, 3)[body["vtkOriginalPointIds"]]
    xyz = d["node_xyz"]
    diag = np.linalg.norm(xyz.max(0) - xyz.min(0))
    scale = 0.08 * diag / max(np.abs(u).max(), 1e-30)
    base = body.points.copy()
    plotter = pv.Plotter(off_screen=True, window_size=size)
    plotter.set_background("white")
    plotter.enable_anti_aliasing("ssaa")
    outline = body.extract_surface()
    plotter.add_mesh(outline, color=GHOST)
    cam = _camera(plotter, d, iso=False)
    images = []
    for t in np.concatenate([np.linspace(0, 1, n // 2), np.linspace(1, 0, n // 2)]):
        plotter.clear()
        plotter.add_mesh(outline, color=GHOST, opacity=0.35)
        moved = body.copy()
        moved.points = base + t * scale * u
        plotter.add_mesh(moved.extract_surface(), color=SOLID, show_edges=True, edge_color=EDGE, line_width=1)
        for a in _arrows(d, "load"):
            plotter.add_mesh(a, color=LOAD)
        for a in _arrows(d, "output", n_max=1):
            plotter.add_mesh(a, color=OUTPUT)
        plotter.add_text("Compliant inverter: input push (red) -> output moves inward (blue)",
                         position=(16, size[1] - 40), font_size=10, color="black", font="times")
        plotter.add_text(f"deformation x{scale:.0f} (scaled to show the motion)", position=(16, 16), font_size=9,
                         color="#333333", font="times")
        plotter.camera_position = cam
        images.append(plotter.screenshot(return_img=True))
    plotter.close()
    _save_gif(images, os.path.join(OUT, f"{case}_deformation.gif"), ms=70, hold_ms=70)
    print(f"{case}: deformation -> {case}_deformation.gif", flush=True)


CODE_COMMON = """from pyto.topopt.spec import OptimizationSpec, ObjectiveSpec, ConstraintSpec, compile_spec
from pyto.topopt.drivers.mma import topopt_mma

"""
CODE_TEMPLATE = """{comment}
spec = OptimizationSpec(
    objective=ObjectiveSpec("{objective}"),
    constraints=[ConstraintSpec("VolumeFraction()", "<=", {vol})])
to_params = compile_spec(spec, base=to_params)     # same mesh, load, supports
topopt_mma(fe_solver, to_params=to_params)         # gradients by automatic differentiation
"""


def render_code(font_size=30):
    """White-background, Python-highlighted images of how each L-bracket objective is defined (as run)."""
    from pygments import highlight
    from pygments.formatters import ImageFormatter
    from pygments.lexers import PythonLexer
    from pygments.styles import get_style_by_name

    class White(get_style_by_name("default")):
        background_color = "#ffffff"
        highlight_color = "#ffffff"

    fmt = dict(style=White, font_name="DejaVu Sans Mono", font_size=font_size, line_numbers=False,
               image_pad=36, line_pad=8)
    fe = None
    blend = _objective_text("LBracket_blend")
    if blend is None:                                        # no blend run yet: values from the full-size model
        fe = _build("LBracket_blend")[0]
        blend = OBJECTIVE_TEXT["LBracket_blend"]
    import re
    c0, s0 = re.findall(r"/ ([0-9.eE+-]+)", blend)[0], re.findall(r"/ ([0-9.eE+-]+)$", blend)[0]
    ref = {"C0": float(c0), "S0": float(s0)}
    snippets = {
        "code_compliance": CODE_COMMON + CODE_TEMPLATE.format(
            comment="# minimize compliance (stiffest design)", objective="Compliance()", vol=VOLUME_LIMIT),
        "code_stress": CODE_COMMON + CODE_TEMPLATE.format(
            comment="# minimize the p-norm of von Mises stress: change one string, nothing derived by hand",
            objective="StressPNorm() / YieldStrength", vol=VOLUME_LIMIT),
        "code_blend": CODE_COMMON + CODE_TEMPLATE.format(
            comment=f"# {100 * W_COMPLIANCE:.0f} % stiffness + {100 * W_STRESS:.0f} % stress, each term scaled by its start value "
                    f"(C0 = {ref['C0']:g}, S0 = {ref['S0']:g})",
            objective=blend, vol=VOLUME_LIMIT),
        "code_switch": "# the only line that changes between the three L-bracket designs\n"
                       'objective = ObjectiveSpec("Compliance()")\n'
                       'objective = ObjectiveSpec("StressPNorm() / YieldStrength")\n'
                       f'objective = ObjectiveSpec("{blend}")\n',
    }
    for name, code in snippets.items():
        path = os.path.join(OUT, f"{name}.png")
        with open(path, "wb") as f:
            f.write(highlight(code, PythonLexer(), ImageFormatter(**fmt)))
        print(f"code image -> {name}.png", flush=True)


def render_history(case, size_in=(9, 7), dpi=100, ms=90, hold_ms=2200):
    """Objective vs iteration as a GIF that plays in step with <case>_optimization.gif: same size, same iterations per
    frame, same frame time and final hold. The whole curve is shown faintly from the start; the part reached so far
    is drawn in colour with a dot at the current iteration. Fonts follow the paper figures (3.5 in wide, labels 8 pt,
    ticks 7 pt), scaled to this 9 in figure. Axis limits are fixed for all frames. Log axis for positive objectives,
    linear otherwise (the inverter's objective is negative)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    path = os.path.join(OUT, case, "history.npz")
    if not os.path.exists(path):
        return
    J = np.asarray(np.load(path)["objective"], float)
    n = len(J)
    it = np.arange(1, n + 1)
    step = 2 if n > 90 else 1                                # same frame selection as the optimization GIF
    picks = list(range(0, n, step))
    if picks[-1] != n - 1:
        picks.append(n - 1)
    K = size_in[0] / 3.5
    rc = {"font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman"], "mathtext.fontset": "stix",
          "font.size": 8 * K, "axes.labelsize": 8 * K, "xtick.labelsize": 7 * K, "ytick.labelsize": 7 * K,
          "axes.linewidth": 0.7 * K, "xtick.major.width": 0.7 * K, "ytick.major.width": 0.7 * K,
          "xtick.minor.width": 0.5 * K, "ytick.minor.width": 0.5 * K, "xtick.major.size": 3.5 * K,
          "ytick.major.size": 3.5 * K, "xtick.minor.size": 2 * K, "ytick.minor.size": 2 * K,
          "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True}
    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=size_in, dpi=dpi)
        fig.subplots_adjust(left=0.15, right=0.95, top=0.84, bottom=0.14)
        ax.plot(it, J, color="0.85", lw=2.0, zorder=1)
        line, = ax.plot([], [], color="#2E8B57", lw=2.6, zorder=2)
        dot, = ax.plot([], [], "o", color="#2E8B57", ms=9, mec="k", mew=0.8, zorder=3)
        ax.set_xlim(0, n + 1)
        if J.min() > 0:
            ax.set_yscale("log")
            ax.set_ylim(J.min() / 1.25, J.max() * 1.25)
        else:
            pad = 0.06 * (J.max() - J.min() or 1.0)
            ax.set_ylim(J.min() - pad, J.max() + pad)
        ax.set_autoscale_on(False)                           # limits never change between frames
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Objective")
        ax.grid(True, which="both", color="0.92", lw=0.8)
        ax.set_axisbelow(True)
        text = _formulation_text(case)
        fig.text(0.15, 0.965, TITLE[case], fontsize=7.5 * K if len(TITLE[case]) < 62 else 6.3 * K, va="top")
        fig.text(0.15, 0.905, f"objective: {text}", fontsize=13 if len(text) < 60 else 10, family="monospace",
                 color="0.25", va="top")
        label = ax.text(0.97, 0.95, "", transform=ax.transAxes, ha="right", va="top", fontsize=7 * K,
                        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.7"))
        images = []
        for i in picks:
            line.set_data(it[: i + 1], J[: i + 1])
            dot.set_data([it[i]], [J[i]])
            label.set_text(f"iteration {it[i]}/{n}\nJ = {J[i]:.4g}")
            fig.canvas.draw()
            images.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)
    # the optimization GIF ends with the final-design frame held; here the converged curve is held for that time
    _save_gif(images, os.path.join(OUT, f"{case}_history.gif"), ms=ms, hold_ms=ms + hold_ms)
    print(f"{case}: {len(images)} frames -> {case}_history.gif", flush=True)


def write_readme():
    lines = ["# AD showcase", "",
             "Made by `pyto.examples_benchmarks.ad_showcase` from the benchmark problems; all gradients by automatic differentiation (MMA). "
             "The two L-bracket runs use the same OptimizationSpec (volume fraction <= 0.25, z-symmetry, same load and "
             "mesh); only the objective string differs. Load arrows (red) point along the applied load, dots mark "
             "supports; the blue arrow on the inverter is the desired output motion. Same colours as the benchmark images "
             "(light-green elements, black edges).", "",
             "| Case | Objective (as written) | dofs | Iterations | Converged | Final objective | Volume | Time (s) |",
             "|---|---|---|---|---|---|---|---|"]
    for case in CASES:
        p = os.path.join(OUT, case, "summary.json")
        if os.path.exists(p):
            s = json.load(open(p))
            conv = {True: "yes", False: f"no (limit {s.get('max_iterations')})"}.get(s.get("converged"), "–")
            lines.append(f"| {case} | `{s['objective_text']}` | {s['dofs']:,} | {s['iterations']} | {conv} | "
                         f"{s['objective_final']:.4g} | {s['volume_fraction']:.3f} | {s['time_s']:.0f} |")
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "README.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def main(argv):
    if not argv or argv[0] not in ("run", "render", "check", "code", "history"):
        raise SystemExit(__doc__)
    cases = [c for c in argv[1:] if c in CASES] or list(CASES)
    if argv[0] in ("run", "check"):
        for case in cases:                    # check: tiny meshes and few iterations, to test the whole pipeline
            if argv[0] == "check":
                run_case(case, ndof=6000, max_iters=4)
            else:
                run_case(case)
    if argv[0] in ("render", "check"):
        for case in cases:
            render_case(case)
    if argv[0] == "history":                     # only the objective-history GIFs (from the saved runs)
        for case in cases:
            render_history(case)
    if argv[0] in ("code", "render"):
        render_code()
    write_readme()


if __name__ == "__main__":
    main(sys.argv[1:])
