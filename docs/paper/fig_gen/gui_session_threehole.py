"""A recorded PyTO GUI session on ThreeHoleBracketThick, with screen captures for the paper's GUI figures.

Drives the real GUI (MainWindow and its windows, on the X display, so the 3D views render) through the same calls
the buttons make:
  load the STL (a copy, so the GUI's .vtu and _optimized.stl land in the session folder, not in Models/)
  -> Material window: Steel -> supports (two left hole surfaces, Fixed XYZ) and load (right hole surface, 10 kN in -y)
  -> TopOpt Options: Z symmetry, keep fixed faces -> mesh, 67 000 elements requested (about 63k elements, as Fig. 1)
  -> TopOpt Execute: Compliance(), VolumeFraction() <= 0.25, MMA with AD -> gradient check -> optimize (topology
     drawn every iteration) -> save the project -> TopOpt Postprocess: STL recovery.
These are the benchmark's settings (createThreeHoleBracketThickProblem, StructuralTOExamples.ThreeHoleBracketThick).

Output: Results/GUISession_ThreeHole/
  raw/*.png               screen captures (2x scale)
  ThreeHoleBracketThick.pyto, .vtu, _optimized.stl     the GUI's own project, density and recovered part
  session_log.txt         the GUI's message log and the optimizer's iteration log
  session.json            iterations, objective history, captured iterations

Run (opens the GUI on the screen for a few minutes):
    cd docs/paper/fig_gen && QT_SCALE_FACTOR=2 python gui_session_threehole.py [--setup-only | --keep-recovered]
--keep-recovered re-records the GUI captures only (e.g. after a GUI change): the session's recovered STL, project,
density and logs are kept (the CNN smoothing is random, so a new recovery would change the verified part), the .vtu
the GUI re-exports is restored, and the logs go to session_rerecord_log.txt / session_rerecord.json.
"""
import json
import os
import shutil
import sys
import time

import numpy as np


def log(*a):
    """To the terminal, never to the GUI: while optimizing, the GUI redirects stdout into its Iterations log."""
    print(*a, file=sys.__stdout__, flush=True)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(REPO, "src"))
OUT = os.path.join(REPO, "Results", "GUISession_ThreeHole")
RAW = os.path.join(OUT, "raw")
STL_SRC = os.path.join(REPO, "Models", "ThreeHoleBracket", "ThreeHoleBracketThick.STL")
HOLES = {"top": (0.015, 0.065), "bottom": (0.015, 0.015), "right": (0.065, 0.015)}
CAPTURE_ITERATION = 15
VIEW = (0.42, -0.62, 1.0)                     # the view of Figs. 1 and 17
GEOMETRY_OPACITY = 0.45

os.makedirs(RAW, exist_ok=True)
from PyQt5 import QtCore, QtWidgets  # noqa: E402

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
import pyto.gui.PyTOGUI as G  # noqa: E402
G.StructuralLoadsWindow.show_load_values = False   # the load is shown by its arrows; no number label in the captures
from pyto.topopt.spec import ConstraintSpec, ObjectiveSpec, OptimizationSpec  # noqa: E402

# message boxes: shown without blocking (so they can be captured), kept for the log
BOXES = []


def _box(parent, title, text, *a, **k):
    box = QtWidgets.QMessageBox(parent)
    box.setWindowTitle(title)
    box.setText(text)
    box.setModal(False)
    box.show()
    BOXES.append(box)
    log(f"[{title}] {text}")
    return QtWidgets.QMessageBox.Ok


QtWidgets.QMessageBox.information = staticmethod(_box)
QtWidgets.QMessageBox.warning = staticmethod(_box)
QtWidgets.QMessageBox.critical = staticmethod(_box)


def pump(seconds):
    t = time.time() + seconds
    while time.time() < t:
        app.processEvents()
        time.sleep(0.02)


def pump_until(cond, timeout):
    t = time.time() + timeout
    while not cond() and time.time() < t:
        app.processEvents()
        time.sleep(0.05)
    return cond()


def grab(widget, name):
    """Screen capture of the widget's client area (screen grab, so the OpenGL 3D view is included)."""
    widget.raise_()
    widget.activateWindow()
    pump(0.8)
    top_left = widget.mapToGlobal(QtCore.QPoint(0, 0))
    pix = app.primaryScreen().grabWindow(0, top_left.x(), top_left.y(), widget.width(), widget.height())
    path = os.path.join(RAW, f"{name}.png")
    pix.save(path)
    log("captured", path, pix.width(), "x", pix.height())


def set_view(plotter, zoom=1.35):
    c = 0.5 * (np.array([0, 0, 0]) + np.array([0.1, 0.1, 0.02]))
    d = np.array(VIEW) / np.linalg.norm(VIEW)
    plotter.camera_position = [tuple(c + 0.3 * d), tuple(c), (0, 1, 0)]
    plotter.reset_camera()
    plotter.camera.zoom(zoom)
    plotter.render()


def grab_main(win, tw, name):
    """The main window (with the TopOpt Execute window, a child dialog that stays on top, hidden meanwhile) and the
    3D view from the plotter's own render buffer (unaffected by overlapping windows)."""
    if tw is not None:
        tw.hide()
        pump(0.5)
    grab(win, name)
    win.plotter.render()
    path = os.path.join(RAW, f"{name}_view.png")
    win.plotter.screenshot(path)
    log("captured", path, "actors:", sorted(win.plotter.actors))
    if tw is not None:
        tw.show()
        pump(0.3)


def hole_triangles(stl_geom, key):
    v = stl_geom.mesh.vectors
    c = v.mean(1)
    n = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    n /= np.linalg.norm(n, axis=1)[:, None]
    cx, cy = HOLES[key]
    r = np.hypot(c[:, 0] - cx, c[:, 1] - cy)
    return [int(i) for i in np.where((np.abs(r - 0.005) < 5e-4) & (np.abs(n[:, 2]) < 1e-6))[0]]


def main():
    stl = os.path.join(OUT, os.path.basename(STL_SRC))
    shutil.copy(STL_SRC, stl)
    keep = "--keep-recovered" in sys.argv
    vtu = os.path.join(OUT, "ThreeHoleBracketThick.vtu")
    if keep:
        shutil.copy(vtu, vtu + ".keep")
    for f in ("ThreeHoleBracketThick.vtu", "ThreeHoleBracketThick_optimized.stl"):
        if "--setup-only" not in sys.argv and not keep and os.path.exists(os.path.join(OUT, f)):
            os.remove(os.path.join(OUT, f))
    record = {"captures": {}}

    win = G.MainWindow()
    win.resize(700, 600)
    win.move(0, 0)
    win.show()
    pump(1.5)

    projects = G.ProjectsWindow(win)
    projects.load_geometry(stl)                                   # Geometry
    mat = G.MaterialWindow(win)                                   # Material: Steel
    mat.material_combo.setCurrentText("Steel")
    mat.apply_material()
    pump(0.5)
    supports = hole_triangles(win.stl_geom, "top") + hole_triangles(win.stl_geom, "bottom")
    projects.restore_structural_loads({"fixed_faces_indices": supports,          # Structural loads
                                       "load_faces_indices": [hole_triangles(win.stl_geom, "right")],
                                       "load_forces": [[0.0, -10000.0, 0.0]]})
    set_view(win.plotter)
    grab_main(win, None, "1_stl_supports_load")
    # Display Options -> geometry transparency: the load arrows act on the right hole's surface, inside the plate
    G.DisplayOptionsWindow(win).set_geometry_transparency(GEOMETRY_OPACITY)
    set_view(win.plotter, zoom=1.05)          # opening Display Options resets the camera; room for the force label
    grab_main(win, None, "1b_stl_supports_load_transparent")
    if "--setup-only" in sys.argv:            # only the set-up captures (the run's STL and logs are kept)
        app.quit()
        return
    G.DisplayOptionsWindow(win).set_geometry_transparency(1.0)

    projects.restore_topopt_options({"symmetry": {"x_symmetry": False, "y_symmetry": False, "z_symmetry": True},
                                     "other": {"keep_fixed_faces": True}})
    win.mesh_elements = 67000                                     # Analysis: mesh (about 63k elements, as Fig. 1)
    analysis = G.AnalysisWindow(win)
    analysis.generate_mesh()
    set_view(win.plotter)
    grab_main(win, None, "2_voxel_mesh")

    win.optimization_spec = OptimizationSpec(objective=ObjectiveSpec("Compliance()"),
                                             constraints=[ConstraintSpec("VolumeFraction()", "<=", 0.25)])
    win.open_structural_topopt_window()                          # TopOpt Execute
    tw = win.structural_topopt_window
    tw.resize(470, 700)                       # the window's minimum width is about 483 logical px
    tw.move(0, 660)
    pump(1.0)
    tw.formulation.setCurrentIndex(0)
    grab(tw, "3_topopt_objective")
    tw.formulation.setCurrentIndex(1)
    grab(tw, "4_topopt_constraints")
    tw.formulation.setCurrentIndex(0)

    tw.check_gradient()                                          # Check gradient
    pump_until(lambda: BOXES, 120)
    pump(0.5)
    box = BOXES[-1]
    box.move(720, 700)
    grab(box, "5_gradient_check")
    record["gradient_check"] = box.text()
    box.close()

    state = {"iteration": 0}
    tw.history_update.connect(lambda h: state.update(iteration=len(h.get("objective", []))))
    tw.live_topology_check.setChecked(True)
    win.plotter.remove_actor("geometry_info")     # Display Options 'geometry text' off: no clash with 'Iteration n'
    win.plotter.render()
    t0 = time.time()
    tw.start_optimization()                                      # Optimize
    pump_until(lambda: state["iteration"] >= CAPTURE_ITERATION, 1800)
    pump(1.0)
    set_view(win.plotter)
    record["captures"]["running_main"] = state["iteration"]
    grab_main(win, tw, "6_running_main")
    grab(tw, "7_running_topopt")
    pump_until(lambda: not tw.optimization_running, 7200)
    record["optimization_time_s"] = time.time() - t0
    pump(2.0)
    set_view(win.plotter)
    grab_main(win, tw, "8_final_main")
    grab(tw, "9_final_topopt")
    res = win.topopt_results
    record.update(iterations=res["iterations"], converged=res["converged"], objective_final=float(res["objective"]),
                  volume_fraction=float(res["volume_fraction"]),
                  objective_history=[float(np.ravel(v)[0]) for v in res["history"]["objective"]])
    if keep:
        shutil.move(vtu + ".keep", vtu)
        with open(os.path.join(OUT, "session_rerecord_log.txt"), "w") as f:
            f.write("=== GUI messages ===\n" + win.message_text.toPlainText() + "\n\n=== Iterations log ===\n"
                    + tw.formulation.log_view.toPlainText() + "\n")
        with open(os.path.join(OUT, "session_rerecord.json"), "w") as f:
            json.dump(record, f, indent=1)
        log("re-recorded:", {k: v for k, v in record.items() if k != "objective_history"})
        for w in app.topLevelWidgets():
            w.close()
        app.quit()
        return
    np.save(os.path.join(OUT, "density.npy"), res["density"])

    QtWidgets.QFileDialog.getSaveFileName = staticmethod(
        lambda *a, **k: (os.path.join(OUT, "ThreeHoleBracketThick.pyto"), ""))
    projects.save_project()                                      # Projects: save

    post = G.TopOptResultsWindow(win)                            # TopOpt Postprocess: STL recovery
    post.apply_recovery()
    pump(1.0)
    set_view(win.plotter)
    grab_main(win, tw, "10_recovered_stl")

    with open(os.path.join(OUT, "session_log.txt"), "w") as f:
        f.write("=== GUI messages ===\n" + win.message_text.toPlainText() + "\n\n=== Iterations log ===\n"
                + tw.formulation.log_view.toPlainText() + "\n")
    with open(os.path.join(OUT, "session.json"), "w") as f:
        json.dump(record, f, indent=1)
    log("done:", {k: v for k, v in record.items() if k != "objective_history"})
    for w in app.topLevelWidgets():
        w.close()
    app.quit()


if __name__ == "__main__":
    main()
