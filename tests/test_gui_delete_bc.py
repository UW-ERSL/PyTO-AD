"""Deleting applied boundary conditions in the GUI (Applied lists of the Structural / Thermal Loads windows, Clear in
the Body force window): data, picked faces, 3D actors, selection names, workflow flags and the project file."""
import contextlib
import io
import json
import os
import types

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PyQt5.QtWidgets")


@pytest.fixture()
def gui():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    import pyvista as pv
    import pyto.gui.PyTOGUI as G
    from pyto.io.stl_reader import STLGeom

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with contextlib.redirect_stdout(io.StringIO()):
        stl = STLGeom(os.path.join(root, "Models", "Cantilever", "Cantilever.STL"))
    parent = QtWidgets.QWidget()
    parent.message_text = QtWidgets.QTextEdit()
    parent.plotter = pv.Plotter(off_screen=True)
    parent.plotter.add_mesh(pv.PolyData(stl.mesh.vectors.reshape(-1, 3),
                                        np.hstack([np.full((len(stl.mesh.vectors), 1), 3),
                                                   np.arange(3 * len(stl.mesh.vectors)).reshape(-1, 3)])),
                            name="stl_geometry")
    parent.settings = G.Settings()
    parent.stl_geom, parent.hex_mesh = stl, None
    parent.icons = {}
    parent.set_sidebar_icon = lambda name, icon: parent.icons.__setitem__(name, icon)
    for name in ("update_LivVar", "set_nested_value", "handle_state_change", "handle_geometry_loaded",
                 "handle_material_defined", "handle_topopt_options_defined", "handle_structural_loads_change",
                 "handle_thermal_loads_applied", "update_workflow_step", "get_button_name_for_step",
                 "refresh_loads_state", "update_highlights", "remove_highlight_actor", "get_highlight_ids",
                 "create_highlight_mesh", "add_highlight_mesh", "restore_camera_and_render", "show_candidate_nodes",
                 "node_selection_preview", "update_point_markers"):
        setattr(parent, name, types.MethodType(getattr(G.MainWindow, name), parent))
    parent.LivVar, parent.WORKFLOW_STEPS = G.MainWindow.create_initial_state(parent), G.MainWindow.WORKFLOW_STEPS
    parent.notify_display_options_update = lambda *a, **k: None
    parent.highlight_mode, parent.highlight_actor, parent.selected_points, parent.selected_regions = "coarse", None, [], []
    parent.constraint_data, parent.force_data, parent.constrained_triangles = [], [], set()
    parent.force_actors, parent.constraint_actors, parent.torque_actors = [], [], []
    parent.thermal_loads_window, parent.body_force, parent.topopt_options = None, None, {}
    cen = stl.mesh.vectors.mean(axis=1)
    face = lambda axis, end: [int(i) for i in np.where(np.abs(cen[:, axis] - end(cen[:, axis])) < 1e-9)[0]]
    faces = {"x0": face(0, np.min), "x1": face(0, np.max), "y1": face(1, np.max), "z1": face(2, np.max)}
    yield app, G, parent, faces
    parent.plotter.close()


def _select(parent, triangles):
    picked = set(triangles)
    parent.stl_geom.tri_highlight[:] = [i in picked for i in range(len(parent.stl_geom.tri_highlight))]


def _apply(window, load_type, triangles, **values):
    _select(window.parent, triangles)
    window.load_type.setCurrentText(load_type)
    for axis, v in values.items():
        (window.torque_magnitude_spin if axis == "torque" else window.force_spinboxes[axis]).setValue(v)
    window.on_apply_clicked()


def _button(window, text):
    return next(b for b in window.findChildren(QtWidgets.QPushButton) if b.text() == text)


def _drawn(parent, actors):
    return all(any(a is b for b in parent.plotter.actors.values()) for a in actors)


def test_delete_structural_loads(gui, monkeypatch, tmp_path):
    app, G, parent, faces = gui
    w = G.StructuralLoadsWindow(parent)
    _apply(w, "Fixed XYZ", faces["x0"])
    _apply(w, "Force", faces["x1"], Y=-100.0)
    _apply(w, "Force", faces["y1"], Z=50.0)
    _apply(w, "Torque", faces["z1"], torque=5.0)
    rows = [w.applied_list.item(i).text() for i in range(w.applied_list.count())]
    assert [r.split(":")[0] for r in rows] == ["Support1", "Load1", "Load2", "Load3"]
    assert "Fixed XYZ" in rows[0] and "Torque" in rows[3]
    assert parent.LivVar["structural_loads"]["applied"] and parent.icons["Structural Loads"] == "check"
    assert len(parent.force_actors) == 4 and len(parent.torque_actors) == 3     # arrows + labels; arc, tip, label
    first_force = list(parent.force_actors[:2])

    # Delete Load1 through the list and its button: Load2 / Load3 become Load1 / Load2
    w.applied_list.item(1).setSelected(True)
    _button(w, "Delete selected").click()
    assert [f.get("type") for f in parent.force_data] == ["force_xyz", "torque"]
    assert parent.force_data[0]["force_z"] == 50.0
    assert len(parent.force_actors) == 2 and len(parent.torque_actors) == 3
    assert _drawn(parent, parent.force_actors + parent.torque_actors + parent.constraint_actors)
    assert not any(a is b for a in first_force for b in parent.plotter.actors.values())
    names = G.StructuralTopOptWindow.selection_names(types.SimpleNamespace(parent=parent))
    assert {"Support1", "Load1", "Load2"} <= set(names) and "Load3" not in names
    assert w.applied_list.item(1).text().startswith("Load1: Force (+0, +0, +50)")
    assert parent.LivVar["structural_loads"]["applied"]

    # The project file is written from the same lists: one face load left, no stale one
    monkeypatch.setattr(G.QtWidgets.QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(tmp_path / "p.pyto"), "")))
    G.ProjectsWindow(parent).save_project()
    bc = json.loads((tmp_path / "p.pyto").read_text())["structuralBC"]
    assert bc["load_forces"] == [[0.0, 0.0, 50.0]] and sorted(bc["fixed_faces_indices"]) == sorted(faces["x0"])

    # Delete the support: its faces can be picked again, the step goes back to not ready
    w.delete_loads([("constraint", 0)])
    assert parent.constraint_data == [] and parent.constrained_triangles == set() and parent.constraint_actors == []
    assert not parent.LivVar["structural_loads"]["fixed_constraints"]
    assert not parent.LivVar["structural_loads"]["applied"]
    assert parent.icons["Structural Loads"] == "arrow" and parent.icons["Analysis"] == "cross"

    # Delete all (confirmed)
    monkeypatch.setattr(G.QtWidgets.QMessageBox, "question", staticmethod(lambda *a, **k: G.QtWidgets.QMessageBox.Yes))
    _button(w, "Delete all").click()
    assert parent.force_data == [] and parent.force_actors == [] and parent.torque_actors == []
    assert w.applied_list.count() == 0 and not parent.LivVar["structural_loads"]["forces_applied"]
    assert G.StructuralTopOptWindow.selection_names(types.SimpleNamespace(parent=parent)) == []


def test_delete_thermal_loads(gui, monkeypatch):
    app, G, parent, faces = gui
    w = G.ThermalLoadsWindow(parent)
    parent.thermal_loads_window = w
    for kind, tris, value in (("Temperature", faces["x0"], 300.0), ("Heat Flux", faces["x1"], 1000.0),
                              ("Total Heat", faces["y1"], 20.0)):
        _select(parent, tris)
        w.thermal_type.setCurrentText(kind)
        getattr(w, w.THERMAL_TYPES[kind]["value_field"]).setValue(value)
        w.on_apply_clicked()
    assert [w.applied_list.item(i).text().split(":")[0] for i in range(3)] == ["FixedTemp1", "HeatSource1", "HeatSource2"]
    assert parent.LivVar["thermal_loads"]["applied"]

    # A Thermal Loads window opened again keeps (and lists) the loads stored on the previous one
    w = G.ThermalLoadsWindow(parent)
    parent.thermal_loads_window = w
    assert w.applied_list.count() == 3 and len(w.heat_source_actors) == 2

    w.delete_thermal_loads([("heat_sources", 0)])
    assert w.thermal_loads["heat_sources"] == [] and len(w.thermal_loads["total_heat_sources"]) == 1
    assert w.heat_source_actors == [] and len(w.total_heat_actors) == 2 and len(w.fixed_temp_actors) == 2
    assert _drawn(parent, w.total_heat_actors + w.fixed_temp_actors)
    assert w.applied_list.item(1).text().startswith("HeatSource1: Total Heat")
    names = G.StructuralTopOptWindow.selection_names(types.SimpleNamespace(parent=parent))
    assert "HeatSource1" in names and "HeatSource2" not in names

    monkeypatch.setattr(G.QtWidgets.QMessageBox, "question", staticmethod(lambda *a, **k: G.QtWidgets.QMessageBox.Yes))
    _button(w, "Delete all").click()
    assert not any(w.thermal_loads.values()) and w.applied_list.count() == 0
    assert not parent.LivVar["thermal_loads"]["applied"] and parent.icons["Thermal Loads"] == "arrow"
    assert parent.icons["Analysis"] == "cross"


def test_clear_body_force(gui):
    app, G, parent, faces = gui
    del parent.body_force                                  # as in the main window: created by the window
    w = G.BodyForceWindow(parent)
    w.x_edit.setText("0"), w.y_edit.setText("-9.81"), w.z_edit.setText("0")
    w.apply_body_force()
    assert "body_force_arrow" in parent.plotter.actors
    w = G.BodyForceWindow(parent)
    _button(w, "Clear").click()
    assert parent.body_force == {"X": 0.0, "Y": 0.0, "Z": 0.0} and parent.body_force_actors == []
    assert "body_force_arrow" not in parent.plotter.actors and "body_force_label" not in parent.plotter.actors
