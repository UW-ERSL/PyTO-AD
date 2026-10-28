"""The real TopOpt Execute window, driven headlessly (offscreen Qt): picked faces -> selections -> FE problem ->
formulation -> optimization -> live history (plan S5/S6)."""
import contextlib
import io
import os
import types

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PyQt5.QtWidgets")


@pytest.fixture(scope="module")
def window_parent():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from pyto.core.hex_mesher import HexMesher
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mesher = HexMesher()
    with contextlib.redirect_stdout(io.StringIO()):
        mesher.createMeshFromSTLFile(os.path.join(root, "Models", "Cantilever", "Cantilever.STL"), 2000)
        mesher.createEdofMatStructural()
        mesher.createEdofMatThermal()
    cen = mesher.stlGeom.mesh.vectors.mean(axis=1)
    x0 = [int(i) for i in np.where(np.abs(cen[:, 0] - cen[:, 0].min()) < 1e-9)[0]]
    x1 = [int(i) for i in np.where(np.abs(cen[:, 0] - cen[:, 0].max()) < 1e-9)[0]]
    parent = QtWidgets.QWidget()
    parent.message_text = QtWidgets.QTextEdit()
    parent.fe_solver, parent.hex_mesh, parent.stl_geom = None, mesher, mesher.stlGeom
    parent.applied_material = {"properties": {"Young's Modulus": 2.1e11, "Poisson's Ratio": 0.3, "Density": 7850,
                                              "Thermal Conductivity": 50.0, "Specific Heat Capacity": 460.0,
                                              "Thermal Expansion": 12e-6, "Price": 1.0, "Yield Strength": 2.5e8}}
    parent.constrained_triangles, parent.topopt_options, parent.optimization_spec = set(x0), {}, None
    parent.faces = (x0, x1)
    return app, parent


def _run(app, parent, spec, structural=True, thermal=None):
    import pyto.gui.PyTOGUI as G
    from pyto.gui.optimization_setup import display_objective, prepare_run
    x0, x1 = parent.faces
    parent.constraint_data = [{"type": "Fixed XYZ", "triangles": x0}] if structural else []
    parent.force_data = ([{"type": "force", "triangles": x1, "force_x": 0.0, "force_y": -1e4, "force_z": 0.0}]
                         if structural else [])
    parent.thermal_loads_window = types.SimpleNamespace(thermal_loads=thermal) if thermal else None
    w = G.StructuralTopOptWindow(parent)
    w.optimization_done.disconnect()
    done = {}
    w.optimization_done.connect(lambda ok, msg, h, u, fe: done.update(ok=ok, msg=msg, h=h))
    w.formulation.set_spec(spec)
    fe, physics, selections = w.build_problem()
    w.formulation.set_context(physics, selections)
    plan = prepare_run(w.formulation.get_spec(), physics, selections)
    assert not plan.errors, plan.issues
    w.spec, w.physics = w.formulation.get_spec(), physics
    with contextlib.redirect_stdout(io.StringIO()):
        w.run_optimization(plan, fe, w.spec)
    app.processEvents()                                   # deliver the queued live-plot updates
    return w, fe, physics, selections, done, display_objective(w.spec, done["h"]["objective"])


def test_structural_user_objective_through_the_window(window_parent):
    from pyto.topopt.spec import ConstraintSpec, MethodSpec, ObjectiveSpec, OptimizationSpec
    app, parent = window_parent
    spec = OptimizationSpec(objective=ObjectiveSpec("Displacement(Load1, y, mean)", "maximize"),
                            constraints=[ConstraintSpec("VolumeFraction()", "<=", 0.5),
                                         ConstraintSpec("StressFailureFactor()", "<=", 2.0)],
                            method=MethodSpec("MMA", 4))
    w, fe, physics, sel, done, J = _run(app, parent, spec)
    assert physics == "structural" and type(fe).__name__ == "HexStructuralFEA"
    assert {"Support1", "Supports", "Load1", "Loads"} <= set(sel) and len(sel["Load1"]["nodes"]) > 0
    assert done["ok"] and J[-1] > J[0]                    # maximized (negative) deflection increased
    # one live point per iteration; the final-design point is added by optimization_completed (disconnected here)
    assert len(w.history_figure.axes[0].lines[0].get_xdata()) == len(J) - 1


def test_thermal_problem_through_the_window(window_parent):
    from pyto.topopt.spec import MethodSpec, ObjectiveSpec, OptimizationSpec
    app, parent = window_parent
    x0, x1 = parent.faces
    thermal = {"fixed_temps": [{"triangles": x0, "temperature": 300.0}],
               "total_heat_sources": [{"triangles": x1, "total_heat": 50.0}]}
    spec = OptimizationSpec(objective=ObjectiveSpec("Temperature(All, pnorm)"), method=MethodSpec("MMA", 4))
    w, fe, physics, sel, done, J = _run(app, parent, spec, structural=False, thermal=thermal)
    assert physics == "thermal" and type(fe).__name__ == "HexThermalFEA" and set(sel) == {"FixedTemp1", "HeatSource1"}
    assert done["ok"] and J[-1] < J[0]


def test_final_design_fields(window_parent, tmp_path):
    """After an optimization: summary on the solid region, and the TopOpt fields render (structural + thermal)."""
    from pyto.gui.optimization_setup import final_design_summary
    from pyto.topopt.spec import MethodSpec, ObjectiveSpec, OptimizationSpec
    app, parent = window_parent
    w, fe, physics, sel, done, J = _run(app, parent, OptimizationSpec(method=MethodSpec("MMA", 10)))
    s = final_design_summary(fe, physics)
    solid = np.asarray(fe.mesh.elemPseudoDensity) > 0.5
    assert 0.3 < s["solid_fraction"] < 0.7 and s["max_temperature"] is None
    assert s["max_von_mises"] == pytest.approx(float(np.max(np.asarray(fe.vonMisesStress)[solid])))
    all_nodes_max = float(fe.deformation.max())
    assert 0 < s["max_displacement"] <= all_nodes_max                 # solid region only
    fe.plot_deformation(save_path=str(tmp_path / "def.png"))
    fe.plot_vonMisesStress(save_path=str(tmp_path / "vm.png"))
    assert (tmp_path / "def.png").exists() and (tmp_path / "vm.png").exists()

    x0, x1 = parent.faces
    thermal = {"fixed_temps": [{"triangles": x0, "temperature": 300.0}],
               "total_heat_sources": [{"triangles": x1, "total_heat": 50.0}]}
    w, fe, physics, sel, done, J = _run(app, parent, OptimizationSpec(objective=ObjectiveSpec("ThermalCompliance()"),
                                                                     method=MethodSpec("MMA", 4)),
                                        structural=False, thermal=thermal)
    s = final_design_summary(fe, physics)
    assert s["max_displacement"] is None and s["max_temperature"] >= 300.0
    fe.plot_elem_field(s["element_temperature"], title="Temperature", save_path=str(tmp_path / "T.png"))
    assert (tmp_path / "T.png").exists()                              # the plot_elem_field wrapper forwards save_path


def test_optimized_design_buttons_after_completion(window_parent, tmp_path):
    """The real completion handler: results stored (also when not converged), result buttons enabled per physics,
    and every field drawn into the main view."""
    import shutil
    import pyvista as pv
    import pyto.gui.PyTOGUI as G
    from pyto.topopt.spec import MethodSpec, ObjectiveSpec, OptimizationSpec

    class ViewStub(pv.Plotter):                        # the GUI's QtInteractor: show() just redraws
        def show(self, *a, **k):
            self.render()

    app, parent = window_parent
    parent.plotter = ViewStub(off_screen=True)
    parent.update_LivVar = lambda *a, **k: None
    parent.set_sidebar_icon = lambda *a, **k: None
    stl = tmp_path / "Cantilever.STL"                   # completion exports a .vtu next to the STL
    shutil.copy(parent.stl_geom.file_path, stl)
    real_path = parent.stl_geom.file_path
    parent.stl_geom.file_path = str(stl)
    try:
        for spec, structural, thermal, expected in [
            (OptimizationSpec(method=MethodSpec("MMA", 3)), True, None,           # stops at the iteration limit
             ["Density", "Deformation", "Von Mises stress"]),
            (OptimizationSpec(objective=ObjectiveSpec("Temperature(All, pnorm)"), method=MethodSpec("MMA", 3)), False,
             {"fixed_temps": [{"triangles": parent.faces[0], "temperature": 300.0}],
              "total_heat_sources": [{"triangles": parent.faces[1], "total_heat": 50.0}]},
             ["Density", "Temperature"]),
        ]:
            parent.topopt_results = None
            w, fe, physics, sel, done, J = _run(app, parent, spec, structural=structural, thermal=thermal)
            w.optimization_completed(done["ok"], done["msg"], done["h"], None, fe)   # the real handler
            assert parent.topopt_results["fe_solver"] is fe and parent.topopt_results["physics"] == physics
            assert G.topopt_fields_available(parent.topopt_results) == expected
            assert {f: b.isEnabled() for f, b in w.field_buttons.items()} == {f: f in expected for f in G.TOPOPT_FIELDS}
            for field in G.TOPOPT_FIELDS:
                msg = G.show_topopt_field(parent, field)
                if field in expected:
                    assert msg is None and len(parent.plotter.actors) > 0, field
                else:
                    assert "not available" in msg
            analysis = G.AnalysisWindow(parent)                 # Analysis window: 'Optimized design' results
            analysis.design_combo.setCurrentText("Optimized design")
            assert {f: b.isEnabled() for f, b in analysis.field_buttons.items()} == \
                {f: f in expected for f in G.RESULT_FIELDS}         # no Density button there
            assert not hasattr(G.TopOptResultsWindow(parent), "field_buttons")   # Postprocess: STL recovery only
    finally:
        parent.stl_geom.file_path = real_path
        parent.plotter.close()


def test_analysis_window_initial_design_results(window_parent, tmp_path):
    """Structural Analysis (gravity applied once), Include Thermal Effect (coupled), and the Initial / Optimized
    result buttons of the Analysis window, through the window's own methods."""
    import pyvista as pv
    import pyto.gui.PyTOGUI as G

    class ViewStub(pv.Plotter):
        def show(self, *a, **k):
            self.render()

    app, parent = window_parent
    x0, x1 = parent.faces
    parent.plotter = ViewStub(off_screen=True)
    parent.update_LivVar = lambda *a, **k: None
    parent.constraint_data = [{"type": "Fixed XYZ", "triangles": x0}]
    parent.force_data = [{"type": "force", "triangles": x1, "force_x": 0.0, "force_y": -1e4, "force_z": 0.0}]
    parent.body_force = {"X": 0.0, "Y": -9.81, "Z": 0.0}
    parent.fe_solver = parent.thermal_fe_solver = parent.topopt_results = None
    parent.thermal_loads_window = types.SimpleNamespace(thermal_loads={
        "fixed_temps": [{"triangles": x0, "temperature": 300.0}, {"triangles": x1, "temperature": 340.0}]})
    try:
        w = G.AnalysisWindow(parent)
        assert not any(b.isEnabled() for b in w.field_buttons.values())            # nothing analysed yet
        w.solver_combo.setCurrentText("SPSOLVE")
        with contextlib.redirect_stdout(io.StringIO()):
            w.run_structural_analysis()
        fe = parent.fe_solver
        assert type(fe).__name__ == "HexStructuralFEA"
        mesh = fe.mesh
        volume = mesh.num_elems * float(np.prod(mesh.elem_size))
        fy = float(fe.total_force.reshape(-1, 3)[:, 1].sum())
        assert fy == pytest.approx(-1e4 - 7850 * 9.81 * volume, rel=1e-9)          # gravity counted once
        assert {f: b.isEnabled() for f, b in w.field_buttons.items()} == \
            {"Deformation": True, "Von Mises stress": True, "Temperature": False}
        for field in ("Deformation", "Von Mises stress"):
            assert G.show_result_field(parent, "initial", field) is None and parent.plotter.actors
        assert "Thermal Analysis" in G.show_result_field(parent, "initial", "Temperature")

        # Thermo-Structural Analysis button -> coupled analysis, temperature of the initial design available
        assert w.thermostructural_button.text() == "Thermo-Structural Analysis"
        w.temp_spin.setValue(300.0)
        with contextlib.redirect_stdout(io.StringIO()):
            w.thermostructural_button.click()
        assert type(parent.fe_solver).__name__ == "ThermoStructuralFEA"
        assert all(b.isEnabled() for b in w.field_buttons.values())
        assert G.show_result_field(parent, "initial", "Temperature") is None
        assert float(np.max(np.asarray(parent.fe_solver.vonMisesStress))) > float(np.max(np.asarray(fe.vonMisesStress)))

        # without thermal loads the button refuses (warning) and keeps the previous results
        warned = []
        saved_warning, saved_loads = G.QtWidgets.QMessageBox.warning, parent.thermal_loads_window
        G.QtWidgets.QMessageBox.warning = lambda *a, **k: warned.append(a[2])
        parent.thermal_loads_window = None
        previous = parent.fe_solver
        try:
            w.thermostructural_button.click()
        finally:
            G.QtWidgets.QMessageBox.warning, parent.thermal_loads_window = saved_warning, saved_loads
        assert warned and "thermal loads" in warned[0] and parent.fe_solver is previous

        # after an optimization, the initial-design view still draws the full design
        parent.topopt_results = {"fe_solver": parent.fe_solver, "physics": "thermo-structural",
                                 "density": np.where(np.arange(mesh.num_elems) % 2 == 0, 1.0, 0.0)}
        assert G.show_result_field(parent, "optimized", "Deformation") is None
        assert np.array_equal(mesh.elemPseudoDensity, parent.topopt_results["density"])
        assert G.show_result_field(parent, "initial", "Deformation") is None
        assert np.all(mesh.elemPseudoDensity == 1.0)
    finally:
        parent.plotter.close()
        parent.body_force = None


def test_topology_every_iteration_option(window_parent):
    """'Show topology every iteration' is off by default (no redraws); when on, MMA redraws the design once per
    iteration in the main view (via iteration_callback)."""
    import pyvista as pv
    import pyto.gui.PyTOGUI as G
    from pyto.topopt.spec import MethodSpec, OptimizationSpec

    class ViewStub(pv.Plotter):
        def show(self, *a, **k):
            self.render()

    app, parent = window_parent
    parent.plotter = ViewStub(off_screen=True)
    try:
        counts = {}

        def run(spec, live):
            import pyto.gui.PyTOGUI as GG
            original = GG.StructuralTopOptWindow.run_optimization

            def spy(self, plan, fe, sp):
                self.optimization_update.connect(lambda mesh, it: counts.setdefault(sp.method.name, []).append(
                    (mesh.n_cells, it)))
                self.live_topology_check.setChecked(live)
                return original(self, plan, fe, sp)
            GG.StructuralTopOptWindow.run_optimization = spy
            try:
                return _run(app, parent, spec)
            finally:
                GG.StructuralTopOptWindow.run_optimization = original

        w, fe, physics, sel, done, J = run(OptimizationSpec(method=MethodSpec("MMA", 4)), live=False)
        assert G.StructuralTopOptWindow(parent).live_topology_check.isChecked() is False   # default: unchecked
        assert "MMA" not in counts                                     # off: no redraws
        w, fe, physics, sel, done, J = run(OptimizationSpec(method=MethodSpec("MMA", 4)), live=True)
        updates = counts["MMA"]
        assert len(updates) == len(J) - 1 and all(n > 0 for n, _ in updates)   # one per iteration (not the final eval)
        assert [it for _, it in updates] == list(range(len(updates)))
        # the main view shows the last drawn design: a mesh with cell data 'density', and the iteration label
        meshes = [a.GetMapper().GetInput() for a in parent.plotter.actors.values()
                  if hasattr(a, "GetMapper") and a.GetMapper() is not None and a.GetMapper().GetInput() is not None]
        assert any(m.GetCellData().HasArray("density") and m.GetNumberOfCells() == updates[-1][0] for m in meshes)
        assert any(type(a).__name__.endswith(("TextActor", "CornerAnnotation", "Text"))
                   for a in parent.plotter.actors.values())
    finally:
        parent.plotter.close()


def test_heat_flux_is_multiplied_by_the_face_area(window_parent):
    """GUI 'Heat Flux' (W per display area) must give the heat of 'Total Heat' = flux x face area, in any unit system."""
    import pyto.gui.PyTOGUI as G
    app, parent = window_parent
    x0, x1 = parent.faces
    area = float(np.sum(np.asarray(parent.stl_geom.tri_areas)[x1]))          # m^2 (STL read as metres)
    assert G.heat_flux_total(2.0, [0.5, 0.25]) == 1.5
    assert G.heat_flux_total(2.0, [1e-6], length_units_per_m=1000.0) == pytest.approx(2.0)   # 2 W/mm^2 on 1 mm^2

    def thermal_force(loads, unit_system=None):
        parent.constraint_data, parent.force_data = [], []
        parent.thermal_loads_window = types.SimpleNamespace(thermal_loads=dict(
            fixed_temps=[{"triangles": x0, "temperature": 300.0}], **loads))
        saved, saved_fe = getattr(parent, "settings", None), parent.fe_solver
        parent.fe_solver = None                      # else the window clears the (closed) plotter of earlier tests
        if unit_system:
            parent.settings = G.Settings()
            parent.settings.unit_system = unit_system
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                fe, physics, _ = G.StructuralTopOptWindow(parent).build_problem()
        finally:
            parent.fe_solver = saved_fe
            if unit_system:
                parent.settings = saved
        return np.asarray(fe.bc.force)

    total = thermal_force({"total_heat_sources": [{"triangles": x1, "total_heat": 50.0}]})
    flux_si = thermal_force({"heat_sources": [{"triangles": x1, "heat_flux": 50.0 / area}]}, "MKS")
    flux_mm = thermal_force({"heat_sources": [{"triangles": x1, "heat_flux": 50.0 / area / 1e6}]}, "mmKS")
    assert total.sum() == pytest.approx(50.0)
    np.testing.assert_allclose(flux_si, total, rtol=1e-12)
    np.testing.assert_allclose(flux_mm, total, rtol=1e-9)
