"""The GUI and demo scripts must call the FE solvers and optimization drivers with their current signatures.

The GUI cannot be exercised headlessly (VTK needs a display), and these call sites broke silently when the drivers
moved to the single-solver torch API. This parses the GUI sources and binds every call's arguments to the current
signature, so a future API change fails here instead of when a user clicks a button.
"""
import ast
import inspect
import os

import pytest
import torch

GUI_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "pyto", "gui")


def _callables():
    from pyto.physics.structural.hex_structural_fea import HexStructuralFEA
    from pyto.physics.thermal.hex_thermal_fea import HexThermalFEA
    from pyto.topopt.drivers.mma import topopt_mma
    from pyto.topopt.drivers.oc import topopt_optimality_criteria
    return {f.__name__: f for f in (topopt_mma, topopt_optimality_criteria,
                                    HexStructuralFEA, HexThermalFEA)}


@pytest.mark.parametrize("source", ["PyTOGUI.py", "PyTO_demos.py"])
def test_gui_calls_match_current_signatures(source):
    funcs = _callables()
    tree = ast.parse(open(os.path.join(GUI_DIR, source)).read())
    mismatches, checked = [], 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", None)
        if name not in funcs:
            continue
        checked += 1
        try:
            inspect.signature(funcs[name]).bind(*[None] * len(node.args),
                                                **{k.arg: None for k in node.keywords if k.arg})
        except TypeError as e:
            mismatches.append(f"{source}:{node.lineno} {name}: {e}")
    assert checked > 0
    assert not mismatches, "\n".join(mismatches)


def test_structural_solve_without_design_is_a_solid_analysis(structural_fe_solver):
    # The GUI's "run structural analysis" calls solve() with no design (as HexThermalFEA.solve allows).
    fe = structural_fe_solver
    u_default = fe.solve()
    u_solid = fe.solve(torch.ones(fe.mesh.num_elems, dtype=torch.float64))
    assert torch.allclose(u_default, u_solid)


def test_plot_temperature_accepts_grad_tracked_solution(thermal_fe_solver, tmp_path):
    # After an optimization the thermal solution carries autograd history; plotting must still work.
    from pyto.autodiff.material_model import MaterialModel
    fe = thermal_fe_solver
    assert fe.plot_temperature(save_path=str(tmp_path / "none.png")) is None   # nothing solved yet
    x = torch.full((fe.mesh.num_elems,), 0.5, dtype=torch.float64, requires_grad=True)
    fe.solve(x, MaterialModel.SIMP)
    fe.plot_temperature(save_path=str(tmp_path / "T.png"))
    assert (tmp_path / "T.png").exists()


def test_multibody_stl_with_open_edges_meshes_and_empty_mesh_raises(capsys):
    # SpindleAssembly: 2 bodies, 200 open edges. The multi-body mesher used to print "Model has open edges" and
    # return with no mesh, and the GUI then failed with "HexMesher has no attribute elemArray".
    from pyto.core.hex_mesher import HexMesher
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mesher = HexMesher()
    mesher.createMeshFromSTLFile(os.path.join(root, "Models", "SpindleAssembly", "SpindleAssembly.STL"), 3000)
    mesher.createEdofMatStructural()
    assert mesher.num_elems > 0 and len(set(mesher.elemComponentId)) == 2
    assert "open edges" in capsys.readouterr().out   # the user is still warned
    empty = HexMesher()
    with pytest.raises(ValueError, match="produced no elements"):
        empty._check_mesh_created("broken.stl")


def test_gui_body_force_reaches_the_solver_input():
    # The Body force window stores accelerations (m/s^2) in parent.body_force; they used to never reach a solver.
    # They must become the same element forces the benchmarks use (GravityPlate: rho * g * element volume).
    import numpy as np
    from types import SimpleNamespace
    from pyto.core.bc import elem_body_force_from_acceleration
    from pyto.examples_benchmarks.hex_structural_examples import StructuralExamples, getStructuralProblem
    mesh, mat, bc, ref_force = getStructuralProblem(StructuralExamples.GravityPlate, nDOFDesired=2000)
    force = elem_body_force_from_acceleration(mesh, mat.mass_density, [0.0, -9.81, 0.0])
    assert np.allclose(force, ref_force)
    assert elem_body_force_from_acceleration(mesh, mat.mass_density, [0, 0, 0]) is None
    pytest.importorskip("PyQt5")
    from pyto.gui.PyTOGUI import gui_elem_body_force
    window = SimpleNamespace(body_force={"X": 0.0, "Y": -9.81, "Z": 0.0})
    assert np.allclose(gui_elem_body_force(window, mesh, mat), ref_force)
    assert gui_elem_body_force(SimpleNamespace(), mesh, mat) is None      # Body force window never opened


def test_run_gui_launcher_check():
    """run_gui.py --check: with the GUI's modules it runs directly; from a Python without them (system python) it
    restarts itself in the PyToLib conda environment."""
    import shutil
    import subprocess
    import sys
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTO_GUI_RELAUNCHED")}
    r = subprocess.run([sys.executable, os.path.join(root, "run_gui.py"), "--check"], capture_output=True, text=True,
                       timeout=120, env=env)
    assert r.returncode == 0 and r.stdout.startswith("OK:"), r.stdout + r.stderr
    bare = "/usr/bin/python3"
    has_gui = subprocess.run([bare, "-c", "import PyQt5"], capture_output=True).returncode == 0 \
        if os.path.isfile(bare) else True
    if has_gui or not (os.environ.get("CONDA_EXE") or shutil.which("conda")):
        pytest.skip("needs a Python without the GUI modules and conda")
    r = subprocess.run([bare, os.path.join(root, "run_gui.py"), "--check"], capture_output=True, text=True,
                       timeout=180, env=env)
    assert r.returncode == 0 and "OK:" in r.stdout and "(environment PyToLib)" in r.stdout, r.stdout + r.stderr
