"""Smoke tests for the MMA driver: does a couple of iterations
run at all, on both physics, without exceptions?

The xfail cases here are not aspirational -- each one was verified to fail
with exactly the stated exception before being marked, and documents a real,
currently-broken path found during the file-restructuring plan's Phase 0.
They are the "did I fix it" signal for later phases (see
~/.claude/plans/cheerful-dazzling-seahorse.md, Phase 4/5): once a phase's
fix lands, its xfail should start showing as an unexpected pass (XPASS) and
be promoted to a normal passing test.
"""
import numpy as np
import pytest
import torch

import pyto.topopt.drivers.mma as topopt_mma
from pyto.topopt.common import FEA_MODE


def test_mma_structural_runs(structural_fe_solver, structural_problem):
    _, _, _, _, to_params = structural_problem
    sol, history, success, error_msg, n_feas = topopt_mma.topopt_mma(
        fe_solver=structural_fe_solver,
        to_params=to_params,
        maxMMAIterations=2,
        print_progress=False,
        plot_progress=False,
    )
    assert len(history["objective"]) >= 2
    assert np.isfinite(history["objective"][-1])


def test_mma_thermal_runs(thermal_fe_solver, thermal_problem):
    # Was xfail through Phase 3 (HexThermalFEA.solve() was pure NumPy/SciPy);
    # fixed in Phase 4 by adding get_thermal_material_model_scaling_torch and
    # rewriting HexThermalFEA.solve()/postprocess()/get_thermoelastic_force()
    # to be torch-native, mirroring HexStructuralFEA.
    _, _, _, _, to_params = thermal_problem
    sol, history, success, error_msg, n_feas = topopt_mma.topopt_mma(
        fe_solver=thermal_fe_solver,
        to_params=to_params,
        maxMMAIterations=2,
        print_progress=False,
        plot_progress=False,
    )
    assert len(history["objective"]) >= 2
    assert np.isfinite(history["objective"][-1])


def test_binarization_acceptable_rule():
    from pyto.topopt.drivers._shared import binarization_acceptable
    assert binarization_acceptable(17.4, 25.5)                      # 0/1 stiffer than the grey design: keep it
    assert binarization_acceptable(40.0, 25.5)                      # worse, within 100 %
    assert not binarization_acceptable(1.1e10, 1539.0)              # load/heat path cut
    assert not binarization_acceptable(float("nan"), 1.0)
    assert binarization_acceptable(-0.9, -1.0)                      # maximized (negated) objective, small loss
    assert not binarization_acceptable(0.5, -1.0)                   # ... sign flipped: rejected
    assert not binarization_acceptable(-0.3, -1.0)                  # maximized value fell below half
    assert not binarization_acceptable(10.0, 100.0, sign=-1.0)      # prescribed-temperature problem (maximized J)
    assert binarization_acceptable(80.0, 100.0, sign=-1.0)


def test_mma_keeps_continuous_design_when_binarization_cuts_the_heat_path():
    """Thin plate, point heat source, default conductivity penalty 1: the design stays grey and the volume-preserving
    threshold disconnects source from sink (thermal compliance 1.5e3 -> 1.1e10). MMA must return the continuous
    design and say why, instead of reporting the broken 0/1 value."""
    import contextlib
    import io
    import pyto.autodiff.sparse_solve as sparse_solve
    import pyto.core.bc as bound_cond
    import pyto.core.mat_lib as mat_lib
    from pyto.core.hex_mesher import HexMesher
    from pyto.physics.thermal.hex_thermal_fea import HexThermalFEA
    from pyto.topopt.common import TOParams, TO_QOI

    mesh = HexMesher()
    mesh.grid_mesh(num_elems=(30, 30, 1), elem_size=(0.1 / 30, 0.1 / 30, 0.005))
    mesh.createEdofMatThermal()
    xyz = mesh.node_xyz
    sink = np.where(xyz[:, 1] > xyz[:, 1].max() - 1e-9)[0]
    source = np.where((xyz[:, 1] < 1e-9) & (np.abs(xyz[:, 0] - 0.05) < 0.005))[0]
    heat = np.zeros(mesh.num_nodes)
    heat[source] = 10.0 / len(source)
    bc = bound_cond.BC(force=heat, fixed_dofs=sink, dirichlet_values=np.full(len(sink), 20.0))
    fe = HexThermalFEA(mesh=mesh, mat_prop=mat_lib.get_material("Steel"), bc=bc, solver=sparse_solve.Solvers.SPSOLVE)
    p = TOParams()
    p.Objective = (TO_QOI.COMPLIANCE, None)
    p.Constraints = [(TO_QOI.VOLUME_FRACTION, None, 0.3)]
    with contextlib.redirect_stdout(io.StringIO()):
        _, history, ok, message, _ = topopt_mma.topopt_mma(fe, to_params=p, maxMMAIterations=30,
                                                           binarize_topology=True, print_progress=False)
    J = [float(v) for v in history["objective"]]
    assert J[-1] == pytest.approx(J[-2], rel=0.05)                # the delivered design is the continuous one
    assert "continuous design is returned" in message
    assert 0.0 < np.min(fe.mesh.elemPseudoDensity) < 1.0 or np.any((fe.mesh.elemPseudoDensity > 0.01) &
                                                                    (fe.mesh.elemPseudoDensity < 0.99))
