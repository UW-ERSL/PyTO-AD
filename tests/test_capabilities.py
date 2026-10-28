"""Method capabilities, gradient source, gradient check and cost estimate (plan S4)."""
import numpy as np
import pytest
import torch

from pyto.topopt.capabilities import (check_gradient, cost_estimate, gradient_source, method_support,
                                      solution_dependent_count)
from pyto.topopt.spec import ConstraintSpec, ObjectiveSpec, OptimizationSpec, compile_spec

CLASSICAL = OptimizationSpec()                                          # min compliance, vf <= 0.5
STRESS = OptimizationSpec(constraints=[ConstraintSpec("VolumeFraction()", "<=", 0.4),
                                       ConstraintSpec("StressFailureFactor()", "<=", 0.5)])


def test_method_support_matrix():
    # MMA is the one optimizer and solves every formulation
    maxi = OptimizationSpec(objective=ObjectiveSpec("Compliance()", "maximize"))
    for spec in (CLASSICAL, STRESS, maxi):
        for physics in ("structural", "thermal", "thermo-structural"):
            assert method_support(spec, physics) == {"MMA": (True, "")}
    assert method_support(CLASSICAL, "structural", body_force=True) == {"MMA": (True, "")}
    assert "autograd" in gradient_source("MMA") and "Analytical" in gradient_source("MMA", "analytical")


def test_cost_estimate_counts_adjoint_solves():
    sel = {"Tip": {"nodes": [1, 2]}}
    spec = OptimizationSpec(constraints=[ConstraintSpec("VolumeFraction()", "<=", 0.4),
                                         ConstraintSpec("Mass()", "<=", 10.0),
                                         ConstraintSpec("MaxStress()", "<=", 2e8),
                                         ConstraintSpec("Displacement(Tip, y, pnorm)", "<=", 1e-3),
                                         ConstraintSpec("Compliance()", "<=", 9.0, enabled=False)])
    assert solution_dependent_count(spec, sel) == 3                      # compliance, stress, displacement
    assert cost_estimate(spec, sel) == "Per iteration: 1 forward FE solve + 3 adjoint solves."
    assert "(about 1.60 s)" in cost_estimate(spec, sel, forward_s=0.4, backward_s=0.4)


def test_gradient_check_passes_for_correct_and_flags_broken_functions(structural_fe_solver):
    fe = structural_fe_solver
    tip = np.unique(np.nonzero(fe.bc.force)[0] // 3)
    sel = {"Tip": {"nodes": tip}}
    spec = OptimizationSpec(objective=ObjectiveSpec("Displacement(Tip, y, mean)", "maximize"),
                            constraints=[ConstraintSpec("VolumeFraction()", "<=", 0.5),
                                         ConstraintSpec("MaxStress()", "<=", 2e8),
                                         ConstraintSpec("Displacement(Tip, magnitude, pnorm)", "<=", 1e-3)])
    results = check_gradient(compile_spec(spec, sel), fe, n_elements=4)
    assert [r.name for r in results] == ["objective", "constraint 1", "constraint 2", "constraint 3"]
    assert all(r.ok for r in results), results

    # a user function that silently drops the displacement part of its gradient (detach) must be flagged
    from pyto.topopt.common import TOParams, TO_QOI
    broken = TOParams()
    broken.Objective = (TO_QOI.GFUNCTION, lambda sol, x, f: sol[3 * tip].detach().sum() * x.mean() * 1e3)
    broken.Constraints = [(TO_QOI.VOLUME_FRACTION, None, 0.5)]
    r = check_gradient(broken, fe, n_elements=4)
    assert not r[0].ok and r[1].ok


def test_gradient_check_covers_filter_symmetry_and_projection(structural_fe_solver):
    # check_gradient perturbs MMA's own design variable, before the density filter (with a symmetry plane) and the
    # Heaviside projection, so finite differences and the gradient (AD or analytical) both go through that chain
    from pyto.topopt.spec import ManufacturingSpec, RegularizationSpec
    fe = structural_fe_solver
    for gradient in ("autodiff", "analytical"):
        spec = OptimizationSpec(regularization=RegularizationSpec(filter_radius=2.5, heaviside=True),
                                manufacturing=ManufacturingSpec(symmetry=["Y"]))
        spec.method.gradient = gradient
        results = check_gradient(compile_spec(spec), fe, n_elements=4)
        assert all(r.ok for r in results), (gradient, results)


def test_heaviside_projection_is_shared_and_maps_unit_interval():
    from pyto.topopt.drivers._shared import heaviside_projection
    x = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
    for beta in (1.0, 8.0, 32.0):
        assert torch.allclose(heaviside_projection(x, beta), x)          # fixed points 0, eta = 0.5, 1
