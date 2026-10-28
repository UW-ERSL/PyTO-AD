"""Thermo-mechanical coupling: differentiable thermal force, coupled solve, gradients."""
import numpy as np
import torch

from pyto.autodiff.material_model import MaterialModel


def _solved_thermal(thermal_fe_solver, seed=0):
    fe = thermal_fe_solver
    rng = np.random.default_rng(seed)
    x = torch.tensor(0.3 + 0.6 * rng.random(fe.mesh.num_elems), dtype=torch.float64)
    T = fe.solve(x, MaterialModel.SIMP)
    return fe, x, T


def test_torch_thermoelastic_force_matches_numpy_reference(thermal_fe_solver):
    # The torch force must reproduce the existing NumPy get_thermoelastic_force to round-off.
    fe, x, T = _solved_thermal(thermal_fe_solver)
    f_np = fe.get_thermoelastic_force(x.numpy(), MaterialModel.SIMP)
    f_t = fe.get_thermoelastic_force_torch(T, x, MaterialModel.SIMP).detach().numpy()
    assert np.max(np.abs(f_t)) > 0
    assert np.allclose(f_t, f_np, rtol=1e-10, atol=1e-10 * np.max(np.abs(f_np)))


def test_torch_thermoelastic_force_gradients_wrt_temperature_and_density(thermal_fe_solver):
    # Autograd must flow through both T and x (checked against central finite differences).
    fe, x, T = _solved_thermal(thermal_fe_solver)
    T = T.detach().clone()
    w = torch.tensor(np.random.default_rng(1).standard_normal(3 * fe.mesh.num_nodes), dtype=torch.float64)

    def scalar(T_, x_):
        return (w * fe.get_thermoelastic_force_torch(T_, x_, MaterialModel.SIMP)).sum()

    T_req, x_req = T.clone().requires_grad_(True), x.clone().requires_grad_(True)
    gT, gx = torch.autograd.grad(scalar(T_req, x_req), (T_req, x_req))
    rng = np.random.default_rng(2)
    for name, base, g in (("T", T, gT), ("x", x, gx)):
        for i in rng.choice(base.numel(), size=3, replace=False):
            step = 1e-3 * max(1.0, abs(float(base[i])))
            up, dn = base.clone(), base.clone()
            up[i] += step; dn[i] -= step
            fd = (scalar(up, x) - scalar(dn, x)).item() / (2 * step) if name == "T" else \
                 (scalar(T, up) - scalar(T, dn)).item() / (2 * step)
            assert abs(fd - float(g[i])) <= 1e-5 * max(abs(fd), 1e-12) + 1e-12, (name, i, fd, float(g[i]))


# ---------------------------------------------------------------- B2: structural solve with a thermal load
def _uniform_temperature_problem(supports, n=(4, 2, 2), dT=40.0, T0=23.0):
    """Small brick, uniform temperature T0 + dT (thermal Dirichlet everywhere), structural supports chosen by
    `supports(mesh) -> list of dof indices`."""
    import pyto.core.bc as bound_cond
    import pyto.core.mat_lib as mat_lib
    import pyto.core.hex_mesher as hex_mesher
    import pyto.solve.numpy_backend as lin
    from pyto.physics.structural.hex_structural_fea import HexStructuralFEA
    from pyto.physics.thermal.hex_thermal_fea import HexThermalFEA

    mesh = hex_mesher.HexMesher()
    mesh.grid_mesh(num_elems=n, elem_size=(0.5, 0.5, 0.5))
    mat = mat_lib.get_material("Steel")
    all_nodes = np.arange(mesh.num_nodes)
    tbc = bound_cond.BC(force=np.zeros(mesh.num_nodes), fixed_dofs=all_nodes, dirichlet_values=np.full(mesh.num_nodes, T0 + dT))
    thermal = HexThermalFEA(mesh=mesh, mat_prop=mat, bc=tbc, solver=lin.Solvers.SPSOLVE, thermoElasticReferenceTemperature=T0)
    fixed = np.array(sorted(set(supports(mesh))))
    sbc = bound_cond.BC(force=np.zeros(3 * mesh.num_nodes), fixed_dofs=fixed, dirichlet_values=np.zeros(len(fixed)))
    structural = HexStructuralFEA(mesh=mesh, mat_prop=mat, bc=sbc, solver=lin.Solvers.SPSOLVE)  # sets mesh.edofMat structural
    x = torch.ones(mesh.num_elems, dtype=torch.float64)
    T = thermal.solve(x, MaterialModel.SIMP)
    f_th = thermal.get_thermoelastic_force_torch(T, x, MaterialModel.SIMP)
    u = structural.solve(x, MaterialModel.SIMP, extra_force=f_th, elem_thermal_strain=thermal.get_element_thermal_strain(T))
    return mesh, mat, structural, u.detach().numpy(), dT


def test_free_thermal_expansion_gives_alpha_dT_times_position():
    # Minimal supports (6 rigid-body modes removed, nothing else): a uniformly heated body expands freely,
    # u = alpha * dT * (X - X_origin), and carries no stress.
    def supports(mesh):
        xyz = mesh.node_xyz
        origin = int(np.argmin(np.linalg.norm(xyz - xyz.min(0), axis=1)))
        on_x = int(np.argmin(np.linalg.norm(xyz - np.array([xyz[:, 0].max(), xyz[origin, 1], xyz[origin, 2]]), axis=1)))
        on_y = int(np.argmin(np.linalg.norm(xyz - np.array([xyz[origin, 0], xyz[:, 1].max(), xyz[origin, 2]]), axis=1)))
        return [3 * origin, 3 * origin + 1, 3 * origin + 2, 3 * on_x + 1, 3 * on_x + 2, 3 * on_y + 2]
    mesh, mat, structural, u, dT = _uniform_temperature_problem(supports)
    expected = mat.thermal_expansion_coefficient * dT * (mesh.node_xyz - mesh.node_xyz.min(0))
    assert np.allclose(u.reshape(-1, 3), expected, rtol=1e-6, atol=1e-6 * np.abs(expected).max())
    structural.postprocess()
    assert np.max(np.abs(structural.vonMisesStress)) < 1e-6 * mat.youngs_modulus * mat.thermal_expansion_coefficient * dT


def test_bar_clamped_at_both_ends_carries_minus_E_alpha_dT():
    # ux fixed on both end faces, lateral directions free: uniaxial stress sigma_xx = -E * alpha * dT.
    def supports(mesh):
        xyz = mesh.node_xyz
        ends = np.where((xyz[:, 0] == xyz[:, 0].min()) | (xyz[:, 0] == xyz[:, 0].max()))[0]
        origin = int(np.argmin(np.linalg.norm(xyz - xyz.min(0), axis=1)))
        corner = int(np.argmin(np.linalg.norm(xyz - np.array([xyz[origin, 0], xyz[:, 1].max(), xyz[origin, 2]]), axis=1)))
        return [3 * n for n in ends] + [3 * origin + 1, 3 * origin + 2, 3 * corner + 2]
    mesh, mat, structural, u, dT = _uniform_temperature_problem(supports)
    structural.postprocess()
    expected = mat.youngs_modulus * mat.thermal_expansion_coefficient * dT
    sxx = np.asarray(structural.stressComponents)[:, 0]
    assert np.allclose(sxx, -expected, rtol=1e-5), (sxx.min(), sxx.max(), -expected)


# ---------------------------------------------------------------- B3: coupled solver object
def _coupled_problem(n=(4, 2, 2), T_hot=80.0, T_cold=23.0):
    """Cantilever-like brick: hot face x=0, cold face x=L (thermal Dirichlet), clamped at x=0, tip load at x=L."""
    import pyto.core.bc as bound_cond
    import pyto.core.mat_lib as mat_lib
    import pyto.core.hex_mesher as hex_mesher
    import pyto.solve.numpy_backend as lin
    from pyto.physics.thermoelastic.thermostructural_fea import ThermoStructuralFEA

    mesh = hex_mesher.HexMesher()
    mesh.grid_mesh(num_elems=n, elem_size=(0.5, 0.5, 0.5))
    mat = mat_lib.get_material("Steel")
    xyz = mesh.node_xyz
    hot, cold = np.where(xyz[:, 0] < 1e-9)[0], np.where(xyz[:, 0] > xyz[:, 0].max() - 1e-9)[0]
    tbc = bound_cond.BC(force=np.zeros(mesh.num_nodes), fixed_dofs=np.concatenate([hot, cold]),
                        dirichlet_values=np.concatenate([np.full(len(hot), T_hot), np.full(len(cold), T_cold)]))
    fixed = np.concatenate([3 * hot, 3 * hot + 1, 3 * hot + 2])
    force = np.zeros(3 * mesh.num_nodes)
    force[3 * cold + 1] = -1e5
    sbc = bound_cond.BC(force=force, fixed_dofs=fixed, dirichlet_values=np.zeros(len(fixed)))
    fe = ThermoStructuralFEA(mesh=mesh, mat_prop=mat, structural_bc=sbc, thermal_bc=tbc,
                             solver=lin.Solvers.SPSOLVE, thermoElasticReferenceTemperature=T_cold)
    return fe


def _energy_compliance(fe, x):
    from pyto.autodiff.qoi.compliance import compute_compliance_torch
    u = fe.solve(x, MaterialModel.SIMP)
    KE = fe.elem_stiff_torch[0]
    return compute_compliance_torch(u, x, fe, KE, MaterialModel.SIMP)


def test_coupled_solver_keeps_structural_edofmat_and_matches_manual_chain():
    fe = _coupled_problem()
    assert np.array_equal(fe.mesh.edofMat, fe.mesh.edofMatStructural)  # thermal ctor must not leave its edofMat behind
    x = torch.tensor(0.3 + 0.6 * np.random.default_rng(0).random(fe.mesh.num_elems), dtype=torch.float64)
    u = fe.solve(x, MaterialModel.SIMP)
    T = fe.thermal_fea.solve(x, MaterialModel.SIMP)
    f = fe.thermal_fea.get_thermoelastic_force_torch(T, x, MaterialModel.SIMP)
    assert torch.allclose(fe.temperature, T)
    assert np.max(np.abs(f.numpy())) > 0
    assert torch.allclose(fe.total_force, torch.as_tensor(fe.bc.force) + f)
    assert fe.elem_thermal_strain is not None


def test_coupled_compliance_gradient_matches_finite_differences():
    # Autograd through thermal adjoint -> thermal force -> structural adjoint, vs central differences.
    fe = _coupled_problem()
    x0 = torch.tensor(0.3 + 0.6 * np.random.default_rng(3).random(fe.mesh.num_elems), dtype=torch.float64)
    x = x0.clone().requires_grad_(True)
    (g,) = torch.autograd.grad(_energy_compliance(fe, x), x)
    for i in np.random.default_rng(4).choice(fe.mesh.num_elems, size=4, replace=False):
        step = 1e-5
        up, dn = x0.clone(), x0.clone()
        up[i] += step; dn[i] -= step
        fd = (_energy_compliance(fe, up) - _energy_compliance(fe, dn)).item() / (2 * step)
        assert abs(fd - float(g[i])) <= 1e-5 * abs(fd) + 1e-9, (i, fd, float(g[i]))


def test_coupled_autograd_gradient_matches_hand_derived_thermoelastic_sensitivity():
    # Independent oracle: ThermoElasticSensitivity (three-term adjoint formula, used by the GUI).
    import contextlib, io
    import pyto.solve.numpy_backend as lin
    from pyto.physics.thermoelastic.topopt_thermostructural_sensitivity import ThermoElasticSensitivity
    fe = _coupled_problem()
    x0 = torch.tensor(0.3 + 0.6 * np.random.default_rng(3).random(fe.mesh.num_elems), dtype=torch.float64)
    x = x0.clone().requires_grad_(True)
    (g,) = torch.autograd.grad(_energy_compliance(fe, x), x)
    with contextlib.redirect_stdout(io.StringIO()):  # the oracle prints debug lines
        ref = ThermoElasticSensitivity(fe.thermal_fea, fe).compute_compliance_sensitivity(
            x0.numpy(), fe.temperature.detach().numpy(), fe.sol.detach().numpy(), solver=lin.Solvers.SPSOLVE)
    assert np.max(np.abs(ref - g.numpy())) <= 1e-6 * np.max(np.abs(g.numpy()))


# ---------------------------------------------------------------- B4: drivers accept the coupled solver
def test_mma_converges_on_the_coupled_solver():
    # Thermal compliance is not monotone in the design: MMA oscillates for a few iterations, then converges.
    from pyto.topopt.common import TOParams, TO_QOI
    from pyto.topopt.drivers.mma import topopt_mma
    params = TOParams()
    params.Objective = (TO_QOI.COMPLIANCE, None)
    params.Constraints = [(TO_QOI.VOLUME_FRACTION, None, 0.5)]
    h = topopt_mma(_coupled_problem(), to_params=params, maxMMAIterations=30, print_progress=False,
                   binarize_topology=False)[1]
    J = np.array(h["objective"]).ravel()
    assert np.all(np.isfinite(J)) and J[-1] < 0.5 * J[0]
    assert h["volfrac"][-1] <= 0.5 + 0.01  # volume constraint is an upper bound


# ---------------------------------------------------------------- B0: problem definitions
def test_thermal_scaling_accepts_per_problem_penalty_and_void_ratio():
    from pyto.autodiff.material_model import get_thermal_material_model_scaling_torch as scaling
    x = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
    s = scaling(x, MaterialModel.SIMP, penalty=3.0, void_ratio=0.03)
    assert torch.allclose(s, torch.tensor([0.03, 0.03 + 0.125 * 0.97, 1.0], dtype=torch.float64))
    assert torch.allclose(scaling(x, MaterialModel.SIMP), scaling(x, MaterialModel.SIMP, None, None))


def test_paper_problem_definitions():
    from pyto.examples_benchmarks.topopt_thermostructural_benchmarks import (ThermoStructuralTOExamples as P,
                                                                               getThermoStructuralTOProblem)
    # Rodrigues & Fernandes Example A: 1000 kgf, vf 0.4, clamped columns kept, walls at T_ref + dT
    for prob, dT in ((P.BiClampDT0, 0), (P.BiClamp, 1), (P.BiClampDT4, 4), (P.BiClampDT10, 10)):
        mesh, mat, sbc, tbc, _, p = getThermoStructuralTOProblem(prob)
        assert abs(np.abs(sbc.force).sum() - 1000 * 9.80665) < 1e-6
        assert p.Constraints[0][2] == 0.4 and p.ElemsToKeep is not None and len(p.ElemsToKeep) > 0
        assert np.allclose(tbc.dirichlet_values, p.ThermalReferenceTemperature + dT)
    # Ooms et al. MBB: concrete, 20/800 C, F_m = 5 kN (half of 10 kN) per 10 mm of thickness, insulating void, exponent 3
    mesh, mat, sbc, tbc, _, p = getThermoStructuralTOProblem(P.MBBBeam)
    thickness = np.ptp(mesh.node_xyz[:, 2])
    assert abs(np.abs(sbc.force).sum() - 5e5 * thickness) < 1e-3
    assert mat.youngs_modulus == 30e9 and p.ThermalReferenceTemperature == 20.0
    assert set(np.unique(tbc.dirichlet_values)) == {20.0, 800.0}
    assert p.ConductivityPenalty == 3.0 and p.ConductivityVoidRatio == 0.03


def test_mma_with_heaviside_projection_runs_and_is_differentiable():
    from pyto.topopt.common import TOParams, TO_QOI
    from pyto.topopt.drivers.mma import topopt_mma
    params = TOParams()
    params.Objective = (TO_QOI.COMPLIANCE, None)
    params.Constraints = [(TO_QOI.VOLUME_FRACTION, None, 0.5)]
    params.HeavisideProjection, params.HeavisideBetaInterval = True, 3  # beta 1, 2, 4 within 9 iterations
    h = topopt_mma(_coupled_problem(), to_params=params, maxMMAIterations=9, print_progress=False,
                   binarize_topology=False)[1]
    J = np.array(h["objective"]).ravel()
    assert np.all(np.isfinite(J)) and h["volfrac"][-1] <= 0.5 + 0.02
