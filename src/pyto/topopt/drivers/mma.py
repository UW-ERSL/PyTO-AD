from pyto.topopt.common import *
from pyto.autodiff.qoi.compliance import compliance_sign
from pyto.autodiff.qoi import *
from pyto.autodiff.material_model import *
from pyto.autodiff.sparse_solve import SparseLinearSolve, Solvers
import math
import time
import matplotlib.pyplot as plt
from pyto.topopt.drivers.mmaWrapper import runMMA
from pyto.topopt.drivers._shared import binarization_acceptable, heaviside_projection, setup_driver_state, torch_sparse_filter
from pyto.topopt.manual_sensitivities import manual_gradients
from pyto.examples_benchmarks.topopt_structural_benchmarks import *
from pyto.examples_benchmarks.topopt_thermal_benchmarks import *


def run_topopt_mma(to_problem):
    """Run topology optimization using the MMA method.

    Builds the mesh and FE solver for the given example, then calls topopt_mma() and plots the result
    (`from pyto.topopt.drivers.mma import run_topopt_mma`, as in main.ipynb's import cell).

    Args:
        to_problem: A StructuralTOExamples or ThermalTOExamples member
            identifying which example problem to set up.

    Returns: None
    """
    if to_problem in StructuralTOExamples:
        mesh, mat_prop, bc, elem_body_force, to_params = getStructuralTOProblem(to_problem)
    elif to_problem in ThermalTOExamples:
        mesh, mat_prop, bc, elem_body_force, to_params = getThermalTOProblem(to_problem)

    print(f"Running {to_problem.name}...")
    print("-" * 50)
    solver = Solvers.PARDISO  # Choose solver. Typically PARDISO, but DPCG for DOF > 200,000
    dsolver = deflation.DeflationSolver()
    if to_params.nDOFDesired > DIRECT_SOLVER_DOF_CUTOFF:
        solver = Solvers.DPCG
        nGroups = min(dsolver.maxGroups, max(dsolver.minGroups, round(3 * mesh.num_nodes / dsolver.dofPerGroup)))
        dsolver.create_deflation_groups(mesh, nGroups)
        dsolver.create_deflation_matrix(mesh)

    if to_problem in StructuralTOExamples:
        fe_solver = hex_structural_fea.HexStructuralFEA(mesh=mesh, mat_prop=mat_prop, bc=bc,
                    solver=solver, dsolver=dsolver, rtol=1e-8, elem_body_force=elem_body_force)
    elif to_problem in ThermalTOExamples:
        fe_solver = hex_thermal_fea.HexThermalFEA(mesh=mesh, mat_prop=mat_prop, bc=bc,
                    solver=solver, dsolver=dsolver, rtol=1e-8, elem_body_force=elem_body_force)

    print('Solver: ', fe_solver.solver.name)
    print("nDOF: ", 3 * fe_solver.mesh.num_nodes)
    print("nElem: ", fe_solver.mesh.num_elems)

    startTime = time.time()
    print("OptimizationMethod: MMA")
    sol, history, success, errorMsg, nFEAs = topopt_mma(fe_solver=fe_solver, to_params=to_params,
                    plot_progress=True, print_progress=True, maxMMAIterations=to_params.MaxIterations)
    timeTaken = time.time() - startTime
    print(f"Time taken: {timeTaken:.0f} s")
    if not success:
        print(f"Error: {errorMsg}")

    title = f"MMA: vol: {history['volfrac'][-1]:0.2f}, J: {history['objective'][-1]:.3g}, nFEA: {len(history['objective']):3d}, time: {timeTaken:.0f} s"
    fe_solver.plot_mesh(title=title, plot_bc=False, save_path=None)

    plt.figure()
    plt.plot(history["objective"], label="Objective")
    plt.xlabel("Iterations")
    plt.ylabel(f"Objective ({to_params.Objective[0].name})")
    plt.grid(True)
    plt.show()


def _stress_limits(to_params, fe_solver):
    """Allowed stress [Pa] of every stress-type constraint (empty when there is none)."""
    out = []
    for c in to_params.Constraints:
        if c[0] is TO_QOI.MAX_VONMISES_STRESS:
            out.append(c[2])
        elif c[0] is TO_QOI.STRESS_FAILURE_FACTOR:
            out.append(fe_solver.mat_prop.yield_strength * c[2])
    return out


def topopt_mma(fe_solver, #hex_structural_fea.HexStructuralFEA or hex_thermal_fea.HexThermalFEA
                           to_params,
                            maxMMAIterations: int = 150, 
                            timeLimitSecs: float = 36000, #10 hour
                            move_limit: float = 0.2,
                            kkt_tol: float = 1.e-6,
                            objective_tol: float = 1.e-4,
                            constraint_tol: float = 1.e-4,
                            print_progress: bool = True,
                            plot_progress: bool = False,
                            binarize_topology: bool = True,   
                            progress_callback=None, 
                            iteration_callback=None,
                            plotter=None  
                             ) -> tuple[np.ndarray, dict]:
    """MMA based topology optimization for minimum compliance.

    Args:
        fe_solver: The structural FEA solver object.
        maxMMAIterations: Maximum number of MMA iterations.
        volfrac: The target volume fraction.
        penal: The penalization factor for the SIMP method.
        move_limit: The maximum change allowed for the design variables in each
            iteration.
        kkt_tol: The tolerance for the KKT conditions.
        step_tol: The tolerance for the step size.

    Returns: The displacement field of the optimized structure.
    """
    def log_message(msg):  # GUI/console logger
        if progress_callback:
            progress_callback(str(msg))
        else:
            print(msg)

    nDOFPerNode = 3 if isinstance(fe_solver, hex_structural_fea.HexStructuralFEA) else 1
    material_model = MaterialModel.SIMP

    tStart = time.time()
    num_elems = fe_solver.mesh.num_elems

    # History buffers
    history = {"objective": [], "volfrac": []}
    for idx, _constraint in enumerate(to_params.Constraints):
        history[f"constraint_{idx+1}"] = []

    if print_progress:
        log_message("Computing Filters ...")

    state = setup_driver_state(fe_solver, to_params, log_message if print_progress else None)
    H, Hs = state["H"], state["Hs"]
    elemsWithForces = state["elemsWithForces"]
    KE = torch.tensor(state["KE"])
    nodal_body_force = state["nodal_body_force"]

    # Torch-sparse copies of the (constant) filter, used inside the autograd
    # chain below -- H is a scipy sparse matrix, which can't be matmul'd
    # against a grad-tracked torch tensor directly.
    H_torch, Hs_torch = torch_sparse_filter(H, Hs)

    success = True
    errorMsg = "No errors."
    nFEAs = 0
    obj0 = None
    stress_limits = _stress_limits(to_params, fe_solver)
    if stress_limits:
        ones = torch.ones(num_elems, dtype=torch.float64)
        with torch.no_grad():
            _p, solid_max_vm = compute_pnorm_stress_autograd(fe_solver.solve(ones, material_model), ones, fe_solver)
        if float(solid_max_vm) > min(stress_limits):
            log_message(f"Warning: the stress limit ({min(stress_limits):.4g} Pa) is below the peak stress of the "
                        f"SOLID design ({float(solid_max_vm):.4g} Pa); the constraint may be unreachable.")
    # -1 only for prescribed-value-driven compliance problems (see compliance_sign)
    sign = compliance_sign(fe_solver) if to_params.Objective[0] is TO_QOI.COMPLIANCE else 1.0
    mmaIterations = 0

    # ----------------- Core chains (x -> obj) and (x -> cons) -----------------

    def obj_cons_function(
        x_raw: torch.Tensor,
    ):
        """
        'x_raw -> x_filtered -> FE solve -> objective, constraints, grads' chain.

        Returns
        -------
        obj_raw       : float
        grad_obj_filt : (n,) ndarray       (gradient w.r.t. FILTERED density)
        c             : (m,1) ndarray      (constraints)
        dcdx_filt     : (m,n) ndarray      (grad w.r.t. FILTERED density)
        sol           : ndarray            (FE solution)
        x_filtered    : (n,) ndarray       (filtered density used in analysis/solve)
        """

        x_raw = x_raw.view(-1)  # ensure 1D

        def chain(x_raw: torch.Tensor):
            # Density filtering for ANALYSIS
            x_filtered = (
                torch.sparse.mm(H_torch, x_raw.unsqueeze(1)).squeeze(1) / Hs_torch
            ) if to_params.APPLY_FILTER_TO_DENSITY else x_raw

            if to_params.APPLY_FILTER_TO_DENSITY and to_params.HeavisideProjection:
                # Smooth Heaviside projection of the filtered density (Ooms et al. 2023, eq. 33, eta = 0.5),
                # continuation in beta from the iteration count. Autograd differentiates through it.
                beta = min(to_params.HeavisideBetaMax, 2.0 ** (mmaIterations // to_params.HeavisideBetaInterval))
                x_filtered = heaviside_projection(x_filtered, beta)

            # Set filtered density on mesh for FE + plotting
            with torch.no_grad():
                fe_solver.mesh.setPseudoDensity(x_filtered.detach())

            # FE solve
            sol = fe_solver.solve(x_filtered, material_model)

            # Objective (Torch scalar)
            obj_raw = compute_objective_and_gradient(
                to_params, sol, x_filtered, fe_solver, KE, material_model
            )

            # Constraints (Torch vector)
            c_t, _ = compute_constraint_and_gradient(
                to_params, sol, x_filtered, fe_solver, KE, material_model
            )
            c_t = c_t.view(-1)  # (m,)

            return obj_raw, c_t, sol, x_filtered

        if gradient_method(to_params.Gradient) is GRADIENT_METHOD.ANALYTICAL:
            return manual_step(x_raw.detach())

        # Forward pass
        obj_raw_t, c_t, sol_t, x_filtered_t = chain(x_raw)

        # Gradient of objective wrt x_raw (like jacrev on a scalar)
        (grad_obj_t,) = torch.autograd.grad(
            outputs=obj_raw_t,
            inputs=x_raw,
            create_graph=False,
            retain_graph=True,
            allow_unused=False,
        )  # (n,)

        # Jacobian of constraints wrt x_raw, row-by-row (like jacrev on a vector)
        grads_c = []
        m = c_t.numel()
        for i in range(m):
            (grad_ci,) = torch.autograd.grad(
                outputs=c_t[i],
                inputs=x_raw,
                create_graph=False,
                retain_graph=(i < m - 1),
                allow_unused=False,
            )
            grads_c.append(grad_ci)  # each (n,)
        dcdx_t = torch.stack(grads_c, dim=0)
        # Convert to numpy with MMA-friendly shapes
        obj_raw = obj_raw_t.detach().cpu().numpy()
        grad_obj_filt = grad_obj_t.detach().cpu().numpy().reshape(-1)       # (n,)
        c_np = c_t.detach().cpu().numpy().reshape(-1, 1)                    # (m,1)
        dcdx_filt = dcdx_t.detach().cpu().numpy()                           # (m,n)
        sol_np = sol_t.detach().cpu().numpy()
        x_filtered_np = x_filtered_t.detach().cpu().numpy()

        return obj_raw, grad_obj_filt, c_np, dcdx_filt, sol_np, x_filtered_np


    def manual_step(x_raw: torch.Tensor):
        """obj_cons_function with analytical gradients (GRADIENT_METHOD.ANALYTICAL): same values, the
        gradient w.r.t. the physical density comes from manual_sensitivities and is chained back through the
        Heaviside projection and the density filter by hand."""
        with torch.no_grad():
            x_f = (torch.sparse.mm(H_torch, x_raw.unsqueeze(1)).squeeze(1) / Hs_torch
                   ) if to_params.APPLY_FILTER_TO_DENSITY else x_raw
            dproj = None
            if to_params.APPLY_FILTER_TO_DENSITY and to_params.HeavisideProjection:
                beta = min(to_params.HeavisideBetaMax, 2.0 ** (mmaIterations // to_params.HeavisideBetaInterval))
                den = math.tanh(beta * 0.5) + math.tanh(beta * 0.5)
                t = torch.tanh(beta * (x_f - 0.5))
                dproj = beta * (1.0 - t ** 2) / den
                x_f = (math.tanh(beta * 0.5) + t) / den
            fe_solver.mesh.setPseudoDensity(x_f)
            sol_t = fe_solver.solve(x_f, material_model)
            obj_t = compute_objective_and_gradient(to_params, sol_t, x_f, fe_solver, KE, material_model)
            c_t, _ = compute_constraint_and_gradient(to_params, sol_t, x_f, fe_solver, KE, material_model)
        d_obj, d_cons = manual_gradients(to_params, sol_t, x_f, fe_solver, material_model)
        g = torch.as_tensor(np.vstack([d_obj[None, :], d_cons]), dtype=x_f.dtype)     # (1+m, n) w.r.t. x_f
        if dproj is not None:
            g = g * dproj
        if to_params.APPLY_FILTER_TO_DENSITY:                                          # x_f = H x / Hs
            g = torch.sparse.mm(H_torch.t(), (g / Hs_torch).t()).t()
        g = g.cpu().numpy()
        return (obj_t.detach().cpu().numpy(), g[0], c_t.detach().cpu().numpy().reshape(-1, 1), g[1:],
                sol_t.detach().cpu().numpy(), x_f.cpu().numpy())

    # ------------------------ MMA objective callback -------------------------

    def optimizationFunction(x):
        nonlocal nFEAs, obj0, mmaIterations

        x = np.asarray(x).flatten()
        x_t = torch.tensor(x, requires_grad=True)

        # Objective chain (filters are inside)
        obj_raw, grad_obj_filt, c, dcdx_filt, sol, x_filtered = obj_cons_function(x_t)

        # Normalization
        if obj0 is None:
            obj0 = abs(obj_raw)
        obj = sign * obj_raw / obj0
        grad_obj = sign * grad_obj_filt / obj0  # still w.r.t. FILTERED density

        # Sensitivity filtering. The gradients above come from autograd w.r.t. the RAW
        # variables, i.e. they already include the chain rule through the density filter
        # when APPLY_FILTER_TO_DENSITY is on -- filtering them again would be a second,
        # incorrect smoothing. Sensitivity filtering is only the right regularization
        # when the density filter is OFF.
        if not to_params.APPLY_FILTER_TO_DENSITY and (to_params.Objective[0] is TO_QOI.COMPLIANCE):
            # Weighted filter
            grad_obj = (H @ (x_filtered * grad_obj)) / Hs / (x_filtered + 1e-12)
        elif not to_params.APPLY_FILTER_TO_DENSITY and (
            to_params.Objective[0] is not TO_QOI.VOLUME_FRACTION
        ):
            # Regular filter
            grad_obj = (H @ grad_obj) / Hs

        # Retain elements with forces / keep-list
        if elemsWithForces.size > 0:
            grad_obj[elemsWithForces] = min(grad_obj)
        if to_params.ElemsToKeep is not None:
            grad_obj[to_params.ElemsToKeep] = min(grad_obj)


        # Sensitivity filtering for constraints (only when the density filter is off;
        # see note on the objective above)
        if not to_params.APPLY_FILTER_TO_DENSITY:
            for m in range(len(to_params.Constraints)):
                if to_params.Constraints[m][0] is TO_QOI.COMPLIANCE:
                    dcdx_filt[m] = (H @ (x_filtered * dcdx_filt[m])) / Hs / (x_filtered + 1e-12)
                elif to_params.Constraints[m][0] not in (TO_QOI.VOLUME_FRACTION, TO_QOI.VOLUME_FRACTION_MIN):
                    dcdx_filt[m] = (H @ dcdx_filt[m]) / Hs
        dcdx = dcdx_filt

        # Book-keeping
        history["objective"].append(sign * obj * obj0)  # store the true (de-normalized) objective
        history["volfrac"].append(np.mean(x_filtered)) # track filtered volume fraction
        for idx, val in enumerate(c.flatten()):
            history[f"constraint_{idx+1}"].append(val)
        if iteration_callback is not None:  # e.g. the GUI's live objective/constraint plot
            iteration_callback(history)

        # Initial-iteration warning on constraints
        if (mmaIterations == 0) and print_progress:
            if np.any(c > 0):
                print("Warning: Constraint(s) violated at start of optimization!")
                print("GCMMA may not converge for this problem.")
                print("Consider changing constraints if convergence issues occur.")

        # Shape for MMA
        grad_obj = grad_obj.reshape(-1, 1)

        # Extract names for printing
        objective_name = getattr(to_params.Objective[0], 'name', str(to_params.Objective[0]))
        constraint_names = [getattr(c[0], 'name', str(c[0])) for c in to_params.Constraints]

        # Print objective and constraints for this iteration
        if (print_progress):
            print(50* '-')
            print(f"Iteration: {mmaIterations}")
            objective_name = getattr(
                to_params.Objective[0], "name", str(to_params.Objective[0])
            )
            labels = getattr(to_params, "ConstraintLabels", None)
            if getattr(to_params, "ObjectiveLabel", None) is not None:   # formulation from the GUI / a spec
                expr, sense = to_params.ObjectiveLabel
                value = sign * obj * obj0 * (-1.0 if sense == "maximize" else 1.0)   # the driver minimizes -f
                print(f"{'Max.' if sense == 'maximize' else 'Min.'} Objective ({expr}): {value:.3g}")
            else:
                print(f"Min. Objective ({objective_name}): {sign * obj * obj0:.3g}")
            for idx, val in enumerate(c.flatten()):
                if labels is not None:
                    # undo compile_spec's normalization: <= b>0: f/b - 1; <= b<=0: f - b; >= b: 1 - f/b
                    expr, op, rhs = labels[idx]
                    value = (1.0 - val) * rhs if op == ">=" else (val + 1.0) * rhs if rhs > 0 else val + rhs
                    status = "ok" if (value >= rhs if op == ">=" else value <= rhs) else "violated"
                    print(f"Constraint {idx+1} ({expr}): {value:.3g} {op} {rhs:.3g}  {status}")
                else:
                    # benchmark problems: every constraint is value/limit - 1 <= 0
                    rhs = to_params.Constraints[idx][2]
                    print(f"Constraint {idx+1} ({constraint_names[idx]}): {(val+1)*rhs:.3g} <= {rhs:.3g}?")

        mmaIterations += 1
        nFEAs += 1

        # Heuristic: enforce constraints more strongly for MMA if requested
        if to_params.Enforce_Constraints_MMA:
            for m in range(len(to_params.Constraints)):
                scaling = max(0.1, min(10, 10 ** c[m]))  # heuristic
                dcdx[m, :] /= scaling

        return obj, grad_obj, c, dcdx


    objective_name = getattr(to_params.Objective[0], 'name', str(to_params.Objective[0]))
    constraint_names = [getattr(c[0], 'name', str(c[0])) for c in to_params.Constraints]
    
    # Start from the volume-fraction limit when there is one (Reference-era behaviour). Starting at
    # 0.5 with a 0.1 limit is an infeasible start, and self-weight problems then collapse to zero material.
    initialDensity = next((c[2] for c in to_params.Constraints if c[0] == TO_QOI.VOLUME_FRACTION), 0.5)
    x0 = initialDensity * np.ones(num_elems, dtype=float).reshape(-1, 1)
    lowerBound = np.zeros(num_elems, dtype=float).reshape(-1, 1)
    upperBound = np.ones(num_elems, dtype=float).reshape(-1, 1)
    nVariables = num_elems
    nConstraints = len(to_params.Constraints)

    (
        xOptimal,
        f0val,
        df0dx,
        gval,
        dgdx,
        nFEAs,
    ) = runMMA(
        nVariables,
        nConstraints,
        optimizationFunction,
        x0,
        lowerBound,
        upperBound,
        maxIterations=maxMMAIterations,
        timeLimitSecs=timeLimitSecs,
        move_limit=move_limit,
        # With a beta continuation the objective jumps at every beta change; run the whole schedule.
        fTolerance=0.0 if to_params.HeavisideProjection else objective_tol,
        gTolerance=constraint_tol,
        kktTol=kkt_tol,
        verbose=False,
        progress_callback=progress_callback,
    )

    # ---------------------------- Finalization -------------------------------

    x = np.asarray(xOptimal).flatten()
    x_t = torch.tensor(x, requires_grad=True)

    # Continuous evaluation at the optimum. x is the RAW design variable; with
    # density filtering on, the PHYSICAL design is the filtered field returned here.
    _obj, _g, _c, _dg, _sol, x_filtered_final = obj_cons_function(x_t)
    x_phys = np.asarray(x_filtered_final).flatten()

    # Grey fraction of the physical design, before binarization
    grey_elements = np.sum((x_phys > 0.1) & (x_phys < 0.9))
    fraction_grey = grey_elements / num_elems

    # Optional volume-preserving binarization of the PHYSICAL density
    x_design = x_phys
    if binarize_topology:
        x_sorted = np.sort(x_phys)
        threshold = x_sorted[int((1 - np.mean(x_phys)) * len(x_phys))]
        x_design = np.where(x_phys < threshold, 0.0, 1.0)

    if to_params.Eliminate_Hanging_Elements:
        # Ensure binarized
        x_sorted = np.sort(x_design)
        threshold = x_sorted[int((1 - np.mean(x_design)) * len(x_design))]
        x_design = np.where(x_design < threshold, 0.0, 1.0)
        fe_solver.mesh.setPseudoDensity(torch.tensor(x_design))

        meshComponents = fe_solver.mesh.find_connected_components()
        if len(meshComponents) > 1:
            if print_progress:
                log_message(50 * "-")
                log_message("Removing hanging elements.")
            largest_component = max(meshComponents, key=len)
            x_design[:] = 0.0
            x_design[list(largest_component)] = 1.0

    # Final FE evaluation of the design actually being returned. x_design is
    # already physical, so evaluate it WITHOUT re-filtering (a second filter pass
    # would blur the binarized topology and misreport its objective/volume).
    filter_setting = to_params.APPLY_FILTER_TO_DENSITY
    to_params.APPLY_FILTER_TO_DENSITY = False
    try:
        x_final_t = torch.tensor(x_design, requires_grad=True)
        obj_raw_final, _gobj_filt, c_final, _dg_filt, sol, x_filtered_final = obj_cons_function(x_final_t)
    finally:
        to_params.APPLY_FILTER_TO_DENSITY = filter_setting

    # A 0/1 design that is much worse than the continuous one (thresholding cut its load or heat paths) is not
    # delivered: keep the continuous design and say so (binarization_acceptable).
    if binarize_topology and history["objective"] and not binarization_acceptable(
            obj_raw_final, history["objective"][-1], sign):
        warning = (f"Warning: the 0/1 (binarized) design was much worse than the continuous one "
                   f"(objective {float(obj_raw_final):.4g} vs {float(history['objective'][-1]):.4g}; grey fraction "
                   f"{fraction_grey:.2f}), so the continuous design is returned.")
        errorMsg = warning if errorMsg == "No errors." else f"{errorMsg} {warning}"
        x_design = x_phys
        to_params.APPLY_FILTER_TO_DENSITY = False
        try:
            obj_raw_final, _g, c_final, _dg, sol, x_filtered_final = obj_cons_function(
                torch.tensor(x_design, requires_grad=True))
        finally:
            to_params.APPLY_FILTER_TO_DENSITY = filter_setting

    # Track the final design in history (store de-normalized to match earlier semantics)
    history["objective"].append(obj_raw_final)
    history["volfrac"].append(np.mean(x_filtered_final))
    for idx, val in enumerate(c_final.flatten()):
        history[f"constraint_{idx+1}"].append(val)

    # Leave the final design on the mesh for plotting
    fe_solver.mesh.setPseudoDensity(torch.tensor(x_design))

    if stress_limits:
        fe_solver.postprocess()  # stresses of the design just evaluated
        solid = x_design > 0.5
        max_vm = float(np.max(np.asarray(fe_solver.vonMisesStress)[solid])) if solid.any() else 0.0
        if max_vm > 1.01 * min(stress_limits):
            errorMsg = (f"Warning: stress limit exceeded by {100 * (max_vm / min(stress_limits) - 1):.1f}% "
                        f"(max von Mises {max_vm:.4g} Pa vs limit {min(stress_limits):.4g} Pa)")

    # Log final line (the separator only follows the per-iteration log)
    if print_progress:
        print("-" * 50)
    log_message(
        f"Final objective: {obj_raw_final:.4g}, vf: {np.mean(x_filtered_final):.3f}, grey: {fraction_grey:.3f}"
    )
    log_message(f"Total Time: {time.time() - tStart:.2f} s")
    log_message(f"Error: {errorMsg}")

    return np.asarray(sol), history, success, errorMsg, nFEAs


# ------------------------------ Script runner -------------------------------

if __name__ == "__main__":
    from pyto.examples_benchmarks.topopt_structural_benchmarks import *
    from pyto.examples_benchmarks.topopt_thermal_benchmarks import *

    print("-" * 50)

    to_problem = StructuralTOExamples.LBracketTopLoadStressObjective # Choose the TO problem
    
    if (to_problem in StructuralTOExamples):
        mesh, mat_prop, bc,elem_body_force, to_params = getStructuralTOProblem(to_problem)
    elif (to_problem in ThermalTOExamples):
        mesh, mat_prop, bc,elem_body_force, to_params = getThermalTOProblem(to_problem)

    print(f"Running {to_problem.name}...")
    print("-" * 50)

    solver = Solvers.SPSOLVE  # default, see below
    dsolver = deflation.DeflationSolver(use_gpu=False)

    # if to_params.nDOFDesired > DIRECT_SOLVER_DOF_CUTOFF:
    #     # Typically PARDISO, but DPCG for large DOF problems
    #     solver = lin_solv.Solvers.DPCG
    #     nGroups = min(
    #         dsolver.maxGroups,
    #         max(
    #             dsolver.minGroups,
    #             round(3 * mesh.num_nodes / dsolver.dofPerGroup),
    #         ),
    #     )
    #     dsolver.create_deflation_groups(mesh, nGroups)
    #     dsolver.create_deflation_matrix(mesh)
    #     dsolver.W = dsolver.W[bc.free_dofs, :]

    if to_problem in StructuralTOExamples:
        fe_solver = hex_structural_fea.HexStructuralFEA(
            mesh=mesh,
            mat_prop=mat_prop,
            bc=bc,
            solver=solver,
            dsolver=dsolver,
            rtol=1e-8,
            elem_body_force=elem_body_force,
        )
    elif to_problem in ThermalTOExamples:
        fe_solver = hex_thermal_fea.HexThermalFEA(
            mesh=mesh,
            mat_prop=mat_prop,
            bc=bc,
            solver=solver,
            dsolver=dsolver,
            rtol=1e-8,
            elem_body_force=elem_body_force,
        )

    print("Solver: ", fe_solver.solver.name)
    print("nDOF: ", 3 * fe_solver.mesh.num_nodes)
    print("nNodes: ", fe_solver.mesh.num_nodes)
    print("nElem: ", fe_solver.mesh.num_elems)    
    
    title = f'nNodes: {fe_solver.mesh.num_nodes}, nElem: {fe_solver.mesh.num_elems}'
    #fe_solver.plot_mesh(title = title, save_path = None)
    
    plot_progress = True
    print_progress = True
    startTime = time.time()
    print("OptimizationMethod: MMA")
    
    u, history,success,errorMsg,nFEAs = topopt_mma(fe_solver = fe_solver,
                                to_params = to_params,
                                plot_progress= plot_progress,
                                print_progress= print_progress,
                                maxMMAIterations= to_params.MaxIterations,)
    timeTaken = time.time() - startTime
    fe_solver.postprocess()
    title = (
        f"MMA: vol: {history['volfrac'][-1]:0.2f}, "
        f"J: {history['objective'][-1]:.3g}, "
        f"nFEA: {len(history['objective']):3d}, time: {timeTaken:.0f} s"
    )
    fe_solver.plot_mesh(title=title, plot_bc=False, save_path=None)
    fe_solver.postprocess()
    fe_solver.plot_vonMisesStress()

    plt.figure()
    plt.plot(history["objective"], label="Objective")
    plt.xlabel("Iterations")
    plt.ylabel(f"Objective ({to_params.Objective[0].name})")
    plt.grid(True)
    plt.show()

    for idx, constraint_name in enumerate(
        [getattr(c[0], "name", str(c[0])) for c in to_params.Constraints]
    ):
        plt.figure()
        val = np.array(history[f"constraint_{idx+1}"])
        constraint_val = (val + 1) * to_params.Constraints[idx][2]
        plt.plot(constraint_val, label=f"Constraint {idx+1} ({constraint_name})")
        plt.xlabel("Iterations")
        plt.ylabel(f"Constraint {idx+1} ({constraint_name})")
        plt.axhline(y=to_params.Constraints[idx][2], color="r", linestyle="--")
        plt.grid(True)
        plt.show()
