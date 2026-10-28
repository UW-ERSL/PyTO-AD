"""Solver scaling study: how far each linear solver goes with automatic differentiation and with analytical
sensitivities, and what it costs in time and memory (structural, hex8 voxel meshes, MMA).

For each problem the mesh is refined step by step (nDOFDesired ladder). At every size each solver runs an MMA
optimization of 10 iterations (--max-iters; final evaluation and binarization as in the sweep) with AUTODIFF and with
ANALYTICAL sensitivities, each in its own process. A (solver, gradient) pair stops climbing the ladder at its first
failure: memory cap or out of memory, an unconverged linear solve, an error, or fewer than MIN_ITERATIONS
iterations within the time limit. By default runs have no time limit; with --time-limit SECONDS every run is
stopped after that much wall clock, whatever it is doing at that moment; its per-iteration cost comes from the iterations it completed, the
summary records how many that was, and est_opt_time_s extrapolates to the full iteration count. A stopped run with
at least MIN_ITERATIONS iterations is TIME_LIMIT and the ladder goes on; with fewer it is TOO_SLOW and the ladder
stops.

Measured per iteration (instrumented from outside the library; the FE and MMA code is the code the sweep runs):
  time      whole iteration, FE solve (assembly + linear solve), factorization, forward solves, gradient
            (autograd backward, or the analytical sensitivities), adjoint solves, MMA update
  memory    peak resident memory of the process during the iteration (kernel high-water mark, reset each
            iteration), memory before the optimization starts, AD graph (bytes of the tensors autograd saves),
            factor memory (PARDISO iparm / SuperLU nnz), GPU memory pool (DPCG on the GPU)
  solver    CG iterations per solve (DPCG/PCG) and the relative residual |Ax - b|/|b| of every solve
  results   objective, volume fraction and constraints of every iteration (iterations.csv), the first-iteration
            gradient fingerprint (all runs of one size start from the same design, so iteration 1 must agree to
            solver accuracy across solvers and gradient methods), final objective and topology image.

Run from the repository root after other benchmarks have finished (timings need the machine to itself):
    export PYTHONPATH=src MPLBACKEND=Agg PYVISTA_OFF_SCREEN=true
    nohup systemd-run --user --scope -p MemoryMax=55G -p MemorySwapMax=0 \\
        python -m pyto.examples_benchmarks.solver_scaling > solver_scaling.log 2>&1 &          # days
    python -m pyto.examples_benchmarks.solver_scaling --problems EdgeCantilever --dofs 25000 50000 --max-iters 3
systemd-run puts the study in its own memory cgroup, so even an allocation spike that beats the in-process memory
watchdog kills only the study, never the desktop or VS Code. Stopped? Start again with the same --tag: recorded
runs are skipped and stopped ladders stay stopped (--rerun repeats them).

Output (Results/SolverScaling_<tag>/):
    solver_scaling_summary.csv   one row per run (times are medians over iterations 2.., memory in GB)
    solver_scaling_checks.csv    objective / gradient agreement of every run with the reference run of its size
    SOLVER_SCALING_<tag>.md      tables, solver limits, plots
    <problem>_scaling.png        time per iteration, peak memory, AD graph, AD/analytical ratio vs dofs
    runs/<problem>/<ndof>/<solver>_<gradient>/  iterations.csv, result.json, topology.png
"""
import argparse
import csv
import datetime
import json
import math
import multiprocessing as mp
import os
import platform
import threading
import time

# PROBLEMS = ("EdgeCantilever", "LBracketThickTopLoad_Stress_Vol", "GEGrabCAD")
# SOLVERS = ("PARDISO", "SPSOLVE", "DPCG", "DPCG_GPU", "PCG")
PROBLEMS = ("EdgeCantilever",)
SOLVERS = ("PARDISO", "DPCG", "DPCG_GPU")
ALL_SOLVERS = ("PARDISO", "SPSOLVE", "DPCG", "DPCG_GPU", "PCG")   # report order; a results folder may hold any of them
GRADIENTS = ("AUTODIFF", "ANALYTICAL")
DOFS = (25000, 50000, 100000, 250000, 500000, 1000000)
MIN_ITERATIONS = 3          # fewer iterations than this within the time limit: the solver's limit is reached
RESIDUAL_WARN = 1e-5        # relative residual above this: LOW_ACCURACY (recorded; the J/gradient checks show the effect)
RESIDUAL_LIMIT = 1e-2       # above this the solve has not converged and gradients are wrong: limit reached
GB = 1024 ** 3
STOP = {"MEMORY_CAP", "OUT_OF_MEMORY", "HARD_TIMEOUT", "TOO_SLOW", "INACCURATE_SOLVE", "ERROR", "CRASHED",
        "GPU_UNAVAILABLE"}


# ------------------------------------------------------------------------------------------------ measurement

def _proc_status(key):
    """VmRSS / VmHWM of this process in bytes (/proc/self/status)."""
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith(key + ":"):
                return int(line.split()[1]) * 1024
    return 0


def _reset_peak():
    """Reset the kernel's peak-RSS counter (VmHWM), so the next read is the peak of the coming phase."""
    try:
        with open("/proc/self/clear_refs", "w") as f:
            f.write("5")
        return True
    except OSError:
        return False


class _Watchdog(threading.Thread):
    """Samples RSS; above the cap it records MEMORY_CAP and ends the process (before the kernel's OOM killer
    picks a victim). Also the fallback peak when VmHWM cannot be reset."""

    def __init__(self, cap_bytes, on_cap):
        super().__init__(daemon=True)
        self.cap, self.on_cap, self.peak = cap_bytes, on_cap, 0

    def run(self):
        while True:
            rss = _proc_status("VmRSS")
            self.peak = max(self.peak, rss)
            if rss > self.cap:
                self.on_cap(rss)
                os._exit(3)
            time.sleep(0.02)


def _tensor_bytes(t):
    import torch
    if t.layout != torch.strided:                               # sparse: indices + values
        return t._indices().numel() * 8 + t._values().numel() * t._values().element_size(), None
    try:
        s = t.untyped_storage()
        return s.nbytes(), s.data_ptr()
    except Exception:
        return t.numel() * t.element_size(), None


def _median(values):
    v = sorted(x for x in values if x is not None and x == x)
    if not v:
        return None
    n = len(v)
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


def _write_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=1, default=float)
    os.replace(tmp, path)


# ----------------------------------------------------------------------------------------------------- one run

def _gpu_probe(queue):
    try:
        import cupy as cp
        float((cp.arange(10) * 2.0).sum())                     # needs NVRTC (elementwise kernel), as DPCG does
        queue.put("")
    except Exception as e:
        queue.put(f"{type(e).__name__}: {str(e).splitlines()[0]}")


def _child(cfg):
    """One optimization (problem, size, solver, gradient) in a fresh process, instrumented."""
    import numpy as np
    import torch
    import scipy.sparse.linalg as spla
    import pyto.autodiff.sparse_solve as ss
    import pyto.solve.deflation as deflation
    import pyto.topopt.drivers.mma as mma
    from pyto.examples_benchmarks.topopt_structural_benchmarks import StructuralTOExamples, getStructuralTOProblem
    from pyto.physics.structural import hex_structural_fea
    from pyto.solve.solvers import Solvers
    from pyto.topopt.common import gradient_method
    from pyto.topopt.manual_sensitivities import manual_support

    run_dir = cfg["run_dir"]
    os.makedirs(run_dir, exist_ok=True)
    info = {"config": {k: v for k, v in cfg.items() if k != "run_dir"}, "status": "RUNNING", "stage": "build",
            "iterations": []}
    progress_path, result_path = os.path.join(run_dir, "progress.json"), os.path.join(run_dir, "result.json")

    def save(final=False):
        _write_json(result_path if final else progress_path, info)

    def on_cap(rss):
        info.update(status="MEMORY_CAP", note=f"resident memory {rss / GB:.1f} GB passed the cap "
                                              f"{cfg['mem_cap_gb']:.1f} GB during {info['stage']}")
        save(final=True)

    watchdog = _Watchdog(cfg["mem_cap_gb"] * GB, on_cap)
    watchdog.start()
    hwm_ok = _reset_peak()
    t_child = time.perf_counter()

    # ---- build (as build_benchmark does, with this size)
    problem = getattr(StructuralTOExamples, cfg["problem"])
    gpu = cfg["solver"] == "DPCG_GPU"
    solver = Solvers.DPCG if gpu else getattr(Solvers, cfg["solver"])
    t0 = time.perf_counter()
    mesh, mat_prop, bc, elem_body_force, to_params = getStructuralTOProblem(problem, nDOFDesired=cfg["ndof_desired"])
    dsolver = deflation.DeflationSolver(use_gpu=gpu)
    t_mesh = time.perf_counter() - t0
    if solver == Solvers.DPCG:
        nGroups = min(dsolver.maxGroups, max(dsolver.minGroups, round(3 * mesh.num_nodes / dsolver.dofPerGroup)))
        dsolver.create_deflation_groups(mesh, nGroups)
        dsolver.create_deflation_matrix(mesh)
    fe = hex_structural_fea.HexStructuralFEA(mesh=mesh, mat_prop=mat_prop, bc=bc, solver=solver, dsolver=dsolver,
                                             rtol=1e-8, elem_body_force=elem_body_force)
    to_params.Gradient = gradient_method(cfg["gradient"])
    if cfg["max_iters"]:
        to_params.MaxIterations = int(cfg["max_iters"])
    info.update(dofs=3 * mesh.num_nodes, nodes=mesh.num_nodes, elements=mesh.num_elems,
                max_iterations=to_params.MaxIterations, build_s=time.perf_counter() - t0, mesh_s=t_mesh,
                rss_setup_bytes=_proc_status("VmRSS"), peak_build_bytes=_proc_status("VmHWM") if hwm_ok else None,
                torch_threads=torch.get_num_threads(), objective=getattr(to_params.Objective[0], "name", ""),
                constraints=[f"{getattr(c[0], 'name', c[0])} <= {c[2]}" for c in to_params.Constraints])
    if cfg["gradient"] == "ANALYTICAL":
        ok, why = manual_support(to_params, fe)
        if not ok:
            info.update(status="NOT_APPLICABLE", note=why)
            save(final=True)
            return
    if gpu:
        import cupy as cp
        pool = cp.get_default_memory_pool()
    save()

    # ---- instrumentation (wrappers around the library calls; the computation itself is unchanged)
    acc = {}

    def reset_acc():
        acc.clear()
        acc.update(fe_solve_s=0.0, factor_s=0.0, fwd_solve_s=0.0, adjoint_s=0.0, grad_s=0.0, n_fe_solves=0,
                   n_adjoint=0, cg_fwd=[], cg_adj=[], max_residual=0.0, max_res_fwd=0.0, max_res_adj=0.0,
                   graph_bytes=0, graph_ptrs=set(),
                   factor_bytes=None, nnz=None, gpu_bytes=None, in_adjoint=False)

    reset_acc()

    cg_orig = spla.cg

    def cg_counted(A, b, *a, **k):                             # PCG (scipy cg, imported at call time)
        n = [0]
        k["callback"] = lambda _xk: n.__setitem__(0, n[0] + 1)
        out = cg_orig(A, b, *a, **k)
        acc["_cg_last"] = n[0]
        return out

    spla.cg = cg_counted
    pending = {}

    def residual(A, x, b, transpose=False):
        try:
            b = np.asarray(b, dtype=float)
            nb = np.linalg.norm(b)
            if nb == 0:
                return 0.0
            Ax = (A.T @ x) if transpose else (A @ x)
            return float(np.linalg.norm(Ax - b) / nb)
        except Exception:
            return None

    def matrix_of(obj):
        return getattr(obj, "A", None) if hasattr(obj, "A") else getattr(getattr(obj, "factors", None), "A", None)

    def timed(fn, obj, b, adjoint):
        acc["_cg_last"] = None
        t = time.perf_counter()
        x = fn(obj, b)
        dt = time.perf_counter() - t
        if solver == Solvers.DPCG:
            its = getattr(dsolver, "last_iterations", None)
        else:
            its = acc.get("_cg_last")
        A = pending.pop("A", None) if not adjoint else None
        A = A if A is not None else matrix_of(obj)
        r = residual(A, np.asarray(x), b, transpose=adjoint) if A is not None else None
        if r is not None:
            acc["max_residual"] = max(acc["max_residual"], r)
            key = "max_res_adj" if adjoint else "max_res_fwd"
            acc[key] = max(acc[key], r)
        if gpu:
            acc["gpu_bytes"] = max(acc["gpu_bytes"] or 0, pool.total_bytes())
        return x, dt, its

    def wrap_class(cls):
        """Time the solves of every solver object of this class. Wrapping the class methods (not each object)
        keeps no reference to the objects, so each factorization is freed as soon as the library drops it,
        exactly as without measurement (per-object wrappers formed reference cycles that kept the factors
        alive until Python's cycle collector ran, and emptied the PARDISO solver pool)."""
        solve, solve_t = cls.solve, cls.solve_transpose

        def solve_w(self, b):
            if acc["in_adjoint"]:                              # _IterativeSolve.solve_transpose -> self.solve
                return solve(self, b)
            x, dt, its = timed(solve, self, b, False)
            acc["fwd_solve_s"] += dt
            if its is not None and its >= 0:
                acc["cg_fwd"].append(its)
            return x

        def solve_t_w(self, b):
            acc["in_adjoint"] = True
            try:
                x, dt, its = timed(solve_t, self, b, True)
            finally:
                acc["in_adjoint"] = False
            acc["adjoint_s"] += dt
            acc["n_adjoint"] += 1
            if its is not None and its >= 0:
                acc["cg_adj"].append(its)
            return x

        cls.solve, cls.solve_transpose = solve_w, solve_t_w

    for cls in (ss._LUFactors, ss._PardisoSolve, ss._IterativeSolve, ss._DirectEachTime):
        wrap_class(cls)

    factorize_orig = ss._factorize

    def factorize_w(A_csc, kind, params):
        t = time.perf_counter()
        obj = factorize_orig(A_csc, kind, params)
        acc["factor_s"] += time.perf_counter() - t
        acc["nnz"] = int(A_csc.nnz)
        if hasattr(obj, "lu"):                                 # SuperLU: values (8 B) + row indices (4 B)
            acc["factor_bytes"] = int((obj.lu.L.nnz + obj.lu.U.nnz) * 12)
        elif hasattr(obj, "factors"):                          # PARDISO: iparm(15..17), kB
            ip = obj.factors.solver.iparm
            acc["factor_bytes"] = int(max(ip[14], ip[15] + ip[16])) * 1024   # iparm is int32: multiply as Python int
        pending["A"] = A_csc                                   # for the residual of the forward solve only
        return obj

    ss._factorize = factorize_w
    fe_solve_orig = fe.solve

    def fe_solve_w(*a, **k):
        t = time.perf_counter()
        out = fe_solve_orig(*a, **k)
        acc["fe_solve_s"] += time.perf_counter() - t
        acc["n_fe_solves"] += 1
        return out

    fe.solve = fe_solve_w
    grad_orig, manual_orig = torch.autograd.grad, mma.manual_gradients

    def grad_w(*a, **k):
        t = time.perf_counter()
        out = grad_orig(*a, **k)
        acc["grad_s"] += time.perf_counter() - t
        return out

    def manual_w(*a, **k):
        t = time.perf_counter()
        out = manual_orig(*a, **k)
        acc["grad_s"] += time.perf_counter() - t
        return out

    torch.autograd.grad, mma.manual_gradients = grad_w, manual_w

    def pack(t):
        nbytes, ptr = _tensor_bytes(t)
        if ptr is None or ptr not in acc["graph_ptrs"]:
            acc["graph_bytes"] += nbytes
            if ptr is not None:
                acc["graph_ptrs"].add(ptr)
        # Detached (same storage, no copy): handing autograd back the tensor itself makes a reference cycle when it is
        # the output of the op (tensor -> grad_fn -> saved tensor), which kept every iteration's graph and factors.
        return t.detach()

    rng = np.random.default_rng(12345)
    probe = rng.standard_normal(mesh.num_elems)
    its = info["iterations"]
    clock = {"t_opt": None, "eval_end": None}
    run_mma_orig = mma.runMMA

    def run_mma_w(*args, **kwargs):
        args = list(args)
        f = args[2]

        def f_w(x):
            now = time.perf_counter()
            if its:                                            # close the previous iteration (MMA update + peak)
                its[-1]["mma_update_s"] = now - clock["eval_end"]
                its[-1]["iter_s"] = now - its[-1]["_start"]
                its[-1]["peak_rss_bytes"] = _proc_status("VmHWM") if hwm_ok else watchdog.peak
                save()
            else:
                info["driver_setup_s"] = now - clock["t_opt"]
                info["peak_driver_setup_bytes"] = _proc_status("VmHWM") if hwm_ok else None
            _reset_peak()
            reset_acc()
            info["stage"] = f"iteration {len(its) + 1}"
            out = f(x)
            clock["eval_end"] = time.perf_counter()
            obj, grad_obj, c, dcdx = out
            rec = {"k": len(its) + 1, "_start": now, "eval_s": clock["eval_end"] - now,
                   "eval_peak_rss_bytes": _proc_status("VmHWM") if hwm_ok else None,
                   **{key: acc[key] for key in ("fe_solve_s", "factor_s", "fwd_solve_s", "adjoint_s", "grad_s",
                                                "n_fe_solves", "n_adjoint", "max_residual", "max_res_fwd",
                                                "max_res_adj", "graph_bytes",
                                                "factor_bytes", "nnz", "gpu_bytes")},
                   "cg_fwd": _median(acc["cg_fwd"]), "cg_adj": _median(acc["cg_adj"]),
                   "objective": float(history_ref["h"]["objective"][-1]),
                   "volfrac": float(history_ref["h"]["volfrac"][-1]),
                   "constraints": [float(history_ref["h"][key][-1]) for key in history_ref["h"]
                                   if key.startswith("constraint_")]}
            if len(its) == 0:                                  # same design for every run of this size
                g = np.asarray(grad_obj, dtype=float).reshape(-1)
                rec["grad_norm"] = float(np.linalg.norm(g))
                rec["grad_probe"] = float(g @ probe)
                rec["cons_grad_probe"] = [float(np.asarray(row).reshape(-1) @ probe) for row in np.atleast_2d(dcdx)]
            its.append(rec)
            info["max_residual"] = max(info.get("max_residual", 0.0), acc["max_residual"])
            return out

        args[2] = f_w
        out = run_mma_orig(*args, **kwargs)
        now = time.perf_counter()
        if its and "iter_s" not in its[-1]:
            its[-1]["mma_update_s"] = now - clock["eval_end"]
            its[-1]["iter_s"] = now - its[-1]["_start"]
            its[-1]["peak_rss_bytes"] = _proc_status("VmHWM") if hwm_ok else watchdog.peak
        clock["mma_end"] = now
        _reset_peak()
        reset_acc()
        info["stage"] = "final evaluation"
        return out

    mma.runMMA = run_mma_w
    history_ref = {}

    def iteration_callback(history):                           # called inside f, after the book-keeping
        history_ref["h"] = history

    # ---- optimization (as in runTOMethodOnBenchmarks)
    print(f"Running {cfg['problem']} problem using DENSITYMMA method and {cfg['solver']} solver ({cfg['gradient']}), "
          f"{info['dofs']:,} dofs, {info['elements']:,} elements", flush=True)
    print("-" * 50, flush=True)
    clock["t_opt"] = time.perf_counter()
    try:
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
            u, history, success, errorMsg, nFEAs = mma.topopt_mma(
                fe, to_params=to_params, maxMMAIterations=to_params.MaxIterations, timeLimitSecs=cfg["time_limit"] or 1e12,
                binarize_topology=not to_params.HeavisideProjection, print_progress=False,
                iteration_callback=iteration_callback)
    except Exception as e:
        import traceback
        oom = "out of memory" in str(e).lower() or isinstance(e, MemoryError)
        info.update(status="OUT_OF_MEMORY" if oom else "ERROR", note=f"{type(e).__name__}: {e}",
                    traceback=traceback.format_exc()[-3000:], peak_rss_bytes=watchdog.peak)
        for r in its:
            r.pop("_start", None)
        save(final=True)
        return
    t_end = time.perf_counter()
    for r in its:
        r.pop("_start", None)
    opt_s = t_end - clock["t_opt"]
    peaks = [info.get("peak_build_bytes"), info.get("peak_driver_setup_bytes"),
             _proc_status("VmHWM") if hwm_ok else None] + [r.get("peak_rss_bytes") for r in its]
    n = len(its)
    info["final_max_residual"] = acc["max_residual"]           # solves of the final (binarized) evaluation
    worst = info.get("max_residual", 0.0)
    if worst > RESIDUAL_LIMIT:
        status, note = "INACCURATE_SOLVE", f"a linear solve did not converge (relative residual {worst:.1e})"
    elif cfg["time_limit"] and n < to_params.MaxIterations and \
            clock["mma_end"] - clock["t_opt"] >= cfg["time_limit"]:  # runMMA stopped
        status, note = ("TIME_LIMIT" if n >= MIN_ITERATIONS else "TOO_SLOW"), \
            f"{n} of {to_params.MaxIterations + 1} evaluations within the {cfg['time_limit']:g} s limit"
    elif worst > RESIDUAL_WARN:
        status, note = "LOW_ACCURACY", f"largest relative residual {worst:.1e}"
    else:
        status, note = "OK", ""
    if info["final_max_residual"] > RESIDUAL_WARN:
        note = (note + "; " if note else "") + \
            f"final-design solve residual {info['final_max_residual']:.1e} (J final less reliable)"
    info.update(status=status, note=note, success=bool(success), errorMsg=errorMsg, n_iterations=n,
                n_fe_solves=int(nFEAs), opt_s=opt_s, final_s=t_end - clock["mma_end"],
                total_s=time.perf_counter() - t_child, objective_final=float(history["objective"][-1]),
                volfrac_final=float(history["volfrac"][-1]),
                peak_rss_bytes=max([p for p in peaks if p] + [watchdog.peak]), stage="done")
    save(final=True)
    try:                                                       # after result.json: a plotting failure loses nothing
        cam = "xy" if to_params.ExtrudeZ else "iso"
        fe.plot_mesh(save_path=os.path.join(run_dir, "topology.png"), plot_bc=None, title=None, camera_position=cam)
    except Exception as e:
        print(f"topology image not written: {e}", flush=True)


# ---------------------------------------------------------------------------------------------------- driver

def _run_dir(root, problem, ndof, solver, gradient):
    return os.path.join(root, "runs", problem, str(ndof), f"{solver}_{gradient}")


def _row(cfg, info, exitcode):
    """Summary row of one run, from its result.json (or progress.json if the process was killed)."""
    its = [r for r in info.get("iterations", []) if "iter_s" in r]
    later = its[1:] if len(its) > 1 else its                   # iteration 1 carries one-time costs
    med = lambda key, rows=later: _median([r.get(key) for r in rows])
    gbm = lambda v: None if v is None else v / GB
    status = info.get("status", "CRASHED")
    max_evals = (info.get("max_iterations") or 0) + 1        # start design + one per MMA iteration
    if cfg.get("_timed_out") and status in ("RUNNING", "OK", "LOW_ACCURACY", "TIME_LIMIT"):  # stopped at the limit
        status = "TIME_LIMIT" if len(its) >= MIN_ITERATIONS else "TOO_SLOW"
        stage = info.get("stage") or ""
        stage = f"iteration {len(its) + 1}" if stage.startswith("iteration") else stage
        info["note"] = (f"stopped at the {cfg['time_limit']:g} s limit during {stage}: "
                        f"{len(its)} of {max_evals} evaluations completed")
    elif status == "RUNNING":                                  # killed from outside
        status = {3: "MEMORY_CAP", -9: "OUT_OF_MEMORY", 137: "OUT_OF_MEMORY"}.get(exitcode, "CRASHED")
        info["note"] = info.get("note") or f"process ended with exit code {exitcode} during {info.get('stage')}"
    peaks = [r.get("peak_rss_bytes") for r in its] + [info.get("peak_rss_bytes"), info.get("peak_build_bytes")]
    first = its[0] if its else {}
    return {
        "problem": cfg["problem"], "solver": cfg["solver"], "gradient": cfg["gradient"],
        "ndof_desired": cfg["ndof_desired"], "dofs": info.get("dofs"), "nodes": info.get("nodes"),
        "elements": info.get("elements"), "nnz": next((r.get("nnz") for r in its if r.get("nnz")), None),
        "status": status, "note": info.get("note", ""), "iterations": len(its),
        "max_iterations": info.get("max_iterations"), "fe_solves": info.get("n_fe_solves"),
        "est_opt_time_s": ((info.get("driver_setup_s") or 0.0) + sum(r["iter_s"] for r in its) +
                           max(0, max_evals - len(its)) * (_median([r["iter_s"] for r in later]) or 0.0))
        if its else None,
        "opt_time_s": info.get("opt_s"), "total_time_s": info.get("total_s"), "build_s": info.get("build_s"),
        "driver_setup_s": info.get("driver_setup_s"), "final_eval_s": info.get("final_s"),
        "s_per_iter_mean": (sum(r["iter_s"] for r in its) / len(its)) if its else None,
        "s_per_iter": med("iter_s"), "eval_s": med("eval_s"), "fe_solve_s": med("fe_solve_s"),
        "factor_s": med("factor_s"), "fwd_solve_s": med("fwd_solve_s"), "grad_s": med("grad_s"),
        "adjoint_s": med("adjoint_s"), "mma_update_s": med("mma_update_s"), "adjoint_solves": med("n_adjoint"),
        "cg_iters_fwd": med("cg_fwd"), "cg_iters_adj": med("cg_adj"),
        "max_residual": max([r.get("max_residual") or 0.0 for r in its], default=None),
        "max_residual_fwd": max([r.get("max_res_fwd") or 0.0 for r in its], default=None),
        "max_residual_adj": max([r.get("max_res_adj") or 0.0 for r in its], default=None),
        "final_max_residual": info.get("final_max_residual"),
        "rss_setup_gb": gbm(info.get("rss_setup_bytes")),
        "peak_rss_gb": gbm(max([p for p in peaks if p], default=0) or None),
        "peak_iter_rss_gb": gbm(med("peak_rss_bytes")),
        "ram_added_gb": gbm(max([r.get("peak_rss_bytes") or 0 for r in its], default=0) -
                            info["rss_setup_bytes"]) if its and info.get("rss_setup_bytes") else None, "ad_graph_gb": gbm(med("graph_bytes")),
        "factor_mem_gb": gbm(med("factor_bytes")), "gpu_mem_gb": gbm(med("gpu_bytes")),
        "mem_cap_gb": cfg["mem_cap_gb"], "objective_iter1": first.get("objective"),
        "grad_norm_iter1": first.get("grad_norm"), "grad_probe_iter1": first.get("grad_probe"),
        "objective_last_iter": its[-1]["objective"] if its else None,
        "objective_final": info.get("objective_final"), "volfrac_final": info.get("volfrac_final"),
        "objective_name": info.get("objective"), "constraints": "; ".join(info.get("constraints", [])),
        "errorMsg": info.get("errorMsg", ""), "torch_threads": info.get("torch_threads"),
        "run_dir": os.path.relpath(cfg["run_dir"], cfg["root"])}


def _load_rows(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _save_rows(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _write_iterations_csv(run_dir):
    for name in ("result.json", "progress.json"):
        p = os.path.join(run_dir, name)
        if os.path.exists(p):
            with open(p) as f:
                its = json.load(f).get("iterations", [])
            if its:
                keys = [k for k in its[0] if k not in ("cons_grad_probe",)]
                with open(os.path.join(run_dir, "iterations.csv"), "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
                    w.writeheader()
                    w.writerows(its)
            return


def _f(v):
    try:
        v = float(v)
        return v if v == v else None
    except (TypeError, ValueError):
        return None


def _iter_objectives(root, row):
    p = os.path.join(root, row["run_dir"], "iterations.csv")
    if not os.path.exists(p):
        return []
    with open(p, newline="") as f:
        return [float(r["objective"]) for r in csv.DictReader(f)]


def _checks(root, rows):
    """Agreement of every run with the reference run of the same problem and size (PARDISO AUTODIFF, else the first
    direct-solver run): objective at iteration 1 (same design), gradient fingerprint at iteration 1, objective at the
    last iteration both runs reached, final objective."""
    out = []
    groups = {}
    for r in rows:
        groups.setdefault((r["problem"], r["ndof_desired"]), []).append(r)
    order = [("PARDISO", "AUTODIFF"), ("PARDISO", "ANALYTICAL"), ("SPSOLVE", "AUTODIFF"), ("SPSOLVE", "ANALYTICAL")]
    for (problem, ndof), rs in groups.items():
        ok = [r for r in rs if _f(r["iterations"]) and _f(r["objective_iter1"]) is not None]
        ref = next((r for key in order for r in ok if (r["solver"], r["gradient"]) == key), ok[0] if ok else None)
        if ref is None:
            continue
        j_ref = _iter_objectives(root, ref)
        rel = lambda a, b: None if a is None or b is None else abs(a - b) / max(abs(b), 1e-300)
        for r in rs:
            if r is ref or not _f(r["iterations"]):
                continue
            j = _iter_objectives(root, r)
            k = min(len(j), len(j_ref))
            out.append({
                "problem": problem, "ndof_desired": ndof, "dofs": r["dofs"], "solver": r["solver"],
                "gradient": r["gradient"], "reference": f"{ref['solver']} {ref['gradient']}",
                "objective_iter1_rel_diff": rel(_f(r["objective_iter1"]), _f(ref["objective_iter1"])),
                "grad_norm_iter1_rel_diff": rel(_f(r["grad_norm_iter1"]), _f(ref["grad_norm_iter1"])),
                "grad_probe_iter1_rel_diff": rel(_f(r["grad_probe_iter1"]), _f(ref["grad_probe_iter1"])),
                "common_iterations": k,
                "objective_at_common_iter_rel_diff": rel(j[k - 1], j_ref[k - 1]) if k else None,
                "objective_final_rel_diff": rel(_f(r["objective_final"]), _f(ref["objective_final"])),
                "max_residual": _f(r["max_residual"])})
    return out


def _plots(root, rows, problem):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sel = [r for r in rows if r["problem"] == problem and _f(r["dofs"]) and _f(r["s_per_iter"])]
    if not sel:
        return None
    colors = dict(zip(ALL_SOLVERS, ("tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple")))
    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    for solver in ALL_SOLVERS:
        for grad, ls, mk in (("AUTODIFF", "-", "o"), ("ANALYTICAL", "--", "s")):
            s = sorted((r for r in sel if r["solver"] == solver and r["gradient"] == grad), key=lambda r: _f(r["dofs"]))
            if not s:
                continue
            x = [_f(r["dofs"]) for r in s]
            kw = dict(color=colors[solver], ls=ls, marker=mk, label=f"{solver} {grad.lower()}")
            ax[0, 0].loglog(x, [_f(r["s_per_iter"]) for r in s], **kw)
            ax[0, 1].loglog(x, [_f(r["peak_rss_gb"]) for r in s], **kw)
            if grad == "AUTODIFF":
                xe = [_f(r["elements"]) for r in s if _f(r["ad_graph_gb"])]
                ax[1, 0].loglog(xe, [_f(r["ad_graph_gb"]) for r in s if _f(r["ad_graph_gb"])], **kw)
    for solver in ALL_SOLVERS:                                     # AD / analytical time per iteration
        pairs = []
        for r in sel:
            if r["solver"] == solver and r["gradient"] == "AUTODIFF":
                an = next((q for q in sel if q["solver"] == solver and q["gradient"] == "ANALYTICAL"
                           and q["ndof_desired"] == r["ndof_desired"]), None)
                if an:
                    pairs.append((_f(r["dofs"]), _f(r["s_per_iter"]) / _f(an["s_per_iter"])))
        if pairs:
            pairs.sort()
            ax[1, 1].semilogx(*zip(*pairs), color=colors[solver], marker="o", label=solver)
    ax[1, 1].axhline(1.0, color="k", lw=0.8)
    titles = [("Time per MMA iteration (median, s)", "dofs"), ("Peak resident memory (GB)", "dofs"),
              ("AD graph: saved tensors per iteration (GB)", "elements"),
              ("AD time / analytical time per iteration", "dofs")]
    for a, (t, xl) in zip(ax.flat, titles):
        a.set_title(t)
        a.set_xlabel(xl)
        a.grid(True, which="both", alpha=0.3)
        if a.get_legend_handles_labels()[0]:
            a.legend(fontsize=7)
    fig.suptitle(problem)
    fig.tight_layout()
    path = os.path.join(root, f"{problem}_scaling.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return os.path.basename(path)


def write_report(root, tag, header=""):
    rows = _load_rows(os.path.join(root, "solver_scaling_summary.csv"))
    checks = _checks(root, rows)
    if checks:
        _save_rows(os.path.join(root, "solver_scaling_checks.csv"), checks)
    fmt = lambda v, spec: "–" if _f(v) is None else format(_f(v), spec)
    md = ["# Solver scaling: automatic differentiation vs analytical sensitivities", "", header, "",
          "Every row is one complete MMA optimization in its own process. Times are medians over iterations 2 and "
          "later (iteration 1 carries one-time set-up). **Peak RAM** is the kernel's peak resident memory of the "
          "process; **Added** is the peak during the iterations minus the memory after set-up; **AD graph** is the bytes of the tensors autograd saves in one iteration (zero for analytical); "
          "**Factors** is the direct solver's factor memory. **J₁ diff** compares the objective at iteration 1 "
          "(same design in every run of a size) with the reference run, **∇ diff** the gradient fingerprint at "
          "iteration 1 (dot product with a fixed random vector): both must be at solver accuracy. **J diff** "
          "compares the objective at the last iteration both runs reached. **Evals** counts objective evaluations "
          "(FE solves): the start design plus one per MMA iteration. A run is stopped at the time limit; its time per "
          "iteration comes from the evaluations it completed, and **~Opt. time** extrapolates to all planned evaluations. **Residual** is the largest relative residual "
          "|Ax − b|/|b| of the forward / adjoint solves.", ""]
    check = {(c["problem"], str(c["ndof_desired"]), c["solver"], c["gradient"]): c for c in checks}
    for problem in dict.fromkeys(r["problem"] for r in rows):
        sel = sorted((r for r in rows if r["problem"] == problem),
                     key=lambda r: (int(r["ndof_desired"]), ALL_SOLVERS.index(r["solver"]), r["gradient"] != "AUTODIFF"))
        first = sel[0]
        md += [f"## {problem}", "", f"Objective: {first.get('objective_name', '')}; constraints: "
               f"{first.get('constraints', '')}.", ""]
        img = _plots(root, rows, problem)
        if img:
            md += [f"![{problem}]({img})", ""]
        md += ["| dofs | elements | Solver | Gradient | Status | Evals done/planned | Opt. time (s) | s/iter | FE solve (s) | "
               "Gradient (s) | Adjoint (s) | CG its fwd/adj | Peak RAM (GB) | Added (GB) | AD graph (GB) | "
               "Factors (GB) | GPU (GB) | Residual fwd/adj | J₁ diff | ∇ diff | J diff | J final | Topology |",
               "|" + "---|" * 23]
        for r in sel:
            c = check.get((problem, str(r["ndof_desired"]), r["solver"], r["gradient"]), {})
            topo = os.path.join(r["run_dir"], "topology.png")
            topo = f'<img src="{topo}" width="120">' if os.path.exists(os.path.join(root, topo)) else "–"
            status = r["status"] + (f" ({r['note']})" if r.get("note") and r["status"] != "OK" else "")
            md.append(
                f"| {r['dofs'] or '–'} | {r['elements'] or '–'} | {r['solver']} | {r['gradient'].lower()} | {status} | "
                f"{r['iterations']}/{int(_f(r['max_iterations']) or 0) + 1} | "
                f"{('~' + fmt(r['est_opt_time_s'], '.0f')) if r['status'] in ('TIME_LIMIT', 'TOO_SLOW') else fmt(r['opt_time_s'], '.0f')} | "
                f"{fmt(r['s_per_iter'], '.3g')} | "
                f"{fmt(r['fe_solve_s'], '.3g')} | {fmt(r['grad_s'], '.3g')} | {fmt(r['adjoint_s'], '.3g')} | "
                f"{fmt(r['cg_iters_fwd'], '.0f')}/{fmt(r['cg_iters_adj'], '.0f')} | {fmt(r['peak_rss_gb'], '.2f')} | "
                f"{fmt(r['ram_added_gb'], '.2f')} | "
                f"{fmt(r['ad_graph_gb'], '.3f')} | {fmt(r['factor_mem_gb'], '.2f')} | {fmt(r['gpu_mem_gb'], '.2f')} | "
                f"{fmt(r['max_residual_fwd'], '.0e')}/{fmt(r['max_residual_adj'], '.0e')} | "
                f"{fmt(c.get('objective_iter1_rel_diff'), '.1e')} | "
                f"{fmt(c.get('grad_probe_iter1_rel_diff'), '.1e')} | "
                f"{fmt(c.get('objective_at_common_iter_rel_diff'), '.1e')} | {fmt(r['objective_final'], '.5g')} | "
                f"{topo} |")
        md += ["", "**Limits** (largest size that ran, and what stopped the next one):", "",
               "| Solver | Gradient | Largest dofs run | s/iter there | Peak RAM there (GB) | Stopped by |",
               "|---|---|---|---|---|---|"]
        for solver in ALL_SOLVERS:
            for grad in GRADIENTS:
                s = [r for r in sel if r["solver"] == solver and r["gradient"] == grad]
                if not s:
                    continue
                good = [r for r in s if r["status"] in ("OK", "TIME_LIMIT", "LOW_ACCURACY")]
                last = max(good, key=lambda r: int(r["ndof_desired"])) if good else None
                stop = next((r for r in s if r["status"] in STOP), None)
                md.append(f"| {solver} | {grad.lower()} | {last['dofs'] if last else '–'} | "
                          f"{fmt(last['s_per_iter'], '.3g') if last else '–'} | "
                          f"{fmt(last['peak_rss_gb'], '.2f') if last else '–'} | "
                          f"{(stop['status'] + ' at ' + str(stop['dofs'] or stop['ndof_desired'])) if stop else 'not reached'} |")
        md.append("")
    path = os.path.join(root, f"SOLVER_SCALING_{tag}.md")
    with open(path, "w") as f:
        f.write("\n".join(md))
    return path


def _mem_available_gb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024 ** 2
    return float("inf")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Solver scaling study (structural, MMA, AD vs analytical).")
    ap.add_argument("--problems", nargs="*", default=list(PROBLEMS), choices=PROBLEMS)
    ap.add_argument("--solvers", nargs="*", default=list(SOLVERS), choices=ALL_SOLVERS)
    ap.add_argument("--gradients", nargs="*", default=list(GRADIENTS), choices=GRADIENTS)
    ap.add_argument("--dofs", nargs="*", type=int, default=list(DOFS), help="nDOFDesired ladder")
    ap.add_argument("--max-iters", type=int, default=10,
                    help="MMA iterations per optimization (default 10, enough for the per-iteration cost; "
                         "0 = each problem's MaxIterations)")
    ap.add_argument("--time-limit", type=float, default=0.0,
                    help="wall-clock limit per run in seconds (default 0 = no limit); the run is stopped there and "
                         "its per-iteration cost is taken from the iterations it completed")
    ap.add_argument("--mem-cap-gb", type=float, default=70.0,
                    help="memory cap per run; lowered to the memory available at the start of the run minus 3 GB")
    ap.add_argument("--tag", default=str(datetime.date.today()), help="output folder tag; reuse it to continue")
    ap.add_argument("--rerun", action="store_true", help="repeat runs already recorded for this --tag")
    args = ap.parse_args(argv)

    root = os.path.join("Results", f"SolverScaling_{args.tag}")
    os.makedirs(root, exist_ok=True)
    summary = os.path.join(root, "solver_scaling_summary.csv")
    rows = _load_rows(summary)
    import torch
    header = (f"*Tag:* {args.tag}  |  *Machine:* {platform.node()}, {os.cpu_count()} CPUs, "
              f"{_mem_available_gb():.0f} GB available at start, torch threads {torch.get_num_threads()}  |  "
              f"*Time limit:* {f'{args.time_limit:g} s per run' if args.time_limit else 'none'}  |  *Memory cap:* {args.mem_cap_gb:g} GB  |  "
              f"*Iterations:* {args.max_iters or 'each problem’s MaxIterations'}")
    print(f"Solver scaling study -> {root}/\n{header}", flush=True)

    gpu_reason = ""
    if "DPCG_GPU" in args.solvers:
        q = mp.get_context("spawn").Queue()
        p = mp.get_context("spawn").Process(target=_gpu_probe, args=(q,))
        p.start()
        try:
            gpu_reason = q.get(timeout=300)
        except Exception:
            gpu_reason = f"GPU check did not finish (exit code {p.exitcode})"
        p.join(10)
        if gpu_reason:
            print(f"DPCG_GPU skipped, the GPU is not usable from CuPy: {gpu_reason}\n"
                  "  (fix for cupy-cuda11x: conda install -n PyToLib -c conda-forge cudatoolkit=11.8)", flush=True)

    stopped = {}
    for r in rows:
        if r["status"] in STOP:
            key = (r["problem"], r["solver"], r["gradient"])
            stopped[key] = min(stopped.get(key, 10 ** 12), int(r["ndof_desired"]))
    done = {(r["problem"], r["solver"], r["gradient"], int(r["ndof_desired"])) for r in rows}

    for problem in args.problems:
        for ndof in sorted(args.dofs):
            for solver in args.solvers:
                for gradient in args.gradients:
                    key = (problem, solver, gradient)
                    run_dir = _run_dir(root, problem, ndof, solver, gradient)
                    if (problem, solver, gradient, ndof) in done and not args.rerun:
                        continue
                    if key in stopped and ndof > stopped[key] and not args.rerun:
                        continue
                    cfg = dict(problem=problem, solver=solver, gradient=gradient, ndof_desired=ndof,
                               max_iters=args.max_iters, time_limit=args.time_limit, run_dir=run_dir, root=root,
                               mem_cap_gb=round(min(args.mem_cap_gb, _mem_available_gb() - 3.0), 1))
                    print("-" * 50 + f"\n[SCALING {problem}] {solver} {gradient}  nDOFDesired={ndof:,}  "
                          f"(memory cap {cfg['mem_cap_gb']:g} GB)", flush=True)
                    for name in ("result.json", "progress.json"):
                        if os.path.exists(os.path.join(run_dir, name)):
                            os.remove(os.path.join(run_dir, name))
                    if solver == "DPCG_GPU" and gpu_reason:
                        info, code = {"status": "GPU_UNAVAILABLE", "note": gpu_reason}, 0
                    else:
                        p = mp.get_context("spawn").Process(target=_child, args=(cfg,))
                        p.start()
                        p.join(args.time_limit or None)
                        if p.is_alive():                       # wall-clock limit of the run reached
                            cfg["_timed_out"] = True
                            p.kill()
                            p.join()
                        code = p.exitcode
                        info = {}
                        for name in ("result.json", "progress.json"):
                            path = os.path.join(run_dir, name)
                            if os.path.exists(path):
                                with open(path) as f:
                                    info = json.load(f)
                                break
                        _write_iterations_csv(run_dir)
                    row = _row(cfg, info, code)
                    rows = [r for r in rows if (r["problem"], r["solver"], r["gradient"], int(r["ndof_desired"]))
                            != (problem, solver, gradient, ndof)] + [row]
                    _save_rows(summary, rows)
                    rows = _load_rows(summary)
                    if row["status"] in STOP:
                        stopped[key] = min(stopped.get(key, 10 ** 12), ndof)
                    fmt = lambda v, s: "–" if v is None else format(v, s)
                    print(f"Iterations: {row['iterations']} of {(row['max_iterations'] or 0) + 1} | s/iteration: {fmt(row['s_per_iter'], '.3g')} "
                          f"(FE solve {fmt(row['fe_solve_s'], '.3g')}, gradient {fmt(row['grad_s'], '.3g')}, "
                          f"adjoint {fmt(row['adjoint_s'], '.3g')}) | peak RAM {fmt(row['peak_rss_gb'], '.2f')} GB "
                          f"| AD graph {fmt(row['ad_graph_gb'], '.3f')} GB | factors {fmt(row['factor_mem_gb'], '.2f')} "
                          f"GB | max residual {fmt(row['max_residual'], '.1e')}", flush=True)
                    print(f"Status: {row['status']}" + (f" ({row['note']})" if row["note"] else ""), flush=True)
                    if row["status"] in STOP:
                        print(f"{solver} {gradient} stops here for {problem}: larger meshes are skipped", flush=True)
                    print("-" * 50, flush=True)
                    write_report(root, args.tag, header)
    path = write_report(root, args.tag, header)
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
