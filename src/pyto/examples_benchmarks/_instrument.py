"""Per-iteration measurement of an optimization run, from outside the library (used by gradient_comparison).

Wraps the library calls of one process and splits each optimizer iteration into phases. The FE and optimizer code
is unchanged; only timers and counters are added:
  fe_solve_s   FE solve: assembly, boundary conditions, linear solve (fe_solver.solve)
  factor_s     factorization / solver set-up inside it      fwd_solve_s  forward solves with those factors
  grad_s       gradient: autograd backward pass (AD) or the analytical sensitivity formula, incl. adjoint solves
  adjoint_s    adjoint solves (with the forward factors)    update_s     the rest: optimizer update, filtering,
                                                                         bookkeeping (iter_s - fe_solve_s - grad_s)
  peak_rss     peak resident memory of the iteration (kernel high-water mark, reset each iteration)
  graph_bytes  bytes of tensors autograd saves in the iteration (zero for analytical sensitivities)
  residuals    |Ax - b| / |b| of every forward and adjoint solve; CG iterations for DPCG/PCG

Iteration boundaries come from the drivers' iteration_callback (MMA calls it once per iteration, after the
FE solve and gradient), so one record spans a whole iteration. Iteration 1 also holds the driver's set-up (filter,
normalisation solves); report medians over iterations 2 and later.
"""
import os
import threading
import time

import numpy as np

GB = 1024 ** 3


def proc_status(key):
    """VmRSS / VmHWM of this process in bytes."""
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith(key + ":"):
                return int(line.split()[1]) * 1024
    return 0


def reset_peak():
    """Reset the kernel's peak-RSS counter (VmHWM) so the next read is the peak of the coming phase."""
    try:
        with open("/proc/self/clear_refs", "w") as f:
            f.write("5")
        return True
    except OSError:
        return False


def median(values):
    v = sorted(x for x in values if x is not None and x == x)
    if not v:
        return None
    n = len(v)
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


class Watchdog(threading.Thread):
    """Samples RSS; above the cap it calls on_cap and ends the process before the kernel's OOM killer acts."""

    def __init__(self, cap_bytes, on_cap):
        super().__init__(daemon=True)
        self.cap, self.on_cap, self.peak = cap_bytes, on_cap, 0

    def run(self):
        while True:
            rss = proc_status("VmRSS")
            self.peak = max(self.peak, rss)
            if rss > self.cap:
                self.on_cap(rss)
                os._exit(3)
            time.sleep(0.02)


def _tensor_bytes(t):
    import torch
    if t.layout != torch.strided:
        return t._indices().numel() * 8 + t._values().numel() * t._values().element_size(), None
    try:
        s = t.untyped_storage()
        return s.nbytes(), s.data_ptr()
    except Exception:
        return t.numel() * t.element_size(), None


class Instrument:
    """Install with Instrument(fe_solver).install(); pass .callback as the driver's iteration_callback and run the
    driver inside `with inst.graph_hooks():`. Records are in .iterations (one dict per iteration)."""

    def __init__(self, fe_solver, on_iteration=None):
        self.fe = fe_solver
        self.on_iteration = on_iteration          # called with the new record (e.g. to save progress)
        self.iterations = []
        self.hwm_ok = reset_peak()
        self.acc = {}
        self._reset()
        self.t_start = None
        self.pending = {}

    def _reset(self):
        self.acc = dict(fe_solve_s=0.0, factor_s=0.0, fwd_solve_s=0.0, adjoint_s=0.0, grad_s=0.0, n_fe_solves=0,
                        n_adjoint=0, cg_fwd=[], cg_adj=[], max_res_fwd=0.0, max_res_adj=0.0, graph_bytes=0,
                        graph_ptrs=set(), in_adjoint=False, cg_last=None)

    # ------------------------------------------------------------------------------------------------ wrappers
    def install(self):
        import scipy.sparse.linalg as spla
        import torch
        import pyto.autodiff.sparse_solve as ss
        import pyto.topopt.drivers.mma as mma
        pending = self.pending
        inst = self

        cg_orig = spla.cg

        def cg_counted(A, b, *a, **k):                # PCG (scipy cg, imported at call time)
            n = [0]
            k["callback"] = lambda _xk: n.__setitem__(0, n[0] + 1)
            out = cg_orig(A, b, *a, **k)
            inst.acc["cg_last"] = n[0]
            return out

        spla.cg = cg_counted

        def residual(A, x, b, transpose=False):
            try:
                b = np.asarray(b, dtype=float)
                nb = np.linalg.norm(b)
                if nb == 0:
                    return 0.0
                return float(np.linalg.norm(((A.T @ x) if transpose else (A @ x)) - b) / nb)
            except Exception:
                return None

        def timed(fn, obj, b, adjoint):
            inst.acc["cg_last"] = None
            t = time.perf_counter()
            x = fn(obj, b)
            dt = time.perf_counter() - t
            dsolver = getattr(obj, "dsolver", None)
            its = getattr(dsolver, "last_iterations", None) if dsolver is not None else inst.acc["cg_last"]
            A = None if adjoint else pending.pop("A", None)
            if A is None:
                A = getattr(obj, "A", None) if hasattr(obj, "A") else getattr(getattr(obj, "factors", None), "A", None)
            r = residual(A, np.asarray(x), b, transpose=adjoint) if A is not None else None
            key = "max_res_adj" if adjoint else "max_res_fwd"
            if r is not None:
                inst.acc[key] = max(inst.acc[key], r)
            return x, dt, its

        def wrap_class(cls):
            """Time the solves of every solver object of this class. Wrapping the class methods (not each object)
            keeps no reference to the objects, so each factorization is freed as soon as the library drops it,
            exactly as without measurement. (Wrappers stored on each object referenced the object itself; those
            cycles kept every iteration's factors, and with AD its graph, alive until Python's cycle collector ran:
            memory grew by about one iteration's worth per iteration.)"""
            solve, solve_t = cls.solve, cls.solve_transpose

            def solve_w(self, b):
                if inst.acc["in_adjoint"]:             # _IterativeSolve.solve_transpose calls self.solve
                    return solve(self, b)
                x, dt, its = timed(solve, self, b, False)
                inst.acc["fwd_solve_s"] += dt
                if its is not None:
                    inst.acc["cg_fwd"].append(its)
                return x

            def solve_t_w(self, b):
                inst.acc["in_adjoint"] = True
                try:
                    x, dt, its = timed(solve_t, self, b, True)
                finally:
                    inst.acc["in_adjoint"] = False
                inst.acc["adjoint_s"] += dt
                inst.acc["n_adjoint"] += 1
                if its is not None:
                    inst.acc["cg_adj"].append(its)
                return x

            cls.solve, cls.solve_transpose = solve_w, solve_t_w

        for cls in (ss._LUFactors, ss._PardisoSolve, ss._IterativeSolve, ss._DirectEachTime):
            wrap_class(cls)

        factorize_orig = ss._factorize

        def factorize_w(A_csc, kind, params):
            t = time.perf_counter()
            obj = factorize_orig(A_csc, kind, params)
            inst.acc["factor_s"] += time.perf_counter() - t
            pending["A"] = A_csc                       # for the residual of the forward solve only
            return obj

        ss._factorize = factorize_w

        fe_solve_orig = self.fe.solve

        def fe_solve_w(*a, **k):
            t = time.perf_counter()
            out = fe_solve_orig(*a, **k)
            inst.acc["fe_solve_s"] += time.perf_counter() - t
            inst.acc["n_fe_solves"] += 1
            return out

        self.fe.solve = fe_solve_w

        def timed_grad(fn):
            def w(*a, **k):
                t = time.perf_counter()
                out = fn(*a, **k)
                inst.acc["grad_s"] += time.perf_counter() - t
                return out
            return w

        torch.autograd.grad = timed_grad(torch.autograd.grad)
        mma.manual_gradients = timed_grad(mma.manual_gradients)
        return self

    def graph_hooks(self):
        import torch
        inst = self

        def pack(t):
            nbytes, ptr = _tensor_bytes(t)
            if ptr is None or ptr not in inst.acc["graph_ptrs"]:
                inst.acc["graph_bytes"] += nbytes
                if ptr is not None:
                    inst.acc["graph_ptrs"].add(ptr)
            # Detached (same storage, no copy): handing autograd back the tensor itself makes a reference cycle when
            # it is the output of the op (tensor -> grad_fn -> saved tensor), which kept every iteration's graph alive.
            return t.detach()

        return torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t)

    # ------------------------------------------------------------------------------------------------ records
    def start(self):
        reset_peak()
        self._reset()
        self.t_start = self.t_last = time.perf_counter()

    def callback(self, history):
        """Driver's iteration_callback: closes the record of the iteration that just finished."""
        now = time.perf_counter()
        a = self.acc
        rec = {"k": len(self.iterations) + 1, "iter_s": now - self.t_last,
               **{key: a[key] for key in ("fe_solve_s", "factor_s", "fwd_solve_s", "adjoint_s", "grad_s",
                                          "n_fe_solves", "n_adjoint", "max_res_fwd", "max_res_adj", "graph_bytes")},
               "cg_fwd": median(a["cg_fwd"]), "cg_adj": median(a["cg_adj"]),
               "peak_rss_bytes": proc_status("VmHWM") if self.hwm_ok else None,
               "objective": float(np.asarray(history["objective"][-1]).reshape(-1)[0]),
               "volfrac": float(history["volfrac"][-1])}
        rec["update_s"] = max(rec["iter_s"] - rec["fe_solve_s"] - rec["grad_s"], 0.0)
        self.iterations.append(rec)
        if self.on_iteration is not None:
            self.on_iteration(rec)
        reset_peak()
        self._reset()
        self.t_last = time.perf_counter()

    def final(self):
        """The part after the last iteration (final evaluation, binarization): time, peak memory, residual."""
        return {"final_s": time.perf_counter() - self.t_last,
                "final_peak_rss_bytes": proc_status("VmHWM") if self.hwm_ok else None,
                "final_max_residual": max(self.acc["max_res_fwd"], self.acc["max_res_adj"])}


def summarize(iterations):
    """Medians over iterations 2.. (iteration 1 holds one-time set-up) and totals."""
    later = iterations[1:] if len(iterations) > 1 else iterations
    med = lambda key: median([r.get(key) for r in later])
    return {
        "iterations": len(iterations),
        "s_per_iter": med("iter_s"), "s_per_iter_mean": (sum(r["iter_s"] for r in iterations) / len(iterations))
        if iterations else None,
        "fe_solve_s": med("fe_solve_s"), "factor_s": med("factor_s"), "fwd_solve_s": med("fwd_solve_s"),
        "grad_s": med("grad_s"), "adjoint_s": med("adjoint_s"), "update_s": med("update_s"),
        "adjoint_solves": med("n_adjoint"), "fe_solves_per_iter": med("n_fe_solves"),
        "cg_iters_fwd": med("cg_fwd"), "cg_iters_adj": med("cg_adj"),
        "max_residual_fwd": max([r["max_res_fwd"] for r in iterations if r.get("max_res_fwd") is not None],
                                default=None),
        "max_residual_adj": max([r["max_res_adj"] for r in iterations if r.get("max_res_adj") is not None],
                                default=None),
        "peak_iter_rss_gb": (med("peak_rss_bytes") or 0) / GB if iterations else None,
        "ad_graph_gb": (med("graph_bytes") or 0) / GB if iterations else None,
        "iter_time_iqr_s": _iqr([r["iter_s"] for r in later]),
    }


def _iqr(values):
    v = sorted(values)
    if len(v) < 4:
        return None
    return float(np.percentile(v, 75) - np.percentile(v, 25))
