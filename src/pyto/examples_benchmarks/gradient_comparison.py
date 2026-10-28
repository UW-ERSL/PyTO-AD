"""Automatic differentiation (AD) vs analytical sensitivities on the benchmark problems, measured per iteration.

Every problem of the benchmark sweep that has analytical sensitivities is optimized with MMA, once with
each gradient method, exactly as the sweep does it (build_benchmark: same mesh, solver, iteration limit, binarization;
the same driver call). Only to_params.Gradient differs, so both runs follow the same optimization path: the same
number of FE solves and the same objective to round-off. That is checked for every pair.

Each run is its own process and is measured per iteration (pyto.examples_benchmarks._instrument): time split into
FE solve (assembly + linear solve), gradient (AD backward pass or analytical formula), adjoint solves and optimizer
update; peak memory per iteration; AD graph size; solve residuals. Reported values are medians over iterations 2 and
later, and over repeats (--repeats). AD and analytical runs alternate (AD, analytical, AD, ...) so slow drifts of the
machine affect both alike. Before each run the script waits until the machine is quiet (total CPU use of other
programs below --max-busy percent, sampled for 2 s; at most --wait-quiet minutes); a run started on a busy machine is
flagged in the report.

Run from the repository root (the machine should be otherwise idle):
    export PYTHONPATH=src MPLBACKEND=Agg PYVISTA_OFF_SCREEN=true
    nohup systemd-run --user --scope -p MemoryMax=55G -p MemorySwapMax=0 \\
        python -m pyto.examples_benchmarks.gradient_comparison --repeats 3 > gradient_comparison.log 2>&1 &
    python -m pyto.examples_benchmarks.gradient_comparison --problems LBracketTopLoad_Stress_Vol MBBBeam --max-iters 5
Stopped? Start it again with the same --tag: recorded runs are skipped (--rerun repeats them).

Output (Results/GradientComparison_<tag>/):
    runs.csv                     one row per run (per-iteration medians, totals, memory, residuals, CPU use before start)
    pairs.csv                    one row per method and problem: AD vs analytical side by side, medians over repeats
    GRADIENT_COMPARISON_<tag>.md the pairs as tables, problems not run and why
    runs/<method>/<problem>/<AUTODIFF|ANALYTICAL>_r<k>/   iterations.csv, result.json, topology.png
"""
import argparse
import contextlib
import csv
import datetime
import io
import json
import multiprocessing as mp
import os
import platform
import time

ENUMS = ("StructuralTOExamples", "ThermalTOExamples", "ThermoStructuralTOExamples")
GRADIENTS = ("AUTODIFF", "ANALYTICAL")
METHODS = {"MMA": "DENSITYMMA"}
GB = 1024 ** 3


# --------------------------------------------------------------------------------------------- problem lookup

def _sweep_problems():
    """(enum name, member name) of every problem in the benchmark sweep, in sweep order, without the problems kept
    for the full sweep only (benchmarks_full_sweep_only: BiClampDT4, BiClampDT10, MBBBeamNoHeat)."""
    import pyto.examples_benchmarks.topopt_run_benchmarks as R
    out = []
    for problem in R.allBenchmarks:
        if problem in R.benchmarks_full_sweep_only:
            continue
        key = (type(problem).__name__, problem.name)
        if key not in out:
            out.append(key)
    return out


def _problem(enum_name, name):
    import pyto.examples_benchmarks.topopt_run_benchmarks as R
    return getattr(getattr(R, enum_name), name)


def _category(problem):
    """The sweep's result subfolder of a problem (as runTOMethodOnBenchmarks names it)."""
    import pyto.examples_benchmarks.topopt_run_benchmarks as R
    for name, group in (("Structural-Compliance2.5D_1", R.benchmarks_structural_2_5D_problems_1),
                        ("Structural-Compliance2.5D_2", R.benchmarks_structural_2_5D_problems_2),
                        ("Thermal-Compliance2.5D", R.benchmarks_thermal_2_5D_problems),
                        ("Structural-Compliance3D", R.benchmarks_structural_3D_problems),
                        ("Structural-NonCompliance", R.benchmarks_structural_noncompliance),
                        ("Structural-BodyForce", R.benchmarks_structural_bodyforce),
                        ("ThermoStructural", R.benchmarks_thermostructural_problems)):
        if problem in group:
            return name
    return "Other"


# ------------------------------------------------------------------------------------------------- workers

def _check(enum_name, name, solver_name, queue):
    """Worker: can this problem use analytical sensitivities? (builds it as the sweep does)"""
    try:
        from pyto.examples_benchmarks.topopt_run_benchmarks import build_benchmark
        from pyto.solve.solvers import Solvers
        from pyto.topopt.manual_sensitivities import manual_support
        with contextlib.redirect_stdout(io.StringIO()):
            fe, to_params, _ = build_benchmark(_problem(enum_name, name),
                                               solver=getattr(Solvers, solver_name) if solver_name else None)
        ok, why = manual_support(to_params, fe)
        queue.put({"analytical": ok, "reason": why})
    except Exception as e:
        queue.put({"analytical": False, "reason": f"{type(e).__name__}: {e}", "error": True})


def _write_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=1, default=float)
    os.replace(tmp, path)


def _child(cfg):
    """One optimization (method, problem, gradient), measured per iteration, in a fresh process."""
    import numpy as np
    import torch
    from pyto.examples_benchmarks._instrument import GB as _GB, Instrument, Watchdog, proc_status, summarize
    from pyto.examples_benchmarks.topopt_run_benchmarks import build_benchmark
    from pyto.solve.solvers import Solvers
    from pyto.topopt.common import FEA_MODE, gradient_method
    from pyto.topopt.drivers.mma import topopt_mma

    run_dir = cfg["run_dir"]
    os.makedirs(run_dir, exist_ok=True)
    info = {"config": {k: v for k, v in cfg.items() if k != "run_dir"}, "status": "RUNNING", "stage": "build",
            "iterations": []}
    progress, result = os.path.join(run_dir, "progress.json"), os.path.join(run_dir, "result.json")

    def on_cap(rss):
        info.update(status="MEMORY_CAP", note=f"resident memory {rss / _GB:.1f} GB passed the cap "
                                              f"{cfg['mem_cap_gb']:.1f} GB during {info['stage']}")
        _write_json(result, info)

    Watchdog(cfg["mem_cap_gb"] * _GB, on_cap).start()
    problem = _problem(cfg["enum"], cfg["problem"])
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        fe, to_params, feaMode = build_benchmark(problem, solver=getattr(Solvers, cfg["solver"])
                                                 if cfg["solver"] else None)
    to_params.Gradient = gradient_method(cfg["gradient"])
    if cfg["max_iters"]:
        to_params.MaxIterations = int(cfg["max_iters"])
    per_node = {FEA_MODE.STRUCTURAL: 3, FEA_MODE.THERMAL: 1, FEA_MODE.THERMO_STRUCTURAL: 4}[feaMode]
    info.update(dofs=per_node * fe.mesh.num_nodes, nodes=fe.mesh.num_nodes, elements=fe.mesh.num_elems,
                physics=feaMode.name, solver=fe.solver.name, build_s=time.perf_counter() - t0,
                max_iterations=to_params.MaxIterations, rss_setup_bytes=proc_status("VmRSS"),
                torch_threads=torch.get_num_threads(),
                objective_type=getattr(to_params.Objective[0], "name", str(to_params.Objective[0])),
                constraints=[f"{getattr(c[0], 'name', c[0])} <= {c[2]}" for c in to_params.Constraints])

    def on_iteration(rec):
        info["stage"] = f"iteration {rec['k'] + 1}"
        _write_json(progress, info)

    binarize = not to_params.HeavisideProjection
    if cfg.get("plain"):
        return _child_plain(cfg, info, fe, to_params, binarize, run_dir, result, on_iteration)
    inst = Instrument(fe, on_iteration=on_iteration).install()
    info["iterations"] = inst.iterations
    print(f"Running {cfg['problem']} problem using {METHODS[cfg['method']]} method and {fe.solver.name} solver "
          f"({cfg['gradient']}), {info['dofs']:,} dofs, {info['elements']:,} elements", flush=True)
    print("-" * 50, flush=True)
    inst.start()
    t_opt = time.perf_counter()
    try:
        with inst.graph_hooks():
            _u, history, success, errorMsg, nFEAs = topopt_mma(
                fe, to_params=to_params, maxMMAIterations=to_params.MaxIterations, binarize_topology=binarize,
                print_progress=False, iteration_callback=inst.callback)
    except Exception as e:
        import traceback
        info.update(status="ERROR", note=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-3000:])
        _write_json(result, info)
        return
    opt_s = time.perf_counter() - t_opt
    final = inst.final()
    peaks = [r.get("peak_rss_bytes") for r in inst.iterations] + [final["final_peak_rss_bytes"]]
    info.update(status="OK", success=bool(success), errorMsg=errorMsg, fe_solves=int(nFEAs), opt_time_s=opt_s,
                objective_final=float(np.asarray(history["objective"][-1]).reshape(-1)[0]),
                volfrac_final=float(history["volfrac"][-1]), peak_rss_bytes=max(p for p in peaks if p) if any(peaks)
                else None, stage="done", **final, **summarize(inst.iterations))
    _write_json(result, info)
    with open(os.path.join(run_dir, "iterations.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(inst.iterations[0]) if inst.iterations else ["k"])
        w.writeheader()
        w.writerows(inst.iterations)
    try:                                               # after result.json: a plotting failure loses nothing
        cam = "xy" if to_params.ExtrudeZ else "iso"
        fe.plot_mesh(save_path=os.path.join(run_dir, "topology.png"), plot_bc=None, title=None, camera_position=cam)
    except Exception as e:
        print(f"topology image not written: {e}", flush=True)


def _child_plain(cfg, info, fe, to_params, binarize, run_dir, result, on_iteration):
    """--plain: the optimization with no instrumentation at all (no solver wrappers, no saved-tensor hooks, no
    residual checks); only a wall-clock stamp per iteration from the driver's own iteration_callback, so AD and
    analytical are timed on exactly the code a user runs. Peak memory: the process high-water mark at the end."""
    import numpy as np
    from pyto.examples_benchmarks._instrument import proc_status, summarize
    from pyto.topopt.drivers.mma import topopt_mma
    iterations, t_last = [], [None]
    info["iterations"] = iterations

    def stamp(history):
        now = time.perf_counter()
        rec = {"k": len(iterations) + 1, "iter_s": now - t_last[0],
               "objective": float(np.asarray(history["objective"][-1]).reshape(-1)[0]),
               "volfrac": float(history["volfrac"][-1])}
        iterations.append(rec)
        on_iteration(rec)
        t_last[0] = time.perf_counter()

    t_opt = t_last[0] = time.perf_counter()
    try:
        _u, history, success, errorMsg, nFEAs = topopt_mma(
            fe, to_params=to_params, maxMMAIterations=to_params.MaxIterations, binarize_topology=binarize,
            print_progress=False, iteration_callback=stamp)
    except Exception as e:
        import traceback
        info.update(status="ERROR", note=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-3000:])
        _write_json(result, info)
        return
    opt_s = time.perf_counter() - t_opt
    info.update(status="OK", success=bool(success), errorMsg=errorMsg, fe_solves=int(nFEAs), opt_time_s=opt_s,
                objective_final=float(np.asarray(history["objective"][-1]).reshape(-1)[0]),
                volfrac_final=float(history["volfrac"][-1]), peak_rss_bytes=proc_status("VmHWM"), stage="done",
                s_per_iter_total=opt_s / max(len(iterations), 1), **summarize(iterations))
    _write_json(result, info)
    with open(os.path.join(run_dir, "iterations.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["k", "iter_s", "objective", "volfrac"])
        w.writeheader()
        w.writerows(iterations)


def environment_record():
    """Code version, library versions, thread settings and hardware, written next to every result set."""
    import subprocess
    import sys
    rec = {"date": datetime.datetime.now().isoformat(timespec="seconds"), "python": sys.version.split()[0],
           "platform": platform.platform()}
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()
    rec["commit"] = git("rev-parse", "HEAD")
    diff = subprocess.run(["git", "diff", "HEAD"], capture_output=True).stdout
    import hashlib
    rec["uncommitted_diff_sha1"] = hashlib.sha1(diff).hexdigest() if diff else None
    rec["uncommitted_files"] = git("status", "--short").splitlines()
    for mod in ("torch", "numpy", "scipy", "pypardiso", "mkl"):
        try:
            m = __import__(mod)
            rec[f"{mod}_version"] = getattr(m, "__version__", "?")
        except Exception:
            rec[f"{mod}_version"] = None
    try:
        import torch
        rec["torch_threads"] = torch.get_num_threads()
        rec["torch_config"] = torch.__config__.parallel_info().splitlines()[:8]
        rec["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        pass
    import pyto
    pkg = os.path.dirname(os.path.abspath(pyto.__file__))      # the code actually imported (may be a frozen copy)
    h = hashlib.sha1()
    for d, _sub, files in sorted(os.walk(pkg)):
        for fn in sorted(f for f in files if f.endswith(".py")):
            with open(os.path.join(d, fn), "rb") as f:
                h.update(os.path.relpath(os.path.join(d, fn), pkg).encode() + f.read())
    rec["pyto_path"], rec["pyto_source_sha1"] = pkg, h.hexdigest()
    try:
        rec["gpu_nvidia_smi"] = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                                               capture_output=True, text=True, timeout=10).stdout.strip() or None
    except Exception:
        rec["gpu_nvidia_smi"] = None
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        rec[k] = os.environ.get(k)
    try:
        with open("/proc/cpuinfo") as f:
            rec["cpu"] = next(l.split(":", 1)[1].strip() for l in f if l.startswith("model name"))
        rec["cpus"] = os.cpu_count()
    except Exception:
        pass
    return rec


# -------------------------------------------------------------------------------------------------- driver

def _busy_pct():
    """CPU use of the whole machine in % of all cores, sampled for 2 s (no run of ours is active at that moment)."""
    import psutil
    return psutil.cpu_percent(interval=2.0)


def _mem_available_gb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024 ** 2
    return float("inf")


def _wait_quiet(max_busy, max_wait_min):
    """Wait until other programs use less than max_busy % of the CPUs. Returns the last sample (%)."""
    t_end = time.time() + 60 * max_wait_min
    busy = _busy_pct()
    while busy > max_busy and time.time() < t_end:
        time.sleep(10)
        busy = _busy_pct()
    return busy


def _rows(path):
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


def _f(v):
    try:
        v = float(v)
        return v if v == v else None
    except (TypeError, ValueError):
        return None


def _median(values):
    v = sorted(x for x in values if x is not None)
    if not v:
        return None
    n = len(v)
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


RUN_KEYS = ("status", "note", "dofs", "nodes", "elements", "physics", "solver", "objective_type", "iterations",
            "max_iterations", "fe_solves", "opt_time_s", "build_s", "s_per_iter", "s_per_iter_mean", "iter_time_iqr_s",
            "fe_solve_s", "factor_s", "fwd_solve_s", "grad_s", "adjoint_s", "update_s", "adjoint_solves",
            "cg_iters_fwd", "cg_iters_adj", "max_residual_fwd", "max_residual_adj", "final_max_residual",
            "peak_iter_rss_gb", "ad_graph_gb", "objective_final", "volfrac_final", "success", "errorMsg",
            "torch_threads")


def _run_row(cfg, info, exitcode, busy_before):
    status = info.get("status", "CRASHED")
    if status == "RUNNING":
        status = {3: "MEMORY_CAP", -9: "OUT_OF_MEMORY"}.get(exitcode, "CRASHED")
        info["note"] = info.get("note") or f"process ended with exit code {exitcode} during {info.get('stage')}"
    row = {"category": cfg["category"], "method": cfg["method"], "problem": cfg["problem"], "gradient": cfg["gradient"],
           "repeat": cfg["repeat"]}
    if isinstance(info.get("iterations"), list):       # run stopped early: summarize the iterations it completed
        from pyto.examples_benchmarks._instrument import summarize
        info = {**info, **summarize(info["iterations"])}
    for k in RUN_KEYS:
        row[k] = info.get(k)
    row["status"] = status
    row["peak_rss_gb"] = info["peak_rss_bytes"] / GB if info.get("peak_rss_bytes") else None
    row["setup_rss_gb"] = info["rss_setup_bytes"] / GB if info.get("rss_setup_bytes") else None
    row.update(cpu_busy_before_pct=round(busy_before, 1), max_busy_pct=cfg["max_busy"],
               busy=busy_before > cfg["max_busy"], run_dir=os.path.relpath(cfg["run_dir"], cfg["root"]))
    return row


def build_pairs(runs):
    """One row per (method, problem): AD and analytical side by side, medians over successful repeats."""
    pairs = []
    keys = list(dict.fromkeys((r["method"], r["problem"]) for r in runs))
    for method, problem in keys:
        sel = {g: [r for r in runs if r["method"] == method and r["problem"] == problem and r["gradient"] == g
                   and r["status"] == "OK"] for g in GRADIENTS}
        if not sel["AUTODIFF"] or not sel["ANALYTICAL"]:
            continue
        m = lambda g, k: _median([_f(r.get(k)) for r in sel[g]])
        ad, an = m("AUTODIFF", "s_per_iter"), m("ANALYTICAL", "s_per_iter")
        j_ad, j_an = m("AUTODIFF", "objective_final"), m("ANALYTICAL", "objective_final")
        spread = lambda g: (max(_f(r["s_per_iter"]) for r in sel[g]) - min(_f(r["s_per_iter"]) for r in sel[g])) / \
            m(g, "s_per_iter") if len(sel[g]) > 1 else None
        first = sel["AUTODIFF"][0]
        fe_ad, fe_an = m("AUTODIFF", "fe_solves"), m("ANALYTICAL", "fe_solves")
        rel = abs(j_ad - j_an) / max(abs(j_an), 1e-300) if j_ad is not None and j_an is not None else None
        pairs.append({
            "category": first["category"], "method": method, "problem": problem, "physics": first["physics"],
            "objective_type": first["objective_type"], "dofs": first["dofs"], "elements": first["elements"],
            "solver": first["solver"], "repeats_ad": len(sel["AUTODIFF"]), "repeats_an": len(sel["ANALYTICAL"]),
            "fe_solves_ad": fe_ad, "fe_solves_an": fe_an, "objective_ad": j_ad, "objective_an": j_an,
            "objective_rel_diff": rel, "same_path": fe_ad == fe_an and rel is not None and rel < 1e-6,
            "s_per_iter_ad": ad, "s_per_iter_an": an, "ad_extra_pct": 100 * (ad / an - 1) if ad and an else None,
            "fe_solve_ad": m("AUTODIFF", "fe_solve_s"), "fe_solve_an": m("ANALYTICAL", "fe_solve_s"),
            "grad_ad": m("AUTODIFF", "grad_s"), "grad_an": m("ANALYTICAL", "grad_s"),
            "adjoint_ad": m("AUTODIFF", "adjoint_s"), "adjoint_an": m("ANALYTICAL", "adjoint_s"),
            "adjoint_solves_ad": m("AUTODIFF", "adjoint_solves"), "adjoint_solves_an": m("ANALYTICAL", "adjoint_solves"),
            "update_ad": m("AUTODIFF", "update_s"), "update_an": m("ANALYTICAL", "update_s"),
            "opt_time_ad": m("AUTODIFF", "opt_time_s"), "opt_time_an": m("ANALYTICAL", "opt_time_s"),
            "peak_rss_ad_gb": m("AUTODIFF", "peak_rss_gb"), "peak_rss_an_gb": m("ANALYTICAL", "peak_rss_gb"),
            "ad_graph_gb": m("AUTODIFF", "ad_graph_gb"),
            "spread_ad": spread("AUTODIFF"), "spread_an": spread("ANALYTICAL"),
            "any_busy": any(str(r.get("busy")) == "True" for g in GRADIENTS for r in sel[g])})
    return pairs


def write_report(root, tag, skipped=(), header=""):
    runs = _rows(os.path.join(root, "runs.csv"))
    pairs = build_pairs(runs)
    if pairs:
        _save_rows(os.path.join(root, "pairs.csv"), pairs)
    fmt = lambda v, spec: "–" if _f(v) is None else format(_f(v), spec)
    md = ["# AD vs analytical sensitivities: per-iteration measurement", "", header, "",
          "Each run is a complete optimization as in the benchmark sweep, in its own process; only the gradient method "
          "differs. Times are medians over iterations 2 and later (iteration 1 holds the driver's set-up) and over "
          "repeats. **Extra** = AD time per iteration ÷ analytical − 1. **FE solve** = assembly + linear solve, "
          "**Gradient** = AD backward pass or analytical formula (includes **Adjoint** solves), **Update** = optimizer "
          "update, filtering and bookkeeping. **Same path**: equal FE-solve count and final objectives within 1e-6. "
          "**Spread** = (max − min) / median of the per-iteration time over repeats. ⚠ = a run started while the machine "
          "was busy.", ""]
    for method in dict.fromkeys(p["method"] for p in pairs):
        sel = sorted((p for p in pairs if p["method"] == method), key=lambda p: -(p["ad_extra_pct"] or 0))
        md += [f"## {method}", "",
               "| Problem | dofs | FE solves AD / AN | Same path | s/iter AD | s/iter AN | **Extra** | FE solve AD / AN | "
               "Gradient AD / AN | Adjoint AD / AN | Update AD / AN | Peak RAM AD / AN (GB) | AD graph (GB) | Spread AD / AN | "
               "Objective rel. diff |",
               "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for p in sel:
            md.append(
                f"| {p['problem']}{' ⚠' if p['any_busy'] else ''} | {p['dofs']} | {fmt(p['fe_solves_ad'], '.0f')} / "
                f"{fmt(p['fe_solves_an'], '.0f')} | {'yes' if p['same_path'] else 'NO'} | {fmt(p['s_per_iter_ad'], '.3g')} | "
                f"{fmt(p['s_per_iter_an'], '.3g')} | **{fmt(p['ad_extra_pct'], '+.0f')}%** | "
                f"{fmt(p['fe_solve_ad'], '.3g')} / {fmt(p['fe_solve_an'], '.3g')} | {fmt(p['grad_ad'], '.3g')} / "
                f"{fmt(p['grad_an'], '.3g')} | {fmt(p['adjoint_ad'], '.3g')} / {fmt(p['adjoint_an'], '.3g')} | "
                f"{fmt(p['update_ad'], '.3g')} / "
                f"{fmt(p['update_an'], '.3g')} | {fmt(p['peak_rss_ad_gb'], '.2f')} / {fmt(p['peak_rss_an_gb'], '.2f')} | "
                f"{fmt(p['ad_graph_gb'], '.3f')} | {fmt(p['spread_ad'], '.0%')} / {fmt(p['spread_an'], '.0%')} | "
                f"{fmt(p['objective_rel_diff'], '.1e')} |")
        md.append("")
    failed = [r for r in runs if r["status"] != "OK"]
    if failed:
        md += ["## Runs that did not finish", "", "| Method | Problem | Gradient | Repeat | Status | Note |",
               "|---|---|---|---|---|---|"]
        md += [f"| {r['method']} | {r['problem']} | {r['gradient']} | {r['repeat']} | {r['status']} | {r.get('note', '')} |"
               for r in failed] + [""]
    if skipped:
        md += ["## Not run (analytical sensitivities not available)", "", "| Problem | Reason |", "|---|---|"]
        md += [f"| {p} | {why} |" for p, why in skipped] + [""]
    path = os.path.join(root, f"GRADIENT_COMPARISON_{tag}.md")
    with open(path, "w") as f:
        f.write("\n".join(md))
    return path, pairs


def main(argv=None):
    ap = argparse.ArgumentParser(description="AD vs analytical sensitivities on the benchmark problems, per iteration.")
    ap.add_argument("--problems", nargs="*", help="problem names, or Enum.Name (default: the whole sweep)")
    ap.add_argument("--methods", nargs="*", default=["MMA"], choices=list(METHODS))
    ap.add_argument("--solver", choices=["PARDISO", "SPSOLVE", "DPCG", "PCG"], default=None,
                    help="linear solver for both modes (default: the sweep's choice, PARDISO or DPCG by size)")
    ap.add_argument("--repeats", type=int, default=1, help="runs per problem and gradient method (3 for the paper)")
    ap.add_argument("--max-iters", type=int, default=None,
                    help="iteration limit for quick trials (default: each problem's, as in the sweep)")
    ap.add_argument("--max-busy", type=float, default=5.0,
                    help="wait before each run until other programs use less than this %% of all CPUs "
                         "(at most 5: above that the timing tables are not valid)")
    ap.add_argument("--wait-quiet", type=float, default=30.0, help="longest wait for a quiet machine (minutes)")
    ap.add_argument("--mem-cap-gb", type=float, default=70.0,
                    help="memory cap per run; lowered to the memory available at the start minus 3 GB")
    ap.add_argument("--tag", default=str(datetime.date.today()), help="output folder tag; reuse it to continue")
    ap.add_argument("--rerun", action="store_true", help="repeat runs already recorded for this --tag")
    ap.add_argument("--plain", action="store_true",
                    help="time whole iterations only, with no instrumentation (no phase split, graph bytes or "
                         "residuals): the timing a user sees")
    args = ap.parse_args(argv)
    if args.max_busy > 5.0:      # a looser quiet-machine check still writes timing tables that look valid but aren't
        ap.error(f"--max-busy {args.max_busy:g} is above 5%: timings would not be comparable")

    root = os.path.join("Results", f"GradientComparison_{args.tag}")
    os.makedirs(root, exist_ok=True)
    env = environment_record()
    env.update(plain=bool(args.plain), argv=list(argv) if argv is not None else None)
    _write_json(os.path.join(root, f"environment_{env['date'].replace(':', '')}.json"), env)
    runs_path = os.path.join(root, "runs.csv")

    problems = _sweep_problems()
    if args.problems:                        # names, or Enum.Name for names used twice (MBBBeam)
        chosen = []
        for n in args.problems:
            enum_name, member = n.split(".", 1) if "." in n else (None, n)
            match = [p for p in problems if p[1] == member and enum_name in (None, p[0])] or \
                    [(e, member) for e in ENUMS if enum_name in (None, e) and hasattr(_problem_enum(e), member)]
            if not match:
                raise SystemExit(f"Unknown problem: {n}")
            chosen.append(match[0])
        problems = chosen

    import torch
    header = (f"*Tag:* {args.tag}  |  *Machine:* {platform.node()}, {os.cpu_count()} CPUs, torch threads "
              f"{torch.get_num_threads()}  |  *Solver:* {args.solver or 'sweep default (PARDISO; DPCG above 100k dofs)'}"
              f"  |  *Iterations:* {args.max_iters or 'each problem’s MaxIterations (as in the sweep)'}  |  "
              f"*Repeats:* {args.repeats}  |  *Quiet machine:* other CPU use < {args.max_busy:g}% before each run")
    print(f"AD vs analytical comparison -> {root}/\n{header}", flush=True)

    eligible, skipped, failed = [], [], []
    for enum_name, name in problems:
        q = mp.get_context("spawn").Queue()
        p = mp.get_context("spawn").Process(target=_check, args=(enum_name, name, args.solver, q))
        p.start()
        info = q.get()
        p.join()
        (eligible.append((enum_name, name)) if info["analytical"] else skipped.append((name, info["reason"])))
        print(f"check {name}: {'analytical OK' if info['analytical'] else 'skipped: ' + info['reason']}", flush=True)
        if info.get("error"):
            failed.append((name, info["reason"]))
    if failed:                                     # a problem that does not build is a broken setup, not "no analytical"
        raise SystemExit("These problems failed to build: " + "; ".join(f"{n}: {why}" for n, why in failed))

    rows = _rows(runs_path)
    done = {(r["method"], r["problem"], r["gradient"], str(r["repeat"])) for r in rows if r["status"] == "OK"}
    for method in args.methods:
        for enum_name, name in eligible:
            problem = _problem(enum_name, name)
            for rep in range(1, args.repeats + 1):
                for gradient in GRADIENTS:              # alternate: AD, analytical, AD, analytical, ...
                    if not args.rerun and (method, name, gradient, str(rep)) in done:
                        continue
                    run_dir = os.path.join(root, "runs", method, name, f"{gradient}_r{rep}")
                    for f in ("result.json", "progress.json"):
                        if os.path.exists(os.path.join(run_dir, f)):
                            os.remove(os.path.join(run_dir, f))
                    busy_before = _wait_quiet(args.max_busy, args.wait_quiet)
                    cfg = dict(enum=enum_name, problem=name, method=method, gradient=gradient, repeat=rep,
                               category=_category(problem), solver=args.solver, max_iters=args.max_iters,
                               max_busy=args.max_busy, run_dir=run_dir, root=root, plain=args.plain,
                               mem_cap_gb=round(min(args.mem_cap_gb, _mem_available_gb() - 3.0), 1))
                    print("-" * 50 + f"\n[{method} {gradient}] {name}  (repeat {rep}/{args.repeats}, CPU in use "
                          f"before start {busy_before:.0f}%{' – BUSY' if busy_before > args.max_busy else ''})",
                          flush=True)
                    p = mp.get_context("spawn").Process(target=_child, args=(cfg,))
                    p.start()
                    p.join()
                    info = {}
                    for f in ("result.json", "progress.json"):
                        path = os.path.join(run_dir, f)
                        if os.path.exists(path):
                            with open(path) as fh:
                                info = json.load(fh)
                            break
                    row = _run_row(cfg, info, p.exitcode, busy_before)
                    rows = [r for r in rows if (r["method"], r["problem"], r["gradient"], str(r["repeat"]))
                            != (method, name, gradient, str(rep))] + [row]
                    _save_rows(runs_path, rows)
                    rows = _rows(runs_path)
                    fm = lambda v, s: "–" if _f(v) is None else format(_f(v), s)
                    print(f"Iterations: {row['iterations']} | s/iteration: {fm(row['s_per_iter'], '.3g')} (FE solve "
                          f"{fm(row['fe_solve_s'], '.3g')}, gradient {fm(row['grad_s'], '.3g')}, adjoint "
                          f"{fm(row['adjoint_s'], '.3g')}, update {fm(row['update_s'], '.3g')}) | peak RAM "
                          f"{fm(row['peak_rss_gb'], '.2f')} GB | AD graph {fm(row['ad_graph_gb'], '.3f')} GB", flush=True)
                    print(f"Status: {row['status']}" + (f" ({row['note']})" if row.get("note") else ""), flush=True)
                    print("-" * 50, flush=True)
                    write_report(root, args.tag, skipped, header)
    path, pairs = write_report(root, args.tag, skipped, header)
    print(f"wrote {path}", flush=True)
    return pairs


def _problem_enum(enum_name):
    import pyto.examples_benchmarks.topopt_run_benchmarks as R
    return getattr(R, enum_name)


if __name__ == "__main__":
    main()
