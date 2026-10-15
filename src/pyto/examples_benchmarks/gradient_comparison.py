"""Full benchmark runs with automatic differentiation and with analytical sensitivities, side by side.

This is the benchmark sweep (topopt_run_benchmarks) run twice, once per gradient method, on the problems where
analytical sensitivities apply, for MMA and OC. Every run is a complete optimization
exactly as in the sweep: same build_benchmark problem, solver, iteration limit, binarization, topology images
and summary CSV/tables. Only to_params.Gradient differs. Order: for each problem MMA with AUTODIFF then
ANALYTICAL; after all MMA problems, OC the same way. Each run is its own process (as in the sweep), so neither
mode benefits from the other's caches.

Run from the repository root, like the sweep:
    export PYTHONPATH=src MPLBACKEND=Agg PYVISTA_OFF_SCREEN=true
    nohup python -m pyto.examples_benchmarks.gradient_comparison > gradient_comparison.log 2>&1 &   # hours
    python -m pyto.examples_benchmarks.gradient_comparison --problems LBracketTopLoad_Stress_Vol MBBBeam
Stopped? Start it again with the same --date: runs already recorded are skipped (--rerun repeats them).
Output:
    Results/Results_<date>_AUTODIFF/ and Results/Results_<date>_ANALYTICAL/   the two sweeps (sweep layout)
    Results/GRADIENT_COMPARISON_<date>.md and .csv                            the two side by side
"""
import argparse
import contextlib
import datetime
import glob
import io
import multiprocessing as mp
import os
import platform

ENUMS = ("StructuralTOExamples", "ThermalTOExamples", "ThermoStructuralTOExamples")
GRADIENTS = ("AUTODIFF", "ANALYTICAL")


def _sweep_problems():
    """(enum name, member name) of every problem in the benchmark sweep, in sweep order."""
    import pyto.examples_benchmarks.topopt_run_benchmarks as R
    out = []
    for problem in R.allBenchmarks:
        key = (type(problem).__name__, problem.name)
        if key not in out:
            out.append(key)
    return out


def _problem(enum_name, name):
    import pyto.examples_benchmarks.topopt_run_benchmarks as R
    return getattr(getattr(R, enum_name), name)


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
        queue.put({"analytical": False, "reason": f"{type(e).__name__}: {e}"})


def _run(enum_name, name, method, gradient, date_str, solver_name, max_iters):
    """Worker: one sweep run (one problem, one method, one gradient method)."""
    from pyto.examples_benchmarks.topopt_run_benchmarks import runTOMethodOnBenchmarks
    from pyto.solve.solvers import Solvers
    from pyto.topopt.common import TO_METHODS
    runTOMethodOnBenchmarks(TO_METHODS[method], problems_override=[_problem(enum_name, name)], date_str=date_str,
                            gradient=gradient, solver=getattr(Solvers, solver_name) if solver_name else None,
                            max_iters=max_iters)


def _spawn(target, *args):
    p = mp.get_context("spawn").Process(target=target, args=args)
    p.start()
    p.join()
    return p.exitcode


def _already_done(date_str, method_name, problem):
    """Whether this run is already recorded in the sweep folder (a restarted comparison continues from there)."""
    import pandas as pd
    for f in glob.glob(os.path.join("Results", f"Results_{date_str}", "*", method_name, f"{method_name}_summary.csv")):
        if problem in set(pd.read_csv(f)["name"]):
            return True
    return False


def _results(date_str):
    """{(category, METHOD, problem): row} from one sweep folder's summary CSVs."""
    import pandas as pd
    out = {}
    root = os.path.join("Results", f"Results_{date_str}")
    for f in glob.glob(os.path.join(root, "*", "*", "*_summary.csv")):
        category, method = f.split(os.sep)[-3:-1]
        for row in pd.read_csv(f).to_dict("records"):
            out[(category, method, row["name"])] = row
    return out


def _num(v):
    try:
        v = float(v)
        return v if v == v else None          # NaN -> None
    except (TypeError, ValueError):
        return None


def write_report(date, skipped=(), header=""):
    """Side-by-side report of the two sweeps: Results/GRADIENT_COMPARISON_<date>.md and .csv."""
    import csv
    runs = {g: _results(f"{date}_{g}") for g in GRADIENTS}
    keys = sorted(set(runs["AUTODIFF"]) | set(runs["ANALYTICAL"]), key=lambda k: (k[1] != "DENSITYMMA", k[0], k[2]))
    rows = []
    for key in keys:
        ad, an = runs["AUTODIFF"].get(key, {}), runs["ANALYTICAL"].get(key, {})
        t_ad, t_an = _num(ad.get("time (s)")), _num(an.get("time (s)"))
        n_ad, n_an = _num(ad.get("#FEAs")), _num(an.get("#FEAs"))
        j_ad, j_an = _num(ad.get("objective")), _num(an.get("objective"))
        rows.append({
            "category": key[0], "method": key[1], "problem": key[2], "ndof": ad.get("ndof") or an.get("ndof"),
            "time_ad_s": t_ad, "time_an_s": t_an, "feas_ad": n_ad, "feas_an": n_an,
            "s_per_fea_ad": t_ad / n_ad if t_ad and n_ad else None, "s_per_fea_an": t_an / n_an if t_an and n_an else None,
            "speedup_total": t_ad / t_an if t_ad and t_an else None,
            "objective_ad": j_ad, "objective_an": j_an,
            "objective_rel_diff": abs(j_an - j_ad) / max(abs(j_ad), 1e-300) if j_ad is not None and j_an is not None else None,
            "volfrac_ad": _num(ad.get("volfrac")), "volfrac_an": _num(an.get("volfrac")),
            "message_ad": ad.get("errorMsg", "not run"), "message_an": an.get("errorMsg", "not run")})
    os.makedirs("Results", exist_ok=True)
    with open(os.path.join("Results", f"GRADIENT_COMPARISON_{date}.csv"), "w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    fmt = lambda v, spec: "–" if v is None else format(v, spec)
    md = ["# Automatic differentiation vs analytical sensitivities: full benchmark runs", "", header,
          "",
          f"The two sweeps: `Results/Results_{date}_AUTODIFF/` and `Results/Results_{date}_ANALYTICAL/` (same layout as "
          "a normal sweep). Every run is a complete optimization as in the sweep, in its own process; only the "
          "gradient method differs.",
          "",
          "- **Time** = the optimizer call (the sweep's `time (s)`); **FEAs** = FE solves (iterations).",
          "- **Speed-up** = AD time ÷ analytical time for the whole run (above 1: analytical faster). "
          "**s/FEA** removes the effect of different iteration counts.",
          "- **Rel. diff** compares the final objectives: the gradients agree to about 1e-12, so a difference "
          "comes from round-off steering the optimizer onto a slightly different path.", ""]
    for method in dict.fromkeys(r["method"] for r in rows):
        sel = [r for r in rows if r["method"] == method]
        md += [f"## {method}", "",
               "| Category | Problem | dofs | FEAs AD / AN | Time AD (s) | Time AN (s) | Speed-up | s/FEA AD | s/FEA AN | "
               "Objective AD | Objective AN | Rel. diff | Vol AD / AN | Topology AD | Topology AN |",
               "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        tot_ad = tot_an = 0.0
        for r in sel:
            img = lambda g: (f'<img src="Results_{date}_{g}/{r["category"]}/{method}/{r["problem"]}.png" width="160">'
                             if os.path.exists(os.path.join("Results", f"Results_{date}_{g}", r["category"], method,
                                                            f"{r['problem']}.png")) else "–")
            if r["time_ad_s"] and r["time_an_s"]:
                tot_ad, tot_an = tot_ad + r["time_ad_s"], tot_an + r["time_an_s"]
            note = "" if r["message_ad"] == r["message_an"] == "No errors." else \
                f" ⚠ AD: {r['message_ad']}; AN: {r['message_an']}"
            md.append(f"| {r['category']} | {r['problem']}{note} | {r['ndof']} | "
                      f"{fmt(r['feas_ad'], '.0f')} / {fmt(r['feas_an'], '.0f')} | {fmt(r['time_ad_s'], '.1f')} | "
                      f"{fmt(r['time_an_s'], '.1f')} | {fmt(r['speedup_total'], '.2f')} | {fmt(r['s_per_fea_ad'], '.3f')} | "
                      f"{fmt(r['s_per_fea_an'], '.3f')} | {fmt(r['objective_ad'], '.6g')} | {fmt(r['objective_an'], '.6g')} | "
                      f"{fmt(r['objective_rel_diff'], '.1e')} | {fmt(r['volfrac_ad'], '.3f')} / {fmt(r['volfrac_an'], '.3f')} | "
                      f"{img('AUTODIFF')} | {img('ANALYTICAL')} |")
        if tot_an:
            md.append(f"| **Total** | | | | **{tot_ad:.0f}** | **{tot_an:.0f}** | **{tot_ad / tot_an:.2f}** | | | | | | | | |")
        md.append("")
    if skipped:
        md += ["## Not run (analytical sensitivities not available)", "", "| Problem | Reason |", "|---|---|"]
        md += [f"| {p} | {why} |" for p, why in skipped] + [""]
    path = os.path.join("Results", f"GRADIENT_COMPARISON_{date}.md")
    with open(path, "w") as f:
        f.write("\n".join(md))
    return path, rows


def main(argv=None):
    ap = argparse.ArgumentParser(description="Benchmark sweep with AD and with analytical sensitivities (MMA, OC).")
    ap.add_argument("--problems", nargs="*", help="problem names, or Enum.Name (default: the whole sweep)")
    ap.add_argument("--methods", nargs="*", default=["MMA", "OC"], choices=["MMA", "OC"])
    ap.add_argument("--solver", choices=["PARDISO", "SPSOLVE", "DPCG"], default=None,
                    help="linear solver for both modes (default: the sweep's choice, PARDISO or DPCG by size)")
    ap.add_argument("--max-iters", type=int, default=None,
                    help="iteration limit for quick trials (default: each problem's, as in the sweep)")
    ap.add_argument("--date", default=str(datetime.date.today()),
                    help="tag of the output folders; rerun with the same --date to continue a stopped comparison")
    ap.add_argument("--rerun", action="store_true", help="repeat runs already recorded for this --date")
    args = ap.parse_args(argv)

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

    eligible, skipped = [], []
    for enum_name, name in problems:
        q = mp.get_context("spawn").Queue()
        p = mp.get_context("spawn").Process(target=_check, args=(enum_name, name, args.solver, q))
        p.start()
        info = q.get()
        p.join()
        (eligible.append((enum_name, name)) if info["analytical"] else skipped.append((name, info["reason"])))
        print(f"check {name}: {'analytical OK' if info['analytical'] else 'skipped: ' + info['reason']}", flush=True)

    method_names = {"MMA": "DENSITYMMA", "OC": "DENSITYOCM"}
    for method in args.methods:
        for enum_name, name in eligible:
            for gradient in GRADIENTS:
                if not args.rerun and _already_done(f"{args.date}_{gradient}", method_names[method], name):
                    print(f"[{method} {gradient}] {name}: already done for {args.date}, skipped", flush=True)
                    continue
                print("-" * 50 + f"\n[{method} {gradient}] {name}", flush=True)
                code = _spawn(_run, enum_name, name, method_names[method], gradient, f"{args.date}_{gradient}",
                              args.solver, args.max_iters)
                if code != 0:
                    print(f"WARNING: {method} {gradient} {name} exited with code {code} (e.g. out of memory)", flush=True)
                print("-" * 50, flush=True)

    from pyto.examples_benchmarks.topopt_run_benchmarks import combine_results, create_summary_tables
    for gradient in GRADIENTS:
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                combine_results(f"{args.date}_{gradient}")
                create_summary_tables(f"{args.date}_{gradient}")
        except Exception as e:
            print(f"summary tables for {gradient} failed: {e}")
    import torch
    header = (f"*Date:* {args.date}  |  *Machine:* {platform.node()}, {os.cpu_count()} CPUs, torch threads "
              f"{torch.get_num_threads()}  |  *Solver:* {args.solver or 'sweep default (PARDISO; DPCG above 100k dofs)'}"
              f"  |  *Iterations:* {args.max_iters or 'each problem’s MaxIterations (as in the sweep)'}")
    path, rows = write_report(args.date, skipped, header)
    print(f"wrote {path}")
    return rows


def _problem_enum(enum_name):
    import pyto.examples_benchmarks.topopt_run_benchmarks as R
    return getattr(R, enum_name)


if __name__ == "__main__":
    main()
