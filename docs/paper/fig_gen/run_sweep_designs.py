"""Sweep designs for the 'Tutorial problems at a glance' gallery, saved with their mesh so they can be drawn from
any view.

The benchmark sweep saved only pictures, drawn from a fixed camera with z up, which lays the thick L-brackets on their
side. These four 3D problems are rerun exactly as the sweep runs them (build_benchmark, MMA with AD gradients, the
problem's own settings, 0/1 binarization), and each final design is saved to _cache/sweep_<problem>.npz with the run's
objective, volume fraction, FE solves and time. fig_sweep_galleries.py draws them from the view of the README GIFs.

Run from docs/paper/fig_gen:  python run_sweep_designs.py [problem ...]     (about 20 min for all four)
"""
import contextlib
import io
import json
import os
import sys
import time

import numpy as np

from _style import CACHE

from pyto.examples_benchmarks.topopt_run_benchmarks import StructuralTOExamples, build_benchmark
from pyto.topopt.drivers.mma import topopt_mma

PROBLEMS = ("LBracketThickMidLoad", "LBracketThickTopLoad", "Multiload", "EdgeCantilever")


def run(name):
    fe, to_params, _ = build_benchmark(getattr(StructuralTOExamples, name))
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        _u, history, success, message, n_fea = topopt_mma(
            fe, to_params=to_params, maxMMAIterations=to_params.MaxIterations,
            binarize_topology=not to_params.HeavisideProjection, print_progress=False)
    stats = dict(problem=name, dofs=int(3 * fe.mesh.num_nodes), fe_solves=int(n_fea),
                 objective=float(np.ravel(history["objective"][-1])[0]), volume=float(history["volfrac"][-1]),
                 success=bool(success), message=message, opt_time_s=time.perf_counter() - t0)
    os.makedirs(CACHE, exist_ok=True)
    np.savez_compressed(os.path.join(CACHE, f"sweep_{name}.npz"), x=np.asarray(fe.mesh.elemPseudoDensity, float),
                        node_xyz=np.asarray(fe.mesh.node_xyz), elems=np.asarray(fe.mesh.elemArray)[:, :8].astype(np.int64),
                        fixed=np.unique(np.asarray(fe.bc.fixed_dofs) // 3), stats=json.dumps(stats))
    print(name, json.dumps(stats), flush=True)


if __name__ == "__main__":
    for name in sys.argv[1:] or PROBLEMS:
        run(name)
