"""STL recovery: CNN smoothing against a Gaussian filter, on the two designs the paper shows recovered.

  quickstart   docs/examples/quickstart.py at nDOFDesired = 80000 (Fig. 2a; docs/paper/fig_gen/_cache, made by
               docs/paper/fig_gen/fig02_quickstart.py), domain ThreeHoleBracketThick.STL
  gui          the recorded GUI session's design (62,849 elements; Fig. 4, Table 11), domain from the session

Methods (pyto.io.topopt_stl_recovery.extract_isosurface, then subtract_voids_from_stl):
  cnn       seeds 0, 1, 2. The library does not seed the CNN fit, so a recovery is random; the seeds here make
            the comparison repeatable and measure the spread between fits.
  gaussian  sigma = 0.5 and 1.0 grid spacings (deterministic)

Measured per run: time of the smoothing + iso-surface step; the part's volume over the voxel design's (elements with
density >= 0.5); overlap with the voxel design (IoU at element centres: inside the part vs density >= 0.5); surface
roughness (area-weighted mean angle between the normals of adjacent triangles, degrees; lower = smoother); one closed
body. For the CNN, the spread between seeds (volume range, IoU of seed 0 with seeds 1 and 2).
gui only: tetrahedral re-analysis of CNN seed 0 and Gaussian sigma 0.5, as the GUI row of Table 11
(pyto.examples_benchmarks.body_fitted_study, case ThreeHoleBracketGUI: benchmark supports and load, surface
simplified to the same triangle count before meshing, compliance corrected by that case's full-domain ratio).

Run from the repository root (a few minutes; the tet solves are about 1.2 M DOF each):
    PYTHONPATH=src MPLBACKEND=Agg PYVISTA_OFF_SCREEN=true python docs/paper/analysis/compare_smoothing.py
Output: Results/SmoothingComparison/current/ (or $PYTO_SMOOTHING_OUT): comparison.json, comparison.md, the STLs.
"""
import contextlib
import io
import json
import os
import sys
import time

import numpy as np
import pyvista as pv
import torch
import trimesh

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(ROOT, "src"))
from pyto.io.topopt_stl_recovery import extract_isosurface, subtract_voids_from_stl  # noqa: E402

OUT = os.environ.get("PYTO_SMOOTHING_OUT") or os.path.join(ROOT, "Results", "SmoothingComparison", "current")
GUI = os.path.join(ROOT, "Results", "GUISession_ThreeHole")
DOMAIN = os.path.join(ROOT, "Models", "ThreeHoleBracket", "ThreeHoleBracketThick.STL")
RUNS = [("cnn", 0), ("cnn", 1), ("cnn", 2), ("gaussian", 0.5), ("gaussian", 1.0)]
TET_KEYS = ("cnn_0", "gaussian_0.5")


def hex_grid(xyz, elems):
    cells = np.hstack([np.full((len(elems), 1), 8), elems]).ravel()
    return pv.UnstructuredGrid(cells, np.full(len(elems), pv.CellType.HEXAHEDRON, dtype=np.uint8), xyz)


def cases():
    q = np.load(os.path.join(ROOT, "docs", "paper", "fig_gen", "_cache", "quickstart_compliance_80000.npz"))
    g = hex_grid(q["node_xyz"].astype(float), q["elems"])
    g.cell_data["density"] = q["x"]
    yield "quickstart", g.cell_data_to_point_data(), pv.read(DOMAIN).clean(), q["x"], g
    vtu = pv.read(os.path.join(GUI, "ThreeHoleBracketThick.vtu"))
    rho = np.load(os.path.join(GUI, "density.npy"))
    yield "gui", vtu, pv.read(os.path.join(GUI, "ThreeHoleBracketThick.STL")).clean(), rho, vtu


def roughness(t):
    """Area-weighted mean angle (deg) between the normals of adjacent triangles."""
    ang = np.degrees(t.face_adjacency_angles)
    w = t.area_faces[t.face_adjacency].mean(1)
    return float((ang * w).sum() / w.sum())


def main():
    os.makedirs(OUT, exist_ok=True)
    out = {}
    for name, vtu, domain, rho, mesh in cases():
        domain = domain.triangulate().compute_normals()
        centres = np.asarray(mesh.cell_centers().points)
        solid = rho >= 0.5
        h = np.prod([np.diff(np.unique(np.round(np.asarray(mesh.points)[:, k], 9)))[0] for k in range(3)])
        v_hex = solid.sum() * h
        res, inside = {}, {}
        for method, par in RUNS:
            key = f"{method}_{par}"
            if method == "cnn":
                torch.manual_seed(par)
                np.random.seed(par)
            t0 = time.time()
            with contextlib.redirect_stdout(io.StringIO()):
                void = extract_isosurface(vtu, 0.5, method=method, sigma=par if method == "gaussian" else 1.0)
            t_smooth = time.time() - t0
            stl = os.path.join(OUT, f"{name}_{key}.stl")
            try:
                subtract_voids_from_stl(domain, void).save(stl)
            except ValueError as e:                  # the Boolean step refuses a void surface that is not closed
                res[key] = {"time_s": t_smooth, "error": f"Boolean step failed: {e}"}
                print(f"{name:10s} {key:13s} {t_smooth:6.1f} s  Boolean step failed: {e}", flush=True)
                continue
            t = trimesh.load(stl)
            inside[key] = t.contains(centres)
            iou = float((inside[key] & solid).sum() / (inside[key] | solid).sum())
            res[key] = {"time_s": t_smooth, "volume_over_voxel": float(t.volume / v_hex), "iou_voxel": iou,
                        "roughness_deg": roughness(t), "triangles": int(len(t.faces)),
                        "one_closed_body": bool(t.is_watertight and len(t.split(only_watertight=False)) == 1)}
            print(f"{name:10s} {key:13s} {t_smooth:6.1f} s  V/Vvox {res[key]['volume_over_voxel']:.3f}  "
                  f"IoU {iou:.3f}  rough {res[key]['roughness_deg']:.2f} deg  closed {res[key]['one_closed_body']}",
                  flush=True)
        seeds = [inside[f"cnn_{s}"] for s in (0, 1, 2)]
        spread = {"volume_range": float(np.ptp([res[f"cnn_{s}"]["volume_over_voxel"] for s in (0, 1, 2)])),
                  "iou_seed0_vs_others": [float((seeds[0] & s).sum() / (seeds[0] | s).sum()) for s in seeds[1:]]}
        out[name] = {"runs": res, "cnn_seed_spread": spread, "voxel_volume_m3": float(v_hex), "elements": int(len(rho))}
        print(f"{name}: CNN seed spread {spread}", flush=True)
    out["gui"]["tet_reanalysis"] = tet_reanalysis()
    with open(os.path.join(OUT, "comparison.json"), "w") as f:
        json.dump(out, f, indent=1)
    write_md(out)


def tet_reanalysis():
    """Corrected tet/hex compliance of the GUI part for CNN seed 0 and Gaussian sigma 0.5, as Table 11's GUI row."""
    import pyto.examples_benchmarks.body_fitted_study as bf
    from pyto.autodiff.material_model import MaterialModel
    case = "ThreeHoleBracketGUI"
    fe_hex, _ = bf._hex(case)
    rho = np.load(os.path.join(GUI, "density.npy"))
    with torch.no_grad():
        u = fe_hex.solve(torch.tensor((rho >= 0.5).astype(float)), MaterialModel.SIMP).numpy()
    c_hex = float(np.asarray(fe_hex.bc.force) @ u)
    with open(bf._path(case, "analysis.json")) as f:
        calib = json.load(f)["comparison"]["compliance_tet_over_hex_full_domain"]
    n = bf.CASES[case].get("tet_surface_triangles")
    out = {}
    for key in TET_KEYS:
        if not os.path.exists(os.path.join(OUT, f"gui_{key}.stl")):
            out[key] = {"error": "no recovered part (Boolean step failed)"}
            continue
        s = pv.read(os.path.join(OUT, f"gui_{key}.stl")).triangulate()
        if n and s.n_cells > n:
            s = s.decimate_pro(1 - n / s.n_cells, preserve_topology=True).clean()
        surf = os.path.join(OUT, f"gui_{key}_tet_surface.stl")
        s.save(surf)
        try:
            _tm, _q, _fe, tet = bf._tet_solve(case, surf, fe_hex, bf.TET_REQUEST)
        except RuntimeError as e:
            out[key] = {"error": str(e).splitlines()[0]}
            print(f"gui {key}: tetrahedral analysis failed ({out[key]['error']})", flush=True)
            continue
        out[key] = {"tet_dofs": tet["dofs"], "compliance_tet": tet["compliance"], "compliance_hex": c_hex,
                    "total_ratio": tet["compliance"] / c_hex, "corrected_ratio": tet["compliance"] / c_hex / calib}
        print(f"gui {key}: tet/hex {out[key]['total_ratio']:.3f}, corrected {out[key]['corrected_ratio']:.3f} "
              f"({tet['dofs']} dofs)", flush=True)
    return out


def write_md(out):
    lines = ["# STL recovery: CNN against Gaussian smoothing", "",
             "Made by `docs/paper/analysis/compare_smoothing.py`. V/Vvox: part volume over the voxel design's (density >= "
             "0.5). IoU: overlap with the voxel design at element centres. Roughness: area-weighted mean angle between "
             "adjacent triangle normals (lower = smoother). Time: smoothing + iso-surface.", ""]
    for name, c in out.items():
        lines += [f"## {name} ({c['elements']} elements)", "",
                  "| Method | Time (s) | V/Vvox | IoU | Roughness (deg) | One closed body |", "|---|---|---|---|---|---|"]
        for k, r in c["runs"].items():
            if "error" in r:
                lines.append(f"| {k.replace('_', ' ')} | {r['time_s']:.1f} | {r['error']} | | | |")
                continue
            lines.append(f"| {k.replace('_', ' ')} | {r['time_s']:.1f} | {r['volume_over_voxel']:.3f} | "
                         f"{r['iou_voxel']:.3f} | {r['roughness_deg']:.2f} | {r['one_closed_body']} |")
        s = c["cnn_seed_spread"]
        lines += ["", f"CNN between seeds: volume range {s['volume_range']:.4f} (fraction of the voxel volume); "
                  f"IoU of seed 0 with seeds 1, 2: {', '.join(f'{v:.3f}' for v in s['iou_seed0_vs_others'])}. "
                  "Gaussian: deterministic.", ""]
        if "tet_reanalysis" in c:
            lines += ["Tetrahedral re-analysis (Table 11 GUI case), corrected tet/hex compliance: " + ", ".join(
                f"{k.replace('_', ' ')} " + (f"{v['corrected_ratio']:.3f}" if "corrected_ratio" in v else v["error"])
                for k, v in c["tet_reanalysis"].items()), ""]
    with open(os.path.join(OUT, "comparison.md"), "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
