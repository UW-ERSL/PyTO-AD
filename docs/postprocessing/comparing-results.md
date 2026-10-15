# Comparing results

This page explains the output of a benchmark sweep and how to compare two sweeps. Typical comparisons are before and after a code change, or one method against another. It is for developers checking that a change didn't break results, and for anyone comparing methods.

How to run a sweep: [Benchmarks from code](../code/benchmarks-from-code.md#run-a-sweep).

## Output layout
```
Results/Results_<date>/
  <category>/                        e.g. Structural-Compliance2.5D_1, Thermal-Compliance2.5D, ThermoStructural
    <METHOD>/                        DENSITYMMA, DENSITYOCM (PARETO, LEVELSET when requested)
      <problem>.png                  the final topology
      <METHOD>_summary.csv           one row per problem
      <METHOD>_summary.png           the same as a table image
    Problems/                        pictures of each problem's setup
    <category>_summary_table.png     all methods side by side, with topology thumbnails
    <category>_compliance_comparison.png, _volume_, _time_, _fea_comparison.png   bar charts per method
```
The summary CSV columns: `name`, `comment`, `ndof`, `volfrac`, `objective`, `#FEAs`, `time (s)`, `success`, `errorMsg`.

`Results/` is not tracked by git. Keep a reference sweep (e.g. `Results/Results_Reference`) to compare new sweeps against.

## Compare two sweeps
```python
import glob
import os
import pandas as pd

def load(folder):
    rows = []
    for f in glob.glob(os.path.join(folder, "*", "*", "*_summary.csv")):
        category, method = f.split(os.sep)[-3:-1]
        rows.append(pd.read_csv(f).assign(category=category, method=method))
    return pd.concat(rows).set_index(["category", "method", "name"])

new, ref = load("Results/Results_2026-09-29"), load("Results/Results_Reference")
both = new[["objective", "volfrac"]].join(ref[["objective", "volfrac"]], rsuffix="_ref", how="inner")
both["change_%"] = 100 * (both["objective"] / both["objective_ref"] - 1)
print(both.sort_values("change_%").to_string())
```

## Reading the comparison
- **Same volume, objective within ±1 %:** a match. Small differences come from the optimizer's path: solver round-off, thread count.
- **Lower objective at the same or lower volume:** a better design, e.g. after a bug fix or a better method.
- **Very large changes** (−100 %, or orders of magnitude): usually the **reference** or the new run evaluated something different, not a real improvement. For example, a reference made before a bug fix, or a disconnected design after thresholding. Look at both topology images before drawing conclusions.
- **Different volume:** the objectives can't be compared directly. Check why the volume changed; OC and Pareto may stop at a slightly different volume.
- **`success` False or an `errorMsg`:** read the message; see [Troubleshooting](../reference/troubleshooting.md).

## Comparing with published results
For the thermo-structural problems reproduced from papers, `make_publication_comparison` places PyTO's topologies next to the paper figures:
```bash
python -m pyto.examples_benchmarks.make_publication_comparison Results/Results_<date>/ThermoStructural/DENSITYMMA
```
It needs the papers' PDFs in `Results/Publications` and `pdftoppm` (poppler-utils). The comparison images for this repository are shown in [Validation](../reference/validation.md).

---
Next: [Part 4: Extending PyTO](../extending/architecture.md) · [Back to contents](../README.md)
