# Benchmark problems

This page lists every ready-made problem in PyTO, with its physics, formulation and sweep group. Use it to pick a starting point, or to see what the benchmark sweep covers. How to build and run them: [Benchmarks from code](../code/benchmarks-from-code.md).

**Columns:**
- **Objective** and **Constraints** use the built-in types (`VOLUME_FRACTION ≤ 0.5` means at most 50 % material).
- **Options:** *extrude z* = constant through the thickness (2.5D problems); *symmetry* = one or more symmetry planes; *body force* = self-weight or centrifugal load; *Heaviside* = projection on.
- **Group:** the `benchmarks_*` list in `topopt_run_benchmarks.py` that includes it in the sweep.

## Structural (`StructuralTOExamples`)
| Problem | Objective | Constraints | Options | Group |
|---|---|---|---|---|
| `Mitchell` | COMPLIANCE | VOLUME_FRACTION ≤ 0.5 | extrude z | structural_2_5D_problems_1 |
| `Mitchell_2` | COMPLIANCE | VOLUME_FRACTION ≤ 0.5 | extrude z | not in sweep |
| `Mitchell_3` | COMPLIANCE | VOLUME_FRACTION ≤ 0.5 | extrude z | not in sweep |
| `ShortCantileverTipLoad` | COMPLIANCE | VOLUME_FRACTION ≤ 0.5 | extrude z | not in sweep (used by the tests) |
| `ShortCantileverMidLoad` | COMPLIANCE | VOLUME_FRACTION ≤ 0.5 | extrude z, symmetry | not in sweep |
| `CantileverTipLoad` | COMPLIANCE | VOLUME_FRACTION ≤ 0.5 | extrude z | structural_2_5D_problems_1 |
| `CantileverMidLoad` | COMPLIANCE | VOLUME_FRACTION ≤ 0.5 | extrude z, symmetry | structural_2_5D_problems_1 |
| `MBBBeam` | COMPLIANCE | VOLUME_FRACTION ≤ 0.5 | extrude z | structural_2_5D_problems_1 |
| `Bridge` | COMPLIANCE | VOLUME_FRACTION ≤ 0.4 | extrude z, symmetry | structural_2_5D_problems_1 |
| `TwoBar` | COMPLIANCE | VOLUME_FRACTION ≤ 0.25 | extrude z, symmetry | structural_2_5D_problems_1 |
| `LBracketTopLoad` | COMPLIANCE | VOLUME_FRACTION ≤ 0.5 | extrude z | structural_2_5D_problems_2 |
| `LBracketMidLoad` | COMPLIANCE | VOLUME_FRACTION ≤ 0.5 | extrude z | structural_2_5D_problems_2 |
| `TorquePlate` | COMPLIANCE | VOLUME_FRACTION ≤ 0.5 | extrude z | structural_2_5D_problems_2 |
| `DistributedLoad` | COMPLIANCE | VOLUME_FRACTION ≤ 0.5 | extrude z, symmetry | structural_2_5D_problems_2 |
| `ThreeHoleBracket` | COMPLIANCE | VOLUME_FRACTION ≤ 0.5 | extrude z | structural_2_5D_problems_2 |
| `TensilePlate` | COMPLIANCE | VOLUME_FRACTION ≤ 0.25 | extrude z, symmetry | not in sweep |
| `EdgeCantilever` | COMPLIANCE | VOLUME_FRACTION ≤ 0.25 | symmetry | structural_3D_problems |
| `EdgeCantileverLargeDOF` | COMPLIANCE | VOLUME_FRACTION ≤ 0.25 | symmetry | not in sweep |
| `EdgeCantileverConstraintMatrix` | COMPLIANCE | VOLUME_FRACTION ≤ 0.5 | symmetry | not in sweep |
| `Multiload` | COMPLIANCE | VOLUME_FRACTION ≤ 0.25 | symmetry | structural_3D_problems |
| `ThreeHoleBracketThick` | COMPLIANCE | VOLUME_FRACTION ≤ 0.25 | symmetry | structural_3D_problems |
| `LBracketThickTopLoad` | COMPLIANCE | VOLUME_FRACTION ≤ 0.25 | symmetry | structural_3D_problems |
| `LBracketThickMidLoad` | COMPLIANCE | VOLUME_FRACTION ≤ 0.25 | symmetry | structural_3D_problems |
| `Table` | COMPLIANCE | VOLUME_FRACTION ≤ 0.15 | symmetry | structural_3D_problems |
| `KnuckleAssembly` | COMPLIANCE | VOLUME_FRACTION ≤ 0.25 | symmetry, multi-body with a kept component | not in sweep |
| `GEGrabCAD` | COMPLIANCE | VOLUME_FRACTION ≤ 0.35 | – | structural_casestudies |
| `CantileverMidLoadVolumeCompliance` | VOLUME_FRACTION | COMPLIANCE ≤ 25 | extrude z, symmetry | structural_noncompliance |
| `LBracketTopLoad_Stress_Vol` | PNORM_STRESS | VOLUME_FRACTION ≤ 0.3 | extrude z | structural_noncompliance |
| `LBracketTopLoad_Vol_Stress` | VOLUME_FRACTION | MAX_VONMISES_STRESS ≤ 4e8 | extrude z | structural_noncompliance |
| `LBracketMidLoad_Vol_Stress` | VOLUME_FRACTION | MAX_VONMISES_STRESS ≤ 2.25e8 | extrude z | structural_noncompliance |
| `LBracketTopLoad_Mass_StressFF` | MASS | STRESS_FAILURE_FACTOR ≤ 0.5 | extrude z | structural_noncompliance |
| `LBracketMidLoad_Vol_StressFailureFactor_Compliance` | VOLUME_FRACTION | STRESS_FAILURE_FACTOR ≤ 0.5, COMPLIANCE ≤ 400 | extrude z | not in sweep |
| `LBracketThickTopLoad_Vol_Stress` | VOLUME_FRACTION | MAX_VONMISES_STRESS ≤ 3e8 | symmetry | not in sweep |
| `LBracketThickTopLoad_Stress_Vol` | PNORM_STRESS | VOLUME_FRACTION ≤ 0.3 | extrude z | not in sweep |
| `Inverter` | GVECTOR (output displacement of a compliant mechanism) | VOLUME_FRACTION ≤ 0.3, COMPLIANCE ≤ 1.5 | extrude z, symmetry | structural_noncompliance |
| `GravityPlate` | COMPLIANCE (self-weight) | VOLUME_FRACTION ≤ 0.1, VOLUME_FRACTION_MIN ≥ 0.095 | body force, symmetry | structural_bodyforce |
| `CentrifugalPlate` | COMPLIANCE (centrifugal load) | VOLUME_FRACTION ≤ 0.5 | body force, extrude z | structural_bodyforce |

## Thermal (`ThermalTOExamples`)
| Problem | Objective | Constraints | Options | Group |
|---|---|---|---|---|
| `HeatPlate` | COMPLIANCE (heat source + heat sink) | VOLUME_FRACTION ≤ 0.5 | extrude z, symmetry | thermal_2_5D_problems |
| `FourCornersThermal` | COMPLIANCE (prescribed temperatures only; maximizes conduction) | VOLUME_FRACTION ≤ 0.75 | extrude z | thermal_2_5D_problems |
| `BridgeThermal` | COMPLIANCE (prescribed temperatures only) | VOLUME_FRACTION ≤ 0.5 | extrude z, symmetry | thermal_2_5D_problems |

## Thermo-structural (`ThermoStructuralTOExamples`)
| Problem | Objective | Constraints | Options | Group |
|---|---|---|---|---|
| `BiClamp` | COMPLIANCE, ΔT = 1 K (Rodrigues & Fernandes 1995, Example A) | VOLUME_FRACTION ≤ 0.4 | extrude z, symmetry | thermostructural_problems |
| `BiClampDT0`, `BiClampDT4`, `BiClampDT10` | the same with ΔT = 0, 4, 10 K | VOLUME_FRACTION ≤ 0.4 | extrude z, symmetry | thermostructural_problems |
| `MBBBeam` | COMPLIANCE, steady heating (Ooms et al. 2023) | VOLUME_FRACTION ≤ 0.4 | extrude z, Heaviside | thermostructural_problems |
| `MBBBeamNoHeat` | the same without heating | VOLUME_FRACTION ≤ 0.4 | extrude z, Heaviside | thermostructural_problems |

## Which methods run which problems in the sweep
| Method | Runs |
|---|---|
| MMA | all |
| OC | compliance problems, except GravityPlate (self-weight) |
| Pareto (only when requested) | compliance problems except thermo-structural |
| LevelSet (only when requested) | compliance problems except body-force and thermo-structural |

---
Next: [Validation](validation.md) · [Back to contents](../README.md)
