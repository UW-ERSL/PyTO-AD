"""One-object thermo-mechanical solver: thermal solve, thermal force, structural solve.

The optimization drivers take a single `fe_solver` and call `fe_solver.solve(x, material_model)`. For a
coupled problem that call has to run the whole chain

    T(x) = thermal solve         K_T(x) T = q
    F_th(x, T)                   thermoelastic nodal force (design dependent)
    u(x) = structural solve      K(x) u = F_mech + F_th

Everything is torch, so autograd differentiates through both adjoints and the drivers need no coupled
sensitivity code. The compliance is the energy form u^T K u (= F_total . u), which is what the shared
qoi code already computes from `sol`.
"""
import torch

from pyto.physics.structural.hex_structural_fea import HexStructuralFEA
from pyto.physics.thermal.hex_thermal_fea import HexThermalFEA


class ThermoStructuralFEA(HexStructuralFEA):
  """Structural FEA whose load includes the thermal expansion force of a solved thermal field.

  Subclasses HexStructuralFEA so every isinstance/attribute the drivers, qoi code and plotters use for a
  structural problem keeps working; `solve` returns the displacement, `self.thermal_fea` holds the thermal
  half and `self.temperature` the last temperature field.
  """

  def __init__(self, mesh, mat_prop, structural_bc, thermal_bc, solver, dsolver=None,
               elem_body_force=None, thermoElasticReferenceTemperature=23.0,
               conductivity_penalty=None, conductivity_void_ratio=None, **kwargs):
    # Thermal first: HexThermalFEA.__init__ points the shared mesh.edofMat at the thermal connectivity, and
    # the structural constructor (which reads mesh.edofMat for its assembly and the qoi code reads
    # afterwards) has to be the last one to set it.
    self.thermal_fea = HexThermalFEA(mesh=mesh, mat_prop=mat_prop, bc=thermal_bc, solver=solver, dsolver=dsolver,
                                     thermoElasticReferenceTemperature=thermoElasticReferenceTemperature,
                                     conductivity_penalty=conductivity_penalty,
                                     conductivity_void_ratio=conductivity_void_ratio, **kwargs)
    super().__init__(mesh=mesh, mat_prop=mat_prop, bc=structural_bc, solver=solver, dsolver=dsolver,
                     elem_body_force=elem_body_force, **kwargs)
    self.temperature = None
    self.thermal_force = None

  def solve(self, x=None, material_model=None, extra_force=None, elem_thermal_strain=None):
    if x is None:  # plain analysis of the full design, as HexStructuralFEA/HexThermalFEA.solve()
      x = torch.ones(self.mesh.num_elems, dtype=torch.float64)
    self.temperature = self.thermal_force = None    # the previous solve's, with its graph (see HexStructuralFEA.solve)
    T = self.thermal_fea.solve(x, material_model)
    self.temperature = T
    f_th = self.thermal_fea.get_thermoelastic_force_torch(T, x, material_model)
    self.thermal_force = f_th          # kept for responses such as MechanicalCompliance (= (F_total - F_th) . u)
    if extra_force is not None:
      f_th = f_th + extra_force
    return super().solve(x, material_model, extra_force=f_th,
                         elem_thermal_strain=self.thermal_fea.get_element_thermal_strain(T))
