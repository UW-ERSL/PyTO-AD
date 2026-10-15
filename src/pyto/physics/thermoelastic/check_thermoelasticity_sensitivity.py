"""
Complete Thermo-Elastic Sensitivity Example

The ThermoElasticSensitivity class this script demonstrates/verifies used
to be defined here as a second, independent, byte-for-byte-identical copy
of pyto.physics.thermoelastic.topopt_thermostructural_sensitivity's class
(confirmed via diff -- flagged as a known duplication through Phase 5/7).
Consolidated by importing it from there instead, since the two class
bodies had already drifted apart from a single common origin rather than
being deliberately different implementations (unlike
pyto.autodiff.reference_adjoint, which is deliberately independent of
pyto.autodiff.qoi so a shared bug can't hide in an FD self-check).
"""
try:
    import numpy as np
    import os
    import sys
    import scipy.sparse as sp
    import pyto.solve.numpy_backend as linear_solvers
    from pyto.autodiff.material_model import MaterialModel
    import pyto.core.bc as bound_cond
    from pyto.physics.thermal.hex_thermal_fea import HexThermalFEA
    from pyto.physics.structural.hex_structural_fea import HexStructuralFEA
    import pyto.core.hex_mesher as hex_mesher
    import pyto.core.mat_lib as mat_lib
    from pyto.physics.thermoelastic.topopt_thermostructural_sensitivity import ThermoElasticSensitivity
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)


# Import required modules


script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def createThermoElasticBarProblem(nDOFDesired: int = 10000,
                                   hot_temperature: float = 100.0,
                                   cold_temperature: float = 20.0,
                                   tensile_force: float = 1000.0):
    """
    Creates a coupled thermo-elastic problem on a bar.
    
    Problem description:
    - Bar fixed at left end (x=0)
    - Hot temperature applied at left end
    - Cold temperature applied at right end
    - Optional tensile force at right end
    - Temperature gradient causes thermal expansion
    - Combined thermal + mechanical loading
    
    Parameters:
    -----------
    nDOFDesired : int
        Desired number of degrees of freedom
    hot_temperature : float
        Temperature at left end (°C)
    cold_temperature : float
        Temperature at right end (°C)
    tensile_force : float
        Applied tensile force at right end (N)
        
    Returns:
    --------
    tuple : (mesh, mat_prop, bc_thermal, bc_structural, reference_temp)
    """
    # Load STL geometry
    stl_file = os.path.join(script_dir, '../Models/Beam/beam.STL')
    
    # Create mesh
    nElemsDesired = nDOFDesired / 3  # Estimate for structural DOFs
    mesh = hex_mesher.HexMesher()
    mesh.createMeshFromSTLFile(stl_file, nElemsDesired=nElemsDesired)

    # =========================================================================
    # THERMAL BOUNDARY CONDITIONS
    # =========================================================================
    
    # Create thermal DOF mapping
    mesh.createEdofMatThermal()
   
    # Fixed temperature at left end (hot)
    hot_nodes = mesh.getNodesOnBoundingBoxPlane(0, True)  # x = 0 plane
    hot_dofs = hot_nodes.astype(int)
    hot_values = hot_temperature * np.ones_like(hot_dofs, dtype=float)
    
    # Fixed temperature at right end (cold)
    cold_nodes = mesh.getNodesOnBoundingBoxPlane(0, False)  # x = xMax plane
    cold_dofs = cold_nodes.astype(int)
    cold_values = cold_temperature * np.ones_like(cold_dofs, dtype=float)
    
    # Combine thermal BCs
    thermal_fixed_dofs = np.concatenate([hot_dofs, cold_dofs])
    thermal_dirichlet_values = np.concatenate([hot_values, cold_values])
    
    # No thermal body forces (heat generation)
    thermal_force = np.zeros(mesh.num_nodes)
    
    bc_thermal = bound_cond.BC(
        force=thermal_force,
        fixed_dofs=thermal_fixed_dofs,
        dirichlet_values=thermal_dirichlet_values
    )

    # =========================================================================
    # STRUCTURAL BOUNDARY CONDITIONS
    # =========================================================================
    
    # Create structural DOF mapping
    mesh.createEdofMatStructural()
    # Fixed displacement at left end (all directions)
    fixed_nodes = mesh.getNodesOnBoundingBoxPlane(0, True)  # x = 0 plane
    fixed_dofs = np.array([
        3 * fixed_nodes,
        3 * fixed_nodes + 1,
        3 * fixed_nodes + 2
    ]).flatten().astype(int)
    dirichlet_values = 0 * np.ones_like(fixed_dofs, dtype=float)

    mesh.node_indices[fixed_nodes, 3] = 1# for plotting

    force = np.zeros(3 * mesh.num_nodes)
    load_nodes = mesh.getNodesOnBoundingBoxPlane(0, False)  # x = xMax plane
    
    force[3 * load_nodes] = tensile_force/len(load_nodes)  # Distribute load evenly
    mesh.node_indices[load_nodes, 3] = 2 # for plotting

    bc_structural = bound_cond.BC(
        force=force,
        fixed_dofs=fixed_dofs,
        dirichlet_values=dirichlet_values
    )
    mat_prop = mat_lib.get_material("Steel")
    # Reference temperature for thermal expansion
    reference_temp = cold_temperature  # Use cold temperature as reference
   
    return mesh, mat_prop, bc_thermal, bc_structural, reference_temp


def run_coupled_analysis(nDOFDesired: int = 10000,
                         hot_temp: float = 100.0,
                         cold_temp: float = 20.0,
                         tensile_force: float = 1000.0,
                         p_simp: float = 3.0,
                         q_simp: float = 1.0,
                         perturbation=1e-4,
                         verify_fd: bool = True):
    """
    Run complete coupled thermo-elastic analysis with sensitivity computation.
    
    Parameters:
    -----------
    nDOFDesired : int
        Target number of DOFs
    hot_temp : float
        Hot temperature (°C)
    cold_temp : float
        Cold temperature (°C)
    tensile_force : float
        Applied tensile force (N)
    p_simp : float
        Structural SIMP penalty
    q_simp : float
        Thermal SIMP penalty
    verify_fd : bool
        Whether to verify with finite differences
    """
    
    # =========================================================================
    # PROBLEM SETUP
    # =========================================================================
    
    mesh, mat_prop, bc_thermal, bc_structural, reference_temp = \
        createThermoElasticBarProblem(
            nDOFDesired=nDOFDesired,
            hot_temperature=hot_temp,
            cold_temperature=cold_temp,
            tensile_force=tensile_force
        )
    
    # Solver
    solver = linear_solvers.Solvers.PARDISO
    
    # =========================================================================
    # THERMAL FEA
    # =========================================================================

    thermal_fea = HexThermalFEA(
        mesh=mesh,
        mat_prop=mat_prop,
        bc=bc_thermal,
        solver=solver,
        thermoElasticReferenceTemperature=reference_temp
    )
    
    # Initialize design variables (uniform density)
    x = 0.1*np.ones(mesh.num_elems) 

    # Solve thermal problem
    T = thermal_fea.solve(x, material_model=MaterialModel.SIMP)
 
    # =========================================================================
    # COMPUTE THERMAL FORCES
    # =========================================================================
    f_thermal = thermal_fea.get_thermoelastic_force(x, material_model=MaterialModel.SIMP)

    # =========================================================================
    # STRUCTURAL FEA
    # =========================================================================
    
    # Add thermal forces to structural FEA
    structural_fea = HexStructuralFEA(
        mesh=mesh,
        mat_prop=mat_prop,
        bc=bc_structural,
        solver=solver,
        thermo_elastic_force=f_thermal
    )
    # Solve structural problem
    d = structural_fea.solve(x, material_model=MaterialModel.SIMP)

    # =========================================================================
    # SENSITIVITY ANALYSIS
    # =========================================================================
    
    # Initialize sensitivity analyzer
    sensitivity = ThermoElasticSensitivity(thermal_fea, structural_fea)
    
    # Compute analytical sensitivities
    print("\nComputing analytical sensitivities using adjoint method...")


    dJdx = sensitivity.compute_compliance_sensitivity(
        x=x,
        T=T,
        d=d,
        p=p_simp,
        q=q_simp,
        material_model=MaterialModel.SIMP,
        solver=solver,
        verbose=True
    )

    # =========================================================================
    # FINITE DIFFERENCE VERIFICATION
    # =========================================================================
    
    if verify_fd:
        print("\nComputing finite difference sensitivities and comparing with analytical results...")
        # Select elements to verify
        num_verify = min(10, mesh.num_elems)
        element_indices = np.linspace(0, mesh.num_elems - 1, num_verify, dtype=int)
            
        sensitivity.verify_sensitivity_fd(
            x=x,
            T=T,
            d=d,
            p=p_simp,
            q=q_simp,
            perturbation=perturbation,
            element_indices=element_indices,
            material_model=MaterialModel.SIMP,
            verbose=True
        )
       

if __name__ == "__main__":

    # Problem parameters
    nDOFDesired = 10000
    hot_temp = 30.0  # °C
    cold_temp = 23.0  # °C
    tensile_force = 0.0  # N
    
    perturbation = 1e-5  # Finite difference perturbation size

    run_coupled_analysis(
            nDOFDesired=nDOFDesired,
            hot_temp=hot_temp,
            cold_temp=cold_temp,
            tensile_force=tensile_force,
            perturbation=perturbation,
            verify_fd=True
        )
