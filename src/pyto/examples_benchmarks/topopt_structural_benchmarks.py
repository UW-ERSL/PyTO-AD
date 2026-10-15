import enum
from pyto.examples_benchmarks.hex_structural_examples import *
from pyto.topopt.common import *

class StructuralTOExamples(enum.Enum):

    # 2.5D Examples
	Mitchell = enum.auto()
	Mitchell_2 = enum.auto()
	Mitchell_3 = enum.auto()       
	ShortCantileverTipLoad = enum.auto()
	ShortCantileverMidLoad = enum.auto()
	CantileverTipLoad = enum.auto()
	CantileverMidLoad = enum.auto()
	MBBBeam = enum.auto()
	LBracketTopLoad = enum.auto()
	LBracketMidLoad = enum.auto()
	TwoBar = enum.auto()
	TorquePlate = enum.auto()
	DistributedLoad = enum.auto()
	TensilePlate = enum.auto()
	ThreeHoleBracket = enum.auto()
	Bridge = enum.auto()

    # 3D Examples
	EdgeCantilever = enum.auto()
	EdgeCantileverLargeDOF = enum.auto()
	EdgeCantileverConstraintMatrix = enum.auto()
	Multiload = enum.auto()
	ThreeHoleBracketThick = enum.auto()
	LBracketThickTopLoad = enum.auto()
	LBracketThickMidLoad = enum.auto()
	Table = enum.auto()
	GEGrabCAD = enum.auto()


    # Non-compliance Examples
	CantileverMidLoadVolumeCompliance = enum.auto()

	LBracketTopLoad_Stress_Vol = enum.auto()
	LBracketTopLoad_Vol_Stress = enum.auto()
	LBracketTopLoad_Mass_StressFF = enum.auto()
	LBracketThickTopLoad_Vol_Stress = enum.auto()
	LBracketThickTopLoad_Stress_Vol = enum.auto()

	LBracketMidLoad_Vol_Stress = enum.auto()
	LBracketMidLoad_Vol_StressFailureFactor_Compliance = enum.auto()
	
	Inverter = enum.auto()

    # Body Force Examples
	CentrifugalPlate = enum.auto()
	GravityPlate = enum.auto()

    # Other Examples
	KnuckleAssembly = enum.auto()

def getSTLPath_TOProblem(to_problem: StructuralTOExamples):
    if to_problem == StructuralTOExamples.Mitchell:
        stl_file = "Models/Mitchell/Mitchell.STL"
    elif to_problem == StructuralTOExamples.Mitchell_2:
        stl_file = "Models/Mitchell/Mitchell.STL"
    elif to_problem == StructuralTOExamples.Mitchell_3:
        stl_file = "Models/Mitchell/Mitchell.STL"
    elif to_problem == StructuralTOExamples.ShortCantileverTipLoad:
        stl_file = "Models/ShortCantilever/ShortCantilever.STL"
    elif to_problem == StructuralTOExamples.ShortCantileverMidLoad:
        stl_file = "Models/ShortCantilever/ShortCantilever.STL"
    elif to_problem == StructuralTOExamples.CantileverTipLoad:
        stl_file = "Models/Cantilever/Cantilever.STL"
    elif to_problem == StructuralTOExamples.CantileverMidLoad:
        stl_file = "Models/Cantilever/Cantilever.STL"
    elif to_problem == StructuralTOExamples.MBBBeam:
        stl_file = "Models/MBBBeam/MBBBeam.STL"
    elif to_problem == StructuralTOExamples.LBracketTopLoad:
        stl_file = "Models/LBracket/LBracket.STL"
    elif to_problem == StructuralTOExamples.LBracketMidLoad:
        stl_file = "Models/LBracket/LBracket.STL"
    elif to_problem == StructuralTOExamples.TwoBar:
        stl_file = "Models/TwoBar/TwoBar.STL"
    elif to_problem == StructuralTOExamples.TorquePlate:
        stl_file = "Models/CircularPlateHole/CircularPlateHole.STL"
    elif to_problem == StructuralTOExamples.DistributedLoad:
        stl_file = "Models/DistributedLoad/DistributedLoad.STL"
    elif to_problem == StructuralTOExamples.TensilePlate:
        stl_file = "Models/TensilePlate/TensilePlate.STL"
    elif to_problem == StructuralTOExamples.ThreeHoleBracket:
        stl_file = "Models/ThreeHoleBracket/ThreeHoleBracket.STL"
    elif to_problem == StructuralTOExamples.EdgeCantilever:
        stl_file = "Models/EdgeCantilever/EdgeCantilever.STL"
    elif to_problem == StructuralTOExamples.EdgeCantileverLargeDOF:
        stl_file = "Models/EdgeCantilever/EdgeCantilever.STL"
    elif to_problem == StructuralTOExamples.EdgeCantileverConstraintMatrix:
        stl_file = "Models/EdgeCantilever/EdgeCantilever.STL"
    elif to_problem == StructuralTOExamples.Multiload:
        stl_file = "Models/Multiload/Multiload.STL"
    elif to_problem == StructuralTOExamples.ThreeHoleBracketThick:
        stl_file = "Models/ThreeHoleBracket/ThreeHoleBracketThick.STL"
    elif to_problem == StructuralTOExamples.LBracketThickTopLoad:
        stl_file = "Models/LBracketThick/LBracketThick.STL"
    elif to_problem == StructuralTOExamples.LBracketThickMidLoad:
        stl_file = "Models/LBracketThick/LBracketThick.STL"
    elif to_problem == StructuralTOExamples.Table:
        stl_file = "Models/Table/Table.STL"
    elif to_problem == StructuralTOExamples.GEGrabCAD:
        stl_file = "Models/GEGrabCad/GEGrabCAD.STL"

    elif to_problem == StructuralTOExamples.CantileverMidLoadVolumeCompliance:
        stl_file = "Models/Cantilever/CantileverMidLoad.STL"
    elif to_problem == StructuralTOExamples.LBracketTopLoad_Stress_Vol:
        stl_file = "Models/LBracket/LBracket.STL"
    elif to_problem == StructuralTOExamples.LBracketTopLoad_Vol_Stress:
        stl_file = "Models/LBracket/LBracket.STL"
    elif to_problem == StructuralTOExamples.LBracketTopLoad_Mass_StressFF:
        stl_file =  "Models/LBracket/LBracket.STL"
    elif to_problem == StructuralTOExamples.LBracketThickTopLoad_Stress_Vol:
        stl_file = "Models/LBracketThick/LBracketThick.STL"
    elif to_problem == StructuralTOExamples.LBracketMidLoad_Vol_Stress:
        stl_file = "Models/LBracket/LBracket.STL"
    elif to_problem == StructuralTOExamples.LBracketMidLoad_Vol_StressFailureFactor_Compliance:
        stl_file = "Models/LBracket/LBracket.STL"
    elif to_problem == StructuralTOExamples.LBracketThickTopLoad_Vol_Stress:
        stl_file = "Models/LBracketThick/LBracketThick.STL"
    elif to_problem == StructuralTOExamples.Inverter:
        stl_file = "Models/Inverter/Inverter.STL"
    elif to_problem == StructuralTOExamples.CentrifugalPlate:
        stl_file = "Models/CircularPlateHole/CircularPlateHole.STL"
    elif to_problem == StructuralTOExamples.GravityPlate:
        stl_file = "Models/GravityPlate/GravityPlate.STL"
    elif to_problem == StructuralTOExamples.KnuckleAssembly:
        stl_file = "Models/KnuckleAssembly/KnuckleAssembly.STL"
    else:
        raise ValueError(f"Unknown problem: {to_problem}")

    return stl_file

def getStructuralTOProblem(to_problem: StructuralTOExamples,nDOFDesired = None, **kwargs):
    """Get the structural topology optimization problem based on the specified example.

    Args:
        problem: The example problem to solve.
        **kwargs: Additional arguments to pass to the problem.

    Returns:
        StructuralTOProblem: The structural topology optimization problem.
    """
    
    to_params = TOParams()
    if to_problem == StructuralTOExamples.Mitchell:
        structural_problem = StructuralExamples.Mitchell
        kwargs['load1'] = 5.6e4
        kwargs['load2'] = 0
        to_params.Comment = "Benchmark 2.5D"
        to_params.ExtrudeZ = True
        to_params.nDOFDesired = 25000 if nDOFDesired is None else nDOFDesired
        to_params.Objective = (TO_QOI.COMPLIANCE, None)
        to_params.Constraints = [(TO_QOI.VOLUME_FRACTION, None, 0.5)]
    elif to_problem == StructuralTOExamples.Mitchell_2:
        structural_problem = StructuralExamples.Mitchell
        kwargs['load1'] = 2.8e4
        kwargs['load2'] = 2.8e4
        to_params.Comment = "Benchmark 2.5D"
        to_params.ExtrudeZ = True
        to_params.nDOFDesired = 25000 if nDOFDesired is None else nDOFDesired
        to_params.Objective = (TO_QOI.COMPLIANCE, None)
        to_params.Constraints = [(TO_QOI.VOLUME_FRACTION, None, 0.5)]
    elif to_problem == StructuralTOExamples.Mitchell_3:
        structural_problem = StructuralExamples.Mitchell
        kwargs['load1'] = 3.72e4
        kwargs['load2'] = 1.86e4
        to_params.Comment = "Benchmark 2.5D"
        to_params.ExtrudeZ = True
        to_params.nDOFDesired = 25000 if nDOFDesired is None else nDOFDesired
        to_params.Objective = (TO_QOI.COMPLIANCE, None)
        to_params.Constraints = [(TO_QOI.VOLUME_FRACTION, None, 0.5)]
    elif to_problem == StructuralTOExamples.ShortCantileverTipLoad:
        structural_problem = StructuralExamples.ShortCantileverTipLoad
        to_params.Comment  = "Benchmark 2.5D"
        to_params.ExtrudeZ = True
        to_params.nDOFDesired = 25000 if nDOFDesired is None else nDOFDesired
        to_params.Objective = (TO_QOI.COMPLIANCE, None)
        to_params.Constraints = [(TO_QOI.VOLUME_FRACTION, None, 0.5)]
    elif to_problem == StructuralTOExamples.ShortCantileverMidLoad:
        structural_problem = StructuralExamples.ShortCantileverMidLoad
        to_params.Comment  = "Benchmark 2.5D"
        to_params.YSymmetry = True  # Symmetry about the Y-axis
        to_params.ExtrudeZ = True
        to_params.nDOFDesired = 25000 if nDOFDesired is None else nDOFDesired
        to_params.Objective = (TO_QOI.COMPLIANCE, None)
        to_params.Constraints = [(TO_QOI.VOLUME_FRACTION, None, 0.5)]
    elif to_problem == StructuralTOExamples.CantileverTipLoad:
        structural_problem = StructuralExamples.CantileverTipLoad
        to_params.Comment  = "Benchmark 2.5D"
        to_params.ExtrudeZ = True
        to_params.nDOFDesired = 25000 if nDOFDesired is None else nDOFDesired
        to_params.Objective = (TO_QOI.COMPLIANCE, None)
        to_params.Constraints = [(TO_QOI.VOLUME_FRACTION, None, 0.5)]
    elif to_problem == StructuralTOExamples.CantileverMidLoad:
        structural_problem = StructuralExamples.CantileverMidLoad
        to_params.Comment  = "Benchmark 2.5D"
        to_params.YSymmetry = True  # Symmetry about the Y-axis
        to_params.ExtrudeZ = True
        to_params.nDOFDesired = 25000 if nDOFDesired is None else nDOFDesired
        to_params.Objective = (TO_QOI.COMPLIANCE, None)
        to_params.Constraints = [(TO_QOI.VOLUME_FRACTION, None, 0.5)]
    elif to_problem == StructuralTOExamples.MBBBeam:
        structural_problem = StructuralExamples.MBBBeam
        to_params.Comment  = "Benchmark 2.5D"
        to_params.nDOFDesired = 50000 if nDOFDesired is None else nDOFDesired
        to_params.ExtrudeZ = True
        to_params.Objective = (TO_QOI.COMPLIANCE, None)
        to_params.Constraints = [(TO_QOI.VOLUME_FRACTION, None, 0.5)]
        to_params.RelativeFilterRadius = 3
    elif to_problem == StructuralTOExamples.Bridge:
        structural_problem = StructuralExamples.Bridge
        to_params.Comment  = "Benchmark 2.5D"
        to_params.XSymmetry = True 
        to_params.ExtrudeZ = True
        to_params.nDOFDesired = 50000 if nDOFDesired is None else nDOFDesired
        to_params.Objective = (TO_QOI.COMPLIANCE, None)
        to_params.Constraints = [(TO_QOI.VOLUME_FRACTION, None, 0.4)]
    elif to_problem == StructuralTOExamples.DistributedLoad:
        structural_problem = StructuralExamples.DistributedLoad
        to_params.Comment  = "Benchmark 2.5D"
        to_params.XSymmetry = True 
        to_params.ExtrudeZ = True
        to_params.nDOFDesired = 50000 if nDOFDesired is None else nDOFDesired
        to_params.Objective = (TO_QOI.COMPLIANCE, None)
        to_params.Constraints = [(TO_QOI.VOLUME_FRACTION, None, 0.5)]
    elif to_problem == StructuralTOExamples.Inverter:
        structural_problem = StructuralExamples.Inverter
        to_params.Comment  = "Compliant Mechanism"
        to_params.Objective = (TO_QOI.GVECTOR, None) # see below for setting the GVECTOR after mesh is created
        to_params.YSymmetry = True
        to_params.ExtrudeZ = True
        to_params.nDOFDesired = 40000 if nDOFDesired is None else nDOFDesired
        to_params.Constraints = [(TO_QOI.VOLUME_FRACTION, None, 0.3), (TO_QOI.COMPLIANCE, None, 1.5)]
    elif to_problem == StructuralTOExamples.LBracketTopLoad:
        structural_problem = StructuralExamples.LBracket
        kwargs['topload'] = 1.5e4
        kwargs['midload'] = 0
        to_params.Comment  = "Benchmark 2.5D"
        to_params.ExtrudeZ = True
        to_params.nDOFDesired = 25000 if nDOFDesired is None else nDOFDesired
        to_params.Objective = (TO_QOI.COMPLIANCE, None)
        to_params.Constraints = [(TO_QOI.VOLUME_FRACTION, None, 0.5)]
    elif to_problem == StructuralTOExamples.LBracketMidLoad:
        structural_problem = StructuralExamples.LBracket
        kwargs['topload'] = 0
        kwargs['midload'] = 1.5e4
        to_params.Comment  = "Benchmark 2.5D"
        to_params.ExtrudeZ = True
        to_params.nDOFDesired = 25000 if nDOFDesired is None else nDOFDesired
        to_params.Objective = (TO_QOI.COMPLIANCE, None)
        to_params.Constraints = [(TO_QOI.VOLUME_FRACTION, None, 0.5)]
    elif to_problem == StructuralTOExamples.TwoBar:
        structural_problem = StructuralExamples.TwoBar
        to_params.Comment  = "Benchmark 2.5D"
        to_params.ExtrudeZ = True
        to_params.YSymmetry = True  # Symmetry about the Y-axis
        to_params.nDOFDesired = 25000 if nDOFDesired is None else nDOFDesired
        to_params.Objective = (TO_QOI.COMPLIANCE, None)
        to_params.Constraints = [(TO_QOI.VOLUME_FRACTION, None, 0.25)]
    elif to_problem == StructuralTOExamples.TorquePlate:
        structural_problem = StructuralExamples.TorquePlate
        to_params.Comment  = "Benchmark 2.5D"
        to_params.ZAxisAngularSymmetry = 4
        to_params.ExtrudeZ = True
        to_params.nDOFDesired = 50000 if nDOFDesired is None else nDOFDesired
        to_params.Objective = (TO_QOI.COMPLIANCE, None)
        to_params.Constraints = [(TO_QOI.VOLUME_FRACTION, None, 0.5)]
    elif to_problem == StructuralTOExamples.TensilePlate:
        structural_problem = StructuralExamples.TensilePlate
        to_params.Comment  = "Benchmark 2.5D"
        to_params.XSymmetry = True
        to_params.ExtrudeZ = True
        to_params.Objective = (TO_QOI.COMPLIANCE, None)
        to_params.Constraints = [(TO_QOI.VOLUME_FRACTION, None, 0.25)]
        to_params.nDOFDesired = 50000 if nDOFDesired is None else nDOFDesired
    elif to_problem == StructuralTOExamples.ThreeHoleBracket:
        structural_problem = StructuralExamples.ThreeHoleBracket
        to_params.Comment  = "Retaining Material"
        to_params.ExtrudeZ = True
        to_params.KeepFixedElems = True
        to_params.nDOFDesired = 75000 if nDOFDesired is None else nDOFDesired
        to_params.Objective = (TO_QOI.COMPLIANCE, None)
        to_params.Constraints = [(TO_QOI.VOLUME_FRACTION, None, 0.5)]
    # 3D Examples
    elif to_problem == StructuralTOExamples.EdgeCantilever:
        structural_problem = StructuralExamples.EdgeCantilever
        to_params.Comment = "Benchmark 3D"
        to_params.ZSymmetry = True
        to_params.nDOFDesired = 75000 if nDOFDesired is None else nDOFDesired
        to_params.Objective = (TO_QOI.COMPLIANCE, None)
        to_params.Constraints = [(TO_QOI.VOLUME_FRACTION, None, 0.25)] 
    elif to_problem == StructuralTOExamples.EdgeCantileverLargeDOF:
        structural_problem = StructuralExamples.EdgeCantilever
        to_params.Comment = "Benchmark 3D"
        to_params.ZSymmetry = True
        to_params.nDOFDesired = 50000 if nDOFDesired is None else nDOFDesired
        to_params.Objective = (TO_QOI.COMPLIANCE, None)
        to_params.Constraints = [(TO_QOI.VOLUME_FRACTION, None, 0.25)] 
    elif to_problem == StructuralTOExamples.EdgeCantileverConstraintMatrix:
        structural_problem = StructuralExamples.EdgeCantileverConstraintMatrix
        to_params.Comment = "Benchmark 3D"
        to_params.YSymmetry = True
        to_params.nDOFDesired = 50000 if nDOFDesired is None else nDOFDesired
        to_params.Objective = (TO_QOI.COMPLIANCE, None)
        to_params.Constraints = [(TO_QOI.VOLUME_FRACTION, None, 0.5)] 
    elif to_problem == StructuralTOExamples.ThreeHoleBracketThick:
        structural_problem = StructuralExamples.ThreeHoleBracketThick
        to_params.Comment  = "Retaining Material"
        to_params.ZSymmetry = True
        to_params.KeepFixedElems = True
        to_params.nDOFDesired = 75000 if nDOFDesired is None else nDOFDesired
        to_params.Objective = (TO_QOI.COMPLIANCE, None)
        to_params.Constraints = [(TO_QOI.VOLUME_FRACTION, None, 0.25)] 
    elif to_problem == StructuralTOExamples.Multiload:
        structural_problem = StructuralExamples.Multiload
        to_params.Comment  = "Multiple Loading"
        to_params.ZSymmetry = True
        to_params.nDOFDesired = 75000
        to_params.Objective = (TO_QOI.COMPLIANCE, None)
        to_params.Constraints = [(TO_QOI.VOLUME_FRACTION, None, 0.25)] 
    elif to_problem == StructuralTOExamples.LBracketThickTopLoad:
        structural_problem = StructuralExamples.LBracketThick
        to_params.Comment  = "3D"
        to_params.ZSymmetry = True
        kwargs['topload'] = 1.5e4
        kwargs['midload'] = 0
        to_params.nDOFDesired = 75000 if nDOFDesired is None else nDOFDesired
        to_params.Objective = (TO_QOI.COMPLIANCE, None)
        to_params.Constraints = [(TO_QOI.VOLUME_FRACTION, None, 0.25)] 
    elif to_problem == StructuralTOExamples.LBracketThickMidLoad:
        structural_problem = StructuralExamples.LBracketThick
        to_params.Comment  = "3D"
        to_params.ZSymmetry = True
        kwargs['topload'] = 0
        kwargs['midload'] = 1.5e4
        to_params.nDOFDesired = 75000 if nDOFDesired is None else nDOFDesired
        to_params.Objective = (TO_QOI.COMPLIANCE, None)
        to_params.Constraints = [(TO_QOI.VOLUME_FRACTION, None, 0.25)] 
    elif to_problem == StructuralTOExamples.Table:
        structural_problem = StructuralExamples.Table
        to_params.Comment = "3D"
        to_params.XSymmetry = True
        to_params.ZSymmetry = True
        to_params.nDOFDesired = 75000 if nDOFDesired is None else nDOFDesired
        to_params.Objective = (TO_QOI.COMPLIANCE, None)
        to_params.MaxIterations = 300  # removes ~85% of the material
        to_params.Constraints = [(TO_QOI.VOLUME_FRACTION, None, 0.15)] 

    elif to_problem == StructuralTOExamples.GEGrabCAD:
        structural_problem = StructuralExamples.GEGrabCAD
        to_params.Comment = "3D"
        to_params.nDOFDesired = 100000 if nDOFDesired is None else nDOFDesired
        to_params.KeepFixedElems = True 
        to_params.Objective = (TO_QOI.COMPLIANCE, None)
        to_params.Constraints = [(TO_QOI.VOLUME_FRACTION, None, 0.35)] 
    # Body Force Examples
    elif to_problem == StructuralTOExamples.GravityPlate:
        structural_problem = StructuralExamples.GravityPlate
        to_params.Comment  = "Body Force"
        to_params.XSymmetry = True
        to_params.Objective = (TO_QOI.COMPLIANCE, None)
        to_params.nDOFDesired = 20000 if nDOFDesired is None else nDOFDesired
        # Self-weight compliance has a trivial near-empty optimum (less material = less load): keep a small band
        to_params.Constraints = [(TO_QOI.VOLUME_FRACTION, None, 0.1), (TO_QOI.VOLUME_FRACTION_MIN, None, 0.095)] 
    elif to_problem == StructuralTOExamples.CentrifugalPlate:
        structural_problem = StructuralExamples.CentrifugalPlate
        to_params.Comment  = "Body Force"
        to_params.ExtrudeZ = True
        to_params.ZAxisAngularSymmetry = 4
        to_params.nDOFDesired = 25000 if nDOFDesired is None else nDOFDesired
        to_params.KeepFixedElems = True  # Keep elements that are fixed in the centrifugal plate example
        to_params.Objective = (TO_QOI.COMPLIANCE, None)
        to_params.Constraints = [(TO_QOI.VOLUME_FRACTION, None, 0.5)]
    # Non-compliance problems
    elif to_problem == StructuralTOExamples.LBracketTopLoad_Stress_Vol:
        structural_problem = StructuralExamples.LBracket
        kwargs['topload'] = 1.5e5
        kwargs['midload'] = 0
        to_params.Comment  = "Stress Minimization"
        to_params.Objective = (TO_QOI.PNORM_STRESS, None) 
        to_params.ExtrudeZ = True
        to_params.RelativeFilterRadius = 2.5
        to_params.nDOFDesired = 25000 if nDOFDesired is None else nDOFDesired
        to_params.Constraints = [(TO_QOI.VOLUME_FRACTION, None, 0.3)]
        to_params.Eliminate_Hanging_Elements = True
    elif to_problem == StructuralTOExamples.LBracketTopLoad_Mass_StressFF:
        structural_problem = StructuralExamples.LBracket
        kwargs['topload'] = 5e4
        kwargs['midload'] = 0
        to_params.Comment  = "Stress Safety Factor"
        to_params.ExtrudeZ = True
        to_params.nDOFDesired = 50000 if nDOFDesired is None else nDOFDesired
        to_params.Eliminate_Hanging_Elements = True
        to_params.Objective = (TO_QOI.MASS, None) 
        to_params.Constraints = [ (TO_QOI.STRESS_FAILURE_FACTOR, None, 0.5)]  # failure limit is the opposite of safety factor
    elif to_problem == StructuralTOExamples.LBracketTopLoad_Vol_Stress:
        structural_problem = StructuralExamples.LBracket
        kwargs['topload'] = 1e5
        kwargs['midload'] = 0
        to_params.Comment  = "Stress Constraint"
        to_params.ExtrudeZ = True
        to_params.Eliminate_Hanging_Elements = True
        to_params.nDOFDesired = 50000 if nDOFDesired is None else nDOFDesired
        to_params.Objective = (TO_QOI.VOLUME_FRACTION, None) 
        to_params.Constraints = [ (TO_QOI.MAX_VONMISES_STRESS, None, 400e6)] 
    elif to_problem == StructuralTOExamples.LBracketThickTopLoad_Stress_Vol:
        structural_problem = StructuralExamples.LBracketThick
        kwargs['topload'] = 1.5e5
        kwargs['midload'] = 0
        to_params.Comment  = "Stress Minimization"
        to_params.ExtrudeZ = True
        to_params.nDOFDesired = 50000 if nDOFDesired is None else nDOFDesired
        to_params.Eliminate_Hanging_Elements = True
        to_params.Objective = (TO_QOI.PNORM_STRESS, None) 
        to_params.Constraints = [(TO_QOI.VOLUME_FRACTION, None, 0.3)]
    elif to_problem == StructuralTOExamples.LBracketThickTopLoad_Vol_Stress:
        structural_problem = StructuralExamples.LBracketThick
        to_params.Comment  =  "Stress Minimization"
        to_params.ZSymmetry = True
        kwargs['topload'] = 1.5e6
        kwargs['midload'] = 0
        to_params.nDOFDesired = 50000 if nDOFDesired is None else nDOFDesired
        to_params.Objective = (TO_QOI.VOLUME_FRACTION, None) # pnorm value
        # user-set limit: 300 MPa (a solid part peaks at ~206 MPa, so the original 200 MPa was unreachable; note the material's yield is 250 MPa)
        to_params.Constraints = [(TO_QOI.MAX_VONMISES_STRESS, None, 300e6)] 
        to_params.Eliminate_Hanging_Elements = True
    elif to_problem == StructuralTOExamples.LBracketThickTopLoad_Stress_Vol:
        structural_problem = StructuralExamples.LBracket
        kwargs['topload'] = 0
        kwargs['midload'] = 1.5e5
        to_params.Comment  = "Stress Minimization"
        to_params.ExtrudeZ = True
        to_params.nDOFDesired = 50000 if nDOFDesired is None else nDOFDesired
        to_params.Objective = (TO_QOI.PNORM_STRESS, None) 
        to_params.Constraints = [(TO_QOI.VOLUME_FRACTION, None, 0.4)] 
    elif to_problem == StructuralTOExamples.LBracketMidLoad_Vol_Stress:
        structural_problem = StructuralExamples.LBracket
        kwargs['topload'] = 0
        kwargs['midload'] = 1.5e5
        to_params.Comment  = "Stress Constraint"
        to_params.ExtrudeZ = True
        to_params.nDOFDesired = 30000 if nDOFDesired is None else nDOFDesired
        to_params.Objective = (TO_QOI.VOLUME_FRACTION, None) 
        to_params.Constraints = [ (TO_QOI.MAX_VONMISES_STRESS, None, 225e6)] 
    elif to_problem == StructuralTOExamples.LBracketMidLoad_Vol_StressFailureFactor_Compliance:
        structural_problem = StructuralExamples.LBracket
        kwargs['topload'] = 0
        kwargs['midload'] = 1.5e5
        to_params.Comment  = "Stress Safety Factor"
        to_params.ExtrudeZ = True
        to_params.nDOFDesired = 30000 if nDOFDesired is None else nDOFDesired
        to_params.Objective = (TO_QOI.VOLUME_FRACTION, None) 
        to_params.Constraints = [ (TO_QOI.STRESS_FAILURE_FACTOR, None, 0.5), (TO_QOI.COMPLIANCE, None, 400)] 
    elif to_problem == StructuralTOExamples.CantileverMidLoadVolumeCompliance:
        structural_problem = StructuralExamples.CantileverMidLoad
        to_params.Comment = "Compliance Constraint"
        to_params.YSymmetry = True  # Symmetry about the Y-axis
        to_params.ExtrudeZ = True
        to_params.Enforce_Constraints_MMA = True
        to_params.nDOFDesired = 25000 if nDOFDesired is None else nDOFDesired
        to_params.Objective = (TO_QOI.VOLUME_FRACTION, None) 
        to_params.Constraints = [(TO_QOI.COMPLIANCE, None, 25)]
    elif to_problem == StructuralTOExamples.KnuckleAssembly:
        structural_problem = StructuralExamples.KnuckleAssembly
        to_params.Comment = "Retaining Components"
        to_params.XSymmetry = True
        to_params.ZSymmetry = True
        to_params.nDOFDesired = 75000 if nDOFDesired is None else nDOFDesired
        to_params.Objective = (TO_QOI.COMPLIANCE, None)
        to_params.Constraints = [(TO_QOI.VOLUME_FRACTION, None, 0.25)] 
    else:
        raise ValueError(f"Unknown problem: {to_problem}")
    
    mesh, mat_prop, bc, elem_body_force = getStructuralProblem(structural_problem, nDOFDesired = to_params.nDOFDesired, **kwargs)

    # Add  elements to keep
    to_params.ElemsToKeep  = None # default value

    # Here we add additional parameters specific to the optimization problem
    if (to_params.KeepFixedElems):
        to_params.ElemsToKeep = find_elements_with_fixedDOF(mesh, bc,nDOFPerNode=3)
        #print("Number of fixed elements", to_params.ElemsToKeep.shape)

    if to_problem == StructuralTOExamples.KnuckleAssembly:
         to_params.ElemsToKeep = np.where(mesh.elemComponentId == 2)[0]
         #print("Elems to keep", to_params.ElemsToKeep.shape)
    if to_problem == StructuralTOExamples.Inverter:
        node_pts = mesh.node_xyz
        xMin = np.min(node_pts[:,0]) 
        xMax = np.max(node_pts[:,0]) 
        yMid = (np.max(node_pts[:,1]) + np.min(node_pts[:,1]))/2
   
        h = mesh.elem_size[0]/2
        outputNodes = np.where((abs(node_pts[:, 0] - xMax) < h) & (abs(node_pts[:, 1] - yMid) < h))[0]

        load_nodes = np.where((abs(node_pts[:, 0] - np.min(node_pts[:, 0])) < h) & (abs(node_pts[:, 1] - np.mean(node_pts[:,1])) < h))[0]
        dof_input = 3*load_nodes # x dof
        dof_output= 3*outputNodes # x dof

        g = np.zeros(3*mesh.num_nodes)

        g[dof_output] = 1

        #to_params.ElemsToKeep, _ = mesh.get_element_containing_point(pt2)
        to_params.Objective = (TO_QOI.GVECTOR, g)
    return mesh, mat_prop, bc, elem_body_force, to_params
