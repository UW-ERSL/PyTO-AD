# %%
import time
import numpy as np # pip install numpy
import matplotlib.pyplot as plt # pip install matplotlib
import pyto.physics.structural.hex_structural_fea as fea
import pyto.solve.deflation as deflation
import pyto.solve.numpy_backend as lin_solv
import os
from pyto.topopt.drivers.mma import topopt_mma
from pyto.examples_benchmarks.hex_structural_examples import *
from pyto.topopt.common import *
from pyto.examples_benchmarks.topopt_structural_benchmarks import *
import itertools

dsolver = deflation.DeflationSolver()

script_dir = os.path.dirname(os.path.abspath(__file__))

to_problem = StructuralTOExamples.EdgeCantilever
linearSolvers = [lin_solv.Solvers.SPSOLVE, lin_solv.Solvers.PYAMG, lin_solv.Solvers.PCG, lin_solv.Solvers.PARDISO, lin_solv.Solvers.DPCG]

optimizationMethod = TO_METHODS.DENSITYMMA
dofs = [5000,10000,25000,50000,75000,100000,150000,250000,500000,1e6,1.5e6,2e6,3e6]
# Set the time limit
timeLimit = 60*5 # seconds
dofActualList = []
solverTime = dict(zip(linearSolvers, [None]*len(linearSolvers)))
for linearSolver in linearSolvers:
	solverTime[linearSolver] = []

continueMeshing = True # set to false to skip to solving the FEA problems
dsolver = deflation.DeflationSolver()
title = f"{to_problem.name} - {optimizationMethod.name}"
print_progress = False
for dofDesired in dofs:
	print("    ")
	if (not continueMeshing):
		break
	print('**************************')
	print("dofDesired: ", dofDesired)
	mesh, mat_prop, bc,elem_body_force, to_params = getStructuralTOProblem(to_problem,nDOFDesired = dofDesired)
	dofActual = 3 * mesh.num_nodes
	print("dofActual: ",dofActual)
	dofActualList.append(dofActual)
	continueMeshing = False
	
	for linearSolver in linearSolvers:
		print('-----------------------')
		print(linearSolver)
		# assuming increasing time with increasing DOF, skip if previous time was too long
		if len(solverTime[linearSolver]) > 0 and solverTime[linearSolver][-1] > timeLimit: 
			print( ' ... skipping')
			continue
		continueMeshing = True
		startTime = time.time()
		if (linearSolver == lin_solv.Solvers.DPCG):
			nGroups =  min(dsolver.maxGroups,max(dsolver.minGroups,round(3*mesh.num_nodes/dsolver.dofPerGroup)))
			dsolver.create_deflation_groups(mesh, nGroups)
			dsolver.create_deflation_matrix(mesh)

		fe_solver = hex_structural_fea.HexStructuralFEA(mesh = mesh,
                    mat_prop = mat_prop,
                    bc = bc,
                    solver = linearSolver,
                    dsolver = dsolver,
                    rtol = 1e-8,
                    elem_body_force = elem_body_force)
		nDOF = 3*fe_solver.mesh.num_nodes
		feaMode = FEA_MODE.STRUCTURAL
		topopt_mma(feaMode,fe_solver,None, to_params = to_params,
                                                    plot_progress= False,
                                                    print_progress= False,
                                                    maxiteration= to_params.MaxIterations,)
	
		totalTime = time.time() - startTime
		
		print('Time: {:.2f}'.format(totalTime))
		solverTime[linearSolver].append(totalTime)
		
marker = itertools.cycle(('dk', '+b','xm', '*g', 'or')) 
colors = itertools.cycle(('k', 'b','m', 'g', 'r')) 
for linearSolver in linearSolvers:
	timing = solverTime[linearSolver]
	plt.loglog(dofActualList[0:len(timing)],timing,next(marker))
plt.legend(linearSolvers,loc = 'upper left')

for linearSolver in linearSolvers:
	timing = solverTime[linearSolver]
	plt.loglog(dofActualList[0:len(timing)],timing,next(colors))

plt.axhline(y=timeLimit, color='black', linestyle=':', label='Time limit')
plt.title(title)
plt.xlabel('DOF')
plt.ylabel('Time (secs)')
plt.grid(True)
plt.show()

