from pyto.examples_benchmarks.topopt_thermostructural_benchmarks import ThermoStructuralTOExamples
from pyto.topopt.common import *
from pyto.physics.thermoelastic.thermostructural_fea import ThermoStructuralFEA
from pyto.topopt.drivers.mma import topopt_mma
from pyto.topopt.drivers.oc import topopt_optimality_criteria	
from pyto.topopt.drivers.pareto import topopt_pareto

from pyto.topopt.drivers.levelset import topopt_levelset	
from pyto.examples_benchmarks.topopt_structural_benchmarks import *
from pyto.examples_benchmarks.topopt_thermal_benchmarks import *
from pyto.examples_benchmarks.topopt_thermostructural_benchmarks import *
import time
import glob
import pandas as pd
import multiprocessing

def _record_result(output_dir, optimizationMethod, entry, results_list):
	"""Append `entry` to results_list and merge it into that method's summary CSV.

	Shared by the success path and the per-problem failure handler in
	runTOMethodOnBenchmarks, so a caught exception is recorded in exactly the
	same CSV a successful run would use (same schema, same update-or-append
	behavior) instead of just being printed and lost.
	"""
	results_list.append(entry)
	os.makedirs(output_dir, exist_ok=True)
	result_csv_file = f"{output_dir}/{optimizationMethod.name}_summary.csv"
	if os.path.exists(result_csv_file):
		existing_df = pd.read_csv(result_csv_file)
		problem_index = existing_df[existing_df['name'] == entry['name']].index
		if len(problem_index) > 0:
			existing_df.loc[problem_index[0]] = entry
		else:
			existing_df = pd.concat([existing_df, pd.DataFrame([entry])], ignore_index=True)
		existing_df.to_csv(result_csv_file, index=False)
	else:
		pd.DataFrame([entry]).to_csv(result_csv_file, index=False)


benchmarks_structural_2_5D_problems_1 = [StructuralTOExamples.Mitchell,
						StructuralTOExamples.CantileverTipLoad,
						StructuralTOExamples.CantileverMidLoad,
						StructuralTOExamples.MBBBeam,
						StructuralTOExamples.Bridge,
						StructuralTOExamples.TwoBar,]

benchmarks_structural_2_5D_problems_2 = [StructuralTOExamples.LBracketTopLoad,
						StructuralTOExamples.LBracketMidLoad,
						StructuralTOExamples.TorquePlate,
						StructuralTOExamples.DistributedLoad,
						StructuralTOExamples.ThreeHoleBracket,]

benchmarks_structural_3D_problems = [StructuralTOExamples.EdgeCantilever,
						StructuralTOExamples.ThreeHoleBracketThick,
						StructuralTOExamples.Multiload,
						StructuralTOExamples.LBracketThickTopLoad,
						StructuralTOExamples.LBracketThickMidLoad,
						StructuralTOExamples.Table]

benchmarks_structural_casestudies = [StructuralTOExamples.GEGrabCAD,]

benchmarks_structural_noncompliance = [StructuralTOExamples.CantileverMidLoadVolumeCompliance,
					StructuralTOExamples.LBracketTopLoad_Stress_Vol,
					StructuralTOExamples.LBracketTopLoad_Vol_Stress,
					StructuralTOExamples.LBracketMidLoad_Vol_Stress,
					StructuralTOExamples.LBracketTopLoad_Mass_StressFF,
					StructuralTOExamples.Inverter]

benchmarks_structural_bodyforce = [StructuralTOExamples.GravityPlate,
					StructuralTOExamples.CentrifugalPlate]

benchmarks_thermostructural_problems = [ThermoStructuralTOExamples.BiClamp,
					ThermoStructuralTOExamples.BiClampDT0,
					ThermoStructuralTOExamples.BiClampDT4,
					ThermoStructuralTOExamples.BiClampDT10,
					ThermoStructuralTOExamples.MBBBeam,
					ThermoStructuralTOExamples.MBBBeamNoHeat]

# Defined and building (checked 2026-09-28) but not part of the standard sweep: variants of listed problems,
# large-DOF versions, or problems with no Reference result. Run them with problems_override=... when needed.
benchmarks_not_in_sweep = [StructuralTOExamples.Mitchell_2,
					StructuralTOExamples.Mitchell_3,
					StructuralTOExamples.ShortCantileverTipLoad,
					StructuralTOExamples.ShortCantileverMidLoad,
					StructuralTOExamples.TensilePlate,
					StructuralTOExamples.EdgeCantileverLargeDOF,
					StructuralTOExamples.EdgeCantileverConstraintMatrix,
					StructuralTOExamples.LBracketThickTopLoad_Vol_Stress,
					StructuralTOExamples.LBracketThickTopLoad_Stress_Vol,
					StructuralTOExamples.LBracketMidLoad_Vol_StressFailureFactor_Compliance,
					StructuralTOExamples.KnuckleAssembly]

benchmarks_thermal_2_5D_problems = [ThermalTOExamples.HeatPlate, ThermalTOExamples.FourCornersThermal,
						 ThermalTOExamples.BridgeThermal]

allBenchmarks = benchmarks_structural_2_5D_problems_1 + \
				benchmarks_structural_2_5D_problems_2 + \
				benchmarks_thermal_2_5D_problems + \
				benchmarks_structural_3D_problems  + \
				benchmarks_structural_noncompliance + \
				benchmarks_structural_bodyforce + \
				benchmarks_thermostructural_problems  + \
				benchmarks_structural_casestudies


def build_benchmark(to_problem, solver=None):
	"""Build the FE solver and TO parameters for one benchmark problem, exactly as the benchmark runner does.

	Shared by runTOMethodOnBenchmarks and main.ipynb, so a notebook run is the same computation as a sweep row.
	Returns (fe_solver, to_params, feaMode). Thermo-structural problems return one coupled ThermoStructuralFEA
	(thermal solve -> thermal force -> structural solve). `solver` overrides the automatic choice (PARDISO,
	or DPCG with deflation above DIRECT_SOLVER_DOF_CUTOFF).
	"""
	if to_problem in StructuralTOExamples:
		mesh, mat_prop, bc, elem_body_force, to_params = getStructuralTOProblem(to_problem)
		feaMode = FEA_MODE.STRUCTURAL
	elif to_problem in ThermalTOExamples:
		mesh, mat_prop, bc, elem_body_force, to_params = getThermalTOProblem(to_problem)
		feaMode = FEA_MODE.THERMAL
	elif to_problem in ThermoStructuralTOExamples:
		mesh, mat_prop, structural_bc, thermal_bc, elem_body_force, to_params = getThermoStructuralTOProblem(to_problem)
		feaMode = FEA_MODE.THERMO_STRUCTURAL
	else:
		raise ValueError(f"Unknown benchmark problem: {to_problem}")

	dsolver = deflation.DeflationSolver()
	if solver is None:
		solver = lin_solv.Solvers.PARDISO if to_params.nDOFDesired <= DIRECT_SOLVER_DOF_CUTOFF else lin_solv.Solvers.DPCG
	if solver == lin_solv.Solvers.DPCG:  # DPCG for large DOF problems
		nGroups = min(dsolver.maxGroups, max(dsolver.minGroups, round(3 * mesh.num_nodes / dsolver.dofPerGroup)))
		dsolver.create_deflation_groups(mesh, nGroups)
		dsolver.create_deflation_matrix(mesh)

	common = dict(mesh=mesh, mat_prop=mat_prop, solver=solver, dsolver=dsolver, rtol=1e-8,
				  elem_body_force=elem_body_force)
	if feaMode == FEA_MODE.STRUCTURAL:
		fe_solver = hex_structural_fea.HexStructuralFEA(bc=bc, **common)
	elif feaMode == FEA_MODE.THERMAL:
		fe_solver = hex_thermal_fea.HexThermalFEA(bc=bc, **common)
	else:
		fe_solver = ThermoStructuralFEA(structural_bc=structural_bc, thermal_bc=thermal_bc,
										thermoElasticReferenceTemperature=to_params.ThermalReferenceTemperature,
										conductivity_penalty=to_params.ConductivityPenalty,
										conductivity_void_ratio=to_params.ConductivityVoidRatio, **common)
	return fe_solver, to_params, feaMode


def runTOMethodOnBenchmarks(optimizationMethod, problems_override: list = None, date_str: str = None,
							gradient=None, solver=None, max_iters=None):
	"""Run one method on benchmark problems and record images and summaries under Results/Results_<date_str>/.

	gradient: None keeps each problem's setting (automatic differentiation); GRADIENT_METHOD.AUTODIFF /
	ANALYTICAL (or their names) set to_params.Gradient for MMA/OC. solver: a Solvers member to use instead of
	build_benchmark's choice. max_iters: overrides to_params.MaxIterations (quick trial runs). All three are used
	by gradient_comparison; a normal sweep passes none of them."""
	# Create a list to store results

	# Resolved once per call (not once per output path -- see
	# run_all_benchmarks_isolated for why a single, caller-supplied
	# date_str matters for a sweep spanning many hours/subprocesses).
	date_str = date_str or time.strftime('%Y-%m-%d')

	saveVTU = False  # Set to True if you want to save the VTU files for MMA method
	binarize_topology = True  # Set to True if you want to binarize the topology for MMA/OCM method
	results_list = []
	dsolver = deflation.DeflationSolver()
	feaMode = FEA_MODE.STRUCTURAL

	for to_problem in (problems_override if problems_override is not None else allBenchmarks):
		if to_problem in benchmarks_structural_2_5D_problems_1:
			subFolder = "Structural-Compliance2.5D_1"
		elif to_problem in benchmarks_structural_2_5D_problems_2:
			subFolder = "Structural-Compliance2.5D_2"
		elif to_problem in benchmarks_thermal_2_5D_problems:
			subFolder = "Thermal-Compliance2.5D"
		elif to_problem in benchmarks_structural_3D_problems:
			subFolder = "Structural-Compliance3D"
		elif to_problem in benchmarks_structural_noncompliance:
			subFolder = "Structural-NonCompliance"
		elif to_problem in benchmarks_structural_bodyforce:
			subFolder = "Structural-BodyForce"
		elif to_problem in benchmarks_thermostructural_problems:
			subFolder = "ThermoStructural"
		else:
			subFolder = "Other"

		try:
			print_progress = False
			fe_solver, to_params, feaMode = build_benchmark(to_problem, solver=solver)
			if gradient is not None:
				to_params.Gradient = gradient_method(gradient)
			if max_iters is not None:
				to_params.MaxIterations = int(max_iters)
			solver = fe_solver.solver

			# Ensure the output directory exists
			output_base = f"./Results/Results_{date_str}/{subFolder}/Problems/"
			os.makedirs(output_base, exist_ok=True)

			if to_params.ExtrudeZ:
				fe_solver.plot_mesh(
					title="Structural Load",
					plot_bc=True,
					camera_position='xy',
					save_path=f"{output_base}/{to_problem.name}_problem.png"
				)
			else:
				fe_solver.plot_mesh(
					title="Structural Load",
					plot_bc=True,
					camera_position='iso',
					save_path=f"{output_base}/{to_problem.name}_problem.png"
				)
			startTime = time.time()

			print(f"Running {to_problem.name} problem using {optimizationMethod.name} method and {solver.name} solver")
			print("-" * 50)
			# Create the directory if it does not exist
			output_dir = f"./Results/Results_{date_str}/{subFolder}/{optimizationMethod.name}"
			if not os.path.exists(output_dir):
				os.makedirs(output_dir)

		
			if optimizationMethod == TO_METHODS.DENSITYMMA:
				u, history,success,errorMsg,nFEAs = topopt_mma(fe_solver,to_params = to_params,
										maxMMAIterations = to_params.MaxIterations,
										binarize_topology = binarize_topology and not to_params.HeavisideProjection,print_progress = print_progress)
			elif optimizationMethod == TO_METHODS.DENSITYOCM:
				# GravityPlate (pure self-weight) is skipped: its compliance is not monotone in the density, OC's
				# update assumes negative sensitivities, and it collapses the design (tried SIMP and RAMP,
				# 2026-09-28). CentrifugalPlate runs fine with OC.
				if to_problem in benchmarks_structural_noncompliance or to_problem == StructuralTOExamples.GravityPlate:
					continue
				u, history, success,errorMsg,nFEAs = topopt_optimality_criteria(fe_solver,to_params = to_params,
												maxIterations = to_params.MaxIterations,
												binarize_topology = binarize_topology and not to_params.HeavisideProjection,print_progress = print_progress)
			elif optimizationMethod == TO_METHODS.PARETO:
				if to_problem in benchmarks_structural_noncompliance or \
						to_problem in benchmarks_thermostructural_problems:
					continue
				# Pareto includes the body-force term in its topological sensitivity and removes hanging elements:
				# it gives the self-weight arch on GravityPlate.
				u, history, success,errorMsg,nFEAs = topopt_pareto(feaMode, fe_solver,
														to_params = to_params,print_progress = print_progress)
			elif optimizationMethod == TO_METHODS.LEVELSET:
				# Body-force problems are skipped: the level-set shape derivative has no design-dependent load term,
				# and adding it (checked against autograd) still does not converge on either plate (2026-09-28).
				if to_problem in benchmarks_structural_noncompliance or \
						to_problem in benchmarks_structural_bodyforce or \
						to_problem in benchmarks_thermostructural_problems:
					continue
				u, history, success,errorMsg,nFEAs = topopt_levelset(feaMode, fe_solver,
															to_params = to_params, maxIterations = to_params.MaxIterations)
			timeTaken = time.time() - startTime

			image_path = f"{output_dir}/{to_problem.name}.png"
			title = f"{optimizationMethod.name}: vol: {history['volfrac'][-1]:0.2f}, J: {history['objective'][-1]:.3g}, nFEA: {len(history['objective']):3d}, time: {timeTaken:.0f} s"
			title = None
			# Same camera rule as the problem image above (ExtrudeZ -> 2D view), so each result sits next to a
			# setup picture drawn the same way (the non-compliance/body-force 2.5D problems used to get 'iso').
			if to_params.ExtrudeZ:
				fe_solver.plot_mesh(save_path=image_path, plot_bc = None, title=title, camera_position='xy')
			else:
				fe_solver.plot_mesh(save_path=image_path, plot_bc = None, title=title,camera_position='iso')
	
			_record_result(output_dir, optimizationMethod, {
				'name': to_problem.name,
				'comment': to_params.Comment,
				'ndof': 3*fe_solver.mesh.num_nodes,
				'volfrac': history['volfrac'][-1],
				'objective': history['objective'][-1],
				'#FEAs': nFEAs,
				'time (s)': timeTaken,
				'success': success,
				'errorMsg': errorMsg
			}, results_list)

		
			# Export the mesh with pseudo density to a .vtu file
			if (saveVTU) and (optimizationMethod == TO_METHODS.DENSITYMMA):
				vtu_dir= f"./Results/VTU"
				if not os.path.exists(vtu_dir):
					os.makedirs(vtu_dir)
				vtu_file = f"{vtu_dir}/{to_problem.name}.vtu"
				fe_solver.mesh.export_vtu_mesh(fe_solver.mesh.elemPseudoDensity,
											file_name = vtu_file,)
		except Exception as e:
			print(f"FAILED: {to_problem.name} with {optimizationMethod.name}: {e}")
			output_dir = f"./Results/Results_{date_str}/{subFolder}/{optimizationMethod.name}"
			_record_result(output_dir, optimizationMethod, {
				'name': to_problem.name,
				'comment': '',
				'ndof': None,
				'volfrac': None,
				'objective': None,
				'#FEAs': None,
				'time (s)': None,
				'success': False,
				'errorMsg': str(e)
			}, results_list)
			continue

	# Convert results_list to a DataFrame for better visualization
	# Read the results from the existing CSV file if it exists, otherwise create a new DataFrame
	result_csv_file = f"{output_dir}/{optimizationMethod.name}_summary.csv"
	if os.path.exists(result_csv_file):
		results_df = pd.read_csv(result_csv_file)
	else:
		results_df = pd.DataFrame(results_list)
	if (results_df.empty):
		return
	# Format
	results_df['volfrac'] = results_df['volfrac'].map(lambda x: f"{x:.2g}")
	results_df['objective'] = results_df['objective'].map(lambda x: f"{x:.3g}")
	results_df['time (s)'] = results_df['time (s)'].map(lambda x: f"{x:.3g}")

	# Plot the results as a table
	fig, ax = plt.subplots(figsize=(10, len(results_list) * 0.5))
	ax.axis('tight')
	ax.axis('off')
	table = ax.table(cellText=results_df.values, colLabels=results_df.columns, loc='center')
	table.auto_set_font_size(False)
	table.set_fontsize(10)
	table.auto_set_column_width(col=list(range(len(results_df.columns))))

	# Make the first row and column bold
	for key, cell in table.get_celld().items():
		if key[0] == 0 or key[1] == 0:  # Header row
			cell.set_text_props(weight='bold')
	
	# Save the table as an image
	results_path = f"{output_dir}/{optimizationMethod.name}_summary.png"

	plt.savefig(results_path, bbox_inches='tight')
	plt.close()  

def combine_results(date_str: str = None):
	"""Build comparison charts from a day's Results/Results_<date>/ folder.

	date_str defaults to today (time.strftime('%Y-%m-%d')), matching where
	runTOMethodOnBenchmarks just wrote its output -- but a run that takes
	multiple hours can finish after midnight, and re-running this later
	(e.g. the next day, to retry after a crash mid-aggregation) needs
	yesterday's date explicitly, since "today" no longer points at that
	data. Pass the exact "Results_<date>" suffix to target a specific run.
	"""
	if date_str is None:
		date_str = time.strftime('%Y-%m-%d')
	# Get the latest results directory
	for subFolder in ["Structural-Compliance2.5D_1",
				   "Structural-Compliance2.5D_2",
				   "Structural-Compliance3D",
				   "Structural-NonCompliance",
				   "Structural-BodyForce",
				   "ThermoStructural",
				   "Thermal-Compliance2.5D"]:
		# Get the latest results directory for the given subfolder
		results_dirs = sorted(glob.glob(f"./Results/Results_{date_str}/{subFolder}"))
		if not results_dirs:
			print(f"No results directory found for {subFolder}. Skipping...")
			continue
		results_dir = results_dirs[-1]
		print(f"Combining results from {results_dir}")
		# Read all CSV files
		dataframes = {}
		for method in TO_METHODS:
			csv_path = f"{results_dir}/{method.name}/{method.name}_summary.csv"
			if os.path.exists(csv_path):
				df = pd.read_csv(csv_path)
				# Convert compliance and time strings to float
				df['objective'] = df['objective'].astype(float)
				df['time (s)'] = df['time (s)'].astype(float)
				dataframes[method.name] = df

		if not dataframes:
			print(f"No method CSVs found in {results_dir}. Skipping...")
			continue
		if 'DENSITYMMA' not in dataframes:
			print(f"No DENSITYMMA results in {results_dir} -- comparison charts are "
				  "relative to DENSITYMMA, so there's nothing to normalize against. Skipping...")
			continue

		# Create compliance summary
		#
		# Every method's per-problem values are keyed by problem name (a dict,
		# not a plain list) so pd.DataFrame(...) aligns rows by name and fills
		# NaN for any problem a given method is missing, instead of assuming
		# every method has exactly the same problems as DENSITYMMA in exactly
		# the same order. That assumption broke in practice: a problem can
		# legitimately succeed for some methods and fail for others (e.g. get
		# OOM-killed) since runTOMethodOnBenchmarks records each problem's
		# result independently -- indexing straight into a DENSITYMMA-only
		# reference dict for a name DENSITYMMA doesn't have raised a bare
		# KeyError that killed this function (and every subfolder after it)
		# the first time a real run actually hit that case.
		reference_compliance = dict(zip(
			dataframes['DENSITYMMA']['name'],
			dataframes['DENSITYMMA']['objective']
		))
		reference_time = dict(zip(
			dataframes['DENSITYMMA']['name'],
			dataframes['DENSITYMMA']['time (s)']
		))
		reference_feas = dict(zip(
			dataframes['DENSITYMMA']['name'],
			dataframes['DENSITYMMA']['#FEAs']
		))

		compliance_data = {}
		time_data = {}
		fea_data = {}
		volume_data = {}

		# Calculate normalized compliance, time, #FEAs, and volume fraction
		# for each method -- only for problems that also have a DENSITYMMA
		# reference value (rows without one are silently omitted, not an
		# error: there is no meaningful "relative to DENSITYMMA" value to
		# compute without one).
		for method, df in dataframes.items():
			compliance_row, time_row, fea_row, volume_row = {}, {}, {}, {}
			for _, row in df.iterrows():
				name = row['name']
				volume_row[name] = float(row['volfrac'])
				if name in reference_compliance:
					compliance_row[name] = min(row['objective'] / reference_compliance[name], 5)
				if name in reference_time:
					time_row[name] = row['time (s)'] / reference_time[name]
				if name in reference_feas:
					fea_row[name] = row['#FEAs'] / reference_feas[name]
			compliance_data[method] = compliance_row
			time_data[method] = time_row
			fea_data[method] = fea_row
			volume_data[method] = volume_row

		# Create and save normalized compliance summary
		compliance_df = pd.DataFrame(compliance_data)
		#compliance_df.to_csv(f"{results_dir}/compliance_summary.csv")

		# Create and save normalized time summary
		time_df = pd.DataFrame(time_data)
		#time_df.to_csv(f"{results_dir}/time_summary.csv")

		# Create and save #FEAs summary
		fea_df = pd.DataFrame(fea_data)
		#fea_df.to_csv(f"{results_dir}/fea_summary.csv")
		# Create separate plots for compliance, time and FEAs
		
		# Plot normalized compliance
		plt.figure(figsize=(12, 6))
		compliance_df.plot(kind='bar', width=0.8)
		plt.title('Relative Compliance', fontsize=14, fontweight='bold')
		plt.ylabel('Relative to DENSITYMMA', fontsize=10)
		plt.xticks(rotation=45, fontsize=8, ha='right')
		plt.legend(title='Method', fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')
		plt.grid(True, alpha=0.3)
		plt.subplots_adjust(right=0.85)  # Make room for legend
		plt.tight_layout()
		plt.savefig(f"{results_dir}/{subFolder}_compliance_comparison.png", dpi=300, bbox_inches='tight')
		plt.close()

		# Plot normalized time
		plt.figure(figsize=(10, 6))
		time_df.plot(kind='bar', width=0.8)
		plt.title('Relative Computation Time', fontsize=14, fontweight='bold') 
		plt.ylabel('Relative to DENSITYMMA', fontsize=10)
		plt.xticks(rotation=45, fontsize=8, ha='right')
		plt.legend(title='Method', fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')
		plt.grid(True, alpha=0.3)
		plt.tight_layout()
		plt.savefig(f"{results_dir}/{subFolder}_time_comparison.png", dpi=300, bbox_inches='tight')
		plt.close()

		# Plot number of FEAs
		plt.figure(figsize=(10, 6))
		fea_df.plot(kind='bar', width=0.8)
		plt.title('Relative Num. of FEAs', fontsize=14, fontweight='bold')
		plt.ylabel('Relative to DENSITYMMA', fontsize=10)
		plt.xticks(rotation=45, fontsize=8, ha='right')
		plt.legend(title='Method', fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')
		plt.grid(True, alpha=0.3)
		plt.tight_layout()
		plt.savefig(f"{results_dir}/{subFolder}_fea_comparison.png", dpi=300, bbox_inches='tight')
		plt.close()

		# Volume fraction summary (volume_data was already built name-keyed
		# in the loop above, alongside compliance/time/#FEAs)
		volume_df = pd.DataFrame(volume_data)

		plt.figure(figsize=(10, 6))
		volume_df.plot(kind='bar', width=0.8)
		plt.title('Volume Fraction', fontsize=14, fontweight='bold')
		plt.ylabel('Volume Fraction', fontsize=10)
		plt.xticks(rotation=45, fontsize=8, ha='right')
		plt.legend(title='Method', fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')
		plt.grid(True, alpha=0.3)
		plt.tight_layout()
		plt.savefig(f"{results_dir}/{subFolder}_volume_comparison.png", dpi=300, bbox_inches='tight')
		plt.close()

def create_summary_tables(date_str: str = None):
	"""Build the per-subfolder image-grid summary table. See combine_results()
	for why date_str (default: today) can matter -- same "today's date"
	dependency, same folder layout."""
	if date_str is None:
		date_str = time.strftime('%Y-%m-%d')
	subfolders = [
		"Structural-Compliance2.5D_1",
		"Structural-Compliance2.5D_2",
		"Structural-Compliance3D",
		"Structural-NonCompliance",
		"Structural-BodyForce",
		"ThermoStructural",
		"Thermal-Compliance2.5D",
	]

	for subFolder in subfolders:
		plt.close('all')
		results_dirs = sorted(glob.glob(f"./Results/Results_{date_str}/{subFolder}"))
		if not results_dirs:
			print(f"No results directory found for {subFolder}. Skipping...")
			continue

		results_dir = results_dirs[-1]
		method_dfs = {}

		for method in TO_METHODS:
			csv_path = f"{results_dir}/{method.name}/{method.name}_summary.csv"
			if os.path.exists(csv_path):
				df = pd.read_csv(csv_path)
				df = df[['name', 'volfrac', 'objective', 'time (s)']]
				df.rename(columns={
					'volfrac': f'{method.name}_volfrac',
					'objective': f'{method.name}_objective',
					'time (s)': f'{method.name}_time (s)'
				}, inplace=True)
				method_dfs[method.name] = df

		if not method_dfs:
			print(f"No method CSVs found in {results_dir}. Skipping...")
			continue

		# Start from the DENSITYMMA dataframe to get the problem list
		base_df = method_dfs.get('DENSITYMMA', next(iter(method_dfs.values()))).copy()
		for _, df in method_dfs.items():
			base_df = base_df.merge(df, on='name', how='outer')

		# Build rows and compute image paths per method
		rows = []
		methods = list(TO_METHODS)

		# Use the problem image saved in the Problems folder
		for _, row in base_df.iterrows():
			name = row['name']
			entry = {'Name': name, 'cells': {}}
			# Problem column uses the problem image generated in runTOMethodOnBenchmarks
			problem_img_path = f"{results_dir}/Problems/{name}_problem.png"
			entry['problem_img'] = problem_img_path if os.path.exists(problem_img_path) else None
			for method in methods:
				img_path = f"{results_dir}/{method.name}/{name}.png"
				entry['cells'][method.name] = {
					'img': img_path if os.path.exists(img_path) else None,
					'text': ""
				}
			rows.append(entry)

		# Create a grid figure with images per cell
		n_rows = len(rows)
		# Columns: Name | Problem | per-method images
		n_cols = 2 + len(methods)
		if n_rows == 0:
			print(f"No rows to plot for {results_dir}. Skipping...")
			continue

		cell_h = 1.5  # height per row in inches
		cell_w = 1.8  # width per image cell
		name_col_w = 2.0
		problem_col_w = 2.0
		fig_w = name_col_w + problem_col_w + len(methods) * cell_w
		fig_h = max(3, n_rows * cell_h)
		_, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h))
		# Normalize axes to 2D array
		if n_rows == 1:
			axes = np.array([axes])
		if n_cols == 1:
			axes = axes.reshape(n_rows, 1)

		# Column headers in the first row's axes titles
		axes[0, 0].set_title(" ", fontsize=12, fontweight='bold')
		axes[0, 1].set_title("Problem", fontsize=12, fontweight='bold')
		for ci, method in enumerate(methods, start=2):
			axes[0, ci].set_title(method.name, fontsize=12, fontweight='bold')

		for ri, entry in enumerate(rows):
			# Name cell
			ax0 = axes[ri, 0]
			ax0.axis('off')
			# Long names (e.g. CantileverMidLoadVolumeCompliance) used to overflow this column; tight_layout then
			# squeezed every image axis to make room and the topologies came out tiny. Break the name at '_'
			# and keep all cell texts out of the layout computation.
			name = entry['Name']
			if len(name) > 18:
				if '_' in name:
					name = name.replace('_', '_\n')
				else:  # CamelCase: break at the capital letter nearest the middle
					caps = [i for i, c in enumerate(name) if c.isupper() and i > 0]
					if caps:
						cut = min(caps, key=lambda i: abs(i - len(name) / 2))
						name = name[:cut] + '\n' + name[cut:]
			ax0.text(0.02, 0.5, name, fontsize=12 if len(entry['Name']) <= 18 else 10, va='center', ha='left',
					 transform=ax0.transAxes).set_in_layout(False)

			# Problem image cell
			ax_prob = axes[ri, 1]
			ax_prob.axis('off')
			if entry['problem_img'] is not None:
				try:
					img = plt.imread(entry['problem_img'])
					ax_prob.imshow(img)
				except Exception:
					ax_prob.text(0.02, 0.5, "Problem image load error", fontsize=10, va='center', ha='left',
								 transform=ax_prob.transAxes).set_in_layout(False)
			else:
				ax_prob.text(0.02, 0.5, "No problem image", fontsize=10, va='center', ha='left',
							 transform=ax_prob.transAxes).set_in_layout(False)

			# Method cells: show figure if available, else text
			for ci, method in enumerate(methods, start=2):
				ax = axes[ri, ci]
				ax.axis('off')
				cell = entry['cells'][method.name]
				if cell['img'] is not None:
					try:
						img = plt.imread(cell['img'])
						ax.imshow(img)
						if cell['text']:
							ax.text(0.02, 0.02, cell['text'], fontsize=10, va='bottom', ha='left',
									color='white', bbox=dict(facecolor='black', alpha=0.4, pad=2),
									transform=ax.transAxes)
					except Exception:
						ax.text(0.02, 0.5, cell['text'] or "Image load error", fontsize=10, va='center', ha='left',
								transform=ax.transAxes).set_in_layout(False)
				else:
					ax.text(0.02, 0.5, cell['text'] or "No image", fontsize=10, va='center', ha='left',
							transform=ax.transAxes).set_in_layout(False)

		plt.tight_layout(pad=0.5)
		out_path = f"{results_dir}/{subFolder}_summary_table.png"
		plt.savefig(out_path, dpi=300, bbox_inches='tight')
		plt.close()

def run_all_benchmarks_isolated(optimizationMethods=None, problems=None):
	"""Run every (method, problem) pair in its own OS process.

	pyvista/torch's per-process caching allocators never return memory to
	the OS once allocated: a single benchmark problem's optimization run
	jumps this process's RSS by several GB, and that memory stays reserved
	even after the FE solver/mesh are deleted and gc.collect() runs (
	confirmed directly -- 4 repeats of the *same*-sized problem plateau at
	the post-first-run level, they don't keep growing). Across ~30
	differently-*sized* benchmark problems in one long-lived process,
	each new size forces its own new allocations on top of the ones
	already cached for every previous size, so usage climbs without bound
	-- observed reaching 63GB before the OS OOM-killed it. Because that
	process inherits whatever cgroup launched it (e.g. VSCode's own memory
	cgroup, when started from its integrated terminal), the kill can take
	the whole editor down with it, not just this script.

	Running each (method, problem) pair in a fresh subprocess (spawn, not
	fork, so it's a genuinely new interpreter rather than a copy of this
	already-large one) guarantees the OS fully reclaims that pair's memory
	the instant it exits, regardless of what pyvista/torch cache
	internally -- bounding this driver's own memory to whatever the
	single largest problem needs, not the sum of all of them.

	The whole sweep's output date (the "Results_<date>" folder every
	problem writes into, and the one combine_results()/create_summary_tables()
	read back from at the end) is resolved exactly once, here, up front --
	not separately by each subprocess or by the final aggregation calls.
	A full sweep runs for hours; letting each piece independently ask "what
	day is it right now" means a run that crosses midnight silently splits
	its own output across two date folders, and the final aggregation
	(computed after everything else, i.e. most likely to land on the far
	side of that boundary) ends up looking at the wrong -- often mostly
	empty -- folder. One date_str threaded through every subprocess and
	into the final combine/summarize calls makes this a non-issue by
	construction, not a mismatch to notice and patch around after a run
	already got split across two days.
	"""
	if optimizationMethods is None:
		# MMA and OC by default; Pareto and LevelSet are secondary now (pass them explicitly to include them)
		optimizationMethods = [TO_METHODS.DENSITYMMA, TO_METHODS.DENSITYOCM]
	if problems is None:
		problems = allBenchmarks

	run_date = time.strftime('%Y-%m-%d')
	print(f"Benchmark sweep output folder: Results/Results_{run_date}/ "
		  "(fixed for this whole run, regardless of how long it takes)")

	ctx = multiprocessing.get_context("spawn")
	for optimizationMethod in optimizationMethods:
		for to_problem in problems:
			print("=" * 60)
			print(f"[isolated] {optimizationMethod.name}: {to_problem.name}")
			print("=" * 60)
			p = ctx.Process(
				target=runTOMethodOnBenchmarks,
				args=(optimizationMethod,),
				kwargs={"problems_override": [to_problem], "date_str": run_date},
			)
			p.start()
			p.join()
			if p.exitcode != 0:
				print(f"WARNING: subprocess for {optimizationMethod.name}/{to_problem.name} "
					  f"exited with code {p.exitcode} (likely killed by the OS, e.g. OOM) -- "
					  "not recorded in the CSV; re-run this pair individually if needed.")
		print("-" * 50)
		print(f"Finished {optimizationMethod.name} tests.")
		print("-" * 50)
		print("\n")

	# Combine results from all methods -- same run_date as every subprocess
	# above wrote into, not whatever "today" happens to be by now.
	combine_results(run_date)

	create_summary_tables(run_date)


if __name__ == "__main__":
	run_all_benchmarks_isolated()
