

import numpy as np
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree
import pyto.core.hex_mesher as hex_mesher


def createSmoothingFilter(mesh: hex_mesher.HexMesher, rel_filter_radius: float = 1.5):
	"""Create a smoothing filter using the provided formula.

	Args:
		mesh: The mesh object.
		rel_filter_radius: The relative minimum radius for the filter; the radius is scaled by elem size.

	Returns:
		tuple containing:
			H: Sparse matrix representing the smoothing filter.
			Hs: Array of row sums of H matrix.
	"""
	num_elems = mesh.num_elems
	r_min = rel_filter_radius * mesh.elem_size[0]
	# all element pairs with center distance <= r_min (each element with itself included), in one KD-tree query
	tree = cKDTree(mesh.elem_centers)
	dist = tree.sparse_distance_matrix(tree, r_min, output_type="coo_matrix")
	weight = (r_min - dist.data)/r_min

	H = coo_matrix((weight, (dist.row, dist.col)), shape=(num_elems, num_elems)).tocsc()
	Hs = np.array(H.sum(1)).squeeze()
	return H, Hs

def createXDerivativeFilter(mesh: hex_mesher.HexMesher) -> tuple[coo_matrix, np.ndarray]:
	"""Create a finite-difference filter matrix approximating the spatial x derivative.
	
	Uses central differences when possible and falls back on one-sided differences at boundaries.
	
	Args:
		mesh: The mesh object.
	
	Returns:
		tuple containing:
			HXD: Sparse matrix operator approximating the x derivative.
			HXD_s: Array of row sums of HXD matrix.
	"""
	num_elems = mesh.num_elems
	dx = mesh.elem_size[0]
	rows = []
	cols = []
	data = []
	
	for i in range(num_elems):
		elemCenter = mesh.elem_centers[i, :]
		neighborElems = mesh.elemNeighborsArray[i]  # 27 elements
		# Find left and right neighbors from the 27-element neighborhood
		# Assuming neighborElems is ordered such that we can identify x-direction neighbors
		left_idx = -1
		right_idx = -1
		
		# Vectorize neighbor search
		valid_neighbors = neighborElems[neighborElems != -1]
		valid_neighbors = valid_neighbors[valid_neighbors != i]
		
		if len(valid_neighbors) > 0:
			neighbor_centers = mesh.elem_centers[valid_neighbors, :]
			deltas = neighbor_centers - elemCenter
			
			# Check for left neighbor (negative x, same y and z)
			left_mask = (np.abs(deltas[:, 0] + dx) < dx * 0.1) & \
						(np.abs(deltas[:, 1]) < dx * 0.1) & \
						(np.abs(deltas[:, 2]) < dx * 0.1)
			if np.any(left_mask):
				left_idx = valid_neighbors[left_mask][0]
			
			# Check for right neighbor (positive x, same y and z)
			right_mask = (np.abs(deltas[:, 0] - dx) < dx * 0.1) & \
						 (np.abs(deltas[:, 1]) < dx * 0.1) & \
						 (np.abs(deltas[:, 2]) < dx * 0.1)
			if np.any(right_mask):
				right_idx = valid_neighbors[right_mask][0]
		
		if left_idx != -1 and right_idx != -1:
			# Central difference: df/dx ≈ (f[i+1] - f[i-1]) / (2*dx)
			rows.extend([i, i])
			cols.extend([left_idx, right_idx])
			data.extend([-1.0 / (2.0 * dx), 1.0 / (2.0 * dx)])
			
		elif right_idx != -1:
			# Forward difference: df/dx ≈ (f[i+1] - f[i]) / dx
			rows.extend([i, i])
			cols.extend([i, right_idx])
			data.extend([-1.0 / dx, 1.0 / dx])
			
		elif left_idx != -1:
			# Backward difference: df/dx ≈ (f[i] - f[i-1]) / dx
			rows.extend([i, i])
			cols.extend([left_idx, i])
			data.extend([-1.0 / dx, 1.0 / dx])
			
		else:
			# No x-direction neighbors found; derivative is zero
			# This can occur at isolated elements or for certain boundary conditions
			rows.append(i)
			cols.append(i)
			data.append(0.0)
	
	HXD = coo_matrix((data, (rows, cols)), shape=(num_elems, num_elems)).tocsc()

	return HXD

def createYDerivativeFilter(mesh: hex_mesher.HexMesher) -> tuple[coo_matrix, np.ndarray]:
	"""Create a finite-difference filter matrix approximating the spatial y derivative.
	
	Uses central differences when possible and falls back on one-sided differences at boundaries.
	
	Args:
		mesh: The mesh object.
	
	Returns:
		tuple containing:
			HYD: Sparse matrix operator approximating the y derivative.
			HYD_s: Array of row sums of HYD matrix.
	"""
	num_elems = mesh.num_elems
	dy = mesh.elem_size[1]
	rows = []
	cols = []
	data = []
	
	for i in range(num_elems):
		elemCenter = mesh.elem_centers[i, :]
		neighborElems = mesh.elemNeighborsArray[i]  # 27 elements
		# Find lower and upper neighbors from the 27-element neighborhood
		# Assuming neighborElems is ordered such that we can identify y-direction neighbors
		lower_idx = -1
		upper_idx = -1
		
		# Vectorize neighbor search
		valid_neighbors = neighborElems[neighborElems != -1]
		valid_neighbors = valid_neighbors[valid_neighbors != i]
		
		if len(valid_neighbors) > 0:
			neighbor_centers = mesh.elem_centers[valid_neighbors, :]
			deltas = neighbor_centers - elemCenter
			
			# Check for lower neighbor (negative y, same x and z)
			lower_mask = (np.abs(deltas[:, 0]) < dy * 0.1) & \
						(np.abs(deltas[:, 1] + dy) < dy * 0.1) & \
						(np.abs(deltas[:, 2]) < dy * 0.1)
			if np.any(lower_mask):
				lower_idx = valid_neighbors[lower_mask][0]
			
			# Check for upper neighbor (positive y, same x and z)
			upper_mask = (np.abs(deltas[:, 0]) < dy * 0.1) & \
							(np.abs(deltas[:, 1] - dy) < dy * 0.1) & \
							(np.abs(deltas[:, 2]) < dy * 0.1)
			if np.any(upper_mask):
				upper_idx = valid_neighbors[upper_mask][0]
		
		if lower_idx != -1 and upper_idx != -1:
			# Use central difference: (f[upper] - f[lower]) / (2dy)
			rows.append(i)
			cols.append(lower_idx)
			data.append(-1.0 / (2 * dy))
			
			rows.append(i)
			cols.append(upper_idx)
			data.append(1.0 / (2 * dy))
		elif upper_idx != -1:
			# Forward difference: (f[upper] - f[i]) / dy
			rows.append(i)
			cols.append(i)
			data.append(-1.0 / dy)
			
			rows.append(i)
			cols.append(upper_idx)
			data.append(1.0 / dy)
		elif lower_idx != -1:
			# Backward difference: (f[i] - f[lower]) / dy
			rows.append(i)
			cols.append(lower_idx)
			data.append(-1.0 / dy)
			
			rows.append(i)
			cols.append(i)
			data.append(1.0 / dy)
		else:
			# No neighbor found in the y direction; set derivative to zero.
			rows.append(i)
			cols.append(i)
			data.append(0.0)
	
	HYD = coo_matrix((data, (rows, cols)), shape=(num_elems, num_elems)).tocsc()

	return HYD

def createZDerivativeFilter(mesh: hex_mesher.HexMesher) -> tuple[coo_matrix, np.ndarray]:
	"""Create a finite-difference filter matrix approximating the spatial z derivative.
	
	Uses central differences when possible and falls back on one-sided differences at boundaries.
	
	Args:
		mesh: The mesh object.
	
	Returns:
		tuple containing:
			HZD: Sparse matrix operator approximating the z derivative.
			HZD_s: Array of row sums of HZD matrix.
	"""
	num_elems = mesh.num_elems
	dz = mesh.elem_size[2]
	rows = []
	cols = []
	data = []
	
	for i in range(num_elems):
		elemCenter = mesh.elem_centers[i, :]
		neighborElems = mesh.elemNeighborsArray[i]  # 27 elements
		# Find lower and upper neighbors from the 27-element neighborhood
		# Assuming neighborElems is ordered such that we can identify z-direction neighbors
		lower_idx = -1
		upper_idx = -1
		
		# Vectorize neighbor search
		valid_neighbors = neighborElems[neighborElems != -1]
		valid_neighbors = valid_neighbors[valid_neighbors != i]
		
		if len(valid_neighbors) > 0:
			neighbor_centers = mesh.elem_centers[valid_neighbors, :]
			deltas = neighbor_centers - elemCenter
			
			# Check for lower neighbor (negative z, same x and y)
			lower_mask = (np.abs(deltas[:, 0]) < dz * 0.1) & \
						(np.abs(deltas[:, 1]) < dz * 0.1) & \
						(np.abs(deltas[:, 2] + dz) < dz * 0.1)
			if np.any(lower_mask):
				lower_idx = valid_neighbors[lower_mask][0]
			
			# Check for upper neighbor (positive z, same x and y)
			upper_mask = (np.abs(deltas[:, 0]) < dz * 0.1) & \
							(np.abs(deltas[:, 1]) < dz * 0.1) & \
							(np.abs(deltas[:, 2] - dz) < dz * 0.1)
			if np.any(upper_mask):
				upper_idx = valid_neighbors[upper_mask][0]
		
		if lower_idx != -1 and upper_idx != -1:
			# Use central difference: (f[upper] - f[lower]) / (2dz)
			rows.append(i)
			cols.append(lower_idx)
			data.append(-1.0 / (2 * dz))
			
			rows.append(i)
			cols.append(upper_idx)
			data.append(1.0 / (2 * dz))
		elif upper_idx != -1:
			# Forward difference: (f[upper] - f[i]) / dz
			rows.append(i)
			cols.append(i)
			data.append(-1.0 / dz)
			
			rows.append(i)
			cols.append(upper_idx)
			data.append(1.0 / dz)
		elif lower_idx != -1:
			# Backward difference: (f[i] - f[lower]) / dz
			rows.append(i)
			cols.append(lower_idx)
			data.append(-1.0 / dz)
			
			rows.append(i)
			cols.append(i)
			data.append(1.0 / dz)
		else:
			# No neighbor found in the z direction; set derivative to zero.
			rows.append(i)
			cols.append(i)
			data.append(0.0)
	
	HZD = coo_matrix((data, (rows, cols)), shape=(num_elems, num_elems)).tocsc()

	return HZD


def createXSymmetryFilter(mesh: hex_mesher.HexMesher) -> tuple[coo_matrix, np.ndarray]:
	"""Create a symmetry filter matrix about X mid-plane.
	
	Args:
		mesh: The mesh object.
	
	Returns:
		tuple containing:
			HX: Sparse matrix that when multiplied with density vector enforces X mid-plane symmetry
			HXs: Array of row sums of HX matrix
	"""
	num_elems = mesh.num_elems
	x_mid = (mesh.elem_centers[:, 0].max() + mesh.elem_centers[:, 0].min()) / 2
	
	# Initialize COO matrix arrays
	rows = []
	cols = []
	data = []
	
	for i in range(num_elems):
		elemCenter = mesh.elem_centers[i, :]
		mirror_x = 2 * x_mid - elemCenter[0]
		otherElemCenter = np.array([mirror_x, elemCenter[1], elemCenter[2]])
		mirror_idx = mesh.get_element_near_point(otherElemCenter)
		if (mirror_idx == -1):
			continue
		# Check if the closest element is the same as the current element
		if (mirror_idx == i):
			rows.append(i)
			cols.append(i)
			data.append(1.0)
		else:
			rows.append(i)
			cols.append(i)
			data.append(0.5)
			rows.append(i)
			cols.append(mirror_idx)
			data.append(0.5)

	HX = coo_matrix((data, (rows, cols)), shape=(num_elems, num_elems)).tocsc()
	return HX

def createYSymmetryFilter(mesh: hex_mesher.HexMesher) -> tuple[coo_matrix, np.ndarray]:
	"""Create a symmetry filter matrix about Y mid-plane.
	
	Args:
		mesh: The mesh object.
	
	Returns:
		tuple containing:
			HY: Sparse matrix that when multiplied with density vector enforces Y mid-plane symmetry
			HYs: Array of row sums of HY matrix
	"""
	num_elems = mesh.num_elems
	y_mid = (mesh.elem_centers[:, 1].max() + mesh.elem_centers[:, 1].min()) / 2
	# Initialize COO matrix arrays
	rows = []
	cols = []
	data = []
	for i in range(num_elems):
		elemCenter = mesh.elem_centers[i, :]
		mirror_y = 2 * y_mid - elemCenter[1]
		otherElemCenter = np.array([elemCenter[0], mirror_y, elemCenter[2]])
		mirror_idy = mesh.get_element_near_point(otherElemCenter)
		if (mirror_idy == -1):
			continue
		if (mirror_idy == i):
			rows.append(i)
			cols.append(i)
			data.append(1.0)
		else:
			rows.append(i)
			cols.append(i)
			data.append(0.5)
			rows.append(i)
			cols.append(mirror_idy)
			data.append(0.5)
	
	HY = coo_matrix((data, (rows, cols)), shape=(num_elems, num_elems)).tocsc()
	return HY
	
def createZSymmetryFilter(mesh: hex_mesher.HexMesher) -> tuple[coo_matrix, np.ndarray]:
	"""Create a symmetry filter matrix about Z mid-plane.
	
	Args:
		mesh: The mesh object.
	
	Returns:
		tuple containing:
			HZ: Sparse matrix that when multiplied with density vector enforces Z mid-plane symmetry
			HZs: Array of row sums of HZ matrix
	"""
	num_elems = mesh.num_elems
	z_mid = (mesh.elem_centers[:, 2].max() + mesh.elem_centers[:, 2].min()) / 2
	
	# Initialize COO matrix arrays
	rows = []
	cols = []
	data = []
	
	for i in range(num_elems):
		elemCenter = mesh.elem_centers[i, :]
		mirror_z = 2 * z_mid - elemCenter[2]
		otherElemCenter = np.array([elemCenter[0], elemCenter[1], mirror_z])
		mirror_idz = mesh.get_element_near_point(otherElemCenter)
		if (mirror_idz == -1):
			continue
		if (mirror_idz == i):
			rows.append(i)
			cols.append(i)
			data.append(1.0)
		else:
			rows.append(i)
			cols.append(i)
			data.append(0.5)
			rows.append(i)
			cols.append(mirror_idz)
			data.append(0.5)

	HZ = coo_matrix((data, (rows, cols)), shape=(num_elems, num_elems)).tocsc()
	return HZ
def createXAngularSymmetryFilter(mesh: hex_mesher.HexMesher, n_fold: int) -> tuple[coo_matrix, np.ndarray]:
	"""Create a filter matrix for n-fold angular symmetry about X axis.
	
	Args:
		mesh: The mesh object.
		n_fold: Number of symmetric segments 
	
	Returns:
		tuple containing:
			HAX: Sparse matrix that enforces angular symmetry
			HAXs: Array of row sums of HAX matrix
	"""

	num_elems = mesh.num_elems
	angle_step = 2 * np.pi / n_fold
	
	rows = []
	cols = []
	data = []
	
	for i in range(num_elems):
		elemCenter = mesh.elem_centers[i, :]
		r = np.sqrt(elemCenter[1]**2 + elemCenter[2]**2)
		theta = np.arctan2(elemCenter[2], elemCenter[1])
		
		weight = 1.0 / n_fold
		rows.append(i)
		cols.append(i)
		data.append(weight)
		
		# Vectorized computation for all k at once
		k_values = np.arange(1, n_fold)
		new_thetas = theta + k_values[:, None] * angle_step
		new_y = r * np.cos(new_thetas)
		new_z = r * np.sin(new_thetas)
		
		# Create array of other element centers for all k
		otherElemCenters = np.column_stack((np.full_like(new_y.flatten(), elemCenter[0]),
											new_y.flatten(),
											new_z.flatten()))
		# Find nearest elements for each symmetric position
		sym_indices = [mesh.get_element_near_point(center) for center in otherElemCenters]
		
		# Append to COO matrix components
		rows.extend([i] * (n_fold - 1))
		cols.extend(sym_indices)
		data.extend([weight] * (n_fold - 1))
	
	HAX = coo_matrix((data, (rows, cols)), shape=(num_elems, num_elems)).tocsc()
	return HAX


def createYAngularSymmetryFilter(mesh: hex_mesher.HexMesher, n_fold: int) -> tuple[coo_matrix, np.ndarray]:
	"""Create a filter matrix for n-fold angular symmetry about Y axis.
	
	Args:
		mesh: The mesh object.
		n_fold: Number of symmetric segments 
	
	Returns:
		tuple containing:
			HA: Sparse matrix that enforces angular symmetry
			HAs: Array of row sums of HA matrix
	"""

	num_elems = mesh.num_elems
	angle_step = 2 * np.pi / n_fold
	
	rows = []
	cols = []
	data = []
	
	for i in range(num_elems):
		elemCenter = mesh.elem_centers[i, :]
		r = np.sqrt(elemCenter[0]**2 + elemCenter[2]**2)
		theta = np.arctan2(elemCenter[2], elemCenter[0])
		
		weight = 1.0 / n_fold
		rows.append(i)
		cols.append(i)
		data.append(weight)
		
		# Vectorized computation for all k at once
		k_values = np.arange(1, n_fold)
		new_thetas = theta + k_values[:, None] * angle_step
		new_x = r * np.cos(new_thetas)
		new_z = r * np.sin(new_thetas)
		
		# Create array of other element centers for all k
		otherElemCenters = np.column_stack((new_x.flatten(),
											np.full_like(new_x.flatten(), elemCenter[1]),
											new_z.flatten()))
		# Find nearest elements for each symmetric position
		sym_indices = [mesh.get_element_near_point(center) for center in otherElemCenters]
		
		# Append to COO matrix components
		rows.extend([i] * (n_fold - 1))
		cols.extend(sym_indices)
		data.extend([weight] * (n_fold - 1))
	
	HAY = coo_matrix((data, (rows, cols)), shape=(num_elems, num_elems)).tocsc()
	return HAY
def createZAngularSymmetryFilter(mesh: hex_mesher.HexMesher, n_fold: int) -> tuple[coo_matrix, np.ndarray]:
	"""Create a filter matrix for n-fold angular symmetry about Z axis.
	
	Args:
		mesh: The mesh object.
		n_fold: Number of symmetric segments 
	
	Returns:
		tuple containing:
			HA: Sparse matrix that enforces angular symmetry
			HAs: Array of row sums of HA matrix
	"""

	num_elems = mesh.num_elems
	angle_step = 2 * np.pi / n_fold
	
	rows = []
	cols = []
	data = []
	
	for i in range(num_elems):
		elemCenter = mesh.elem_centers[i, :]
		r = np.sqrt(elemCenter[0]**2 + elemCenter[1]**2)
		theta = np.arctan2(elemCenter[1], elemCenter[0])
		
		weight = 1.0 / n_fold
		rows.append(i)
		cols.append(i)
		data.append(weight)
		
		# Vectorized computation for all k at once
		k_values = np.arange(1, n_fold)
		new_thetas = theta + k_values[:, None] * angle_step
		new_x = r * np.cos(new_thetas)
		new_y = r * np.sin(new_thetas)
		
		# Create array of other element centers for all k
		otherElemCenters = np.column_stack((new_x.flatten(), 
										   new_y.flatten(), 
										   np.full_like(new_x.flatten(), elemCenter[2])))
		# Find nearest elements for each symmetric position
		sym_indices = [mesh.get_element_near_point(center) for center in otherElemCenters]
		
		# Append to COO matrix components
		rows.extend([i] * (n_fold - 1))
		cols.extend(sym_indices)
		data.extend([weight] * (n_fold - 1))
	
	HAZ = coo_matrix((data, (rows, cols)), shape=(num_elems, num_elems)).tocsc()
	return HAZ


def createXExtrudeFilter(mesh: hex_mesher.HexMesher):	
	"""Create a filter matrix for extruding elements all the way through in the X direction.
	
	Args:
		mesh: The mesh object.
	
	Returns:
		tuple containing:
			HXE: Sparse matrix that when multiplied with density vector enforces X extrusion
			HXEs: Array of row sums of HXE matrix
	"""
	num_elems = mesh.num_elems
	x_max = mesh.elem_centers[:, 0].max()
	x_min = mesh.elem_centers[:, 0].min()
	# Initialize COO matrix arrays
	rows = []
	cols = []
	data = []
	elemsProcessed = []
	for i in range(num_elems):
		elemCenter = mesh.elem_centers[i, :]
		# sufficient to process elements with x = x_min 
		if i in elemsProcessed:
			continue
		# Find all elements along the x-axis for current y,z position
		# Count unique x positions in mesh
		num_x_layers = mesh.grid[0]
		x_positions = np.linspace(x_min, x_max, num_x_layers)
		matching_elems = []
		for x in x_positions:
			test_point = np.array([x, elemCenter[1], elemCenter[2]])
			elem_id = mesh.get_element_near_point(test_point)
			if (elem_id == -1):
				continue
			if elem_id not in matching_elems: # Avoid duplicates
				matching_elems.append(elem_id)
	
		matching_elems.append(i)
		# Add all matching elements to processed list to avoid redundant processing
		elemsProcessed.extend(matching_elems)
		# Set equal weights for all matching elements
		weight = 1.0 / len(matching_elems)
		for m1 in matching_elems:
			for m2 in matching_elems:
				rows.append(m1)
				cols.append(m2)
				data.append(weight)

	HXE = coo_matrix((data, (rows, cols)), shape=(num_elems, num_elems)).tocsc()
	return HXE
def createYExtrudeFilter(mesh: hex_mesher.HexMesher):
	"""Create a filter matrix for extruding elements all the way through in the Y direction.
	
	Args:
		mesh: The mesh object.
	
	Returns:
		tuple containing:
			HYE: Sparse matrix that when multiplied with density vector enforces Y extrusion
			HYEs: Array of row sums of HYE matrix
	"""
	num_elems = mesh.num_elems
	y_max = mesh.elem_centers[:, 1].max()
	y_min = mesh.elem_centers[:, 1].min()
	# Initialize COO matrix arrays
	rows = []
	cols = []
	data = []
	elemsProcessed = []
	for i in range(num_elems):
		elemCenter = mesh.elem_centers[i, :]
		# sufficient to process elements with y = y_min 
		if i in elemsProcessed:
			continue
		# Find all elements along the y-axis for current x,z position
		# Count unique y positions in mesh
		num_y_layers = mesh.grid[1]
		y_positions = np.linspace(y_min, y_max, num_y_layers)
		matching_elems = []
		for y in y_positions:
			test_point = np.array([elemCenter[0], y, elemCenter[2]])
			elem_id = mesh.get_element_near_point(test_point)
			if (elem_id == -1):
				continue
			if elem_id not in matching_elems: # Avoid duplicates
				matching_elems.append(elem_id)
	
		matching_elems.append(i)
		# Add all matching elements to processed list to avoid redundant processing
		elemsProcessed.extend(matching_elems)
		# Set equal weights for all matching elements
		weight = 1.0 / len(matching_elems)
		for m1 in matching_elems:
			for m2 in matching_elems:
				rows.append(m1)
				cols.append(m2)
				data.append(weight)

	HYE = coo_matrix((data, (rows, cols)), shape=(num_elems, num_elems)).tocsc()
	return HYE
def createZExtrudeFilter(mesh: hex_mesher.HexMesher):
	"""Create a filter matrix for extruding elements all the way through in the Z direction.
	
	Args:
		mesh: The mesh object.
	
	Returns:
		tuple containing:
			HZE: Sparse matrix that when multiplied with density vector enforces Z extrusion
			HZEs: Array of row sums of HZE matrix
	"""
	num_elems = mesh.num_elems
	z_max = mesh.elem_centers[:, 2].max()
	z_min = mesh.elem_centers[:, 2].min()
	# Initialize COO matrix arrays
	rows = []
	cols = []
	data = []
	elemsProcessed = []
	for i in range(num_elems):
		elemCenter = mesh.elem_centers[i, :]
		# sufficient to process elements with z = z_min 
		if i in elemsProcessed:
			continue
		# Find all elements along the z-axis for current x,y position
		# Count unique z positions in mesh
		num_z_layers = mesh.grid[2]
		z_positions = np.linspace(z_min, z_max, num_z_layers)
		matching_elems = []
		for z in z_positions:
			test_point = np.array([elemCenter[0], elemCenter[1], z])
			elem_id = mesh.get_element_near_point(test_point)
			if (elem_id == -1):
				continue
			if elem_id not in matching_elems: # Avoid duplicates
				matching_elems.append(elem_id)
	
		matching_elems.append(i)
		# Add all matching elements to processed list to avoid redundant processing
		elemsProcessed.extend(matching_elems)
		# Set equal weights for all matching elements
		weight = 1.0 / len(matching_elems)
		for m1 in matching_elems:
			for m2 in matching_elems:
				rows.append(m1)
				cols.append(m2)
				data.append(weight)

	HZE = coo_matrix((data, (rows, cols)), shape=(num_elems, num_elems)).tocsc()
	return HZE
