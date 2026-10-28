import pyvista as pv
import numpy as np
import trimesh
import torch
import torch.nn as nn
from pyto.examples_benchmarks.topopt_structural_benchmarks import *


class CNN3D(nn.Module):
    def __init__(self):
        super().__init__()
        # A simple 3D Autoencoder-like structure for smoothing
        # Input: 1 channel (density), Output: 1 channel (smoothed density)
        
        self.encoder = nn.Sequential(
            # Layer 1: Input -> 16 channels
            nn.Conv3d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(16),
            
            # Layer 2: 16 -> 32 channels
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(32),
        )
        
        self.decoder = nn.Sequential(
            # Layer 3: 32 -> 16 channels
            nn.Conv3d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(16),
            
            # Layer 4: Output layer (1 channel)
            nn.Conv3d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid() # Output between 0 and 1
        )

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # float32 explicitly: pyto.autodiff.material_model sets torch's default dtype to float64 when imported (the
        # FE solves need it), which made these layers float64 while the input is float32 ("Input type (float) and
        # bias type (double) should be the same" in the GUI's STL recovery).
        self.to(device=self.device, dtype=torch.float32)

    def forward(self, x):
        x = x.to(device=self.device, dtype=torch.float32)
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
SMOOTHING_METHODS = ("cnn", "gaussian")    # STL recovery: how the voxel density is smoothed before the isosurface
DEFAULT_SMOOTHING = "cnn"
DEFAULT_GAUSSIAN_SIGMA = 0.5               # Gaussian width, grid spacings; wider erodes thin members (Results/SmoothingComparison)


def extract_isosurface(vtu, isovalue=0.5, method=DEFAULT_SMOOTHING, sigma=DEFAULT_GAUSSIAN_SIGMA):
    """Void isosurface of the design for STL recovery, after smoothing the density by `method`:
      'cnn'       a small 3D convolutional autoencoder fitted to this density (stops at MSE 2e-4 or 400 epochs);
                  its imperfect reconstruction is the smoothed field
      'gaussian'  a Gaussian filter of width `sigma` grid spacings (scipy.ndimage), deterministic
    Both share the grid preparation (_void_field) and the marching cubes at `isovalue`."""
    if method == "cnn":
        return extract_isosurface_cnn(vtu, isovalue)
    if method == "gaussian":
        return extract_isosurface_gaussian(vtu, isovalue, sigma)
    raise ValueError(f"unknown smoothing method {method!r}; choose one of {SMOOTHING_METHODS}")


def _void_field(vtu):
    """The density on the full regular grid of the mesh nodes, padded and inverted (1 = void, 0 = material).
    Returns (field, origin, spacing) of the padded grid."""
    # --- STEP 1: DETECT GRID DIMENSIONS ---
    x_coords = np.unique(vtu.points[:, 0])
    y_coords = np.unique(vtu.points[:, 1])
    z_coords = np.unique(vtu.points[:, 2])
    
    nx, ny, nz = len(x_coords), len(y_coords), len(z_coords)
    print(f"Original Grid: {nx} x {ny} x {nz}")
    
    # Calculate spacing (robustly)
    dx = (x_coords[-1] - x_coords[0]) / (nx - 1) if nx > 1 else 1.0
    dy = (y_coords[-1] - y_coords[0]) / (ny - 1) if ny > 1 else 1.0
    dz = (z_coords[-1] - z_coords[0]) / (nz - 1) if nz > 1 else 1.0

    # --- STEP 2: PREPARE DATA (RESHAPE OR RESAMPLE) ---
    if "density" not in vtu.point_data:
        vtu = vtu.cell_data_to_point_data()
    
    raw_density = vtu.point_data["density"]
    expected_size = nx * ny * nz

    #just giving a chk point to see, to make sure
    # Check if the VTU is a perfect grid or sparse/unstructured
    if raw_density.size == expected_size:
        # structured grid
        grid_density = raw_density.reshape((nx, ny, nz), order='F')
        #grid_density = raw_density.reshape((nx, ny, nz), order='C')

    else:
        print(f"Mismatch: Points {raw_density.size} vs Grid {expected_size}. Resampling to structured grid...")
        # Create the full target grid
        grid = pv.ImageData(dimensions=(nx, ny, nz), 
                            spacing=(dx, dy, dz),
                            origin=(x_coords[0], y_coords[0], z_coords[0]))
        
        # Sample the VTU onto the full structured grid
        # This maps values from the mesh to the grid points
        grid = grid.sample(vtu)
        
        grid_density = grid.point_data["density"]
        # Points outside the mesh (holes and cut-outs of a non-box design domain) count as material, like the padding
        # below: as void they put a void surface on the domain's own boundary, and subtracting it from the domain STL
        # cut slivers off the part and split it into several bodies.
        grid_density = np.nan_to_num(grid_density, nan=1.0)
        
        # Reshape
        grid_density = grid_density.reshape((nx, ny, nz), order='F')

    # --- ADD PADDING --- due to marching cubes issues at boundaries and previous code failed for torque plate 
    # We pad with 1.0 (Solid) so the Void (0.0) is capped inside.
    # np.pad adds layers around the 3D array.
    # ((1,1), (1,1), (1,1)) means add 1 layer before and 1 layer after for each axis.
    padded_density = np.pad(grid_density, ((1,1), (1,1), (1,1)), mode='constant', constant_values=1.0)
    
    # Update dimensions for the new padded grid
    pnx, pny, pnz = padded_density.shape
    print(f"Padded Grid: {pnx} x {pny} x {pnz}")

    # Normalize and Invert (unnecessary if input is already 0-1)
    # (If not, normalize first)
    padded_density = (padded_density - padded_density.min()) / (padded_density.max() - padded_density.min() + 1e-12)
    padded_density = 1.0 - padded_density # Invert: 1=Void, 0=Material (Padding becomes 0)
    # Origin of the padded grid: shifted back by one spacing unit
    return padded_density, (x_coords[0] - dx, y_coords[0] - dy, z_coords[0] - dz), (dx, dy, dz)


def _contour(field, origin, spacing, isovalue):
    grid = pv.ImageData(dimensions=field.shape, spacing=spacing, origin=origin)
    grid.point_data["smoothed"] = np.asarray(field, dtype=float).flatten(order='F')
    return grid.contour([isovalue], scalars="smoothed", method='marching_cubes')


def extract_isosurface_gaussian(vtu, isovalue=0.5, sigma=DEFAULT_GAUSSIAN_SIGMA):
    from scipy.ndimage import gaussian_filter
    padded_density, origin, spacing = _void_field(vtu)
    # Smooth the grid without its padding layer, extending the field by its edge values (mode='nearest'): blurring the
    # padding (material) into a thin design shrank the voids that reach the domain faces, e.g. through the
    # thickness of a 2.5D design. Then put the padding back.
    inner = gaussian_filter(padded_density[1:-1, 1:-1, 1:-1], sigma=sigma, mode='nearest')
    return _contour(np.pad(inner, 1, constant_values=0.0), origin, spacing, isovalue)


def extract_isosurface_cnn(vtu, isovalue=0.5):
    print("Starting Direct 3D CNN extraction with PADDING...")
    padded_density, origin, spacing = _void_field(vtu)
    input_tensor = torch.tensor(padded_density).float().unsqueeze(0).unsqueeze(0)

    # --- STEP 3: TRAIN CNN ---
    model = CNN3D()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    
    print("Training CNN...")
    model.train()
    for epoch in range(400): # Fast training
        optimizer.zero_grad()
        output = model(input_tensor)
        loss = loss_fn(output, input_tensor.to(model.device))
        loss.backward()
        optimizer.step()
        if loss.item() < 0.0002:
            print(f"Converged at epoch {epoch}, Loss: {loss.item():.6f}")
            break
        if epoch % 20 == 0: print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    # --- STEP 4: INFERENCE ---
    model.eval()
    with torch.no_grad():
        smoothed = model(input_tensor).cpu().numpy()[0, 0]
    return _contour(smoothed, origin, spacing, isovalue)

def visualize(original_stl, other_surface,other_surface_label="Void Isosurface"):
    p = pv.Plotter()
    p.add_mesh(original_stl, color='lightblue', opacity=0.3, label="Original STL")
    p.add_mesh(other_surface, color='red', opacity=0.9, label=other_surface_label)
    p.add_legend()
    p.camera_position = 'iso'
    p.show()

def to_trimesh(mesh):
    """Convert a PyVista mesh to Trimesh."""
    faces = mesh.faces.reshape(-1, 4)[:, 1:4]
    return trimesh.Trimesh(vertices=mesh.points, faces=faces)

def subtract_voids_from_stl(stl, void_surface):
    # Load STL as PyVista, convert both to Trimesh

    tm_base = to_trimesh(stl)
    tm_void = to_trimesh(void_surface)
    
    # Fill holes and fix normals for robust boolean
    trimesh.repair.fill_holes(tm_base)
    trimesh.repair.fix_normals(tm_base)
    trimesh.repair.fill_holes(tm_void)
    trimesh.repair.fix_normals(tm_void)
    
    # Do boolean subtraction (STL - voids)
    result_tm = trimesh.boolean.difference([tm_base, tm_void])
    if result_tm is not None and result_tm.faces.shape[0] > 0:
        # Where a void surface grazes a domain face the Boolean leaves sliver triangles with edges ~1e-6 of the model
        # size. Saved as STL (float32) or merged, they collapse and the part becomes non-manifold, which TetGen
        # refuses. Merge vertices closer than 1e-5 of the model size and drop the collapsed triangles.
        tol = 1e-5 * float(np.linalg.norm(result_tm.bounds[1] - result_tm.bounds[0]))
        merged = pv.PolyData(result_tm.vertices, faces=np.hstack(
            [np.full((len(result_tm.faces), 1), 3), result_tm.faces])).clean(tolerance=tol, absolute=True)
        f = merged.faces.reshape(-1, 4)[:, 1:]
        f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]
        cleaned = trimesh.Trimesh(np.asarray(merged.points), f)
        cleaned.update_faces(cleaned.unique_faces())
        cleaned.remove_unreferenced_vertices()
        if cleaned.is_watertight:
            result_tm = cleaned
    if result_tm is None or result_tm.faces.shape[0] == 0:
        # Used to print a message and return the input STL unchanged, which the GUI then saved as the "optimized"
        # part. An empty result (e.g. the voids cover the whole domain) is a failure, not a part.
        raise RecoveryError("STL recovery failed: the Boolean subtraction returned an empty part.")
    return pv.PolyData(result_tm.vertices, faces=np.hstack([np.full((len(result_tm.faces), 1), 3), result_tm.faces]))


class RecoveryError(RuntimeError):
    """STL recovery produced no usable part (empty Boolean result)."""


def recovered_volume_check(part, voxel_volume, rel_tol=0.3):
    """(ok, ratio): the recovered part's volume against the voxel design's (elements with density >= 0.5).
    A ratio outside 1 +- rel_tol means the recovery removed or kept far more than the design (a wrong iso-level,
    a domain STL that does not match the mesh, a void surface that leaked); the part should not be used unchecked."""
    vol = float(abs(to_trimesh(part).volume))
    ratio = vol / voxel_volume if voxel_volume > 0 else float("inf")
    return abs(ratio - 1.0) <= rel_tol, ratio

if __name__ == "__main__":


    # 3x dirname (was 1x before this file moved from src/ to src/pyto/io/ in
    # Phase 6): points back at src/, matching the '../Models/...' joins below.
    script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print(f"Script directory: {script_dir}")
    design_domain_stl_file = os.path.join(script_dir, '../Models/BrakePedal/BRAKES_BRAKE_PEDAL.STL')
    print(f"Design Domain STL file: {design_domain_stl_file}")
    vtu_file= os.path.join(script_dir, '../Models/BrakePedal/BRAKES_BRAKE_PEDAL.vtu')
    output_file = os.path.join(script_dir, '../Models/BrakePedal/BRAKES_BRAKE_PEDAL_recovered.STL')
    
    design_domain = pv.read(design_domain_stl_file).triangulate().compute_normals()  # Read and triangulate the STL file
    vtu = pv.read(vtu_file)

    void_region_stl = extract_isosurface_cnn(vtu)
    visualize(design_domain, void_region_stl,"Void Isosurface")

    optimized_topology_stl = subtract_voids_from_stl(design_domain, void_region_stl)
    visualize(design_domain, optimized_topology_stl,"Optimized STL")
    optimized_topology_stl.save(output_file)
