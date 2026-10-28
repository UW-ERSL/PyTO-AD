"""STL recovery: the CNN must work while torch's default dtype is float64 (set by pyto.autodiff.material_model); the
Gaussian smoother gives a closed, deterministic part, also on a non-box domain."""
import numpy as np
import pytest
import torch


def test_cnn_runs_with_float64_default_dtype():
    import pyto.autodiff.material_model  # noqa: F401  (sets the float64 default, as in the GUI)
    from pyto.io.topopt_stl_recovery import CNN3D
    assert torch.get_default_dtype() == torch.float64
    model = CNN3D()
    assert all(p.dtype == torch.float32 for p in model.parameters())
    out = model(torch.rand(1, 1, 6, 5, 4, dtype=torch.float64))      # any input dtype is accepted
    assert out.shape == (1, 1, 6, 5, 4) and out.dtype == torch.float32


pv = pytest.importorskip("pyvista")
trimesh = pytest.importorskip("trimesh")


def _plate_with_hole(n=(16, 10, 3), h=0.01, outside=None):
    """Hex grid of a plate whose density is 0 in a through hole; `outside` removes elements (non-box domain)."""
    x, y, z = (np.arange(k + 1) * h for k in n)
    grid = pv.RectilinearGrid(x, y, z).cast_to_unstructured_grid()
    c = grid.cell_centers().points
    rho = np.where(np.hypot(c[:, 0] - 0.08, c[:, 1] - 0.05) < 0.025, 0.0, 1.0)
    if outside is not None:
        keep = ~outside(c)
        grid, rho = grid.extract_cells(np.nonzero(keep)[0]), rho[keep]
    grid.cell_data["density"] = rho
    return grid.cell_data_to_point_data(), grid


def _recover(vtu, domain):
    from pyto.io.topopt_stl_recovery import extract_isosurface, subtract_voids_from_stl
    part = subtract_voids_from_stl(domain.extract_surface().triangulate().compute_normals(),
                                   extract_isosurface(vtu, 0.5, method="gaussian"))
    return trimesh.Trimesh(np.asarray(part.points), part.faces.reshape(-1, 4)[:, 1:])


def test_gaussian_recovery_is_closed_deterministic_and_cuts_the_hole():
    vtu, grid = _plate_with_hole()
    a, b = _recover(vtu, grid), _recover(vtu, grid)
    assert a.is_watertight and len(a.split(only_watertight=False)) == 1
    assert np.isclose(a.volume, b.volume, rtol=0, atol=0)                  # same input, same part
    full = 0.16 * 0.10 * 0.03
    assert 0.6 * full < a.volume < 0.95 * full                            # the through hole is removed


def test_points_outside_a_non_box_domain_are_material():
    # an L-shaped domain: the missing corner is outside the mesh; it must not become a void surface on the boundary
    corner = lambda c: (c[:, 0] > 0.12) & (c[:, 1] > 0.06)
    vtu, grid = _plate_with_hole(outside=corner)
    part = _recover(vtu, grid)
    assert part.is_watertight and len(part.split(only_watertight=False)) == 1


def test_unknown_smoothing_method_is_rejected():
    from pyto.io.topopt_stl_recovery import extract_isosurface
    vtu, _ = _plate_with_hole()
    with pytest.raises(ValueError, match="smoothing method"):
        extract_isosurface(vtu, method="median")


def test_empty_boolean_is_an_error_not_the_input_stl():
    """All void: the void surface encloses the whole domain, the Boolean returns nothing. That used to return the
    input STL unchanged (and the GUI saved it as the optimized part); it must raise."""
    from pyto.io.topopt_stl_recovery import RecoveryError, extract_isosurface, subtract_voids_from_stl
    vtu, grid = _plate_with_hole()
    vtu.point_data["density"] = np.zeros(vtu.n_points)
    with pytest.raises(RecoveryError):
        subtract_voids_from_stl(grid.extract_surface().triangulate().compute_normals(),
                                extract_isosurface(vtu, 0.5, method="gaussian"))


def test_non_manifold_domain_raises():
    """Two cubes sharing only an edge: not a volume. The Boolean must refuse it, not return a part."""
    from pyto.io.topopt_stl_recovery import subtract_voids_from_stl
    a = pv.Cube(center=(0.5, 0.5, 0.5)).triangulate()
    b = pv.Cube(center=(1.5, 1.5, 0.5)).triangulate()
    domain = (a + b).clean().extract_surface().triangulate().compute_normals()
    void = pv.Sphere(radius=0.2, center=(0.5, 0.5, 0.5)).triangulate()
    with pytest.raises(ValueError, match="volumes"):
        subtract_voids_from_stl(domain, void)


def test_recovered_volume_check():
    from pyto.io.topopt_stl_recovery import recovered_volume_check
    cube = pv.Cube(x_length=2.0, y_length=1.0, z_length=1.0).triangulate()      # volume 2
    assert recovered_volume_check(cube, 2.0)[0]
    ok, ratio = recovered_volume_check(cube, 1.0)
    assert not ok and abs(ratio - 2.0) < 1e-9
