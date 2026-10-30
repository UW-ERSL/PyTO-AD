"""Setup shared by the optimizer drivers (mma.py, oc.py): density filters, elements-with-forces, element stiffness
matrix, nodal body force mapping. Defined once here so the drivers cannot diverge.
"""
import warnings

import numpy as np
import torch

import pyto.physics.hex_element_stiffness as hex_element_stiffness
import pyto.physics.structural.hex_structural_fea as hex_structural_fea
import pyto.physics.thermal.hex_thermal_fea as hex_thermal_fea
from pyto.topopt.common import createFilters, find_elements_with_forces


def compute_element_stiffness(fe_solver, log_message=None):
    """Element stiffness matrix KE for fe_solver's physics and material(s).

    Multi-material meshes use only the first material's KE, matching the
    simplification already made throughout the optimization pipeline
    (e.g. the torch-native solve()s only ever use elem_stiff_torch's
    broadcast against a single per-physics stiffness tensor).
    """
    if isinstance(fe_solver.mat_prop, list):  # multiple materials (assume same props)
        if isinstance(fe_solver, hex_structural_fea.HexStructuralFEA):
            KE_list = [
                hex_element_stiffness.hex8_stiffness_matrix_structural(
                    mp.youngs_modulus, mp.poissons_ratio, fe_solver.mesh.elem_size
                )
                for mp in fe_solver.mat_prop
            ]
        elif isinstance(fe_solver, hex_thermal_fea.HexThermalFEA):
            KE_list = [
                hex_element_stiffness.hex8_stiffness_matrix_thermal(
                    mp.thermal_conductivity, fe_solver.mesh.elem_size
                )
                for mp in fe_solver.mat_prop
            ]
        else:
            raise TypeError(f"Unsupported fe_solver type: {type(fe_solver)}")
        if log_message:
            log_message("Assuming all elements have the same material properties")
        return KE_list[0]

    if isinstance(fe_solver, hex_structural_fea.HexStructuralFEA):
        return hex_element_stiffness.hex8_stiffness_matrix_structural(
            fe_solver.mat_prop.youngs_modulus,
            fe_solver.mat_prop.poissons_ratio,
            fe_solver.mesh.elem_size,
        )
    if isinstance(fe_solver, hex_thermal_fea.HexThermalFEA):
        return hex_element_stiffness.hex8_stiffness_matrix_thermal(
            fe_solver.mat_prop.thermal_conductivity, fe_solver.mesh.elem_size
        )
    raise TypeError(f"Unsupported fe_solver type: {type(fe_solver)}")


def compute_nodal_body_force(fe_solver):
    """Map fe_solver.elem_body_force (per-element, structural only) to a
    per-node force vector, or None if the problem has no body force.
    """
    if fe_solver.elem_body_force is None:
        return None
    elem_force = fe_solver.elem_body_force.copy()
    nNodes = fe_solver.mesh.num_nodes
    nodal_body_force = np.zeros((nNodes * 3,))
    nodal_body_force[0::3] = fe_solver.mesh.elem_to_node_field_mapping @ elem_force[0::3]
    nodal_body_force[1::3] = fe_solver.mesh.elem_to_node_field_mapping @ elem_force[1::3]
    nodal_body_force[2::3] = fe_solver.mesh.elem_to_node_field_mapping @ elem_force[2::3]
    return nodal_body_force


def setup_driver_state(fe_solver, to_params, log_message=None):
    """Everything an optimizer driver needs before its iteration loop starts.

    Returns a dict with:
      H, Hs               -- density filter (scipy sparse, numpy row sums)
      elemsWithForces      -- element indices with a forced node (retained during filtering)
      KE                   -- element stiffness matrix (single material / first of multi)
      nodal_body_force      -- per-node body force vector, or None
    """
    nDOFPerNode = 3 if isinstance(fe_solver, hex_structural_fea.HexStructuralFEA) else 1

    if log_message:
        log_message("Computing Filters ...")
    H, Hs = createFilters(fe_solver, to_params)

    elemsWithForces = find_elements_with_forces(
        fe_solver.mesh, fe_solver.bc.force, nDOFPerNode
    )

    KE = compute_element_stiffness(fe_solver, log_message)
    nodal_body_force = compute_nodal_body_force(fe_solver)

    return {
        "H": H,
        "Hs": Hs,
        "elemsWithForces": elemsWithForces,
        "KE": KE,
        "nodal_body_force": nodal_body_force,
    }


def heaviside_projection(x_f, beta, eta=0.5):
    """Smooth Heaviside projection of the filtered density (Ooms et al. 2023, eq. 33), differentiable by autograd.
    Shared by the MMA driver and check_gradient, so the gradient check covers the code the driver runs."""
    import math
    import torch
    return (math.tanh(beta * eta) + torch.tanh(beta * (x_f - eta))) / (math.tanh(beta * eta) + math.tanh(beta * (1.0 - eta)))


def torch_sparse_filter(H, Hs):
    """Convert the (constant) scipy-sparse density filter to torch, for use
    inside an autograd chain -- H @ x_raw can't be matmul'd against a
    grad-tracked torch tensor directly (see plan Phase 0's topopt_mma.py fix).
    """
    import torch

    H_coo = H.tocoo()
    H_torch = torch.sparse_coo_tensor(
        torch.tensor(np.vstack([H_coo.row, H_coo.col]), dtype=torch.long),
        torch.tensor(H_coo.data, dtype=torch.float64),
        H_coo.shape,
    ).coalesce()
    Hs_torch = torch.tensor(np.asarray(Hs).flatten(), dtype=torch.float64)
    return H_torch, Hs_torch


class _CsrMatmul(torch.autograd.Function):
    """H @ x, with HT @ g as the backward; both are torch CSR products. torch.sparse.mm on a COO H is ~25x slower
    forward, and its backward transposes and re-sorts H on every call."""

    @staticmethod
    def forward(ctx, x, H, HT):
        ctx.HT = HT
        return H @ x

    @staticmethod
    def backward(ctx, g):
        return ctx.HT @ g, None, None


class TorchDensityFilter:
    """x -> H x / Hs for the constant scipy-sparse density filter, differentiable by autograd. H and H^T are held as
    torch CSR (H^T stored on its own: symmetry and extrusion filters make H unsymmetric)."""

    def __init__(self, H, Hs):
        def csr(A):
            A = A.tocsr()
            A.sort_indices()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)      # "sparse CSR support is in beta state"
                return torch.sparse_csr_tensor(torch.from_numpy(A.indptr.astype(np.int64)),
                                               torch.from_numpy(A.indices.astype(np.int64)),
                                               torch.from_numpy(A.data.astype(np.float64)), A.shape)
        self.H, self.HT = csr(H), csr(H.T)
        self.Hs = torch.tensor(np.asarray(Hs).flatten(), dtype=torch.float64)

    def __call__(self, x):
        return _CsrMatmul.apply(x, self.H, self.HT) / self.Hs

    def transpose(self, g):
        """H^T (g / Hs) for each row of g, shape (k, n): the chain rule through __call__ by hand."""
        return (self.HT @ (g / self.Hs).T).T


BINARIZATION_MAX_LOSS = 1.0   # reject a 0/1 design more than (1 + this) times worse than the continuous one


def binarization_acceptable(objective_binary, objective_continuous, sign=1.0, max_loss=BINARIZATION_MAX_LOSS):
    """Whether the thresholded 0/1 design may replace the continuous one.

    Volume-preserving thresholding keeps the densest elements. For a mostly grey design (e.g. heat conduction with
    the default conductivity penalty 1) those need not connect loads to supports or heat sources to sinks, and the
    0/1 design's objective blows up (seen: thermal compliance 1.5e3 -> 1.1e10). The drivers then keep the
    continuous design.

    `sign` is the direction the driver minimizes (compliance_sign for compliance), m = sign * J. "Worse by more than
    a factor (1 + max_loss)" means m_binary > (1 + max_loss) * m_continuous when m_continuous >= 0 (minimized
    positive quantity), and m_binary > m_continuous / (1 + max_loss) when m_continuous < 0 (a maximized positive
    quantity, or a negated one): the 0/1 value fell below 1/(1 + max_loss) of the continuous value.
    """
    mb, mc = sign * float(objective_binary), sign * float(objective_continuous)
    if not np.isfinite(mb):
        return False
    limit = (1.0 + max_loss) * mc if mc >= 0 else mc / (1.0 + max_loss)
    return mb <= limit
