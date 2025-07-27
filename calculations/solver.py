"""
solver.py ― Exact-diagonalisation helpers for benchmarking NetKet VMC
=====================================================================

**Purpose.**  
For *small Hilbert spaces* the Fröhlich Hamiltonians can still be
diagonalised exactly, allowing us to **validate** the variational
Monte-Carlo (VMC) results.  This module provides two thin wrappers:

1.  **`exact_dense`**  
    Converts the NetKet :pyclass:`~netket.operator.LocalOperator` into a
    dense NumPy array and calls :pyfunc:`numpy.linalg.eigh`.  
    → *O(dim³)* memory/time – feasible only for very small systems.

2.  **`aprox_sol_sparse`**  
    Uses NetKet’s Lanczos routine (`nk.exact.lanczos_ed`) to obtain the
    lowest few eigenpairs in *sparse* representation.  
    → Much cheaper, but still scales poorly for large lattices.

The third helper **`compute_overlap`** calculates the squared overlap
‖⟨ψ₀|ψ_VMC⟩‖² between the exact ground state and a variational state,
providing a stringent quality metric.

In practice you would:

* call one of the two ED functions for **toy-size** test cases, and
* inspect the energy difference *and* overlap between the exact
  and variational states.  

For realistic 3-D Fröhlich lattices the Hilbert-space dimension explodes
((N_max+1)^{N_modes}), so exact methods quickly become impossible – the
VMC engine then remains the only option.

Layout
------
1.  Imports
2.  Dense exact diagonalisation          (``exact_dense``)
3.  Sparse Lanczos approximation         (``aprox_sol_sparse``)
4.  Exact–variational overlap utility    (``compute_overlap``)
"""

# ──────────────────────────────────────────────────────────────
# 1. IMPORTS
# ──────────────────────────────────────────────────────────────
import numpy as np
import netket as nk


# ──────────────────────────────────────────────────────────────
# 2. DENSE EXACT DIAGONALISATION
# ──────────────────────────────────────────────────────────────
def exact_dense(H):
    """
    Exact ground-state energy/vector via **dense** diagonalisation.

    Parameters
    ----------
    H : nk.operator.LocalOperator
        Hamiltonian in NetKet representation.

    Returns
    -------
    e_0 : float
        Lowest eigenvalue.
    v_0 : np.ndarray
        Corresponding (normalised) eigenvector.
    """
    dense_H = H.to_dense()
    eig_vals, eig_vecs = np.linalg.eigh(dense_H)  # Hermitian solver
    e_0 = eig_vals[0]
    print("Exact ground state energy (dense):", e_0)
    return e_0, eig_vecs[:, 0]


# ──────────────────────────────────────────────────────────────
# 3. SPARSE LANCZOS APPROXIMATION
# ──────────────────────────────────────────────────────────────
def aprox_sol_sparse(H):
    """
    Ground-state energy/vector using NetKet’s **Lanczos ED** (sparse).

    Useful for intermediate system sizes where dense diagonalisation is
    already infeasible but the Hilbert space still fits in memory.
    """
    eig_vals, eig_vecs = nk.exact.lanczos_ed(H, k=5, compute_eigenvectors=True)
    print(f"Eigenvalues with exact Lanczos (sparse): {eig_vals[0]}")
    return eig_vals[0], eig_vecs[:, 0]


# ──────────────────────────────────────────────────────────────
# 4. OVERLAP WITH VARIATIONAL STATE
# ──────────────────────────────────────────────────────────────
def compute_overlap(v_0, vstate):
    """
    Squared overlap ‖⟨ψ₀|ψ_VMC⟩‖² between exact and variational states.

    Parameters
    ----------
    v_0 : np.ndarray
        Exact ground-state vector.
    vstate : nk.vqs.AbstractVariationalState
        Variational state produced by NetKet.

    Returns
    -------
    overlap : float
        Probability‐like measure in [0, 1].
    """
    overlap = np.abs(np.vdot(v_0, vstate.to_array()))**2
    return overlap
