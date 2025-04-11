# calculations/solver.py
import numpy as np
import netket as nk


def exact_dense(H):
    """
    Computes the exact solution of the Hamiltonian H using dense matrix methods.
    """
    dense_H = H.to_dense()
    eig_vals,eig_vecs = np.linalg.eigh(dense_H)
    e_0 = eig_vals[0]
    print("Exact ground state energy (dense):", e_0)
    return e_0, eig_vecs[:, 0]


def aprox_sol_sparse(H):
    """
    Computes the approximate solution of the Hamiltonian H.
    """
    eig_vals, eig_vecs = nk.exact.lanczos_ed(H, k=5, compute_eigenvectors=True)
    print(f"Eigenvalues with exact lanczos (sparse): {eig_vals[0]}")
    return eig_vals[0],eig_vecs[:, 0] 


def compute_overlap(v_0,vstate):
    """
    Computes the overlap between the exact ground state and the variational state.
    """
    overlap = np.abs(np.vdot(v_0, vstate.to_array()))**2
    #print(f"Overlap: {overlap}")
    return overlap