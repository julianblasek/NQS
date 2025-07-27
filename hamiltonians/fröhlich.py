"""
frohlich.py -- Hamiltonian builders for 1-D / 3-D Froehlich polarons
====================================================================

This module provides *factory functions* that construct NetKet
LocalOperator instances for several flavours of the Froehlich
impurity-phonon Hamiltonian used in the project:

* **1-D static model**  – zero impurity momentum
* **1-D dynamic model** – finite impurity momentum
* **3-D static model**  – spherically symmetric UV / IR cut-off shell
* **3-D dynamic model** – finite-momentum impurity

Each builder returns a tuple ``(H, e_0)`` where

* ``H``   is the fully assembled NetKet operator, ready for VMC, and
* ``e_0`` is the analytic (or perturbative) ground-state energy
  used as a benchmark reference.

Layout
------
1.  Imports
2.  1-D Hamiltonians
    2.1  Dynamic  (``build_hamilton_dynamic_1d``)
    2.2  Static   (``build_hamilton_1d``)
3.  3-D Hamiltonians
    3.1  Dynamic  (``build_hamilton_dynamic_3d``)
    3.2  Static   (``build_hamilton_3d``)
4.  Utility helpers
    4.1  ``map_k_to_index`` –  (k1,k2,k3) -> flat Hilbert-space index
    4.2  ``norm_k``         –  Euclidean norm |k| in 3-D
"""

# ------------------------------------------------------------------
# 1. IMPORTS
# ------------------------------------------------------------------
# bosonic ladder ops
from netket.operator.boson import create, destroy, number
# NetKet container   
from netket.operator import LocalOperator
# numerical helpers                   
import numpy as np                                          


# ------------------------------------------------------------------
# 2. 1-D HAMILTONIANS
# ------------------------------------------------------------------
# 2.1 Dynamic 1-D
def build_hamilton_dynamic_1d(hi, N_modes, N_max, omega, 
                              alpha_coupling, P):
    """
    Froehlich Hamiltonian in **one dimension** for a *moving* impurity
    with momentum *P*.  Momentum conservation appears via
    (P - k n_k)^2.

    Parameters
    ----------
    hi : nk.hilbert.Fock
        Bosonic Hilbert space with cutoff N_max.
    N_modes : int
        Number of discrete momentum modes (even).
    omega, alpha_coupling : float
        Phonon frequency and dimensionless coupling.
    P : float
        Total momentum carried by the impurity.

    Returns
    -------
    (H, e_0) : tuple[LocalOperator, float]
        NetKet operator and perturbative ground-state energy.
    """
    # Allowed momenta k in {...,-2,-1,1,2,...}; k = 0 excluded
    k_vals = np.concatenate(
        [np.arange(-N_modes // 2, 0, dtype=int),
         np.arange(1,  N_modes // 2 + 1, dtype=int)]
    )

    def coupling(k):
        """Linear Froehlich coupling g_k proportional to 1/k 
        (complex)."""
        z = 0 - 1j
        return alpha_coupling * z / k

    # Kinetic term of the impurity: (P - sum_k k n_k)^2
    H   = LocalOperator(hi, dtype=np.complex128)
    kin = LocalOperator(hi, dtype=np.complex128)
    kin += sum(P - k * number(hi, i) for i, k in enumerate(k_vals))

    # Phonon energy omega * sum_k n_k  +  g_k (b_k† - b_k)
    H += sum(omega * number(hi, i) for i in range(N_modes))
    H += sum(coupling(k) * (-create(hi, i) + destroy(hi, i))
             for i, k in enumerate(k_vals))

    # Complete Hamiltonian
    H = kin @ kin + H

    # Large-alpha (mean-field) estimate of ground-state energy
    e_0 = -alpha_coupling + P**2 / (1 + alpha_coupling / 6)

    return H, e_0


# 2.2 Static 1-D
def build_hamilton_1d(hi, N_modes, N_max, omega, alpha_coupling):
    """
    Froehlich Hamiltonian in **one dimension** for a *static* impurity
    (P = 0).  Simpler than the dynamic variant: no kinetic back-action.
    """
    
    k_vals = np.concatenate(
        [np.arange(-N_modes // 2, 0, dtype=int),
         np.arange(1,  N_modes // 2 + 1, dtype=int)]
    )

    def coupling(k):
        z = 0 - 1j
        return alpha_coupling * z / k

    H = LocalOperator(hi, dtype=np.complex128)
    H += sum(omega * number(hi, i) for i in range(N_modes))
    H += sum(coupling(k) * (-create(hi, i) + destroy(hi, i))
             for i, k in enumerate(k_vals))

    # Weak-coupling perturbative ground-state energy
    e_0 = -1 * sum(alpha_coupling**2 / (omega * abs(k)**2) 
                   for k in k_vals)

    return H, e_0


# ------------------------------------------------------------------
# 3. 3-D HAMILTONIANS
# ------------------------------------------------------------------
# 3.1 Dynamic 3-D
def build_hamilton_dynamic_3d(hi, N_modes, N_max, omega,
                              alpha_coupling, P, m_b):
    """
    3-D Froehlich Hamiltonian with *finite-momentum* impurity.

    Sums are restricted to a cubic momentum lattice
    k in [-N_modes/2, ..., N_modes/2]^3 and additionally to the
    spherical shell  k_min <= |k| <= k_max  (UV / IR cut-off).
    """
    
    k_min = 1        # IR cut-off   |k| >= 1
    k_max = 2        # UV cut-off   |k| <= 2

    k1 = np.arange(-N_modes // 2, N_modes // 2 + 1, dtype=int)
    k2 = k1
    k3 = k1
    
    print("k1:", k1)

    # Re-scale alpha for the continuum dispersion convention
    alpha_coupling = alpha_coupling * np.sqrt(m_b / (2 * omega))

    def v_k(norm_k):
        """3-D Froehlich vertex V_k proportional to 1/|k|."""
        z = 0 - 1j
        return (z / norm_k) * omega * 2 * np.sqrt(alpha_coupling) * \
               (1 / (2 * m_b * omega)) ** 0.25

    H   = LocalOperator(hi, dtype=np.complex128)
    kin = LocalOperator(hi, dtype=np.complex128)
    
    # will accumulate perturbative contribution
    e_0 = 0  

    # Variational parameter  eta = alpha / 6  /  (1 + alpha / 6)
    eta = (alpha_coupling / 6) / (1 + alpha_coupling / 6)

    for i in k1:
        for j in k2:
            for k in k3:
                current_norm = norm_k(i, j, k)
                # Skip modes outside the spherical shell
                if current_norm < k_min or current_norm > k_max:
                    continue

                v   = v_k(current_norm)
                idx = map_k_to_index(i, j, k, N_modes)

                # Impurity-phonon momentum coupling
                kin += 2 * P * (i + j + k) * number(hi, idx)

                # Phonon dispersion  omega + k^2 / (2 m_b)
                H += (omega + current_norm**2 / (2 * m_b)) * number(hi, idx)
                H += v * (-create(hi, idx) + destroy(hi, idx))

                # Rayleigh-Schroedinger 2nd-order energy shift
                denom = (omega
                         - P * (i + j + k) * (1 - eta)
                         + current_norm**2 / (2 * m_b))
                e_0 += v * np.conj(v) / denom

    # Complete Hamiltonian:  (P - sum k)^2 / (2 m_b)  +  phonon terms
    H = (P**2 - kin) / (2 * m_b) + H

    # Analytic ground-state energy estimate
    e_0 = P**2 / (2 * m_b) * (1 - eta**2) - e_0

    return H, e_0.real


# 3.2 Static 3-D
def build_hamilton_3d(hi, N_modes, N_max, omega, alpha_coupling):
    """
    3-D Froehlich Hamiltonian for a *static* impurity (P = 0)
    with spherical UV / IR cut-off.  Simpler than the dynamic case.
    """
    
    k_min = 1
    k_max = 2

    k1 = np.arange(-N_modes // 2, N_modes // 2 + 1, dtype=int)
    k2 = k1
    k3 = k1
    
    print("k1:", k1)

    def coupling(k1, k2, k3):
        """Static 3-D coupling g_k proportional to  -alpha / |k|."""
        return -1 * alpha_coupling / norm_k(k1, k2, k3)

    H   = 0
    e_0 = 0
    for i in k1:
        for j in k2:
            for k in k3:
                current_norm = norm_k(i, j, k)
                if current_norm < k_min or current_norm > k_max:
                    continue

                idx = map_k_to_index(i, j, k, N_modes)
                H  += omega * number(hi, idx)
                H  += coupling(i, j, k) * (create(hi, idx) + destroy(hi, idx))
                e_0 += -1 * alpha_coupling**2 / (omega * current_norm**2)

    return H, e_0


# ------------------------------------------------------------------
# 4. UTILITY HELPERS
# ------------------------------------------------------------------
def map_k_to_index(k1, k2, k3, N_modes):
    """
    Map lattice momentum (k1,k2,k3) in [-N/2, ..., N/2]^3 onto a *flat*
    index in the 1-D NetKet Fock-space layout:

        (k1,k2,k3)  ->  i1 * N^2 + i2 * N + i3
        with  i1, i2, i3 in {0,...,N}

    where N = N_modes + 1 and i_x = k_x + offset centres the range at 0.
    """
    
    N      = N_modes + 1
    offset = N // 2     # shift so that k = -N//2 maps to 0

    i1 = k1 + offset
    i2 = k2 + offset
    i3 = k3 + offset

    return i1 * (N**2) + i2 * N + i3


def norm_k(k1, k2, k3):
    """Euclidean norm |k| in lattice units."""
    return (k1**2 + k2**2 + k3**2) ** 0.5
