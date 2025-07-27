"""
fröhlich.py ― Hamiltonian builders for 1-D/3-D Fröhlich polarons
=================================================================

This module provides **factory functions** that construct NetKet
:pyclass:`LocalOperator` instances for various flavours of the Fröhlich
impurity–phonon Hamiltonian that are used throughout the project:

* **1-D static model** – zero impurity momentum *P = 0*
* **1-D dynamic model** – finite impurity momentum *P ≠ 0*
* **3-D static model** – spherically-symmetric UV/IR-cut-off shell
* **3-D dynamic model** – finite-momentum impurity with continuum
  dispersion ϵₖ ≈ k²/(2m_b)

Each builder returns a tuple ``(H, e_0)`` where

* ``H`` is the fully assembled NetKet operator, ready for VMC, and  
* ``e_0`` is the **analytical (or perturbative) ground-state energy**
  used as a benchmark reference.

No global state is kept – every call is independent.

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
    4.1  ``map_k_to_index`` –   (k₁,k₂,k₃) ↦ flat Hilbert-space index  
    4.2  ``norm_k``         –   Euclidean norm |k| in 3-D
"""

# ──────────────────────────────────────────────────────────────
# 1. IMPORTS
# ──────────────────────────────────────────────────────────────
from netket.operator.boson import create, destroy, number   # bosonic ladder ops
from netket.operator import LocalOperator                   # NetKet container
import numpy as np                                          # numerical helpers


# ──────────────────────────────────────────────────────────────
# 2. 1-D HAMILTONIANS
# ──────────────────────────────────────────────────────────────
# 2.1 Dynamic 1-D
def build_hamilton_dynamic_1d(hi, N_modes, N_max, omega, alpha_coupling, P):
    """
    Fröhlich Hamiltonian in **one dimension** for a *moving* impurity
    with momentum *P*.  Momentum conservation appears via the
    (P − k n̂ₖ)² term.

    Parameters
    ----------
    hi : nk.hilbert.Fock
        Bosonic Hilbert space with cutoff *N_max*.
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
    # Allowed momenta k ∈ {…,-2,-1,1,2,…}; k = 0 excluded
    k_vals = np.concatenate(
        [np.arange(-N_modes / 2, 0, dtype=int),
         np.arange(1, N_modes / 2 + 1, dtype=int)]
    )

    def coupling(k):
        """Linear Fröhlich coupling g_k ∝ 1/k (complex)."""
        z = 0 - 1j
        return alpha_coupling * z / k

    # Kinetic term of the impurity: (P − Σ_k k n̂ₖ)²
    H = LocalOperator(hi, dtype=np.complex128)
    kin = LocalOperator(hi, dtype=np.complex128)
    kin += sum(P - k * number(hi, i) for i, k in enumerate(k_vals))

    # Phonon energy ω Σ_k n̂ₖ  +  g_k (b̂ₖ† − b̂ₖ)
    H += sum(omega * number(hi, i) for i in range(N_modes))
    H += sum(coupling(k) * (-create(hi, i) + destroy(hi, i))
             for i, k in enumerate(k_vals))

    # Complete Hamiltonian
    H = kin @ kin + H

    # Large-α (mean-field) estimate of ground-state energy
    e_0 = -alpha_coupling + P**2 / (1 + alpha_coupling / 6)

    return H, e_0


# 2.2 Static 1-D
def build_hamilton_1d(hi, N_modes, N_max, omega, alpha_coupling):
    """
    Fröhlich Hamiltonian in **one dimension** for a *static* impurity
    (P = 0).  Simpler than the dynamic variant: no kinetic back-action.
    """
    k_vals = np.concatenate(
        [np.arange(-N_modes / 2, 0, dtype=int),
         np.arange(1, N_modes / 2 + 1, dtype=int)]
    )

    def coupling(k):
        z = 0 - 1j
        return alpha_coupling * z / k

    H = LocalOperator(hi, dtype=np.complex128)
    H += sum(omega * number(hi, i) for i in range(N_modes))
    H += sum(coupling(k) * (-create(hi, i) + destroy(hi, i))
             for i, k in enumerate(k_vals))

    # Weak-coupling perturbative ground-state energy
    e_0 = -1 * sum(alpha_coupling**2 / (omega * abs(k)**2) for k in k_vals)

    return H, e_0


# ──────────────────────────────────────────────────────────────
# 3. 3-D HAMILTONIANS
# ──────────────────────────────────────────────────────────────
# 3.1 Dynamic 3-D
def build_hamilton_dynamic_3d(hi, N_modes, N_max, omega,
                              alpha_coupling, P, m_b):
    """
    3-D Fröhlich Hamiltonian with **finite-momentum impurity**.

    Sums are restricted to a cubic momentum lattice
    k ∈ [−N_modes/2, …, N_modes/2]³ and additionally to the
    spherical shell *k_min ≤ |k| ≤ k_max* (UV/IR cut-off).
    """
    k_min = 1        # IR cut-off  |k| ≥ 1
    k_max = 2        # UV cut-off  |k| ≤ 2

    k1 = np.arange(-N_modes // 2, N_modes // 2 + 1, dtype=int)
    k2 = k1
    k3 = k1
    print("k1:", k1)

    # Re-scale α for the continuum dispersion convention
    alpha_coupling = alpha_coupling * np.sqrt(m_b / (2 * omega))

    def v_k(norm_k):
        """3-D Fröhlich vertex V_k ∝ 1/|k|."""
        z = 0 - 1j
        return (z / norm_k) * omega * 2 * np.sqrt(alpha_coupling) * \
               (1 / (2 * m_b * omega)) ** 0.25

    H = LocalOperator(hi, dtype=np.complex128)
    kin = LocalOperator(hi, dtype=np.complex128)
    e_0 = 0  # will accumulate perturbative contribution

    # Variational parameter η = α/6 / (1 + α/6)
    eta = (alpha_coupling / 6) / (1 + alpha_coupling / 6)

    for i in k1:
        for j in k2:
            for k in k3:
                current_norm = norm_k(i, j, k)
                # Skip modes outside the spherical shell
                if current_norm < k_min or current_norm > k_max:
                    continue

                v = v_k(current_norm)
                idx = map_k_to_index(i, j, k, N_modes)

                # Impurity–phonon momentum coupling
                kin += 2 * P * (i + j + k) * number(hi, idx)

                # Phonon dispersion ω + k²/(2m_b)
                H += (omega + current_norm**2 / (2 * m_b)) * number(hi, idx)
                H += v * (-create(hi, idx) + destroy(hi, idx))

                # Rayleigh–Schrödinger 2nd-order energy shift
                denom = (omega
                         - P * (i + j + k) * (1 - eta)
                         + current_norm**2 / (2 * m_b))
                e_0 += v * np.conj(v) / denom

    # Complete Hamiltonian  ( (P − Σ k)² / 2m_b  +  phonon terms )
    H = (P**2 - kin) / (2 * m_b) + H

    # Analytic ground-state energy estimate
    e_0 = P**2 / (2 * m_b) * (1 - eta**2) - e_0

    return H, e_0.real


# 3.2 Static 3-D
def build_hamilton_3d(hi, N_modes, N_max, omega, alpha_coupling):
    """
    3-D Fröhlich Hamiltonian for a *static* impurity (P = 0) with
    spherical UV/IR cut-off.  Simpler than the dynamic case.
    """
    k_min = 1
    k_max = 2

    k1 = np.arange(-N_modes // 2, N_modes // 2 + 1, dtype=int)
    k2 = k1
    k3 = k1
    print("k1:", k1)

    def coupling(k1, k2, k3):
        """Static 3-D coupling g_k ∝ −α/|k|."""
        return -1 * alpha_coupling / norm_k(k1, k2, k3)

    H = 0
    e_0 = 0
    for i in k1:
        for j in k2:
            for k in k3:
                current_norm = norm_k(i, j, k)
                if current_norm < k_min or current_norm > k_max:
                    continue

                idx = map_k_to_index(i, j, k, N_modes)
                H += omega * number(hi, idx)
                H += coupling(i, j, k) * (create(hi, idx) + destroy(hi, idx))
                e_0 += -1 * alpha_coupling**2 / (omega * current_norm**2)

    return H, e_0


# ──────────────────────────────────────────────────────────────
# 4. UTILITY HELPERS
# ──────────────────────────────────────────────────────────────
def map_k_to_index(k1, k2, k3, N_modes):
    """
    Map lattice momentum (k₁,k₂,k₃) ∈ [−N/2,…,N/2]³ onto a *flat*
    index in the 1-D NetKet Fock space layout:

        (k₁,k₂,k₃)  ↦  i₁·N² + i₂·N + i₃   with
        i₁, i₂, i₃ ∈ {0,…,N}

    where N = N_modes + 1 and iₓ = kₓ + offset centres the range at 0.
    """
    N = N_modes + 1
    offset = N // 2       # shift so that k = −N//2 ↦ 0

    i1 = k1 + offset
    i2 = k2 + offset
    i3 = k3 + offset

    return i1 * (N**2) + i2 * N + i3


def norm_k(k1, k2, k3):
    """
    Euclidean norm |k| in lattice units.
    """
    return (k1**2 + k2**2 + k3**2) ** 0.5
