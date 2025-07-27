# hamiltonians/froehlich.py

from netket.operator.boson import create, destroy, number
from netket.operator import LocalOperator
import numpy as np


def build_hamilton_1d(hi, N_modes, N_max, omega, alpha_coupling):
    """
    Constructs the 1D Fröhlich polaron Hamiltonian using a discretized momentum grid.

    Args:
        hi: NetKet Hilbert space object
        N_modes: Number of phonon modes (size of k-space)
        N_max: Maximum phonon occupation (unused here but required for interface compatibility)
        omega: Phonon frequency (assumed constant)
        alpha_coupling: Coupling strength α (∝ 1 / |k|)

    Returns:
        H: NetKet LocalOperator representing the Hamiltonian
        e_0: Analytic ground-state energy from the infinite mass reference model
    """
    # Define discrete momenta: skip k=0 to avoid divergence
    k_vals = np.concatenate([
        np.arange(-N_modes // 2, 0, dtype=int),
        np.arange(1, N_modes // 2 + 1, dtype=int)
    ])

    def coupling(k):
        return -1j * alpha_coupling / k

    H = LocalOperator(hi, dtype=np.complex128)
    H += sum(omega * number(hi, i) for i in range(N_modes))
    H += sum(coupling(k) * (-create(hi, i) + destroy(hi, i)) for i, k in enumerate(k_vals))

    e_0 = -sum((alpha_coupling**2) / (omega * abs(k)**2) for k in k_vals)
    return H, e_0


def build_hamilton_3d(hi, N_modes, N_max, omega, alpha_coupling):
    """
    Constructs the 3D Fröhlich polaron Hamiltonian using a cubic momentum grid.

    Only modes with norm k satisfying k_min <= |k| <= k_max are included (IR/UV cutoff).

    Args:
        hi: NetKet Hilbert space object
        N_modes: Momentum cutoff per direction (defines the 3D grid)
        N_max: Max phonon occupation (unused here)
        omega: Phonon frequency (constant)
        alpha_coupling: Coupling constant (∝ 1 / |k|)

    Returns:
        H: Hamiltonian operator
        e_0: Analytic reference energy
    """
    k_min = 1
    k_max = 2

    k_range = np.arange(-N_modes // 2, N_modes // 2 + 1, dtype=int)
    H = LocalOperator(hi, dtype=np.complex128)
    e_0 = 0

    for i in k_range:
        for j in k_range:
            for k in k_range:
                norm = norm_k(i, j, k)

                if k_min <= norm <= k_max:
                    idx = map_k_to_index(i, j, k, N_modes)
                    H += omega * number(hi, idx)
                    H += (-alpha_coupling / norm) * (create(hi, idx) + destroy(hi, idx))
                    e_0 += -(alpha_coupling**2) / (omega * norm**2)

    return H, e_0


def norm_k(k1, k2, k3):
    """
    Computes the Euclidean norm of a 3D wavevector.

    Returns:
        |k| = sqrt(kx^2 + ky^2 + kz^2)
    """
    return np.sqrt(k1**2 + k2**2 + k3**2)


def map_k_to_index(k1, k2, k3, N_modes):
    """
    Maps a 3D momentum tuple (k1, k2, k3) to a 1D index suitable for NetKet operators.

    Args:
        k1, k2, k3: Momentum components in [-N//2, ..., N//2]
        N_modes: Number of phonon modes per direction

    Returns:
        Unique integer index ∈ [0, (N+1)^3 - 1]
    """
    N = N_modes + 1
    offset = N // 2

    i1 = k1 + offset
    i2 = k2 + offset
    i3 = k3 + offset

    return i1 * (N**2) + i2 * N + i3
