# hamiltonians/fröhlich.py
from netket.operator.boson import create, destroy, number
from netket.operator import LocalOperator
import numpy as np








# Alternative Hamiltonian with coupling depending on k
def build_hamilton_1d(hi,N_modes, N_max, omega, alpha_coupling):
    """
    Build the Hamiltonian for the Fröhlich model.
    """

    k_vals = np.concatenate([np.arange(-N_modes/2, 0, dtype=int),np.arange(1, N_modes/2 + 1, dtype=int)])
    #print("k_vals:", k_vals)
    def coupling(k):
            z=0-1j
            return alpha_coupling*z / k 
        
    # Hamiltonian
    H = LocalOperator(hi,dtype=np.complex128)
    H += sum(omega * number(hi, i) for i in range(N_modes))
    H += sum(coupling(k) * (-create(hi, i) + destroy(hi, i)) for i, k in enumerate(k_vals))

    e_0 = -1* sum(alpha_coupling**2 /(omega* abs(k)**2) for k in k_vals)
    
    #print("E_0:", e_0)
    #print("\nE_0_Model/E_exact",e_0/(-np.pi**2/3))
    return H, e_0



def build_hamilton_3d(hi,N_modes, N_max, omega, alpha_coupling):
    """
    Build the Hamiltonian for the Fröhlich model.
    """
    k_min = 1 #IR cutoff
    k_max = 2 #UV cutoff
    k1 = np.arange(-N_modes // 2, N_modes // 2 + 1, dtype=int)
    k2=k1
    k3=k1
    print("k1:", k1)
    
    
    def coupling(k1,k2,k3):
            return -1*alpha_coupling / norm_k(k1,k2,k3) 
        
    H=0
    e_0=0
    for i in k1:
        for j in k2:
            for k in k3:
                current_norm = norm_k(i, j, k)
                # Mode überspringen, falls die Norm außerhalb des erlaubten Intervalls liegt.
                if current_norm < k_min or current_norm > k_max:
                    continue

                # Ermittle den eindeutigen Index mittels unseres einfachen Mappings.
                ijk = map_k_to_index(i, j, k, N_modes)
                # Füge den Zahloperator-Term hinzu: omega * number(hi, idx)
                H += omega * number(hi, ijk)
                H += coupling(i,j,k) * (create(hi, ijk) + destroy(hi, ijk))
                e_0 += -1*alpha_coupling**2 /(omega* current_norm**2)
    
    #print("e_0:", e_0)
    return H, e_0

def map_k_to_index(k1, k2, k3, N_modes):
    """
    Mappt (k1, k2, k3) ∈ [-N//2, ..., N//2]^3 auf Index [0, (N+1)^3 - 1]
    """
    N = N_modes + 1  # Gesamtanzahl an diskreten k-Werten pro Richtung
    offset = N // 2  # Damit -N//2 ↔ 0, ..., 0 ↔ N//2

    # Shifted Indices: -N//2 → 0, ..., 0 → N//2
    i1 = k1 + offset
    i2 = k2 + offset
    i3 = k3 + offset

    return i1 * (N**2) + i2 * N + i3

def norm_k(k1,k2,k3):
    """
    Norm of k in 3D.
    """
    return (k1**2 + k2**2 + k3**2)**0.5
