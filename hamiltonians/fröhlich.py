# hamiltonians/fröhlich.py
from netket.operator.boson import create, destroy, number
from netket.operator import LocalOperator
import numpy as np


#---------------------------------------------- 1D ----------------------------------------------

# Dynamic Hamiltonian 1D
def build_hamilton_dynamic_1d(hi,N_modes, N_max, omega, alpha_coupling,P):
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
    kin=LocalOperator(hi,dtype=np.complex128)
    kin+= sum(P - k* number(hi, i) for i, k in enumerate(k_vals))
    H += sum(omega * number(hi, i) for i in range(N_modes))
    H += sum(coupling(k) * (-create(hi, i) + destroy(hi, i)) for i, k in enumerate(k_vals))


    H= kin @ kin + H
    e_0 = -alpha_coupling+P**2/(1+alpha_coupling/6)
    
    #print("E_0:", e_0)
    #print("\nE_0_Model/E_exact",e_0/(-np.pi**2/3))
    return H, e_0



# Static Hamiltonian 1D
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


#---------------------------------------------- 3D ----------------------------------------------
# Dynamic Hamiltonian 3D
def build_hamilton_dynamic_3d(hi,N_modes, N_max, omega, alpha_coupling,P,m_b):
    """
    Build the Hamiltonian for the Fröhlich model.
    """
    k_min = 1 #IR cutoff
    k_max = 2 #UV cutoff
    k1 = np.arange(-N_modes // 2, N_modes // 2 + 1, dtype=int)
    k2=k1
    k3=k1
    print("k1:", k1)
    
    
    alpha_coupling =alpha_coupling* np.sqrt(m_b/(2*omega))  # Anpassung fürkonstanten wahl
    
    def v_k(norm_k):
            z=0-1j
            return ((z/ norm_k) * omega*2*np.sqrt(alpha_coupling)*(1/(2*m_b*omega))**(0.25))
        
    H = LocalOperator(hi,dtype=np.complex128)
    kin=LocalOperator(hi,dtype=np.complex128)
    e_0=0
    eta=(alpha_coupling / 6)/(1+alpha_coupling / 6) 
    for i in k1:
        for j in k2:
            for k in k3:
                current_norm = norm_k(i, j, k)
                # Mode überspringen, falls die Norm außerhalb des erlaubten Intervalls liegt.
                if current_norm < k_min or current_norm > k_max:
                    continue
                v = v_k(current_norm)
                # Ermittle den eindeutigen Index mittels unseres einfachen Mappings.
                ijk = map_k_to_index(i, j, k, N_modes)
                # Füge den Zahloperator-Term hinzu: omega * number(hi, idx)
                kin+=2*P*(i+j+k)* number(hi, ijk)
                H += (omega+current_norm**2/(2*m_b)) * number(hi, ijk)
                H += v * (-create(hi, ijk) + destroy(hi, ijk))
                e_0 += v*np.conj(v) /(omega-P*(i+j+k)*(1-eta)+current_norm**2/(2*m_b))
    H= (P**2-kin)/(2*m_b) + H
    e_0=P**2/(2*m_b)*(1-eta**2)-e_0
    
    return H, e_0.real

# Static Hamiltonian 3D
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
