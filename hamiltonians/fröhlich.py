# hamiltonians/fröhlich.py
from netket.operator.boson import create, destroy, number
import numpy as np

k_max = 10  # Maximaler Wert von k (je nach UV-Cutoff)
k_vals = np.linspace(0.01, k_max, 100)

def build_hamilton(hi,N_modes, N_max, omega, alpha_coupling):
    """
    Build the Hamiltonian for the Fröhlich model.
    """

    # Hamiltonian
    H = sum(omega * number(hi, i) for i in range(N_modes))
    H += sum(alpha_coupling * (create(hi, i) + destroy(hi, i)) for i in range(N_modes))

    return H



# Alternative Hamiltonian with coupling depending on k
def build_hamilton2(hi,N_modes, N_max, omega, alpha_coupling):
    """
    Build the Hamiltonian for the Fröhlich model.
    """
    def coupling(k):
            return alpha_coupling / k 
        
    # Hamiltonian
    H = sum(omega * number(hi, i) for i in range(N_modes))
    H += sum(coupling(k) * (create(hi, i) + destroy(hi, i)) for i, k in enumerate(k_vals))

    return H