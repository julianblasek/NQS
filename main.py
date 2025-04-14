# main.py
import matplotlib.pyplot as plt
import numpy as np
import netket as nk
import warnings
from netket.errors import HolomorphicUndeclaredWarning
warnings.filterwarnings("ignore", category=HolomorphicUndeclaredWarning)

# Import Modules
from config import N_max, N_modes, omega, alpha_coupling, alpha, beta, n_samples, lr, n_iter
from models.ffn import FFN, DeepFFN
from hamiltonians.fröhlich import build_hamilton
from calculations.solver import exact_dense, aprox_sol_sparse, compute_overlap
from plotting.visual import energy_convergence, state_chart
from calculations.simulate import run_all



    
def main():
    hi = nk.hilbert.Fock(n_max=N_max, N=N_modes)
    H = build_hamilton(hi, N_modes, N_max, omega, alpha_coupling)
    
    #e_0, v_0 = exact_dense(H)
    e_0, v_0 = aprox_sol_sparse(H)
    #e_0, v_0 = None, None
    
    
    models_to_run = [
        {
            "name": "FFN_Adam",
            "class": FFN,
            "net_params": {"alpha": alpha+1, "beta": beta+1},
            "train_params": {"lr": lr, "n_samples": n_samples, "n_iter": n_iter},
        },
    ]

    results = run_all(models_to_run,hi,H,e_0, v_0)
    

    # --- Vergleichsplots ---
    energy_convergence(results, e_0,n_iter)
    #state_chart(results, v_0,hi)
    return results

if __name__ == "__main__":
    results=main()
    

