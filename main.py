# main.py
import matplotlib.pyplot as plt
import numpy as np
import netket as nk
import warnings
from netket.errors import HolomorphicUndeclaredWarning
warnings.filterwarnings("ignore", category=HolomorphicUndeclaredWarning)

# Import Modules
from config import N_max, N_modes, omega, alpha_coupling, alpha, beta, n_samples, lr, n_iter, sr, m_b
from models.ffn import FFN, DeepFFN
from hamiltonians.fröhlich import build_hamilton_1d, build_hamilton_3d, build_hamilton_dynamic_1d,build_hamilton_dynamic_3d
from calculations.solver import exact_dense, aprox_sol_sparse, compute_overlap
from plotting.visual import energy_convergence, state_chart
from calculations.simulate import run_all

def evaluation(energy_list, error_list, e_0):
    energies = np.array(energy_list)
    errors = np.array(error_list)
    
    M = len(energies)
    e_mean = np.mean(energies)

    # Fehler aus Run-to-Run-Streuung (Standardabweichung der Mittelwerte)
    run_to_run_var = np.sum((energies - e_mean) ** 2) / (M * (M - 1))

    mean_sampling_err = np.sum(errors**2) / M**2

    # Kombinierter Fehler
    total_error = np.sqrt(run_to_run_var + mean_sampling_err)

    # Relative Abweichung zum analytischen Ergebnis
    perc_dev = 100 * abs(e_mean - e_0) / abs(e_0)

    print(f"Mean energy: {e_mean:.6f} ± {total_error:.6f}")
    print(f"Analytic solution: {e_0:.6f}")
    print(f"Mean energy deviation: {perc_dev:.3f}%")

    return e_mean, total_error


def main1d(i):
    hi = nk.hilbert.Fock(n_max=N_max, N=N_modes) # 1D
    
    H, e_0 = build_hamilton_1d(hi, N_modes, N_max, omega, alpha_coupling)
    #H,e_0 = build_hamilton_dynamic_1d(hi, N_modes, N_max, omega, alpha_coupling,P=0.1)
    #e_0, v_0 = exact_dense(H)
    #e_0, v_0 = aprox_sol_sparse(H)
    v_0 = None
    print("Energy to approximate: ", round(e_0,4))
    dev = 100 * (abs(e_0 + np.pi**2 / 3) / (np.pi**2 / 3))
    print("Deviation from analytic solution:", round(dev, 2), "%")
    
    models_to_run = [
        {
            "name": "FFN_1D",
            "class": FFN,
            "net_params": {"alpha": alpha, "beta": beta},
            "train_params": {"lr": lr, "n_samples": n_samples, "n_iter": n_iter, "sr": sr},
        },

    ]

    results = run_all(models_to_run,hi,H,e_0, v_0)
    nqs_deviation = 100 * (abs(e_0 - results[0]["energy"]) / (-e_0))
    print(f"NQS({i+1}) Deviation:", round(nqs_deviation, 2), "%")
    # --- Vergleichsplots ---
    energy_convergence(results, e_0,n_iter)
    #state_chart(results, v_0,hi)
    return results, e_0


def main3d(i):
    hi = nk.hilbert.Fock(n_max=N_max, N=(N_modes+1)**3) # 3D
    #H, e_0 = build_hamilton_3d(hi, N_modes, N_max, omega, alpha_coupling)
    H, e_0 = build_hamilton_dynamic_3d(hi, N_modes, N_max, omega, alpha_coupling,P=0.1,m_b=m_b)
    print("Hilbert dimension: ", hi.size)
    print("Energy to approximate: ", round(e_0,4))
    #e_0, v_0 = exact_dense(H)
    #e_0, v_0 = aprox_sol_sparse(H)
    v_0 = None
    
    
    models_to_run = [
        {
            "name": "FFN_3D",
            "class": FFN,
            "net_params": {"alpha": alpha, "beta": beta},
            "train_params": {"lr": lr, "n_samples": n_samples, "n_iter": n_iter, "sr": sr},
        },

    ]

    results = run_all(models_to_run,hi,H,e_0, v_0)
    nqs_deviation = 100 * (abs(e_0 - results[0]["energy"]) / (-e_0))
    print(f"NQS({i+1}) Deviation:", round(nqs_deviation, 2), "%")
    # --- Vergleichsplots ---
    energy_convergence(results, e_0,n_iter)
    #state_chart(results, v_0,hi)
    return results, e_0




if __name__ == "__main__":
    energy_list = []
    error_list = []
    for i in range(2):
        print(f"Run {i+1}")
        #results,e_0 = main3d(i)
        results,e_0 = main1d(i)
        energy_list.append(results[0]["energy"])
        error_list.append(results[0]["error"])
        
    e_mean, total_error=evaluation(energy_list, error_list,e_0)
