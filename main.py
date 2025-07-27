"""
main.py ― Driver for Fröhlich-polaron VMC
====================================================

This script is the **entry point** for all numerical simulations performed in the
project.  It sets up the desired Hilbert space, constructs the corresponding
Fröhlich(-type) Hamiltonian (static or dynamical, 1D or 3D), selects a neural-
quantum-state (NQS) ansatz, and launches the NetKet variational Monte-Carlo
(VMC) optimisation.  After several independent runs it aggregates statistics
and prints a concise benchmark summary.

Layout
------
1.  Imports
    1.1  Scientific Python stack
    1.2  Third-party libraries
    1.3  Project-local modules
2.  Helper utilities
3.  Workflows for specific geometries   (``main1d()``, ``main3d()``)
4.  Script entry point                 (``if __name__ == "__main__":``)

"""

# ──────────────────────────────────────────────────────────────
# 1. IMPORTS
# ──────────────────────────────────────────────────────────────

# 1.1 Scientific Python stack
import matplotlib.pyplot as plt        # plotting backend used by helpers
import numpy as np                     # numerical routines

# 1.2 Third-party libraries
import netket as nk                    # variational-quantum-Monte-Carlo engine
import warnings
from netket.errors import HolomorphicUndeclaredWarning
warnings.filterwarnings("ignore", category=HolomorphicUndeclaredWarning)  # silence benign NetKet warning

# 1.3 Project-local modules & global parameters
from config import (
    N_max, N_modes, omega, alpha_coupling,    # model parameters
    alpha, beta, n_samples, lr, n_iter, sr, m_b   # network & training hyper-params
)
from models.ffn import FFN, DeepFFN                          # feed-forward NQS variants
from hamiltonians.fröhlich import (
    build_hamilton_1d, build_hamilton_3d,                    # static models
    build_hamilton_dynamic_1d, build_hamilton_dynamic_3d     # dynamical (finite-momentum) models
)
from calculations.solver import exact_dense, aprox_sol_sparse, compute_overlap
from plotting.visual import energy_convergence, state_chart
from calculations.simulate import run_all


# ──────────────────────────────────────────────────────────────
# 2. HELPER UTILITIES
# ──────────────────────────────────────────────────────────────
def evaluation(energy_list, error_list, e_0):
    """
    Aggregate results from several independent VMC runs.

    Parameters
    ----------
    energy_list : list[float]
        Mean energies obtained in each run.
    error_list : list[float]
        Corresponding one-sigma statistical uncertainties reported by NetKet.
    e_0 : float
        Reference (analytic or exact-diagonalisation) ground-state energy.

    Returns
    -------
    tuple (e_mean, total_error)
        • ``e_mean``: run-averaged energy  
        • ``total_error``: combined uncertainty (run-to-run variance ⊕ sampling error)
    """
    energies = np.array(energy_list)
    errors = np.array(error_list)

    M = len(energies)
    e_mean = np.mean(energies)

    # Run-to-run fluctuation (standard error of the mean)
    run_to_run_var = np.sum((energies - e_mean) ** 2) / (M * (M - 1))

    # Average squared single-run statistical error
    mean_sampling_err = np.sum(errors ** 2) / M ** 2

    total_error = np.sqrt(run_to_run_var + mean_sampling_err)

    # Percentage deviation from reference value
    perc_dev = 100 * abs(e_mean - e_0) / abs(e_0)

    print(f"Mean energy:          {e_mean:.6f} ± {total_error:.6f}")
    print(f"Analytic solution:    {e_0:.6f}")
    print(f"Mean energy deviation:{perc_dev:.3f}%\n")

    return e_mean, total_error


# ──────────────────────────────────────────────────────────────
# 3. WORKFLOWS
# ──────────────────────────────────────────────────────────────
def main1d(i: int):
    """
    Single VMC run for the **1-D static Fröhlich model**.

    Parameters
    ----------
    i : int
        Index of the current repetition (for logging only).

    Returns
    -------
    results : list[dict]
        Output of :pyfunc:`calculations.simulate.run_all` for the selected NQS.
    e_0 : float
        Reference ground-state energy used for benchmarking.
    """
    # 3.1 Build Hilbert space and Hamiltonian
    hi = nk.hilbert.Fock(n_max=N_max, N=N_modes)    # local phonon cutoff

    H, e_0 = build_hamilton_1d(hi, N_modes, N_max, omega, alpha_coupling)
    
    # For dynamical Fröhlich (finite impurity momentum) replace with:
    # H, e_0 = build_hamilton_dynamic_1d(hi, N_modes, N_max, omega, alpha_coupling, P=0.1)

    v_0 = None  # eigenvector not required for this benchmark

    print("Energy to approximate:           ", round(e_0, 4))
    dev = 100 * (abs(e_0 + np.pi ** 2 / 3) / (np.pi ** 2 / 3))
    print("Deviation from analytic solution:", round(dev, 2), "%")

    # 3.2 Register NQS models to run
    models_to_run = [
        {
            "name": "FFN_1D",
            "class": FFN,
            "net_params": {"alpha": alpha, "beta": beta},
            "train_params": {
                "lr": lr,
                "n_samples": n_samples,
                "n_iter": n_iter,
                "sr": sr,
            },
        },
    ]

    # 3.3 Launch optimisation
    results = run_all(models_to_run, hi, H, e_0, v_0)

    # 3.4 Quick performance metric
    nqs_deviation = 100 * (abs(e_0 - results[0]["energy"]) / (-e_0))
    print(f"NQS({i+1}) Deviation:            ", round(nqs_deviation, 2), "%")

    # 3.5 Diagnostic plots
    energy_convergence(results, e_0, n_iter)
    # state_chart(results, v_0, hi)   # wave-function comparison (optional)

    return results, e_0


def main3d(i: int):
    """
    Single VMC run for the **3-D dynamic Fröhlich model** (default).

    Parameters
    ----------
    i : int
        Index of the current repetition (for logging only).

    Returns
    -------
    results : list[dict]
        Output of :pyfunc:`calculations.simulate.run_all` for the selected NQS.
    e_0 : float
        Reference ground-state energy used for benchmarking.
    """
    # 3.1 Build Hilbert space and Hamiltonian
    hi = nk.hilbert.Fock(n_max=N_max, N=(N_modes + 1) ** 3)         # 3-D grid (cube incl. k=0)
    
    # Static (P=0) variant:
    H, e_0 = build_hamilton_3d(hi, N_modes, N_max, omega, alpha_coupling)
    # Dynamical variant:
    #H, e_0 = build_hamilton_dynamic_3d(hi, N_modes, N_max, omega, alpha_coupling, P=0.1, m_b=m_b)

    print("Hilbert dimension:              ", hi.size)
    print("Energy to approximate:           ", round(e_0, 4))
    v_0 = None  # eigenvector not required for this benchmark

    # 3.2 Register NQS models to run
    models_to_run = [
        {
            "name": "FFN_3D",
            "class": FFN,
            "net_params": {"alpha": alpha, "beta": beta},
            "train_params": {
                "lr": lr,
                "n_samples": n_samples,
                "n_iter": n_iter,
                "sr": sr,
            },
        },
    ]

    # 3.3 Launch optimisation
    results = run_all(models_to_run, hi, H, e_0, v_0)

    # 3.4 Quick performance metric
    nqs_deviation = 100 * (abs(e_0 - results[0]["energy"]) / (-e_0))
    print(f"NQS({i+1}) Deviation:            ", round(nqs_deviation, 2), "%")

    # 3.5 Diagnostic plots
    energy_convergence(results, e_0, n_iter)
    # state_chart(results, v_0, hi)   # wave-function comparison (optional)

    return results, e_0


# ──────────────────────────────────────────────────────────────
# 4. SCRIPT ENTRY POINT
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    """
    Perform two independent VMC runs and print aggregated statistics.

    Switch between 1-D and 3-D experiments by commenting/uncommenting the
    respective ``mainXd`` call below.
    """
    energy_list = []
    error_list = []

    for i in range(2):  # number of independent repetitions
        print(f"\nRun {i + 1}")
        # 1-D benchmark:
        results, e_0 = main1d(i)
        # 3-D benchmark:
        # results, e_0 = main3d(i)

        energy_list.append(results[0]["energy"])
        error_list.append(results[0]["error"])

    # Final aggregated report
    e_mean, total_error = evaluation(energy_list, error_list, e_0)
