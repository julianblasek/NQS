# calculations/simulate.py

import os
import numpy as np
import pandas as pd
import netket as nk
from .solver import compute_overlap

# Path to save tabulated results
log_path = "/Users/julianblasek/master_local/praktikum/plots/results.tsv"


def run_single(model_class, model_kwargs, train_params, hi, H, seed=None):
    """
    Executes a single VMC run for a given model class and training configuration.

    Args:
        model_class: Flax module class defining the network (e.g., FFN)
        model_kwargs: Dictionary with network hyperparameters (e.g., alpha, beta)
        train_params: Dictionary with VMC parameters (lr, n_samples, n_iter, sr)
        hi: NetKet Hilbert space
        H: Hamiltonian operator
        seed: Optional random seed

    Returns:
        vstate: Final variational state after training
        log: NetKet RuntimeLog object containing training history
    """
    lr, n_samples, n_iter, sr = (
        train_params[k] for k in ("lr", "n_samples", "n_iter", "sr")
    )

    # Initialize model and VMC state
    model = model_class(**model_kwargs)
    sampler = nk.sampler.Metropolis(
        hi, n_chains=n_samples, rule=nk.sampler.rules.LocalRule()
    )
    vstate = nk.vqs.MCState(sampler, model, n_samples=n_samples, seed=seed)

    # Optimizer and stochastic reconfiguration (SR)
    optimizer = nk.optimizer.Sgd(learning_rate=lr)
    preconditioner = nk.optimizer.SR(diag_shift=sr)
    gs = nk.driver.VMC(H, optimizer, variational_state=vstate, preconditioner=preconditioner)

    log = nk.logging.RuntimeLog()
    gs.run(n_iter=n_iter, out=log)

    return vstate, log


def run_all(models_to_run, hi, H, e_0, v_0):
    """
    Runs multiple VMC models and logs their energies and overlaps.

    Args:
        models_to_run: List of dictionaries containing model setup
        hi: NetKet Hilbert space
        H: Hamiltonian
        e_0: Analytic ground-state energy for comparison
        v_0: Exact ground-state wavefunction (optional)

    Returns:
        results: List of dictionaries containing metrics and trained states
    """
    results = []
    rows = []

    for model_info in models_to_run:
        print(f"\n--- Running model: {model_info['name']} ---")

        vstate, log = run_single(
            model_info["class"],
            model_info["net_params"],
            model_info["train_params"],
            hi,
            H,
        )

        energy = vstate.expect(H)
        energy_mean = energy.mean.real
        energy_err = np.sqrt(energy.error_of_mean.real)

        if v_0 is not None:
            overlap = compute_overlap(v_0, vstate)
            overlap_display = f"{overlap:.2f}"
        else:
            overlap_display = "---"

        results.append({
            "name": model_info["name"],
            "vstate": vstate,
            "log": log,
            "energy": energy_mean,
            "error": energy_err,
        })

        rows.append({
            "name": model_info["name"],
            "E_model": f"{energy_mean:.3f}",
            "E_exact": f"{e_0:.3f}",
            "Error": f"{energy_err:.3f}",
            "Overlap (%)": overlap_display,
        })

    # Tabulate results and optionally append to disk
    table = pd.DataFrame(rows)

    if os.path.exists(log_path):
        existing_log = pd.read_csv(log_path, sep="\t")
        full_log = pd.concat([existing_log, table], ignore_index=True)
    else:
        full_log = table

    full_log.to_csv(log_path, sep="\t", index=False)

    return results
