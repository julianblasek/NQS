"""
simulate.py ― Thin wrapper that launches VMC optimisation *and*
                aggregates/export results
===============================================================

This module keeps **all run-time logistics** in one place:

* **`run_single`** – create a NetKet MCState, optimise it via VMC,  
  and return the converged variational state *plus* the in-memory log.
* **`run_all`**   – convenience loop over several ansätze defined
  by `models_to_run`; computes final observables, appends them to a
  human-readable TSV file, and returns a list of dictionaries
  compatible with the plotting utilities.


Layout
------
1.  Imports & global settings
2.  Low-level VMC helper        (``run_single``)
3.  Multi-model orchestration   (``run_all``)
"""

# ──────────────────────────────────────────────────────────────
# 1. IMPORTS & GLOBAL SETTINGS
# ──────────────────────────────────────────────────────────────
import os
import numpy as np
import pandas as pd
from .solver import compute_overlap            # exact-state overlap helper
import netket as nk

# Absolute path of the cumulative results table (TSV, one line per run)
log_path = "/Users/julianblasek/local/praktikum/plots/results.tsv"


# ──────────────────────────────────────────────────────────────
# 2. LOW-LEVEL VMC HELPER
# ──────────────────────────────────────────────────────────────
def run_single(model_class, model_kwargs, train_params, hi, H, seed=None):
    """
    Optimise one *specific* NQS ansatz with NetKet VMC.

    Parameters
    ----------
    model_class : type
        Python class (sub-class of `nk.models.AbstractModule`) to instantiate.
    model_kwargs : dict
        Keyword arguments forwarded to the model constructor.
    train_params : dict
        ``{"lr", "n_samples", "n_iter", "sr"}`` – learning rate, number of
        Monte-Carlo samples, optimisation iterations, and SR shift.
    hi : nk.hilbert.AbstractHilbert
        Hilbert space on which the Hamiltonian `H` acts.
    H : nk.operator.LocalOperator
        Many-body Hamiltonian.
    seed : int | None
        Optional RNG seed for reproducibility.

    Returns
    -------
    vstate : nk.vqs.MCState
        Converged variational state.
    log : nk.logging.RuntimeLog
        NetKet run-time log containing per-iteration data.
    """
    # ── 2.1 Unpack optimisation hyper-parameters ────────────────────
    lr, n_samples, n_iter, sr = (
        train_params[k] for k in ("lr", "n_samples", "n_iter", "sr")
    )

    # ── 2.2 Build model, sampler, and MCState ───────────────────────
    model = model_class(**model_kwargs)

    # ExactSampler would enumerate *all* Fock states (expensive);
    # Metropolis with local moves is far cheaper in practice.
    # sampler = nk.sampler.ExactSampler(hi)
    sampler = nk.sampler.Metropolis(
        hi,
        n_chains=n_samples,
        rule=nk.sampler.rules.LocalRule(),
    )
    vstate = nk.vqs.MCState(sampler, model, n_samples=n_samples, seed=seed)

    # ── 2.3 Optimiser & SR preconditioner ───────────────────────────
    optimizer = nk.optimizer.Sgd(learning_rate=lr)
    preconditioner = nk.optimizer.SR(diag_shift=sr)

    # ── 2.4 Drive optimisation and capture log ─────────────────────
    gs = nk.driver.VMC(
        H,
        optimizer,
        variational_state=vstate,
        preconditioner=preconditioner,
    )
    log = nk.logging.RuntimeLog()
    gs.run(n_iter=n_iter, out=log)

    return vstate, log


# ──────────────────────────────────────────────────────────────
# 3. MULTI-MODEL ORCHESTRATION
# ──────────────────────────────────────────────────────────────
def run_all(models_to_run, hi, H, e_0, v_0):
    """
    Loop over a *list* of model specifications, optimise each, compute
    diagnostics, and append results to a persistent TSV file.

    Parameters
    ----------
    models_to_run : list[dict]
        Each entry must provide keys ``{"name","class","net_params",
        "train_params"}`` (see `main.py` for an example).
    hi, H : NetKet hilbert/operator
        Same objects forwarded to `run_single`.
    e_0 : float
        Reference ground-state energy for error assessment.
    v_0 : ndarray | None
        Exact eigenvector; required only if overlap is desired.

    Returns
    -------
    results : list[dict]
        Summary (state, log, mean energy, error, …) for every model.
    """
    results = []
    tabelle = []     # one row per model → later written to TSV

    for model_info in models_to_run:
        print(f"\n--- Running model: {model_info['name']} ---")

        # ── 3.1 Optimise the current model ──────────────────────────
        vstate, log = run_single(
            model_info["class"],
            model_info["net_params"],
            model_info["train_params"],
            hi,
            H,
        )

        # ── 3.2 Final observables ──────────────────────────────────
        energy = vstate.expect(H)

        # Overlap w/ exact state if provided
        overlap = None if v_0 is None else compute_overlap(v_0, vstate)

        # ── 3.3 Collect in-memory results ───────────────────────────
        results.append(
            {
                "name": model_info["name"],
                "vstate": vstate,
                "log": log,
                "energy": energy.mean.real,
                "error": energy.error_of_mean.real,
            }
        )

        # ── 3.4 Prepare human-readable table row ────────────────────
        energy_round_model = f"{energy.mean.real:.3f}"
        energy_round_exact = f"{e_0:.3f}"
        error_round = f"{np.sqrt(energy.error_of_mean.real):.3f}"
        overlap_round = f"{overlap:.2f}" if overlap is not None else "  --- "

        tabelle.append(
            {
                "name": model_info["name"],
                "E_model": energy_round_model,
                "E_exact": energy_round_exact,
                "Error": error_round,
                "Overlap (%)": overlap_round,
            }
        )

    # ── 3.5 Persist table (append-or-create) ────────────────────────
    tabelle = pd.DataFrame(tabelle)

    if os.path.exists(log_path):
        existing_log = pd.read_csv(log_path, sep="\t")
        full_log = pd.concat([existing_log, tabelle], ignore_index=True)
    else:
        full_log = tabelle

    full_log.to_csv(log_path, sep="\t", index=False)

    return results
