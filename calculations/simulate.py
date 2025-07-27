# calulations/simulate.py
import os
import numpy as np
import pandas as pd
from .solver import compute_overlap
import netket as nk


log_path = "/Users/julianblasek/local/praktikum/plots/results.tsv"

def run_single(model_class, model_kwargs, train_params, hi, H, seed=None):
    # --- VMC-Setup ---

    lr, n_samples, n_iter, sr = (train_params[k] for k in ("lr", "n_samples", "n_iter","sr"))
    
    model = model_class(**model_kwargs)
    #sampler = nk.sampler.ExactSampler(hi) #Erzeugt alle zulässigen Konfigurationen
    sampler = nk.sampler.Metropolis(hi, n_chains=n_samples, rule=nk.sampler.rules.LocalRule()) #Erzeugt Zufalls-Konfigurationen (MCMC)
    vstate = nk.vqs.MCState(sampler, model, n_samples=n_samples,seed=seed)
    
    
    optimizer = nk.optimizer.Sgd(learning_rate=lr)
    preconditioner = nk.optimizer.SR(diag_shift=sr)
    gs = nk.driver.VMC(H, optimizer, variational_state=vstate, preconditioner=preconditioner)
    log = nk.logging.RuntimeLog()
    
    
    gs.run(n_iter=n_iter, out=log)
    return vstate, log

def run_all(models_to_run,hi,H,e_0, v_0):
    results = []
    tabelle=[]
    for model_info in models_to_run:
        print(f"\n--- Running model: {model_info['name']} ---")
        vstate, log = run_single(model_info["class"], model_info["net_params"],model_info["train_params"], hi, H)

        
        energy = vstate.expect(H)

        
        if v_0 is None:
            overlap = None
        else:
            overlap = compute_overlap(v_0, vstate)

        results.append({
            "name": model_info["name"],
            "vstate": vstate,
            "log": log,
            "energy": energy.mean.real,
            "error": energy.error_of_mean.real,        
        })
        energy_round_model=f"{energy.mean.real:.3f}"
        energy_round_exact=f"{e_0:.3f}"
        error_round = f"{np.sqrt(energy.error_of_mean.real):.3f}" 

        if overlap is not None:
            overlap_round=f"{overlap:.2f}"
        else:
            overlap_round="  --- "
        
    tabelle.append({
                "name": model_info["name"],
                "E_model": energy_round_model,
                "E_exact": energy_round_exact,
                "Error": error_round,
                "Overlap (%)": overlap_round,
            })    
        
        
        
    tabelle = pd.DataFrame(tabelle)
    
    if os.path.exists(log_path):
        existing_log = pd.read_csv(log_path, sep="\t")
        # Neue Zeilen anhängen
        full_log = pd.concat([existing_log, tabelle], ignore_index=True)
    else:
        full_log = tabelle

    # Datei neu schreiben mit allen Zeilen
    full_log.to_csv(log_path, sep='\t', index=False)
    return results