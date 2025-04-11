# calulations/simulate.py
import pandas as pd
from .solver import compute_overlap
import netket as nk

def run_single(model_class, model_kwargs, train_params, hi, H, seed=None):
    # --- VMC-Setup ---

    lr, n_samples, n_iter = (train_params[k] for k in ("lr", "n_samples", "n_iter"))
    
    model = model_class(**model_kwargs)
    sampler = nk.sampler.ExactSampler(hi) #Erzeugt alle zulässigen Konfigurationen
    #sampler = nk.sampler.MetropolisExchangeSampler(hi, n_chains=n_samples) #Erzeugt Zufalls-Konfigurationen (MCMC)
    vstate = nk.vqs.MCState(sampler, model, n_samples=n_samples,seed=seed)
    
    
    optimizer = nk.optimizer.Sgd(learning_rate=lr)
    preconditioner = nk.optimizer.SR(diag_shift=0.1)
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
        error = abs((energy.mean - e_0) / e_0)
        overlap = compute_overlap(v_0, vstate)

        results.append({
            "name": model_info["name"],
            "vstate": vstate,
            "log": log,
            "energy": energy.mean.real,
            "error": error,
            "overlap": overlap,
        })
        
        energy_round=f"{energy.mean.real:.4f}"
        error_round=f"{error * 100:.4f}"
        overlap_round=f"{overlap * 100:.4f}"
        tabelle.append({
            "name": model_info["name"],
            "energy": energy_round,
            "error (%)": error_round,
            "overlap (%)": overlap_round,
        })
        
    tabelle = pd.DataFrame(tabelle)
    tabelle.to_csv("/Users/julianblasek/master_local/praktikum/plots/results.csv", index=False)
    return results