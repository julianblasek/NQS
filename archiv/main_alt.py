import numpy as np
import matplotlib.pyplot as plt
import netket as nk
import warnings
from netket.errors import HolomorphicUndeclaredWarning
warnings.filterwarnings("ignore", category=HolomorphicUndeclaredWarning)

# Import Modules
from config import N_max, N_modes, omega, alpha_coupling, alpha, beta, n_samples, lr, n_iter
from models.ffn import FFN
from hamiltonians.fröhlich import build_hamilton
from calculations.solver import exact_dense, aprox_sol_sparse, compute_overlap
from plotting.visual import energy_convergence_plot, state_chart_plot

def main():
    # --- Hilbertraum ---
    hi = nk.hilbert.Fock(n_max=N_max, N=N_modes)

    # --- Hamiltonian aufbauen ---
    H = build_hamilton(hi,N_modes, N_max, omega, alpha_coupling)

    # --- Ground State Calculation ---
    #e_0, v_0=exact_dense(H)
    e_0, v_0=aprox_sol_sparse(H)


    # --- NQS mit FFN ---
    model = FFN(alpha=alpha, beta=beta)
    sampler = nk.sampler.ExactSampler(hi) #Erzeugt alle zulässigen Konfigurationen
    #sampler = nk.sampler.MetropolisExchangeSampler(hi, n_chains=n_samples) #Erzeugt Zufalls-Konfigurationen (MCMC)
    vstate = nk.vqs.MCState(sampler, model, n_samples=n_samples)

    # --- VMC-Setup ---
    optimizer = nk.optimizer.Sgd(learning_rate=lr)
    preconditioner = nk.optimizer.SR(diag_shift=0.1)
    gs = nk.driver.VMC(H, optimizer, variational_state=vstate, preconditioner=preconditioner)
    log = nk.logging.RuntimeLog()
    
    # Training
    gs.run(n_iter=n_iter, out=log)

    # --- Ergebnisse auswerten ---
    ffn_energy = vstate.expect(H)
    error = abs((ffn_energy.mean - e_0) / e_0)
    print(f"Optimized energy: {ffn_energy} \nRelative error: {error*100:.2f}%")

    # --- Plotting ---
    data = log.data  
    energy_convergence_plot(data, e_0)
    state_chart_plot(hi, vstate, v_0)
    compute_overlap(vstate, v_0)

if __name__ == "__main__":
    main()