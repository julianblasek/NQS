import numpy as np
import netket as nk
import jax.numpy as jnp
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
from netket.operator.boson import create, destroy, number
import jax

# --- Parameter des Modells ---
N_modes = 3           # Anzahl der Phonon-Moden
omega = 1.0           # Phononenfrequenz
alpha_coupling = 1.0  # Kopplungsstärke V_q

# --- Hilbert-Raum der Phononen ---
hi = nk.hilbert.Fock(n_max=2, N=N_modes)

# --- Hamiltonian aufstellen ---
H = sum(omega * number(hi, i) for i in range(N_modes))
H += sum(alpha_coupling * (create(hi, i) + destroy(hi, i)) for i in range(N_modes))

# --- Diagonalisierung ---
sp_h = H.to_sparse()
eig_vals, eig_vecs = eigsh(sp_h, k=2, which="SA")
e_0 = eig_vals[0]
print("Exact ground state energy:", e_0)

# --- Gemeinsame Parameter für VMC ---
n_samples = 512
n_iter = 500
learning_rate = 0.1

# --- Sampler ---
sampler = nk.sampler.MetropolisLocal(hi)

# --- Vorgefertigtes MLP-Modell ---
model = nk.models.MLP(
    output_activation=None,    # Keine Aktivierung an der Ausgangsschicht (linear)
)

# --- VMC Setup ---
vstate = nk.vqs.MCState(sampler=sampler, model=model, n_samples=n_samples, seed=4321)

optimizer_fnn = nk.optimizer.Adam(learning_rate=learning_rate)
gs_fnn = nk.driver.VMC(
    H,
    optimizer_fnn,
    variational_state=vstate,
    preconditioner=nk.optimizer.SR(diag_shift=0.1),
)

log_fnn = nk.logging.RuntimeLog()
gs_fnn.run(n_iter=n_iter, out=log_fnn)
energy_fnn = vstate.expect(H)
rel_error_fnn = abs((energy_fnn.mean - e_0) / e_0)
print("FNN: Optimized energy =", energy_fnn.mean, "Relative error =", rel_error_fnn)

# --- Plotten der Konvergenz ---
iters_fnn = log_fnn.data["Energy"].iters
mean_energy_fnn = log_fnn.data["Energy"].Mean
sigma_energy_fnn = log_fnn.data["Energy"].Sigma

plt.figure(figsize=(8, 6))
plt.errorbar(iters_fnn, mean_energy_fnn, yerr=sigma_energy_fnn, label="FNN", fmt='s-', capsize=3)
plt.hlines(energy_fnn.mean, xmin=0, xmax=n_iter, color="gray", linestyle="--", label="FNN (2 Schichten) Endwert")
plt.hlines(e_0, xmin=0, xmax=n_iter, color="red", linestyle=":", label="Exakte Diagonalisierung")

plt.xlabel("Iterationen")
plt.ylabel("Ground-State Energie")
plt.title("Konvergenz der Ground-State Energie (FNN vs. exakt)")
plt.legend()
plt.grid()
plt.savefig("/Users/julianblasek/master_local/praktikum/plots/energy_convergence_fnn_comparison.pdf")
plt.show()
