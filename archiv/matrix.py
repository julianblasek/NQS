import numpy as np
import matplotlib.pyplot as plt
import netket as nk
from netket.operator.boson import create, destroy, number
from scipy.sparse.linalg import eigsh
import flax.linen as nn
import jax.numpy as jnp
import warnings
from netket.errors import HolomorphicUndeclaredWarning
warnings.filterwarnings("ignore", category=HolomorphicUndeclaredWarning)

plt.ion()
#-----------------------------  Funktionen & Klassen  ---------------------------------------------------
def exact_sol_sparse(H):
    """
    Computes the exact solution of the Hamiltonian H.
    """
    sp_h = H.to_sparse()
    eig_vals, eig_vecs = eigsh(sp_h, k=3, which="SA")
    e_0 = eig_vals[0]
    print("Exact ground state energy:", e_0)
    return e_0, eig_vecs[:, 0]

def aprox_sol_sparse(H):
    """
    Computes the approximate solution of the Hamiltonian H.
    """
    eig_vals, eig_vecs = nk.exact.lanczos_ed(H, k=5, compute_eigenvectors=True, scipy_args={"tol": 1e-8})
    print(f"Eigenvalues with exact lanczos (sparse): {eig_vals[0]}")
    return eig_vals[0],eig_vecs[:, 0] 

def exact_sol_dense(H):
    """
    Computes the exact solution of the Hamiltonian H.
    """
    dense_H = H.to_dense()
    eig_vals= np.linalg.eigvalsh(dense_H)
    e_0 = eig_vals[0]
    print("Exact ground state energy:", e_0)
    return e_0

class FFN(nn.Module):
    # alpha ist der Faktor für die Neuronenzahl im ersten Dense-Layer
    alpha: int = 1
    # beta ist der Faktor für die Neuronenzahl im zweiten Dense-Layer (optional, default = 1)
    beta: int = 1

    @nn.compact
    def __call__(self, x):
        # Erstes Dense-Layer
        dense1 = nn.Dense(features=self.alpha * x.shape[-1], param_dtype=complex)
        y = dense1(x)
        y = nn.relu(y)

        # Zweites Dense-Layer
        dense2 = nn.Dense(features=self.beta * x.shape[-1], param_dtype=complex)
        y = dense2(y)
        y = nn.relu(y)

        # Summe am Ende
        return jnp.sum(y, axis=-1)

#-----------------------------------------------------------------------------------------------
# --- Parameter des Modells ---
N_max=3                 # Maximale Besetzungszahl der Moden
N_modes =3              # Anzahl der Phonon-Moden (Gitterpunkte im Impulsraum)
omega = 1.0             # Phononenfrequenz
alpha_coupling = 1.0    # Kopplungsstärke V_q


alpha=1
beta=1
n_samples=2**13
lr=0.01
n_iter=500

# --- Hilbert-Raum der Phononen ---
hi = nk.hilbert.Fock(n_max=N_max, N=N_modes)
#print(hi.shape) #Form des Hilbertraums
#print(hi.size)  #Anzahl der Basisvektoren
#print(hi.n_states)  #Anzahl der Zustände

# --- Hilbert-Raum Elektron ---


# --- Hamiltonian aufstellen ---
H = sum(omega * number(hi, i) for i in range(N_modes))
#H += sum(alpha_coupling * (create(hi, i) + destroy(hi, i)) for i in range(N_modes))

# --- Diagonalisierung ---
e_0,v_0=exact_sol_dense(H)
#e_0,v_0=aprox_sol(H)

# --- VMC ---

model = FFN(alpha=alpha, beta=beta)
sampler = nk.sampler.ExactSampler(hi)
vstate = nk.vqs.MCState(sampler, model, n_samples=n_samples)


optimizer = nk.optimizer.Sgd(learning_rate=lr) 

preconditioner = nk.optimizer.SR(diag_shift=0.1)

# driver for running the simulation
gs = nk.driver.VMC(H, optimizer, variational_state=vstate, preconditioner=preconditioner)
log = nk.logging.RuntimeLog()

gs.run(n_iter=n_iter, out=log)
ffn_energy = vstate.expect(H)
error = abs((ffn_energy.mean - e_0) / e_0)
print(f"Optimized energy: {ffn_energy} \nRelative error: {error*100:.2f}%")

data_FFN = log.data

fig, ax = plt.subplots(figsize=(12, 7))

ax.errorbar(
    data_FFN["Energy"].iters,
    data_FFN["Energy"].Mean.real,
    yerr=data_FFN["Energy"].Sigma,
    label="FFN",
)
ax.axhline([e_0], xmin=0, xmax=700, color="red", label="Exact")
ax.legend()
ax.grid()
ax.set_title("Convergence of the FFN")
ax.set_ylim(-4.5, 4.5)
ax.set_xlabel("Iterations")
ax.set_ylabel("Energy (Re)")
plt.savefig("/Users/julianblasek/master_local/praktikum/plots/energy_convergence_FFN.pdf")
plt.show()

fig, ax = plt.subplots(figsize=(12, 7))
x = np.arange(hi.n_states)
width = 0.4
# NQS (FNN) leicht nach links
ax.bar(x - width/2, np.abs(vstate.to_array()), width=width, label="FNN")
# Exakter Grundzustand leicht nach rechts
ax.bar(x + width/2, np.abs(v_0), width=width, label="Exact GS")
ax.set_xlabel("basis coordinate")
ax.set_ylabel(r"$\mid \psi \mid$")
ax.set_title("Variational state")
#ax.set_xlim(-0.5, 10)
ax.legend(loc=0)
plt.grid(True)
plt.savefig("/Users/julianblasek/master_local/praktikum/plots/state_chart.pdf")
plt.show()
#overlap = np.abs(np.vdot(v_0, vstate.to_array()))**2
#print(f"Overlap mit Grundzustand: {overlap:.5f}")
