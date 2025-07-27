# config.py

# ------------------------------
# Fock Space Parameters
# ------------------------------

N_max = 5         # Maximum occupation number per phonon mode (e.g., 5 for 1D, 4 for 3D)
N_modes = 24      # Number of phonon modes (e.g., 24 for 1D, 2 for 3D)

# ------------------------------
# Hamiltonian Parameters
# ------------------------------

omega = 1.0            # Phonon frequency
alpha_coupling = 1.0   # Impurity-phonon coupling strength (α) (typ. 1 for both 1D and 3D)

# ------------------------------
# Network and Training Parameters
# ------------------------------

alpha = 2              # Scaling factor for the width of the first dense (hidden) layer
beta = 1               # Scaling factor for the width of the second dense layer
n_samples = 2**12      # Number of Monte Carlo samples per VMC iteration (e.g., 4096)
lr = 0.01              # Learning rate for stochastic optimization
n_iter = 200           # Number of training iterations (e.g., 200 for minimal 1D run)
sr = 0.01              # Diagonal shift (ε_SR) for stochastic reconfiguration (preconditioning)
m_b = 0.5              # Impurity mass (relevant for dynamical Hamiltonians in 1D only)
