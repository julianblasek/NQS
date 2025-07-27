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

# Netzwerk-/Trainingsparameter
alpha = 2           # Faktor für Anzahl der Neuronen im 1. Dense-Layer
beta = 1           # Faktor für Anzahl der Neuronen im 2. Dense-Layer
n_samples = 2**14   # Anzahl der Samples im VMC 2^14 1d
lr = 0.01         # Lernrate 0.01 1d
n_iter = 500        # Anzahl der Trainingsiterationen 500 1d
sr=0.01          # Diagonalverschiebung für den SR-Preconditioner 0.01 1d
