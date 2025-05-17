# config.py
# Parameter für den Fock-Hilbertraum
N_max = 5       # Maximale Besetzungszahl pro Mode (5 1d, 4:3d)
N_modes = 24    # Anzahl der Phonon-Moden  (24:1d, 2:3d)

# Hamilton Parameter
omega = 1.0           # Phononenfrequenz
alpha_coupling = 1  # Kopplungsstärke (1:1d, 1:3d)

# Netzwerk-/Trainingsparameter
alpha = 2           # Faktor für Anzahl der Neuronen im 1. Dense-Layer
beta = 1           # Faktor für Anzahl der Neuronen im 2. Dense-Layer
n_samples = 2**14   # Anzahl der Samples im VMC 2^14 1d
lr = 0.01         # Lernrate 0.01 1d
n_iter = 500        # Anzahl der Trainingsiterationen 500 1d
sr=0.01          # Diagonalverschiebung für den SR-Preconditioner 0.01 1d
