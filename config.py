# config.py
# Parameter für den Fock-Hilbertraum
N_max = 5       # Maximale Besetzungszahl pro Mode
N_modes = 5     # Anzahl der Phonon-Moden (Gitterpunkte im k-Raum)

# Hamilton Parameter
omega = 1.0           # Phononenfrequenz
alpha_coupling = 1.0  # Kopplungsstärke (optional, kann später aktiviert werden)

# Netzwerk-/Trainingsparameter
alpha = 1           # Faktor für Anzahl der Neuronen im 1. Dense-Layer
beta = 1            # Faktor für Anzahl der Neuronen im 2. Dense-Layer
n_samples = 2**12   # Anzahl der Samples im VMC
lr = 0.015           # Lernrate
n_iter = 500        # Anzahl der Trainingsiterationen