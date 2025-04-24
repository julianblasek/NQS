# config.py
# Parameter für den Fock-Hilbertraum
N_max = 3       # Maximale Besetzungszahl pro Mode
N_modes = 4     # Anzahl der Phonon-Moden (Gitterpunkte im k-Raum)

# Hamilton Parameter
omega = 1.0           # Phononenfrequenz
alpha_coupling = 0.75  # Kopplungsstärke (optional, kann später aktiviert werden)

# Netzwerk-/Trainingsparameter
alpha = 2           # Faktor für Anzahl der Neuronen im 1. Dense-Layer
beta = 1            # Faktor für Anzahl der Neuronen im 2. Dense-Layer
n_samples = 2**12   # Anzahl der Samples im VMC
lr = 0.01           # Lernrate
n_iter = 400        # Anzahl der Trainingsiterationen