"""
config.py ─ Centralised parameters for the Fröhlich-polaron VMC runs
====================================================================

Everything numeric lives here; import this file wherever parameters
are needed and keep the rest of the code free of hard-coded numbers.

Layout
------
1.  Fock-space truncation           (N_max, N_modes)
2.  Hamiltonian constants           (omega, alpha_coupling)
3.  Network / training hyper-params (alpha, beta, n_samples, lr,
                                     n_iter, sr, m_b)

Conventions
-----------
* Natural units:  ħ = ω = 1.
* All quantities are dimensionless under this choice.
* ``m_b`` is given in units of the bath-boson mass *m_B*.

Default presets
---------------
The values below reproduce the benchmark results quoted in the report.

* 1D static Fröhlich model  
  N_max = 5, N_modes = 24, n_iter = 500, n_samples = 2**12

* 3D static Fröhlich model  
  N_max = 5, N_modes = 2, same optimisation parameters as above
"""

# ──────────────────────────────────────────────────────────────
# 1. FOCK-SPACE PARAMETERS
# ──────────────────────────────────────────────────────────────

#: Maximum phonon occupation per mode.
N_max: int = 5

#: Number of discrete momentum modes.
N_modes: int = 24


# ──────────────────────────────────────────────────────────────
# 2. HAMILTONIAN CONSTANTS
# ──────────────────────────────────────────────────────────────

#: Bare phonon frequency (set to 1 in natural units).
omega: float = 1.0

#: Dimensionless impurity–phonon coupling strength.
alpha_coupling: float = 1.0


# ──────────────────────────────────────────────────────────────
# 3. NETWORK / TRAINING HYPER-PARAMETERS
# ──────────────────────────────────────────────────────────────

#: Width multiplier for the first dense layer.
alpha: int = 2

#: Width multiplier for the second dense layer.
beta: int = 1

#: Monte-Carlo samples per VMC iteration.
n_samples: int = 2 ** 12      # 4096

#: Stochastic-gradient learning rate.
lr: float = 0.01

#: VMC optimisation iterations.
n_iter: int = 500

#: Diagonal shift ε_SR for stochastic-reconfiguration preconditioning.
sr: float = 0.01

#: Impurity mass (relative to bath bosons); used only in dynamical runs.
m_b: float = 0.5
