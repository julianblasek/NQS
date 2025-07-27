"""
visual.py ― Plotting utilities for NetKet-VMC benchmarks
========================================================

All figure generation for the project is centralised here so the rest of
the code base can stay completely free of *matplotlib* calls.

Routines
--------
1.  `energy_convergence` – line plot (with error bars) that shows how the
    VMC energy estimator approaches the reference value over iterations.
2.  `state_chart`        – bar-chart comparison of *|ψ|* amplitudes between
    the exact ground state and the learned variational states.

Both functions save a *PDF* copy of the figure to the local *plots/*
directory **and** show an interactive window if running in a desktop
environment.  Modify the hard-coded path strings if you need a different
output location.
"""

# ──────────────────────────────────────────────────────────────
# 1. IMPORTS
# ──────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import numpy as np


# ──────────────────────────────────────────────────────────────
# 2. ENERGY-CONVERGENCE PLOT
# ──────────────────────────────────────────────────────────────
def energy_convergence(results, e_0, n_iter):
    """
    Plot the energy convergence for *each* model in ``results``.

    Parameters
    ----------
    results : list[dict]
        Output of *simulate.run_all*; every entry must contain keys  
        ``{"name", "log"}``, where ``log`` is a NetKet RuntimeLog object.
    e_0 : float
        Reference (exact or perturbative) ground-state energy shown as
        a horizontal dotted line.  Pass *None* to skip this marker.
    n_iter : int
        Total number of VMC iterations – used only to set the x-axis limit.
    """
    plt.ion()            # interactive mode → window refreshes automatically
    plt.close("all")     # close leftovers from previous figures
    fig, ax = plt.subplots(figsize=(12, 7))

    # ── 2.1 Draw one error-bar line per model ──────────────────────
    for res in results:
        data = res["log"].data
        name = res["name"]

        ax.errorbar(
            data["Energy"].iters,         # iteration index
            data["Energy"].Mean.real,     # ⟨E⟩
            yerr=data["Energy"].Sigma,    # 1σ error from blocking analysis
            label=name,
        )

    # ── 2.2 Optional reference line ───────────────────────────────
    if e_0 is not None:
        ax.axhline(
            e_0,
            color="red",
            linestyle="dotted",
            label=f"Exact (e₀ = {e_0:.4f})",
        )

    # ── 2.3 Aesthetics ────────────────────────────────────────────
    ax.set_title("Energy Convergence of Models")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Energy (Re)")
    ax.set_xlim(0, n_iter)
    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    plt.savefig("/Users/julianblasek/local/praktikum/plots/energy_convergence.pdf")
    plt.show()
    return  # explicit for clarity


# ──────────────────────────────────────────────────────────────
# 3. WAVE-FUNCTION BAR CHART
# ──────────────────────────────────────────────────────────────
def state_chart(results, v_0, hi):
    """
    Bar-chart visualisation of *|ψ|* coefficients for each variational
    state versus the exact ground state.

    Parameters
    ----------
    results : list[dict]
        Same structure as above; needs keys ``{"name", "vstate"}``.
    v_0 : np.ndarray
        Exact ground-state vector in the computational basis.
    hi : nk.hilbert.AbstractHilbert
        Provides the dimension (``hi.n_states``) of the Fock space.
    """
    num_models = len(results)
    n_states = hi.n_states
    x = np.arange(n_states)     # basis-index axis
    width = 0.4                 # bar width (shared for all panels)

    plt.close("all")
    fig, axes = plt.subplots(
        nrows=num_models,
        ncols=1,
        figsize=(12, 4 * num_models),
        sharex=True,
    )

    # Ensure axes is iterable even for a single subplot
    if num_models == 1:
        axes = [axes]

    # ── 3.1 One subplot per model ─────────────────────────────────
    for i, res in enumerate(results):
        ax = axes[i]
        name = res["name"]
        vstate = res["vstate"]

        # Absolute amplitudes of VMC and exact state
        v_array = np.abs(vstate.to_array())
        v0_array = np.abs(v_0)

        # Overlayed bar plot for visual comparison
        ax.bar(x - width / 2, v_array, width=width, label=f"{name}")
        ax.bar(x + width / 2, v0_array, width=width, label="Exact GS")

        ax.set_ylabel(r"$|\psi|$")
        ax.set_title(f"State Comparison: {name}")
        ax.set_xlim(-0.5, min(n_states, 15))  # show first few basis states
        ax.grid(True)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Basis coordinate")

    plt.tight_layout()
    plt.savefig("/Users/julianblasek/local/praktikum/plots/state_chart.pdf")
    plt.show()
    return
