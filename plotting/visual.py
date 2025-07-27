# plotting/visual.py
import matplotlib.pyplot as plt
import numpy as np


def energy_convergence(results, e_0,n_iter):
    """
    Plots the energy convergence of multiple models from the results list.

    Parameters:
    - results: list of dicts, each containing keys like "name", "log"
    - e_0: float, exact ground state energy
    """
    plt.ion()
    plt.close("all")
    fig, ax = plt.subplots(figsize=(12, 7))

    for res in results:
        data = res["log"].data
        name = res["name"]

        # Konvergenzplot mit Fehlerbalken
        ax.errorbar(
            data["Energy"].iters,
            data["Energy"].Mean.real,
            yerr=data["Energy"].Sigma,
            label=name,
        )

    # Exakter Energiewert als Referenzlinie
    if e_0 is not None:
        ax.axhline(e_0, color="red", linestyle="dotted", label=f"Exact (e₀ = {e_0:.4f})")


    # Formatierung
    ax.set_title("Energy Convergence of Models")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Energy (Re)")
    ax.legend()
    ax.set_xlim(0, n_iter)
    ax.grid(True)

    plt.tight_layout()
    plt.savefig("/Users/julianblasek/local/praktikum/plots/energy_convergence.pdf")
    plt.show()
    return



def state_chart(results, v_0, hi):
    """
    Creates subplots comparing the variational states of all models to the exact ground state.

    Parameters:
    - results: list of dicts, each with keys like "name", "vstate"
    - v_0: np.ndarray, exact ground state
    - hi: NetKet Hilbert space (used for dimension)
    """

    num_models = len(results)
    n_states = hi.n_states
    x = np.arange(n_states)
    width = 0.4

    plt.close("all")
    fig, axes = plt.subplots(
        nrows=num_models, ncols=1, figsize=(12, 4 * num_models), sharex=True
    )

    # Falls nur ein Modell → axes ist kein Array → in Liste verpacken
    if num_models == 1:
        axes = [axes]

    for i, res in enumerate(results):
        ax = axes[i]
        name = res["name"]
        vstate = res["vstate"]

        v_array = np.abs(vstate.to_array())
        v0_array = np.abs(v_0)

        ax.bar(x - width/2, v_array, width=width, label=f"{name}")
        ax.bar(x + width/2, v0_array, width=width, label="Exact GS")

        ax.set_ylabel(r"$|\psi|$")
        ax.set_title(f"State Comparison: {name}")
        ax.set_xlim(-0.5, min(n_states, 15))  # nur ersten 20 Basiszustände anzeigen
        ax.grid(True)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Basis coordinate")
    plt.tight_layout()
    plt.savefig("/Users/julianblasek/local/praktikum/plots/state_chart.pdf")
    plt.show()
    return