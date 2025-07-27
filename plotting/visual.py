# plotting/visual.py

import matplotlib.pyplot as plt
import numpy as np


def energy_convergence(results, e_0, n_iter):
    """
    Plots the energy convergence of one or more models during VMC training.

    Args:
        results (list): List of dictionaries, each containing model "name" and "log" with energy data.
        e_0 (float): Analytic or exact ground-state energy for reference.
        n_iter (int): Total number of training iterations.
    """
    plt.ion()
    plt.close("all")
    fig, ax = plt.subplots(figsize=(12, 7))

    for res in results:
        data = res["log"].data
        name = res["name"]

        ax.errorbar(
            data["Energy"].iters,
            data["Energy"].Mean.real,
            yerr=data["Energy"].Sigma,
            label=name,
        )

    # Add reference line for exact energy
    if e_0 is not None:
        ax.axhline(e_0, color="red", linestyle="dotted", label=f"Exact (e₀ = {e_0:.4f})")

    # Plot formatting
    ax.set_title("Energy Convergence of Models")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Energy (Re)")
    ax.legend()
    ax.set_xlim(0, n_iter)
    ax.grid(True)

    plt.tight_layout()
    plt.savefig("/Users/julianblasek/master_local/praktikum/plots/energy_convergence.pdf")
    plt.show()
