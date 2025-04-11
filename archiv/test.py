import numpy as np
import netket as nk
from matplotlib import pyplot as plt
import jax
from netket.operator.spin import sigmax, sigmaz
from scipy.sparse.linalg import eigsh
import jax.numpy as jnp
import flax
from flax import nnx

class MF(nnx.Module):
    """
    A class implementing a uniform mean-field model.
    """

    # The __init__ function is used to define the parameters of the model
    # The RNG argument is used to initialize the parameters of the model.
    def __init__(self, *, rngs: nnx.Rngs):
        # To generate random numbers we need to extract the key from the
        # `rngs` object.
        key = rngs.params()
        # We store the log-wavefunction on a single site, and we call it
        # `log_phi_local`. This is a variational parameter, and it will be
        # optimized during training.
        #
        # We store a single real parameter, as we assume the wavefunction
        # is normalised, and initialise it according to a normal distibution.
        self.log_phi_local = nnx.Param(jax.random.normal(key, (1,)))

    # The __call__(self, x) function should take as
    # input a batch of states x.shape = (n_samples, L)
    # and should return a vector of n_samples log-amplitudes
    def __call__(self, x: jax.Array):

        # compute the probabilities
        p = nnx.log_sigmoid(self.log_phi_local * x)

        # sum the output
        return 0.5 * jnp.sum(p, axis=-1)

N = 20
hi = nk.hilbert.Spin(s=1 / 2, N=N)

hi.random_state(jax.random.key(0), 3)


#specifying the Hamiltonian
Gamma = -1

# Hamiltonian for field interaction
H = sum([Gamma * sigmax(hi, i) for i in range(N)])

# Hamiltonian for nearest neighbor interaction
V = -1
H += sum([V * sigmaz(hi, i) * sigmaz(hi, (i + 1) % N) for i in range(N)])


#Diagonalizing the Hamiltonian
sp_h = H.to_sparse()
sp_h.shape


eig_vals, eig_vecs = eigsh(sp_h, k=2, which="SA")

print("eigenvalues with scipy sparse:", eig_vals)

E_gs = eig_vals[0]

mf_model = MF(rngs=nnx.Rngs(0))

sampler = nk.sampler.MetropolisLocal(hi)

vstate = nk.vqs.MCState(sampler, mf_model, n_samples=512)

print(vstate.parameters)

E = vstate.expect(H)
print(E)
print("Mean                  :", E.mean)
print("Error                 :", E.error_of_mean)
print("Variance              :", E.variance)
print("Convergence indicator :", E.R_hat)
print("Correlation time      :", E.tau_corr)




# First we reset the parameters to run the optimisation again
vstate.init_parameters()

# Then we create an optimiser from the standard library.
# You can also use optax.
optimizer = nk.optimizer.Sgd(learning_rate=0.05)

# build the optimisation driver
gs = nk.driver.VMC(H, optimizer, variational_state=vstate)

# run the driver for 300 iterations. This will display a progress bar
# by default.
gs.run(n_iter=300)

mf_energy = vstate.expect(H)
error = abs((mf_energy.mean - eig_vals[0]) / eig_vals[0])
print("Optimized energy and relative error: ", mf_energy, error)
print("Final optimized parameter: ", vstate.parameters["log_phi_local"])
eig_vals[0]




class JasShort(nnx.Module):

    def __init__(self, *, rngs: nnx.Rngs):

        # Define two parameters j1, and j2.
        # Initialise them with a random normal distribution of standard deviation
        # 0.01
        # We must get a different key for each parameter, otherwise they will be
        # initialised with the same value.
        self.j1 = nnx.Param(0.01 * jax.random.normal(rngs.params(), (1,)), dtype=float)
        self.j2 = nnx.Param(0.01 * jax.random.normal(rngs.params(), (1,)), dtype=float)

    def __call__(self, x: jax.Array):

        # compute the nearest-neighbor correlations
        corr1 = x * jnp.roll(x, -1, axis=-1)
        corr2 = x * jnp.roll(x, -2, axis=-1)

        # sum the output
        return jnp.sum(self.j1 * corr1 + self.j2 * corr2, axis=-1)


# Initialise the model wtih seed 1
model = JasShort(rngs=nnx.Rngs(1))

vstate = nk.vqs.MCState(sampler, model, n_samples=1008)

optimizer = nk.optimizer.Sgd(learning_rate=0.01)

gs = nk.driver.VMC(
    H,
    optimizer,
    variational_state=vstate,
    preconditioner=nk.optimizer.SR(diag_shift=0.1),
)

# construct the logger
log = nk.logging.RuntimeLog()

# One or more logger objects must be passed to the keyword argument `out`.
gs.run(n_iter=300, out=log)

print(f"Final optimized parameters: j1={vstate.parameters['j1']}, j2={vstate.parameters['j2']}")

jas_energy = vstate.expect(H)
error = abs((jas_energy.mean - eig_vals[0]) / eig_vals[0])
print(f"Optimized energy : {jas_energy}")
print(f"relative error   : {error}")


data_jastrow = log.data
print(data_jastrow)




plt.errorbar(
    data_jastrow["Energy"].iters,
    data_jastrow["Energy"].Mean,
    yerr=data_jastrow["Energy"].Sigma,
)
plt.xlabel("Iterations")
plt.ylabel("Energy")
plt.show()



class FFN(nnx.Module):

    def __init__(self, N: int, alpha: int = 1, *, rngs: nnx.Rngs):
        """
        Construct a Feed-Forward Neural Network with a single hidden layer.

        Args:
            N: The number of input nodes (number of spins in the chain).
            alpha: The density of the hidden layer. The hidden layer will have
                N*alpha nodes.
            rngs: The random number generator seed.
        """
        self.alpha = alpha

        # We define a linear (or dense) layer with `alpha` times the number of input nodes
        # as output nodes.
        # We must pass forward the rngs object to the dense layer.
        self.linear = nnx.Linear(in_features=N, out_features=alpha * N, rngs=rngs)

    def __call__(self, x: jax.Array):

        # we apply the linear layer to the input
        y = self.linear(x)

        # the non-linearity is a simple ReLu
        y = nnx.relu(y)

        # sum the output
        return jnp.sum(y, axis=-1)


model=FFN(N=N, alpha=1, rngs=nnx.Rngs(2))

vstate = nk.vqs.MCState(sampler, model, n_samples=1008)

optimizer = nk.optimizer.Sgd(learning_rate=0.1)

# Notice the use, again of Stochastic Reconfiguration, which considerably improves the optimisation
gs = nk.driver.VMC(
    H,
    optimizer,
    variational_state=vstate,
    preconditioner=nk.optimizer.SR(diag_shift=0.1),
)

log = nk.logging.RuntimeLog()
gs.run(n_iter=300, out=log)

ffn_energy = vstate.expect(H)
error = abs((ffn_energy.mean - eig_vals[0]) / eig_vals[0])
print("Optimized energy and relative error: ", ffn_energy, error)


data_FFN = log.data

plt.errorbar(
    data_jastrow["Energy"].iters,
    data_jastrow["Energy"].Mean,
    yerr=data_jastrow["Energy"].Sigma,
    label="Jastrow",
)
plt.errorbar(
    data_FFN["Energy"].iters,
    data_FFN["Energy"].Mean,
    yerr=data_FFN["Energy"].Sigma,
    label="FFN",
)
plt.hlines([E_gs], xmin=0, xmax=300, color="black", label="Exact")
plt.legend()

plt.xlabel("Iterations")
plt.ylabel("Energy")
plt.title("Energy convergence")
plt.grid()
plt.show()
plt.savefig("/Users/julianblasek/master_local/praktikum/plots/energy_convergence.pdf")