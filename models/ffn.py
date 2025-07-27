# models/ffn.py

import flax.linen as nn
import jax.numpy as jnp


class FFN(nn.Module):
    """
    Two-layer complex-valued feedforward neural network (FFN) used
    for variational wavefunction representation.

    Args:
        alpha (int): Scaling factor for the width of the first dense layer
        beta (int): Scaling factor for the width of the second dense layer
    """
    alpha: int = 1
    beta: int = 1

    @nn.compact
    def __call__(self, x):
        # First dense layer with complex-valued parameters
        y = nn.Dense(features=self.alpha * x.shape[-1], param_dtype=complex)(x)
        y = nn.relu(y)

        # Second dense layer
        y = nn.Dense(features=self.beta * x.shape[-1], param_dtype=complex)(y)
        y = nn.relu(y)

        # Return scalar output by summing over last axis (as required by NetKet)
        return jnp.sum(y, axis=-1)
