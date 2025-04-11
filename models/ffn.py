# models/ffn.py
import flax.linen as nn
import jax.numpy as jnp


class FFN(nn.Module):
    # alpha ist der Faktor für die Neuronenzahl im ersten Dense-Layer
    alpha: int = 1
    # beta ist der Faktor für die Neuronenzahl im zweiten Dense-Layer (optional, default = 1)
    beta: int = 1

    @nn.compact
    def __call__(self, x):
        # Erstes Dense-Layer
        dense1 = nn.Dense(features=self.alpha * x.shape[-1], param_dtype=complex)
        y = dense1(x)
        y = nn.relu(y)

        # Zweites Dense-Layer
        dense2 = nn.Dense(features=self.beta * x.shape[-1], param_dtype=complex)
        y = dense2(y)
        y = nn.relu(y)

        # Summe am Ende
        return jnp.sum(y, axis=-1)