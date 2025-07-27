"""
ffn.py ― Collection of neural-quantum-state (NQS) network architectures
======================================================================

This module bundles *several* feed-forward and convolutional ansätze
that can be plugged into NetKet’s :pyclass:`~netket.vqs.MCState`.

Implemented models
------------------
1. **FFN**       –   two fully-connected layers with ReLU  
2. **DeepFFN**   –   three fully-connected layers with ReLU  
3. **DeepFFN2**      –   four layers + *log cosh* non-linearity
4. **Conv**      –   1-D convolution followed by `log cosh` activation 
5. **DeepConv**  –   complex-valued two-stage CNN with custom ReLU
6. **RBM**       –   Restricted Boltzmann Machine (RBM) with complex weights

All classes inherit from *Flax* ``nn.Module``; therefore the function
body focuses on tensor operations and **does not** touch optimisation 
or sampling.

Layout
------
1.  Imports
2.  Dense-based networks   (FFN, DeepFFN, DeepFFN2)
3.  Convolutional networks (Conv, DeepConv)
4.  Restricted Boltzmann Machine (RBM)


"""

# ──────────────────────────────────────────────────────────────
# 1. IMPORTS
# ──────────────────────────────────────────────────────────────
import flax.linen as nn
import jax.numpy as jnp
import jax 


# ──────────────────────────────────────────────────────────────
# 2. DENSE-BASED NETWORKS
# ──────────────────────────────────────────────────────────────
class FFN(nn.Module):
    # alpha is the factor for the number of neurons in the
    # first dense layer
    alpha: int = 1
    # beta is the factor for the number of neurons in 
    # the second dense layer (optional, default = 1)
    beta: int = 1

    @nn.compact
    def __call__(self, x):
        """
        Forward pass: Dense(α·dim) → ReLU → Dense(β·dim) → ReLU → Σᵢ yᵢ.

        Notes
        -----
        * ``param_dtype=complex`` lets the network learn 
        *complex-valued* weights, enabling a direct representation
        of the log-amplitude.
        """
        # First dense layer
        dense1 = nn.Dense(features=self.alpha * x.shape[-1], 
                          param_dtype=complex)
        
        y = dense1(x)
        y = nn.relu(y)

        # Second dense layer
        dense2 = nn.Dense(features=self.beta * x.shape[-1], 
                          param_dtype=complex)
        
        y = dense2(y)
        y = nn.relu(y)

        # Sum at the end
        # Return scalar complex log-amplitude (per sample)
        return jnp.sum(y, axis=-1)


class DeepFFN(nn.Module):
    # alpha is the factor for the number of neurons in the 
    # first dense layer
    alpha: int = 1
    # beta is the factor for the number of neurons in the 
    # second dense layer (optional, default = 1)
    beta: int = 1
    gamma: int = 1  # third-layer width multiplier

    @nn.compact
    def __call__(self, x):
        """
        Forward pass: Dense(α·dim) → ReLU → Dense(β·dim) → ReLU
                      → Dense(γ·dim) → ReLU → Σᵢ yᵢ
        """
        # First dense layer
        dense1 = nn.Dense(features=self.alpha * x.shape[-1], 
                          param_dtype=complex)
        
        y = dense1(x)
        y = nn.relu(y)

        # Second dense layer
        dense2 = nn.Dense(features=self.beta * x.shape[-1], 
                          param_dtype=complex)
        
        y = dense2(y)
        y = nn.relu(y)

        # Third dense layer
        dense3 = nn.Dense(features=self.gamma * x.shape[-1], 
                          param_dtype=complex)
        
        y = dense3(y)
        y = nn.relu(y)

        # Sum at the end
        return jnp.sum(y, axis=-1)


class DeepFFN2(nn.Module):
    """
    Four-layer fully-connected network that uses the 
    physically motivated
    non-linearity ``log cosh``.
    """
    alpha: int = 2
    beta: int = 2
    gamma: int = 2
    delta: int = 2

    @nn.compact
    def __call__(self, x):
        # Dense-1
        dense1 = nn.Dense(features=self.alpha * x.shape[-1])
        
        y = dense1(x)
        y = jnp.log(jnp.cosh(y))

        # Dense-2
        dense2 = nn.Dense(features=self.beta * x.shape[-1])
        
        y = dense2(y)
        y = jnp.log(jnp.cosh(y))

        # Dense-3
        dense3 = nn.Dense(features=self.gamma * x.shape[-1])
        
        y = dense3(y)
        y = jnp.log(jnp.cosh(y))

        # Dense-4
        dense4 = nn.Dense(features=self.delta * x.shape[-1])
        
        y = dense4(y)
        y = jnp.log(jnp.cosh(y))

        # Final linear read-out → scalar log-amplitude
        out = nn.Dense(features=1)(y)
        return out.squeeze(-1)


# ──────────────────────────────────────────────────────────────
# 3. CONVOLUTIONAL NETWORKS
# ──────────────────────────────────────────────────────────────
class Conv(nn.Module):
    features: int = 32  # Anzahl der Filter

    @nn.compact
    def __call__(self, x):
        """
        One-layer 1-D CNN:

            x … (batch, N) → expand-dim → Conv(valid) → log cosh
            → sum-pool
        """
        # Input: shape (batch, N)
        x = x[..., None]  # (batch, N, 1)

        # Convolution without padding
        y = nn.Conv(
            features=self.features,
            kernel_size=(3,),
            padding="VALID"
        )(x)

        # log(cosh) activation
        y = jnp.log(jnp.cosh(y))

        # Global sum pooling
        y = jnp.sum(y, axis=(1, 2))

        return y


class DeepConv(nn.Module):
    """
    Complex-valued two-layer CNN with a polar-ReLU activation that
    keeps phase information intact (|z| → ReLU(|z|) · e^{i arg(z)}).
    """
    channels1: int = 4   # Number of filters in Conv1
    channels2: int = 8   # Number of filters in Conv2
    dense_features: int = 16  # Neurons in Dense Layer

    @nn.compact
    def __call__(self, x):
        # Input: shape (batch, length), real-valued
        x = x[..., None]  # shape (batch, length, 1)

        # First Conv Layer (real + imag)
        conv1_real = nn.Conv(features=self.channels1, 
                             kernel_size=(4,), padding="SAME")
        
        conv1_imag = nn.Conv(features=self.channels1, 
                             kernel_size=(4,), padding="SAME")
        

        y_real1 = conv1_real(x)
        y_imag1 = conv1_imag(x)
        z1 = y_real1 + 1j * y_imag1

        # Activation: ReLU(|z|) * e^{i arg(z)}
        mag1 = jnp.abs(z1)
        phase1 = jnp.angle(z1)
        activated1 = nn.relu(mag1) * jnp.exp(1j * phase1)

        # Second Conv Layer
        conv2_real = nn.Conv(features=self.channels2, 
                             kernel_size=(2,), padding="SAME")
        
        conv2_imag = nn.Conv(features=self.channels2, 
                             kernel_size=(2,), padding="SAME")
        

        y_real2 = conv2_real(activated1.real)
        y_imag2 = conv2_imag(activated1.imag)
        z2 = y_real2 + 1j * y_imag2

        # Activation
        mag2 = jnp.abs(z2)
        phase2 = jnp.angle(z2)
        activated2 = nn.relu(mag2) * jnp.exp(1j * phase2)

        # Flatten
        flat_real = activated2.real.reshape((activated2.shape[0], -1))
        flat_imag = activated2.imag.reshape((activated2.shape[0], -1))
        flat_z = flat_real + 1j * flat_imag  # preserved for completeness

        # Dense Layer (real + imag)
        dense_real = nn.Dense(features=self.dense_features)
        dense_imag = nn.Dense(features=self.dense_features)

        y_dense_real = dense_real(flat_real)
        y_dense_imag = dense_imag(flat_imag)
        z_dense = y_dense_real + 1j * y_dense_imag

        # Activation
        mag_dense = jnp.abs(z_dense)
        phase_dense = jnp.angle(z_dense)
        activated_dense = nn.relu(mag_dense) * jnp.exp(1j * phase_dense)

        # Output: Sum of all components results in a complex scalar
        output = jnp.sum(activated_dense, axis=-1)

        return output

# ──────────────────────────────────────────────────────────────
# 4. RESTRICTED BOLTZMANN MACHINE
# ──────────────────────────────────────────────────────────────
class RBM(nn.Module):
    """
    Complex RBM log-amplitude

        log ψ(x) = Σ_i a_i x_i +
                   Σ_j log cosh( b_j + Σ_i W_{ij} x_i )

    Default assumes binary ±1 variables; works for spins or
    truncated Fock occupations.
    """
    n_hidden: int = 32   # number of hidden units

    @nn.compact
    def __call__(self, x):
        # affine transform:  x W  +  b   (complex params)
        W = self.param("W", nn.initializers.normal(), (x.shape[-1],
                                                      self.n_hidden),
                       complex)
        b = self.param("b", nn.initializers.normal(), (self.n_hidden,),
                       complex)
        a = self.param("a", nn.initializers.normal(), (x.shape[-1],),
                       complex)

        pre_act = b + jnp.dot(x, W)      # shape (..., n_hidden)
        hidden  = jnp.log(jnp.cosh(pre_act))

        log_amp = jnp.sum(a * x, axis=-1) + jnp.sum(hidden, axis=-1)
        return log_amp                    # complex scalar per sample