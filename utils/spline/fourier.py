##
#
#  Fourier series spline implementation.
#  NOTE: only for synthesizing PERIODIC trajectories.
##

# for base class
from utils.spline.base import Base_Spline

# standard imports
import numpy as np

# jax imports
import jax
import jax.numpy as jnp


#############################################################
# Fourier Series Spline
#############################################################

class Fourier_Spline(Base_Spline):
    """
    Vectorized Fourier series spline (more efficient).
    """

    def __init__(self, Y0: jnp.ndarray, 
                       T: float,
                       periodic: bool = True):
        
        super().__init__(Y0, T)

        if self.K < 1:
            raise ValueError(f"Fourier spline requires at least 1 coefficient (K >= 1)."
                             f" Got K={self.K}.")

        if self.K % 2 == 0:
            raise ValueError(f"K should be odd (K = 1 + 2*n_harmonics). Got K={self.K}.")

        self.periodic = periodic
        self.n_harmonics = (self.K - 1) // 2
        
        # Use different frequency based on periodicity
        if periodic:
            self.omega = 2 * np.pi / T  # f(0) = f(T)
        else:
            self.omega = np.pi / T      # No periodicity constraint

        print(f"Fourier Spline ({'periodic' if periodic else 'non-periodic'}) "
              f"initialized with {self.n_harmonics} harmonics.")


    def evaluate(self, times: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluate the Fourier series at given times.
        
        f(t) = a0 + sum_{n=1}^{N} [a_n * cos(n*omega*t) + b_n * sin(n*omega*t)]
        
        Args:
            times: jnp.array, shape (M,), times to evaluate the spline at.

        Returns:
            Y_eval: jnp.array, shape (B, M, dim), spline values at the given times.
        """
        
        if not self.periodic:
            times = jnp.clip(times, 0.0, self.T)
        
        M = times.shape[0]
        
        # Create basis matrix: shape (K, M)
        basis = jnp.zeros((self.K, M))
        
        # DC term
        basis = basis.at[0, :].set(1.0)
        
        # Harmonic terms
        n_vals = jnp.arange(1, self.n_harmonics + 1)  # (n_harmonics,)
        omega_t = self.omega * times  # (M,)
        
        # Cosine terms: indices 1, 3, 5, ...
        cos_indices = 2 * n_vals - 1
        cos_basis = jnp.cos(n_vals[:, None] * omega_t[None, :])  # (n_harmonics, M)
        basis = basis.at[cos_indices, :].set(cos_basis)
        
        # Sine terms: indices 2, 4, 6, ...
        sin_indices = 2 * n_vals
        sin_basis = jnp.sin(n_vals[:, None] * omega_t[None, :])  # (n_harmonics, M)
        basis = basis.at[sin_indices, :].set(sin_basis)
        
        # Evaluate: (K, M) x (B, K, dim) -> (B, M, dim)
        Y_eval = jnp.einsum("km,bkd->bmd", basis, self.Y)
        
        return Y_eval


#############################################################
# EXAMPLE USAGE
#############################################################

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import numpy as np
    import time

    seed = int(time.time())

    # Create a Fourier spline
    B = 64
    n_harmonics = 5
    K = 1 + 2 * n_harmonics  # = 11 (1 DC + 5 cos + 5 sin)
    nu = 2
    T = 10.0
    
    # Initialize with small random coefficients
    Y0 = jax.random.normal(jax.random.PRNGKey(seed), (B, K, nu)) * 0.3
    
    # Set DC term to center the curve
    Y0 = Y0.at[:, 0, :].set(0.0)
    
    spline = Fourier_Spline(Y0, T, periodic=True)

    # Print some info
    print(f"  Total time: {spline.T}")
    print(f"  Batch size: {spline.B}")
    print(f"  Num coefficients: {spline.K}")
    print(f"  Dimensionality: {spline.dim}")

    t_eval = jnp.linspace(0.0, T, 500)
    Y_eval = spline.evaluate(t_eval)
    
    # Convert to numpy for plotting
    t_eval = np.array(t_eval)
    Y_eval = np.array(Y_eval)

    # Plot batch 0
    plt.figure(figsize=(12, 4))

    # Plot as parametric curve in 2D
    plt.subplot(1, 2, 1)
    plt.plot(Y_eval[0, :, 0], Y_eval[0, :, 1], color='blue', linewidth=2, label="Fourier curve")
    plt.scatter(Y_eval[0, 0, 0], Y_eval[0, 0, 1], color='red', s=100, zorder=5, 
                marker='o', label="Start/End (periodic)")
    plt.title("Fourier Series Curve (Batch 0)")
    plt.xlabel("dim 0")
    plt.ylabel("dim 1")
    plt.legend()
    plt.grid()
    plt.axis("equal")

    # Plot each dimension vs time
    plt.subplot(1, 2, 2)
    plt.plot(t_eval, Y_eval[0, :, 0], color='blue', linewidth=2, label="dim 0")
    plt.plot(t_eval, Y_eval[0, :, 1], color='orange', linewidth=2, label="dim 1")
    plt.title("Fourier Series vs Time (Batch 0)")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()