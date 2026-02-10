##
#
#  Linear spline (piecewise linear interpolation).
#  
##

# for base class
from utils.spline.base import Base_Spline

# standard imports
import numpy as np

# jax imports
import jax
import jax.numpy as jnp


#############################################################
# Linear Spline
#############################################################

class Linear_Spline(Base_Spline):
    """
    Linear spline (first-order hold).
    
    The knot points are uniformly distributed over [0, T].
    Values are linearly interpolated between consecutive knots.
    The curve passes through all knot points and stays within their bounds.
    """

    def __init__(self, Y0: jnp.ndarray, T: float):
        
        # initialize parent class
        super().__init__(Y0, T)

        # linear spline requires at least 2 knots
        if self.K < 2:
            raise ValueError(f"Linear spline requires at least 2 knots. Got K={self.K}.")
        
        # create uniform knot times
        self.t_knots = jnp.linspace(0.0, self.T, self.K)  # shape (K,)
        self.dt = self.T / (self.K - 1)  # spacing between knots

        print("Linear Spline initialized.")

    def evaluate(self, times: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluate the linear spline at given times.
        
        Uses linear interpolation between consecutive knots:
        y(t) = y_i + (y_{i+1} - y_i) * (t - t_i) / (t_{i+1} - t_i)

        Args:
            times: jnp.array, shape (M,), times to evaluate the spline at.
        Returns:
            Y_eval: jnp.array, shape (B, M, dim), spline values at the given times.
        """
        # Clip times to [0, T]
        times = jnp.clip(times, 0.0, self.T)
        
        # Find which segment each time belongs to
        segment_idx = jnp.floor(times / self.dt).astype(jnp.int32)
        segment_idx = jnp.clip(segment_idx, 0, self.K - 2)  # last segment is K-2
        
        # Local time within segment: t - t_i, normalized to [0, 1]
        local_t = times - self.t_knots[segment_idx]  # shape (M,)
        alpha = local_t / self.dt  # interpolation parameter in [0, 1]
        
        # Get knot values for start and end of each segment
        y_start = self.Y[:, segment_idx, :]  # shape (B, M, dim)
        y_end = self.Y[:, segment_idx + 1, :]  # shape (B, M, dim)
        
        # Linear interpolation: y = (1-alpha)*y_start + alpha*y_end
        Y_eval = (1.0 - alpha)[None, :, None] * y_start + alpha[None, :, None] * y_end
        
        return Y_eval


#############################################################
# EXAMPLE USAGE
#############################################################

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import numpy as np
    import time

    seed = int(time.time())

    # Create a linear spline
    B = 64
    K = 7
    nu = 2
    T = 10.0
    Y0 = jax.random.uniform(jax.random.PRNGKey(seed), (B, K, nu), minval=-1.0, maxval=1.0)
    
    spline = Linear_Spline(Y0, T)

    # Print info
    print(f"  Total time: {spline.T}")
    print(f"  Batch size: {spline.B}")
    print(f"  Num knots: {spline.K}")
    print(f"  Dimensionality: {spline.dim}")

    # Evaluate
    t_eval = jnp.linspace(0.0, T, 500)
    Y_eval = spline.evaluate(t_eval)
    
    # Convert to numpy for plotting
    t_eval = np.array(t_eval)
    Y_eval = np.array(Y_eval)
    Y_knots = np.array(spline.Y)
    t_knots = np.array(spline.t_knots)

    # Plot batch 0
    plt.figure(figsize=(10, 5))

    # Plot both dimensions
    plt.plot(t_eval, Y_eval[0, :, 0], label="dim 0", linewidth=2)
    plt.plot(t_eval, Y_eval[0, :, 1], label="dim 1", linewidth=2)
    plt.scatter(t_knots, Y_knots[0, :, 0], color='red', s=60, zorder=5)
    plt.scatter(t_knots, Y_knots[0, :, 1], color='red', s=60, marker='x', zorder=5)
    plt.title("Linear Spline (Batch 0)")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()