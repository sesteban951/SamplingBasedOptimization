##
#
#  Cubic spline with continuous second derivatives (Natural boundary conditions).
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
# Cubic Spline with Continuous Second Derivatives
#############################################################

class Cubic_Spline(Base_Spline):
    """
    Cubic spline with continuous second derivatives.
    
    The knot points are uniformly distributed over [0, T].
    Interior segments have continuous values, first, and second derivatives.
    
    Uses natural boundary conditions: second derivatives are zero at endpoints.
    """

    def __init__(self, Y0: jnp.ndarray, 
                       T: float):
        
        # initialize parent class
        super().__init__(Y0, T)

        # cubic spline requires at least 2 knots (1 segment)
        if self.K < 2:
            raise ValueError(f"Cubic spline requires at least 2 knots. Got K={self.K}.")
        
        # create uniform knot times
        self.t_knots = jnp.linspace(0.0, self.T, self.K)  # shape (K,)
        self.dt = self.T / (self.K - 1)  # spacing between knots

        # compute cubic polynomial coefficients
        # shape: (B, K-1, 4, dim) for each segment
        # polynomials are: a + b*(t-t_i) + c*(t-t_i)^2 + d*(t-t_i)^3
        self.coeffs = self._polynomial_coefficients()

        print(f"Cubic Spline initialized.")

    def _polynomial_coefficients(self):
        """
        Compute cubic polynomial coefficients for each segment.
        
        Returns:
            coeffs: jnp.array, shape (B, K-1, 4, dim)
                    coeffs[b, i, :, d] = [a, b, c, d] for segment i, batch b, dimension d
        """
        
        # We'll compute coefficients for each dimension independently
        # Then stack them together
        all_coeffs = []
        
        for d in range(self.dim):

            # Extract this dimension: shape (B, K)
            y = self.Y[:, :, d]
            
            # Solve for second derivatives at each knot
            # This gives us M values at each knot point
            M = self._solve_for_second_derivatives(y)
            
            # Build cubic coefficients from M values
            # For segment i (between knots i and i+1):
            # S_i(t) = a + b*(t-t_i) + c*(t-t_i)^2 + d*(t-t_i)^3
            h = self.dt  # uniform spacing
            
            # Coefficients for each segment
            a = y[:, :-1]  # shape (B, K-1)
            d = (M[:, 1:] - M[:, :-1]) / (6 * h)  # shape (B, K-1)
            b = (y[:, 1:] - y[:, :-1]) / h - h * (2*M[:, :-1] + M[:, 1:]) / 6  # shape (B, K-1)
            c = M[:, :-1] / 2  # shape (B, K-1)
            
            # Stack coefficients: shape (B, K-1, 4)
            coeffs_d = jnp.stack([a, b, c, d], axis=2)
            all_coeffs.append(coeffs_d)
        
        # Stack all dimensions: shape (B, K-1, 4, dim)
        coeffs = jnp.stack(all_coeffs, axis=3)
        
        return coeffs

    def _solve_for_second_derivatives(self, y):
        """
        Solve tridiagonal system for second derivatives M at each knot.
        Natural boundary conditions: M[0] = M[-1] = 0
        
        Args:
            y: jnp.array, shape (B, K), values at knots for one dimension
            
        Returns:
            M: jnp.array, shape (B, K), second derivatives at each knot
        """
        B, K = y.shape
        h = self.dt
        
        if K == 2:
            # Just a line segment
            return jnp.zeros((B, K))
        
        # Right-hand side
        rhs = jnp.zeros((B, K))
        rhs = rhs.at[:, 1:-1].set(6 / h**2 * (y[:, :-2] - 2*y[:, 1:-1] + y[:, 2:]))
        
        # Solve tridiagonal system
        M = self._solve_tridiagonal(rhs, K)
        
        return M

    def _solve_tridiagonal(self, rhs, K):
        """
        Solve tridiagonal system for natural boundary conditions.
        Thomas algorithm for tridiagonal matrix.
        
        Matrix is: [4 1 0 ... ; 1 4 1 0 ... ; ... ; 0 ... 1 4]
        for interior points M[1] to M[K-2]
        """
        B = rhs.shape[0]
        M = jnp.zeros((B, K))
        
        n = K - 2  # number of interior unknowns
        if n == 0:
            return M
        
        # Diagonals: lower=1, main=4, upper=1
        c_prime = jnp.zeros((B, n))
        d_prime = jnp.zeros((B, n))
        
        # Forward sweep
        c_prime = c_prime.at[:, 0].set(1.0 / 4.0)
        d_prime = d_prime.at[:, 0].set(rhs[:, 1] / 4.0)
        
        for i in range(1, n):
            denom = 4.0 - c_prime[:, i-1]
            c_prime = c_prime.at[:, i].set(1.0 / denom)
            d_prime = d_prime.at[:, i].set((rhs[:, i+1] - d_prime[:, i-1]) / denom)
        
        # Back substitution
        M = M.at[:, -2].set(d_prime[:, -1])
        for i in range(n-2, -1, -1):
            M = M.at[:, i+1].set(d_prime[:, i] - c_prime[:, i] * M[:, i+2])
        
        return M
    

    def update_knots(self, Y_new: jnp.ndarray):
        """
        Update the spline knot points and recompute polynomial coefficients.

        Args:
            Y_new: jnp.array, shape (B, K, dim), new knot points.
        """
        # Call parent class method to update Y
        super().update_knots(Y_new)
        
        # Recompute polynomial coefficients with new knot points
        self.coeffs = self._polynomial_coefficients()


    def evaluate(self, times: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluate the cubic spline at given times.

        Args:
            times: jnp.array, shape (M,), times to evaluate the spline at.
        Returns:
            Y_eval: jnp.array, shape (B, M, dim), spline values at the given times.
        """
        # Clip times to [0, T]
        times = jnp.clip(times, 0.0, self.T)
        
        # Find which segment each time belongs to
        # segment_idx = floor((time / T) * (K-1))
        segment_idx = jnp.floor(times / self.dt).astype(jnp.int32)
        segment_idx = jnp.clip(segment_idx, 0, self.K - 2)  # last segment is K-2
        
        # Local time within segment: t - t_i
        local_t = times - self.t_knots[segment_idx]  # shape (M,)
        
        # Evaluate polynomial: a + b*dt + c*dt^2 + d*dt^3
        # coeffs shape: (B, K-1, 4, dim)
        # We need coeffs[:, segment_idx, :, :]
        
        # Gather coefficients for each time point
        coeffs_for_times = self.coeffs[:, segment_idx, :, :]  # shape (B, M, 4, dim)
        
        # Compute powers of local_t
        powers = jnp.stack([
            jnp.ones_like(local_t),
            local_t,
            local_t**2,
            local_t**3
        ], axis=0)  # shape (4, M)
        
        # Evaluate: sum over polynomial coefficients
        # (B, M, 4, dim) * (4, M) -> (B, M, dim)
        Y_eval = jnp.einsum('bmcd,cm->bmd', coeffs_for_times, powers)
        
        return Y_eval


#############################################################
# EXAMPLE USAGE
#############################################################

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import numpy as np
    import time

    seed = int(time.time())

    # Create a cubic spline
    B = 64
    K = 20
    nu = 5
    T = 10.0
    Y0 = jax.random.uniform(jax.random.PRNGKey(seed), (B, K, nu), minval=-1.0, maxval=1.0)
    
    # Create natural cubic spline
    spline = Cubic_Spline(Y0, T)

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
    plt.title("Natural Cubic Spline (Batch 0)")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()