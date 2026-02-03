##
#
#  Bezier curve implementation.
#  
##

# for base class
from utils.spline.base import Base_Spline

# standard imports
import numpy as np
import math

# jax imports
import jax
import jax.numpy as jnp


#############################################################
# Bezier Curve
#############################################################

class Bezier_Spline(Base_Spline):
    """
    Bezier curve.

    Here, Y is the control points with shape (B, K, dim) where K are num of control points.
    The curve is smooth polynomial.
    """

    def __init__(self, Y0: jnp.ndarray, 
                       T: float):
        
        # initialize parent class
        super().__init__(Y0, T)

        # bezier curve requires at least 2 control points
        if self.K < 2:
            raise ValueError(f"Bezier spline requires at least 2 control points (K >= 2)."
                             f" Got K={self.K}.")

        # degree of the polynomial
        self.deg = self.K - 1  

        # precompute binomial coefficients
        self.coeffs = self._binomial_coefficients(self.K - 1)
    
        print("Bezier Spline initialized.")

    # compute the binomial coefficients (n choose k) for k=0..n
    def _binomial_coefficients(self, n: int) -> jnp.ndarray:
        """
        Compute binomial coefficients C(n, k) for k=0..n. (n choose k is combination)

        Args:
            n: int, degree of the polynomial.
        Returns:
            coeffs: jnp.array, shape (n+1,), binomial coefficients.
        """
        # choose values (k = 0, 1, ..., n)
        k = np.arange(n + 1, dtype=np.int64)

        # binomial coefficients
        # WARNING: high num knots can give rise to huge coeff values
        coeffs = jnp.array([math.comb(n, ki) for ki in k], dtype=jnp.float64)

        return coeffs
    
    # evaluate the spline at given times
    def evaluate(self, times: jnp.ndarray) -> jnp.ndarray:

        # clip times to [0, T] and normalize to [0, 1]
        times_ = jnp.clip(times, 0.0, self.T)
        t = jnp.clip(times_ / self.T, 0.0, 1.0)  # (M,)

        # Bernstein polynomial basis
        # B_i(t) = C(n, i) * t^i * (1 - t)^(n - i), for i = 0..n
        i = jnp.arange(self.K)                                              # (K,)
        t_powers = jnp.power(t[None, :], i[:, None])                        # (K, M)
        omt_powers = jnp.power((1.0 - t)[None, :], (self.deg - i)[:, None]) # (K, M)
        basis = self.coeffs[:, None] * t_powers * omt_powers                # (K, M)

        # evaluate the Bezier curve: (K,M) x (B, K, dim) -> (B, M, dim)
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

    # create a ZOH spline
    B = 64
    K = 5
    nu = 2
    T = 10.0
    Y0 = jax.random.uniform(jax.random.PRNGKey(seed), (B, K, nu), minval=-1.0, maxval=1.0)
    spline = Bezier_Spline(Y0, T)

    # print some info
    print(f"  Total time: {spline.T}")
    print(f"  Batch size: {spline.B}")
    print(f"  Num knots: {spline.K}")
    print(f"  Dimensionality: {spline.dim}")

    t_eval = jnp.linspace(0.0, T, 500)
    Y_eval = spline.evaluate(t_eval)
    
    # convert to numpy for plotting
    t_eval = np.array(t_eval)
    Y_eval = np.array(Y_eval)
    Y_knots = np.array(spline.Y)   # shape (B, K, nu)

    # plot batch 0
    plt.figure(figsize=(8,4))

    # plot batch 0 as a parametric curve in 2D
    plt.plot(Y_eval[0, :, 0], Y_eval[0, :, 1], color='blue', linewidth=2, label="Bezier curve")
    plt.plot(Y_knots[0, :, 0], Y_knots[0, :, 1], color='gray', linewidth=1, linestyle='--')
    plt.scatter(Y_knots[0, :, 0], Y_knots[0, :, 1], color='red', s=60, zorder=5, label="Control points")
    for k in range(K):
        plt.annotate(f"P{k}", (Y_knots[0, k, 0], Y_knots[0, k, 1]),
                     textcoords="offset points", xytext=(8, 8), fontsize=10)
    plt.title("Bezier Curve (Batch 0)")
    plt.xlabel("dim 0")
    plt.ylabel("dim 1")
    plt.legend()
    plt.grid()
    plt.axis("equal")  # keep aspect ratio so the curve isn't distorted
    plt.tight_layout()
    plt.show()
