##
#
#  Zero-order hold spline implementations.
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
# Zero-Order Hold Spline
#############################################################

class ZOH_Spline(Base_Spline):
    """
    Zero-order hold spline.

    The knot points are uniformly over [0, T].
    Knot points are aligned to the left.
    """

    def __init__(self, Y0: jnp.ndarray, 
                       T: float):
        
        # initialize parent class
        super().__init__(Y0, T)

        # create the uniform knot times
        self.dt = self.T / self.K
        self.t_knots = jnp.arange(self.K) * self.dt   # [0, dt, 2dt, ... (K-1)dt]

        print("ZOH Spline initialized.")

    # evaluate the spline at given times
    def evaluate(self, times: jnp.ndarray) -> jnp.ndarray:

        # ensure times is an array and clip to [0, T]
        times = jnp.clip(times, 0.0, self.T)

        # interval index k = floor(t / dt)
        k = jnp.floor(times / self.dt).astype(jnp.int32)
        k = jnp.clip(k, 0, self.K - 1)

        # Gather along axis=1 (knot axis): (B, K, dim) -> (B, M, dim)
        return jnp.take(self.Y, k, axis=1)
    

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
    spline = ZOH_Spline(Y0, T)

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
    t_knots = np.array(spline.t_knots)  # shape (K,)

    # plot batch 0
    plt.figure(figsize=(8,4))

    # continuous ZOH-evaluated curve
    plt.plot(t_eval, Y_eval[0, :, 0], label="dim 0")
    plt.plot(t_eval, Y_eval[0, :, 1], label="dim 1")
    plt.scatter(t_knots, Y_knots[0, :, 0], color='red', s=40, label="knots dim 0")
    plt.scatter(t_knots, Y_knots[0, :, 1], color='red', s=40, marker='x', label="knots dim 1")
    plt.title("ZOH Spline Evaluation (Batch 0)")
    plt.xlabel("Time")
    plt.ylabel("Spline Value")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
