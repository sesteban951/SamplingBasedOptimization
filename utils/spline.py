##
#
#  Different spline implementations.
#  
##

# jax imports
import jax
import jax.numpy as jnp


#############################################################
# ZOH Spline
#############################################################

class ZOH_Spline:
    """
    Zero-order hold spline.

    The knot points are uniformly over [0, T].
    Knot points are aligned to the left.
    """

    # spline parameters
    T: float                    # total time
    Y: jnp.ndarray              # knot points
    t_knots: jnp.ndarray        # knot times

    def __init__(self, Y0: jnp.ndarray, 
                       T: float):
        
        # get sizes from the initial knots
        self.B, self.K, self.dim = Y0.shape # (B, num_knots, dim)
        
        # set parameters
        self.Y = Y0

        # create the uniform knot times
        self.T = T
        self.t_knots = jnp.linspace(0.0, T, self.K) # (K,)

        # initialize jit functions
        self.evaluate = jax.jit(self._evaluate)

        print("ZOH Spline initialized.")

    # evaluate the spline at given times
    def _evaluate(self, times):
        """
        Given, a set of times, evaluate the ZOH spline at those times.
        
        Args: 
            times: jnp.array, shape (M,) - times to evaluate the spline at.
        Returns:
            Y_eval: jnp.array, shape (B, M, nu) - spline values at the given times.
        """

        # ensure times is an array and clip to [0, T]
        times = jnp.clip(times, 0.0, self.T)

        # Map times -> knot index k in [0, K-1]
        k = jnp.searchsorted(self.t_knots, times, side="right") - 1
        k = jnp.clip(k, 0, self.K - 1)  # (M,)

        # Gather along axis=1 (knot axis): (B, K, dim) -> (B, M, dim)
        return jnp.take(self.Y, k, axis=1)
    
    # update spline knot points
    def update_knots(self, Y_new: jnp.ndarray):
        """
        Update the spline knot points.

        Args:
            Y_new: jnp.array, shape (B, K, dim) - new knot points.
        """
        self.Y = Y_new

    
#############################################################
# EXAMPLE USAGE
#############################################################

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import numpy as np

    # create a ZOH spline
    B = 64
    K = 10
    nu = 2
    T = 5.0
    Y0 = jax.random.uniform(jax.random.PRNGKey(0), (B, K, nu), minval=-1.0, maxval=1.0)
    spline = ZOH_Spline(Y0, T)

    # print some info
    print("ZOH Spline:")
    print(f"  Total time: {spline.T}")
    print(f"  Batch size: {spline.B}")
    print(f"  Num knots: {spline.K}")
    print(f"  Dimensionality: {spline.dim}")
    print(f"  Knot times: {spline.t_knots}")

    t_eval = jnp.linspace(0.0, T, 500)
    Y_eval = spline.evaluate(t_eval)
    
    # convert to numpy for plotting
    t_eval = np.array(t_eval)
    Y_eval = np.array(Y_eval)
    t_knots = np.array(spline.t_knots)
    Y_knots = np.array(spline.Y)   # shape (B, K, nu)

    # plot batch 0
    plt.figure(figsize=(8,4))

    # continuous ZOH-evaluated curve
    plt.plot(t_eval, Y_eval[0, :, 0], label="dim 0")
    plt.plot(t_eval, Y_eval[0, :, 1], label="dim 1")

    # knot points (batch 0) in red
    plt.scatter(t_knots, Y_knots[0, :, 0], color='red', s=40, label="knots dim 0")
    plt.scatter(t_knots, Y_knots[0, :, 1], color='red', s=40, marker='x', label="knots dim 1")

    plt.title("ZOH Spline Evaluation (Batch 0)")
    plt.xlabel("Time")
    plt.ylabel("Spline Value")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


