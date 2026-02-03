##
#
#  Different spline implementations.
#  
##

# for base class
from __future__ import annotations
from abc import ABC, abstractmethod

# standard imports
import numpy as np

# jax imports
import jax
import jax.numpy as jnp


#############################################################
# Base Spline Class
#############################################################

class Base_Spline(ABC):
    """
    Base class for control/trajectory splines.
    """

    # spline parameters
    Y: jnp.ndarray  # spline points
    T: float        # total time horizon
    B: int          # batch size
    K: int          # number of knots
    dim: int        # dimensionality of the spline

    # initialize the spline class
    def __init__(self, Y0: jnp.ndarray, T: float):

        # set the spline parameters parameters
        self.Y = Y0
        self.T = np.round(T, decimals=6)

        # sizes from initial knots
        self.B, self.K, self.dim = Y0.shape

    # abstract method to evaluate the spline
    @abstractmethod
    def evaluate(self, times: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluate the spline at given times.

        Args:
            times: jnp.array, shape (M,), times to evaluate the spline at.

        Returns:
            Y_eval: jnp.array, shape (B, M, dim), spline values at the given times.
        """
        raise NotImplementedError("evaluate method must be implemented in spline subclass.")

    # update spline knot points
    def update_knots(self, Y_new: jnp.ndarray):
        """
        Update the spline knot points.

        Args:
            Y_new: jnp.array, shape (B, K, dim), new knot points.
        """

        # check that the new shape matches
        if Y_new.shape != self.Y.shape:
            raise ValueError(f"Y_new shape {Y_new.shape} != existing shape {self.Y.shape}")
        
        # set the new knots
        self.Y = Y_new

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
    def evaluate(self, times):

        # ensure times is an array and clip to [0, T]
        times = jnp.clip(times, 0.0, self.T)

        # interval index k = floor(t / dt)
        k = jnp.floor(times / self.dt).astype(jnp.int32)
        k = jnp.clip(k, 0, self.K - 1)

        # Gather along axis=1 (knot axis): (B, K, dim) -> (B, M, dim)
        return jnp.take(self.Y, k, axis=1)
    

#############################################################
# Bezier Curve
#############################################################

# class Bezier_Spline:
#     """
#     Bezier curve spline.

#     The knot points are uniformly over [0, T].
#     Knot points are aligned to the left.
#     """

#     def __init__(self, Y0: jnp.ndarray, 
#                        T: float):
        
#         # get sizes from the initial knots
#         self.B, self.K, self.dim = Y0.shape # (B, num_knots, dim)
        
#         # set parameters
#         self.Y = Y0

#         # create the uniform knot times
#         self.T = T

#         # initialize jit functions
#         # self.evaluate = jax.jit(self._evaluate)
#         self.evaluate = self._evaluate

#         print("ZOH Spline initialized.")

#     # evaluate the spline at given times
#     def _evaluate(self, times):
#         """
#         Given, a set of times, evaluate the ZOH spline at those times.
        
#         Args: 
#             times: jnp.array, shape (M,) - times to evaluate the spline at.
#         Returns:
#             Y_eval: jnp.array, shape (B, M, nu) - spline values at the given times.
#         """

#         # ensure times is an array and clip to [0, T]
#         times = jnp.clip(times, 0.0, self.T)

#         # Map times -> knot index k in [0, K-1]
#         k = jnp.searchsorted(self.t_knots, times, side="right") - 1
#         k = jnp.clip(k, 0, self.K - 1)  # (M,)

#         # Gather along axis=1 (knot axis): (B, K, dim) -> (B, M, dim)
#         return jnp.take(self.Y, k, axis=1)



#############################################################
# EXAMPLE USAGE
#############################################################

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import numpy as np

    # create a ZOH spline
    B = 64
    K = 50
    nu = 2
    T = 10.0
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


