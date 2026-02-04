##
#
#  Base spline implementation.
#  
##

# for base class
from __future__ import annotations
from abc import ABC, abstractmethod

# standard imports
import numpy as np

# jax imports
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

        M = len(times)

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