##
#
# Annealing schedules for Sampling Based Optimization algorithms.
#
##

# standard imports
import numpy as np

# JAX imports
import jax.numpy as jnp


######################################################
# LINEAR
######################################################

def linear_annealing(itr, I, alpha_m=1.0):
    """
    Linear annealing schedule.
    α(i) = αₘ * (1 − i / I)
    where: αₘ ∈ [0, 1] is the maximum value of the annealing coefficient
           I is the total number of iterations

    Args:
        iteration: (int) Current iteration number.
    Returns:
        alpha: (float) Annealing coefficient.
    """
    # clip the maximum value
    alpha_m = np.clip(alpha_m, 0.0, 1.0)

    # compute the annealing coefficient
    alpha = alpha_m * (1 - itr / I)

    return alpha

######################################################
# EXPONENTIAL
######################################################

def exponential_annealing(itr, I, alpha_m=1.0, lam=0.4):
    """
    Exponential annealing schedule.
    α(i) = αₘ * [ e^(−λ * i) − e^(−λ * I) ] / [ 1 − e^(−λ * I) ]
    where: αₘ ∈ [0, 1] is the maximum value of the annealing coefficient
           λ ≠ 0 is the decay rate

    Args:
        iteration: (int) Current iteration number.
    Returns:
        alpha: (float) Annealing coefficient.
    """
    # clip the maximum value
    alpha_m = np.clip(alpha_m, 0.0, 1.0)

    # compute the annealing coefficient
    alpha = alpha_m * (np.exp(-lam * itr) - np.exp(-lam * I)) / (1 - np.exp(-lam * I))

    return alpha
