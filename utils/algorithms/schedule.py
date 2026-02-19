##
#
# Schedules for Sampling Based Optimization algorithms.
# https://www.desmos.com/calculator/hddzxvbzo0
# 
##

# standard imports
import numpy as np

######################################################
# LINEAR
######################################################

def linear_schedule(itr, I, alpha_max=1.0):
    """
    Linear schedule.
    α(i) = αₘ * (1 − i / I)
    where: αₘ ∈ [0, 1] is the maximum value of the schedule coefficient
           I is the total number of iterations

    Args:
        itr: (int) Current iteration number.
        I: (int) Total number of iterations.
        alpha_max: (float) Maximum value of the schedule coefficient.
    Returns:
        alpha: (float) Schedule coefficient.
    """

    # ensure that total iterations is positive
    if I <= 0:
        raise ValueError("Total iterations (I) must be positive.")

    # clip the maximum value
    alpha_max = np.clip(alpha_max, 0.0, 1.0)

    # compute the schedule coefficient
    alpha = alpha_max * (1 - itr / I)

    return alpha

######################################################
# EXPONENTIAL
######################################################

def exponential_schedule(itr, I, alpha_max=1.0, lam=5.0):
    """
    Exponential schedule.
    α(i) = αₘ * [ e^(−λ * i /I) − e^(−λ) ] / [ 1 − e^(−λ) ]
    where: αₘ ∈ [0, 1] is the maximum value of the schedule coefficient
           λ ≠ 0 is the decay rate

    Args:
        itr: (int) Current iteration number.
        I: (int) Total number of iterations.
        alpha_max: (float) Maximum value of the schedule coefficient.
        lam: (float) Decay rate (λ).
    Returns:
        alpha: (float) Schedule coefficient.
    """
    # ensure that total iterations is positive
    if I <= 0:
        raise ValueError("Total iterations (I) must be positive.")
    
    # clip the maximum value
    alpha_max = np.clip(alpha_max, 0.0, 1.0)

    # compute the schedule coefficient
    alpha = alpha_max * (np.exp(-lam * itr) - np.exp(-lam * I)) / (1 - np.exp(-lam * I))

    return alpha

######################################################
# COSINE
######################################################

def cosine_schedule(itr, I, alpha_max=1.0):
    """
    Cosine schedule.
    α(i) = 0.5 * αₘ * (1 + cos(π * i / I))
    where: αₘ ∈ [0, 1] is the maximum value of the schedule coefficient

    Args:
        itr: (int) Current iteration number.
        I: (int) Total number of iterations.
        alpha_max: (float) Maximum value of the schedule coefficient.
    Returns:
        alpha: (float) Schedule coefficient.
    """
    # ensure that total iterations is positive
    if I <= 0:
        raise ValueError("Total iterations (I) must be positive.")
    
    # clip the maximum value
    alpha_max = np.clip(alpha_max, 0.0, 1.0)

    # compute the schedule coefficient
    alpha = 0.5 * alpha_max * (1 + np.cos(np.pi * itr / I))

    return alpha

######################################################
# TANH
######################################################

def tanh_schedule(itr, I, alpha_max=1.0, sigma=5.0):
    """
    Tanh schedule.
    α(i) = 0.5 * αₘ * (1 − tanh(σ * (i / I − 0.5)) / tanh(σ * 0.5))
    where: αₘ ∈ [0, 1] is the maximum value of the schedule coefficient
           σ > 0 is the steepness of the tanh function

    Args:
        itr: (int) Current iteration number.
        I: (int) Total number of iterations.
        alpha_max: (float) Maximum value of the schedule coefficient.
        sigma: (float) Steepness of the tanh function.
    Returns:
        alpha: (float) Schedule coefficient.
    """

    # ensure that total iterations is positive
    if I <= 0:
        raise ValueError("Total iterations (I) must be positive.")
    
    # clip the maximum value
    alpha_max = np.clip(alpha_max, 0.0, 1.0)

    # compute the schedule coefficient
    term1 = np.tanh(sigma * (itr/I - 0.5))
    term2 = np.tanh(sigma * 0.5)
    alpha = 0.5 * alpha_max * (1 - term1 / term2)

    return alpha

######################################################
# EXAMPLE USAGE
######################################################

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    
    I_tot = 100
    iters = np.arange(I_tot)
    alpha_max = 1.0

    # ------------------- Linear schedule -------------------
    alpha_linear = np.zeros(I_tot)
    for i in range(I_tot):
        alpha_linear[i] = linear_schedule(i, I_tot, alpha_max=alpha_max)

    # ------------------- Exp schedule -------------------
    alpha_exp = np.zeros(I_tot)
    for i in range(I_tot):
        alpha_exp[i] = exponential_schedule(i, I_tot, alpha_max=alpha_max, lam=0.05)

    # ------------------- Cosine schedule -------------------
    alpha_cosine = np.zeros(I_tot)
    for i in range(I_tot):
        alpha_cosine[i] = cosine_schedule(i, I_tot, alpha_max=alpha_max)

    # ------------------- Tanh schedule -------------------
    alpha_tanh = np.zeros(I_tot)
    for i in range(I_tot):
        alpha_tanh[i] = tanh_schedule(i, I_tot, alpha_max=alpha_max, sigma=5.0)

    # ------------------- Plotting -------------------
    plt.figure()
    plt.plot(iters, alpha_linear, label='Linear Schedule')
    plt.plot(iters, alpha_exp, label='Exponential Schedule')
    plt.plot(iters, alpha_cosine, label='Cosine Schedule')
    plt.plot(iters, alpha_tanh, label='Tanh Schedule')
    plt.xlabel('Iteration')
    plt.ylabel('Alpha')
    plt.title('Schedules')
    plt.legend()
    plt.grid()
    plt.show()

