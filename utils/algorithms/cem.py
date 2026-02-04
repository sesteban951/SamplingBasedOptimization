##
#
# Cross Entropy Method (CEM) for trajectory optimization.
#
##

# for base class
from __future__ import annotations
from abc import ABC, abstractmethod

# standard imports
from dataclasses import dataclass

# jax imports
import jax
import jax.numpy as jnp

# custom imports
from utils.simulation.simulation import *
from utils.spline.bezier import *
from utils.spline.zoh import *


#############################################################
# CEM Optimizer
#############################################################

# CEM configuration dataclass
@dataclass
class CrossEntropyMethod_Config:

    # random key
    rng: jax.random.PRNGKey  # random key for sampling

    # planning horizon
    T: float         # total time horizon
    N_knots: int     # number of spline knot points

    # basic CEM parameters
    iterations: int  # number of CEM iterations
    N_elite: int     # number of elite samples

    # spline params
    spline_type: str = "ZOH"   # "ZOH" | "Bezier"


# CEM optimizer class
class CrossEntropyMethod(ABC):

    # constructor
    def __init__(self, model_config: Model_Config,
                       sim_config: ParallelSim_Config,
                       cem_config: CrossEntropyMethod_Config):
        
        # create the parallel sim object 
        self.sim = ParallelSim(model_config, sim_config)
        
        # store configs
        self.cem_config = cem_config
        
        # internal rng
        self.rng = cem_config.rng

        # check that the input params make sense
        self._check_valid_params()

        # construct the time array for simulation
        N = int(round(self.cem_config.T / self.sim.dt)) # integer number of sim steps
        self.t_sim = self.sim.dt * jnp.arange(N + 1)    # shape (N+1,)
        self.T_eff = N * self.sim.dt                         # effective total time

        # initialize the spline knots points
        self._initialize_spline_knots()

        # empty distribution
        self.mu = None
        self.Sigma = None

        print("CEM Optimizer initialized.")

    # check that the input params make sense
    def _check_valid_params(self):
        """ 
        Check that the input params make sense. 
        """

        # check that there is atleast two envs
        if self.sim.B < 2:
            raise ValueError(f"Batch size [B = {self.sim.B}] must be at least 2.")
        
        # check positive interval
        if self.cem_config.T <= 0.0:
            raise ValueError(f"Total time [T = {self.cem_config.T}] must be positive.")
        
        # check number of knots
        if self.cem_config.N_knots < 2:
            raise ValueError(f"Number of knots [N_knots = {self.cem_config.N_knots}]"
                             f" must be greater than 1.")
        
        # check that there are enough elite samples
        if self.cem_config.N_elite < 2:
            raise ValueError(f"Number of elite samples [N_elite = {self.cem_config.N_elite}]"
                             f" must be at least 2.")

        # check that number of elite samples is less than batch size
        if self.cem_config.N_elite > self.sim.B:
            raise ValueError(f"Number of elite samples [N_elite = {self.cem_config.N_elite}]" 
                             f" must be less than parallel sim envs [B = {self.sim.B}].")


    # initialize the spline knot points
    def _initialize_spline_knots(self):
        """ 
        Depending on the actuation mode, initialize the spline knot points
        within the position or control limits. NOTE: does not use a
        pre-defined mu and Sigma.
        """

        # initialize an empty array for the knot points
        y_size = (self.sim.B, self.cem_config.N_knots, self.sim.nu)
        Y0 = jnp.zeros(y_size)

        # knot points represent desired positions
        if self.sim.use_pd == True:
            # get the position limits at actuated joints
            pos_limits = self.sim.pos_limits  # shape (nu, 2)

            # create initial knot points within the position limits
            for i in range(self.sim.nu):
                
                # get low and high limits
                pos_lo = pos_limits[i, 0]
                pos_hi = pos_limits[i, 1]

                # error out if no position limits are defined
                if abs(pos_hi) <1e-6 and abs(pos_lo) <1e-6:
                    raise ValueError(f"No position limits defined at actuated actuator index {i}.")

                # split the key
                self.rng, subkey = jax.random.split(self.rng)

                # replace the ith dimension of all knot points
                Y0 = Y0.at[:, :, i].set(
                    jax.random.uniform(
                        subkey,
                        shape=(self.sim.B, self.cem_config.N_knots),
                        minval=pos_lo,
                        maxval=pos_hi
                    )
                )

        # knot points represent direct torques
        else:
            # get the control limits at actuated joints
            ctrl_limits = self.sim.ctrl_limits  # shape (nu, 2)

            # create initial knot points within the control limits
            for i in range(self.sim.nu):
                
                # get low and high limits
                tau_lo = ctrl_limits[i, 0]
                tau_hi = ctrl_limits[i, 1]

                # error out if no control limits are defined
                if abs(tau_hi) <1e-6 and abs(tau_lo) <1e-6:
                    raise ValueError(f"No control limits defined at actuated actuator index {i}.")

                # split the key
                self.rng, subkey = jax.random.split(self.rng)

                # replace the ith dimension of all knot points
                Y0 = Y0.at[:, :, i].set(
                    jax.random.uniform(
                        subkey,
                        shape=(self.sim.B, self.cem_config.N_knots),
                        minval=tau_lo,
                        maxval=tau_hi
                    )
                )

        # create the spline object
        if self.cem_config.spline_type == "ZOH":
            self.spline = ZOH_Spline(Y0, self.T_eff)
        elif self.cem_config.spline_type == "Bezier":
            self.spline = Bezier_Spline(Y0, self.T_eff)
        else:
            raise NotImplementedError(f"Spline type [{self.cem_config.spline_type}] not implemented.")

        # update the spline knot points
        self.spline.update_knots(Y0)

    
    # sample knot_points from the current distribution
    def _sample_knot_points(self):
        """
        Sample knot points from the current distribution.

        Returns:
            Y_samples: jnp.array, shape (B, N_knots, nu) - sampled spline knot points.
        """

        # split the key
        self.rng, subkey = jax.random.split(self.rng)

        # numerical conditioning
        epsilon = 1e-6
        Sigma_cond = 0.5 * (self.Sigma + self.Sigma.T) + epsilon * jnp.eye(self.Sigma.shape[0], dtype=self.Sigma.dtype)

        # cholesky decomposition of covariance
        L = jnp.linalg.cholesky(Sigma_cond)  # shape (N_knots*nu, N_knots*nu)

        # sample from standard normal
        Y_std = jax.random.normal(
            subkey,
            shape=(self.sim.B, self.cem_config.N_knots * self.sim.nu)
        )  # shape (B, N_knots*nu)

        # transform to desired distribution
        Y_flat = self.mu[None, :] + Y_std @ L.T  # shape (B, N_knots*nu)

        # reshape back to knot point matrices
        Y_samples = jnp.reshape(Y_flat, (self.sim.B, self.cem_config.N_knots, self.sim.nu)) # shape (B, N_knots, nu)

        return Y_samples


    # update the distribution based on elite samples
    def _update_distribution(self, Y_elite):
        """
        Update the disitrbution based on elite samples.
        
        Args:
            Y_elite: jnp.array, shape (N_elite, N_knots, nu) - elite spline knot points.
        """

        # NOTE: consider adding smoothing to the distribution update:
        # alpha = 0.1  # or 0.2
        # self.mu = (1 - alpha) * self.mu + alpha * mu_
        # self.Sigma = (1 - alpha) * self.Sigma + alpha * Sigma_

        # Flatten each elite's knot matrix (N_knots, nu) into a vector (N_knots*nu) in row-major order.
        # Example: if Y_elite[k] = [[1,2,3],
        #                           [4,5,6]]  (N_knots=2, nu=3)
        # then Y_flat[k] = [1,2,3,4,5,6].
        K = Y_elite.shape[0]
        Y_flat = jnp.reshape(Y_elite, (K, -1))  # shape (N_elite, N_knots * nu)

        # compute mean along the elite samples
        mu_ = jnp.mean(Y_flat, axis=0)  # shape (N_knots * nu,)

        # center the knots about the mean
        Y_centered = Y_flat - mu_[None, :]  # shape (N_elite, N_knots * nu)

        # compute the unbiased sample covariance
        Sigma_ = (Y_centered.T @ Y_centered) / (K - 1)  # shape (N_knots * nu, N_knots * nu)

        # for numerical stability, add a small value to the diagonal
        epsilon = 1e-6
        Sigma_ = Sigma_ + epsilon * jnp.eye(Sigma_.shape[0], dtype=Sigma_.dtype)

        # store the updated distribution
        self.mu = mu_
        self.Sigma = Sigma_


    # cost function
    @abstractmethod
    def cost(self, q, v, tau):
        """
        Cost function to evaluate the rollouts.

        Args:
            q: jnp.array, shape (B, N+1, nq) - generalized positions trajectory.
            v: jnp.array, shape (B, N+1, nv) - generalized velocities trajectory.
            tau: jnp.array, shape (B, N, nu) - control inputs trajectory.
        Returns:
            costs: jnp.array, shape (B,) - cost for each rollout.
        """
        raise NotImplementedError("cost method must be implemented in CEM subclass.")
    

    # CEM optimization
    def optimize(self, q0, v0):
        """
        Perform CEM optimization.

        Args:
            q0: jnp.array, shape (B, nq) - initial generalized positions.
            v0: jnp.array, shape (B, nv) - initial generalized velocities.
        Returns:
            q_opt: jnp.array, shape (N+1, nq) - optimal generalized positions trajectory.
            v_opt: jnp.array, shape (N+1, nv) - optimal generalized velocities trajectory.
            tau_opt: jnp.array, shape (N, nu) - optimal control inputs trajectory.
        """

        # initialize the optimal solution
        J_opt = jnp.inf
        q_opt = None
        v_opt = None
        tau_opt = None

        # perform CEM iterations
        for itr in range(self.cem_config.iterations):

            # evaluate the spline at simulation times
            y_val = self.spline.evaluate(self.t_sim[:-1])  # shape (B, N, nu)

            # do forward rollout
            q_log, v_log, tau_log = self.sim.rollout(q0, v0, y_val)
            q_log.block_until_ready()

            # compute costs
            J = self.cost(q_log, v_log, tau_log)  # shape (B,)
            J.block_until_ready()

            # select elite samples
            _, elite_idx = jax.lax.top_k(-J, self.cem_config.N_elite)
            
            # select the elite splines
            Y_elite = jnp.take(self.spline.Y, elite_idx, axis=0)  # shape (N_elite, N_knots, nu)

            # update the distribution
            self._update_distribution(Y_elite)

            # sample new knot points from the updated distribution
            Y_samples = self._sample_knot_points()  # shape (B, N_knots, nu)
            self.spline.update_knots(Y_samples)

            # compute the norm of the covariance for monitoring
            cov_norm = jnp.linalg.norm(self.Sigma, ord='fro')

            # record the best solution found so far
            if J.min() < J_opt:
                J_opt = J.min()
                idx_opt = jnp.argmin(J)
                q_opt = q_log[idx_opt, :, :]
                v_opt = v_log[idx_opt, :, :]
                tau_opt = tau_log[idx_opt, :, :]

            print(f"Iteration {itr+1}/{self.cem_config.iterations}, Best Cost: {J.min():.4f}, Cov Norm: {cov_norm:.4f}")

        return q_opt, v_opt, tau_opt
    