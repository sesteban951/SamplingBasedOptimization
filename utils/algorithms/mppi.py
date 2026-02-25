##
#
# Model Predictive Path Integral (MPPI) for trajectory optimization.
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
from utils.algorithms.schedule import *
from utils.spline import zoh, linear, cubic, bezier, fourier
from utils.logging.logger import Logger, Logger_Config


#############################################################
# MPPI Optimizer
#############################################################


@dataclass
class MPPI_Config:

    # random key
    rng: jax.random.PRNGKey  # random key for sampling

    # planning horizon
    T: float         # total time horizon
    N_knots: int     # number of spline knot points

    # basic MPPI parameters
    iterations: int  # number of MPPI iterations
    lam: float       # temperature — lower = greedier, higher = more uniform
    sigma: float     # fixed noise std for sampling

    # spline params
    spline_type: str = "ZOH"   # "ZOH" | "Linear" | "Cubic" | "Bezier" | "Fourier"

    # sampling range scaling for both the torque and position knots
    initial_action_range_scale: float = 1.0  

    # covariance contraction params
    use_cov_contraction: bool = False    # whether to contract covariance
    sigma_min: float = 0.01              # minimum noise std

class MPPI(ABC):


    def __init__(self, model_config: Model_Config,
                       sim_config: ParallelSim_Config,
                       mppi_config: MPPI_Config,
                       log_config: Logger_Config = None):
        
        # create the parallel sim object 
        self.sim = ParallelSim(model_config, sim_config)

        # create the logger object
        if log_config is not None:
            self.logger = Logger(log_config)
        else:
            self.logger = None
        
        # store configs
        self.mppi_config = mppi_config
        
        # internal rng
        self.rng = mppi_config.rng

        # check that the input params make sense
        self._check_valid_params()

        # construct the time array for simulation
        self.N = int(round(self.mppi_config.T / self.sim.dt)) # integer number of sim steps
        self.t_sim = self.sim.dt * jnp.arange(self.N + 1)    # shape (N+1,)
        self.T_eff = self.N * self.sim.dt                    # effective total time

        # empty distribution
        self.mu = None
        self.Sigma = None
        self.sigma = None

        # initialize the spline knots points
        self._initialize_spline_knots()

        print("MPPI Optimizer initialized.")


    def _check_valid_params(self):
        """ 
        Check that the input params make sense. 
        """

        # check that there is atleast two envs
        if self.sim.B < 2:
            raise ValueError(f"Batch size [B = {self.sim.B}] must be at least 2.")
        
        # check positive interval
        if self.mppi_config.T <= 0.0:
            raise ValueError(f"Total time [T = {self.mppi_config.T}] must be positive.")
        
        # check number of knots
        if self.mppi_config.N_knots < 2:
            raise ValueError(f"Number of knots [N_knots = {self.mppi_config.N_knots}]"
                             f" must be greater than 1.")

        # check that temperature is positive
        if self.mppi_config.lam <= 0.0:
            raise ValueError(f"Temperature [lam = {self.mppi_config.lam}] must be positive.")
        
        # check that noise std is positive
        if self.mppi_config.sigma <= 0.0:
            raise ValueError(f"Noise std [sigma = {self.mppi_config.sigma}] must be positive.")
        
        # ensure that the initial action range scale makes sense
        if (   self.mppi_config.initial_action_range_scale <= 0.0
            or self.mppi_config.initial_action_range_scale > 1.0):
            raise ValueError(f"Initial action range scale [{self.mppi_config.initial_action_range_scale}]"
                             f" must be in (0, 1].")


    def _initialize_spline_knots(self):
        """ 
        Depending on the actuation mode, initialize the spline knot points
        within the position or control limits. NOTE: does not use a
        pre-defined mu and Sigma.
        """

        # initialize an empty array for the knot points
        y_size = (self.sim.B, self.mppi_config.N_knots, self.sim.nu)
        Y0 = jnp.zeros(y_size)

        # knot points represent desired positions
        if self.sim.use_pd == True:
            # get the position limits at actuated joints
            pos_limits = self.sim.pos_limits  # shape (nu, 2)

            # create initial knot points within the position limits
            for i in range(self.sim.nu):
                
                # get low and high limits
                pos_lo = pos_limits[i, 0] * self.mppi_config.initial_action_range_scale
                pos_hi = pos_limits[i, 1] * self.mppi_config.initial_action_range_scale

                # error out if no position limits are defined
                if abs(pos_hi) <1e-6 and abs(pos_lo) <1e-6:
                    raise ValueError(f"No position limits defined at actuated actuator index {i}.")

                # split the key
                self.rng, subkey = jax.random.split(self.rng)

                # replace the ith dimension of all knot points
                Y0 = Y0.at[:, :, i].set(
                    jax.random.uniform(
                        subkey,
                        shape=(self.sim.B, self.mppi_config.N_knots),
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
                tau_lo = ctrl_limits[i, 0] * self.mppi_config.initial_action_range_scale
                tau_hi = ctrl_limits[i, 1] * self.mppi_config.initial_action_range_scale

                # error out if no control limits are defined
                if abs(tau_hi) <1e-6 and abs(tau_lo) <1e-6:
                    raise ValueError(f"No control limits defined at actuated actuator index {i}.")

                # split the key
                self.rng, subkey = jax.random.split(self.rng)

                # replace the ith dimension of all knot points
                Y0 = Y0.at[:, :, i].set(
                    jax.random.uniform(
                        subkey,
                        shape=(self.sim.B, self.mppi_config.N_knots),
                        minval=tau_lo,
                        maxval=tau_hi
                    )
                )

        # create the spline object
        if self.mppi_config.spline_type == "ZOH":
            self.spline = zoh.ZOH_Spline(Y0, self.T_eff)
        elif self.mppi_config.spline_type == "Linear":
            self.spline = linear.Linear_Spline(Y0, self.T_eff)
        elif self.mppi_config.spline_type == "Cubic":
            self.spline = cubic.Cubic_Spline(Y0, self.T_eff)
        elif self.mppi_config.spline_type == "Bezier":
            self.spline = bezier.Bezier_Spline(Y0, self.T_eff)
        elif self.mppi_config.spline_type == "Fourier":
            self.spline = fourier.Fourier_Spline(Y0, self.T_eff, periodic=False)
        else:
            raise NotImplementedError(f"Spline type [{self.mppi_config.spline_type}] not implemented.")

        # update the spline knot points
        self.spline.update_knots(Y0)

        # update the distribution with the initial knot points
        self._update_distribution(Y0)


    def _sample_knot_points(self):
        """
        Sample knot points from the current distribution.

        Returns:
            Y_samples: jnp.array, shape (B, N_knots, nu) - sampled spline knot points.
        """

        # split the key
        self.rng, subkey = jax.random.split(self.rng)

        # cholesky decomposition of covariance
        L = jnp.linalg.cholesky(self.Sigma)  # shape (N_knots*nu, N_knots*nu)

        # sample from standard normal
        Y_std = jax.random.normal(
            subkey,
            shape=(self.sim.B, L.shape[0])
        )  # shape (B, N_knots*nu)

        # transform to desired distribution
        Y_flat = self.mu[None, :] + Y_std @ L.T  # shape (B, N_knots*nu)

        # reshape back to knot point matrices
        action_dim = L.shape[0] // self.mppi_config.N_knots
        Y_samples = jnp.reshape(Y_flat, (self.sim.B, self.mppi_config.N_knots, action_dim)) # shape (B, N_knots, nu)

        return Y_samples


    def _update_distribution(self, Y_samples, J=None):
        """
        Update the distribution parameters (mu and Sigma) based on the sampled knot points.

        Args:
            Y_samples: jnp.array, shape (B, N_knots, nu)
            J:         jnp.array, shape (B,) or None on init
        Returns:
            weights_normalized: jnp.array, shape (B,)
        """

        B = Y_samples.shape[0]
        Y_flat = jnp.reshape(Y_samples, (B, -1))  # shape (B, N_knots*nu)
        dim = self.mppi_config.N_knots * Y_samples.shape[2]

        # compute sigma for this iteration — always sets self.sigma
        if not self.mppi_config.use_cov_contraction or J is None:
            self.sigma = self.mppi_config.sigma
        else:
            sigma_max  = self.mppi_config.sigma
            sigma_min  = self.mppi_config.sigma_min
            progress   = self.itr / max(self.mppi_config.iterations - 1, 1)
            self.sigma = float(sigma_max * (sigma_min / sigma_max) ** progress)

        # initialization path
        if J is None:
            self.mu    = jnp.mean(Y_flat, axis=0)
            self.Sigma = (self.sigma ** 2) * jnp.eye(dim)
            return jnp.ones((B,)) / B

        # compute softmax weights
        weights            = jnp.exp(-(J - jnp.min(J)) / self.mppi_config.lam)
        weights_normalized = weights / jnp.sum(weights)

        # update mu and Sigma
        self.mu    = jnp.einsum('b,bd->d', weights_normalized, Y_flat)
        self.Sigma = (self.sigma ** 2) * jnp.eye(dim)

        return weights_normalized


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
        raise NotImplementedError("cost method must be implemented in MPPI subclass.")
    

    def optimize(self, q0, v0):
        """
        Perform MPPI optimization.

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

        # for printing iterations
        itr_width = len(str(self.mppi_config.iterations))

        try:
            # perform MPPI iterations
            for itr in range(self.mppi_config.iterations):

                # set the iterations
                self.itr = itr

                # evaluate the spline at simulation times
                y_val = self.spline.evaluate(self.t_sim[:-1])  # shape (B, N, nu)

                # do forward rollout
                q_log, v_log, tau_log = self.sim.rollout(q0, v0, y_val)
                q_log.block_until_ready()

                # compute costs
                J = self.cost(q_log, v_log, tau_log)  # shape (B,)
                J.block_until_ready()

                # update the distribution using softmax weights over all samples
                weights_normalized = self._update_distribution(self.spline.Y, J)

                # sample new knot points from the updated distribution
                Y_samples = self._sample_knot_points()  # shape (B, N_knots, nu)
                self.spline.update_knots(Y_samples)

                # compute the effective sample size (ESS) to monitor exploration vs exploitation
                ESS = 1.0 / jnp.sum(weights_normalized ** 2)
                ESS_percent = (ESS / self.sim.B) * 100.0

                # compute entropy of the distribution (up to a constant) to monitor convergence
                entropy = -jnp.sum(weights_normalized * jnp.log(weights_normalized + 1e-9))

                # record the best solution found so far
                J_min = jnp.min(J)
                J_min_val = float(J_min)
                J_opt_val = float(J_opt)
                if J_min_val < J_opt_val:
                    J_opt = J_min_val  # store as Python float for printing
                    idx_opt = int(jnp.argmin(J))
                    q_opt = q_log[idx_opt, :, :]
                    v_opt = v_log[idx_opt, :, :]
                    tau_opt = tau_log[idx_opt, :, :]

                # print iteration info
                sigma_str = f" | σ: {self.sigma:.4f}" if self.mppi_config.use_cov_contraction else ""
                print(f"Iteration {itr+1:0{itr_width}d}/{self.mppi_config.iterations} | "
                    f"J_mean: {jnp.mean(J):.2f} | "
                    f"J_best: {J_opt:.2f} | "
                    f"ESS: {ESS_percent:.1f}% | "
                    f"Entropy: {entropy:.2f}"
                    f"{sigma_str}"
                )

                # tensorboard logging
                if self.logger is not None:
                    # build metrics dict
                    metrics = {
                        "J_mean":  float(jnp.mean(J)),
                        "J_best":  float(J_opt),
                        "ESS":     float(ESS_percent),
                        "entropy": float(entropy),
                    }
                    if self.mppi_config.use_cov_contraction:
                        metrics["sigma"] = float(self.sigma)
                    
                    # log
                    self.logger.log(metrics, step=itr)

        finally:
            if self.logger is not None:
                self.logger.close()


        return q_opt, v_opt, tau_opt