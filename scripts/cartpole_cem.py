##
#
# Cross Entropy Method (CEM) for trajectory optimization.
#
##

# standard imports
import numpy as np
from dataclasses import dataclass
import time

# jax imports
import jax
import jax.numpy as jnp

# cusotm imports
from utils.simulation.simulation import *
from utils.spline.bezier import *
from utils.spline.zoh import *


#############################################################
# CEM Optimizer
#############################################################

@dataclass
class CEM_Config:

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
class CEM_Optimizer:

    def __init__(self, model_config: ModelConfig,
                       sim_config: ParallelSimConfig,
                       cem_config: CEM_Config):
        
        # store configs
        self.cem_config = cem_config

        # internal rng
        self.rng = cem_config.rng

        # create the parallel sim object 
        self.sim = ParallelSim(model_config, sim_config)

        # check that the input params make sense
        self._check_valid_params()

        # initialize the spline knots points
        self._initialize_spline_knots()

        # construct the time array for simulation
        N = int(round(self.cem_config.T / self.sim.dt)) # integer number of sim steps
        self.t_sim = self.sim.dt * jnp.arange(N + 1)    # shape (N+1,)

        # empty distribution
        self.mu = None
        self.Sigma = None

        print("CEM Optimizer initialized.")

    # check that the input params make sense
    def _check_valid_params(self):

        # check that there is atleast two envs
        if self.sim.B < 2:
            raise ValueError(f"Batch size [B = {self.sim.B}] must be at least 2.")
        
        # check positive interval
        if self.cem_config.T <= 0.0:
            raise ValueError(f"Total time [T = {self.cem_config.T}] must be positive.")
        
        # check number of knots
        if self.cem_config.N_knots <= 1:
            raise ValueError(f"Number of knots [N_knots = {self.cem_config.N_knots}]"
                             f" must be greater than 1.")

        # check that number of elite samples is less than batch size
        if self.cem_config.N_elite > self.sim.B:
            raise ValueError(f"Number of elite samples [N_elite = {self.cem_config.N_elite}]" 
                             f" must be less than parallel sim envs [B = {self.sim.B}].")

    # initialize the spline knot points
    def _initialize_spline_knots(self):

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
            self.spline = ZOH_Spline(Y0, self.cem_config.T)
        elif self.cem_config.spline_type == "Bezier":
            self.spline = Bezier_Spline(Y0, self.cem_config.T)
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

    # running cost
    def _cost(self, q, v, tau):
        """
        cost function.

        Args:
            q: jnp.array, shape (B, N+1, nq) - generalized position trajectory.
            v: jnp.array, shape (B, N+1, nv) - generalized velocity trajectory.
            tau: jnp.array, shape (B, N, nu) - control input trajectory.
        Returns:
            J: jnp.array, shape (B,) - cost for each batch.
        """

        # cost weights
        w_pos = 10.0
        w_vel = 0.1
        w_tau = 0.01
        wf_pos = 10.0 * w_pos 
        wf_vel = 10.0 * w_vel
    
        # running costs over t=0..N-1 (exclude terminal state)
        q_t = q[:, :-1, :]   # (B, N, nq)
        v_t = v[:, :-1, :]   # (B, N, nv)

        # get pole angle positions
        theta_t = q_t[:, :, 1]  # pole angle at time t
        sin_t = jnp.sin(theta_t)
        cos_t = jnp.cos(theta_t)

        # desired angles error
        theta_des = 0.0
        sin_des = jnp.sin(theta_des)
        cos_des = jnp.cos(theta_des)
        ang_err = jnp.stack([sin_t - sin_des, cos_t - cos_des], axis=-1)  # shape (B, N, 2)

        # other desired states (cart position)
        q_des_other = jnp.array([0.0])     # (1,)
        q_other_t = q_t[:, :, 0:1]         # (B, N, 1)
        other_err = q_other_t - q_des_other  # (B, N, 1)

        # total running position penalty
        running_pos = w_pos * (
            jnp.sum(ang_err**2, axis=(1, 2)) +
            jnp.sum(other_err**2, axis=(1, 2))
        )

        # ----------------------------
        # velocity + control running cost
        # ----------------------------
        v_des = jnp.array([0.0, 0.0])  # cart vel, pole ang vel
        running_vel = w_vel * jnp.sum((v_t - v_des) ** 2, axis=(1, 2))
        running_tau = w_tau * jnp.sum(tau ** 2, axis=(1, 2))

        running_cost = self.sim.dt * (running_pos + running_vel + running_tau)

        # ----------------------------
        # terminal cost at t = N
        # ----------------------------
        q_T = q[:, -1, :]   # (B, nq)
        v_T = v[:, -1, :]   # (B, nv)

        theta_T = q_T[:, 1]  # (B,)
        ang_err_T = jnp.stack(
            [jnp.sin(theta_T) - sin_des, jnp.cos(theta_T) - cos_des],
            axis=-1
        )  # (B, 2)

        other_err_T = q_T[:, 0:1] - q_des_other  # (B, 1)

        terminal_pos = wf_pos * (
            jnp.sum(ang_err_T**2, axis=1) +
            jnp.sum(other_err_T**2, axis=1)
        )
        terminal_vel = wf_vel * jnp.sum((v_T - v_des) ** 2, axis=1)

        J = running_cost + terminal_pos + terminal_vel

        return J

    # perform CEM optimization
    def optimize(self, q0, v0):

        # perform CEM iterations
        for itr in range(self.cem_config.iterations):

            # evaluate the spline at simulation times
            y_val = self.spline.evaluate(self.t_sim[:-1])  # shape (B, N, nu)

            # do forward rollout
            q_log, v_log, tau_log = self.sim.rollout(q0, v0, y_val)
            q_log.block_until_ready()
            v_log.block_until_ready()
            tau_log.block_until_ready()

            # compute costs
            J = self._cost(q_log, v_log, tau_log)  # shape (B,)

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

            print(f"Iteration {itr+1}/{self.cem_config.iterations}, Best Cost: {J.min():.4f}, Cov Norm: {cov_norm:.4f}")

        # extract optimal solution
        q_opt = q_log[jnp.argmin(J), :, :]
        v_opt = v_log[jnp.argmin(J), :, :]
        tau_opt = tau_log[jnp.argmin(J), :, :]

        return q_opt, v_opt, tau_opt

#############################################################
# EXAMPLE USAGE
#############################################################

if __name__ == "__main__":

    import matplotlib.pyplot as plt

    # print deivce that we will use
    print(f"Using device: {jax.default_backend()}")
    if jax.default_backend() == "gpu":
        gpu_info = jax.devices("gpu")[0]
        print(f"GPU device: {gpu_info}")

    # fix the random seed
    np.random.seed(0)

    # model config
    model_config = ModelConfig(
        xml_path="./models/cartpole/cartpole.xml",
        Kp=[500.0], 
        Kd=[50.0],  
        q_actuated_idx=[0],
        v_actuated_idx=[0],
        action_mode="pos"
    )

    # parallel sim config
    sim_config = ParallelSimConfig(
        batch_size = 4096,
    )

    # cem config
    cem_rng = jax.random.PRNGKey(42)
    cem_config = CEM_Config(
        rng=cem_rng,
        T=5.0,
        iterations=300,
        N_elite=2048,
        N_knots=5*5,
        spline_type="ZOH",
        # N_knots=20,
        # spline_type="Bezier",
    )

    # create the CEM optimizer
    cem_optimizer = CEM_Optimizer(
        model_config=model_config,
        sim_config=sim_config,
        cem_config=cem_config
    )

    # optimize from an initial state
    q_opt, v_opt, tau_opt = cem_optimizer.optimize(
        q0=jnp.array([0.0, np.pi + 0.1]),   # initial position
        v0=jnp.array([0.0, 0.0])            # initial velocity
    )
    times = cem_optimizer.t_sim

    # convert to numpy for plotting
    times = np.array(times)
    q_opt = np.array(q_opt)
    v_opt = np.array(v_opt)
    tau_opt = np.array(tau_opt)

    # save as csv files in the results folder
    time_file = "./results/cartpole/times.csv"
    q_file = "./results/cartpole/q_opt.csv"
    v_file = "./results/cartpole/v_opt.csv"
    tau_file = "./results/cartpole/tau_opt.csv"
    np.savetxt(time_file, times, delimiter=",")
    np.savetxt(q_file, q_opt, delimiter=",")
    np.savetxt(v_file, v_opt, delimiter=",")
    np.savetxt(tau_file, tau_opt, delimiter=",")

    print(times.shape)
    print(q_opt.shape)
    print(v_opt.shape)
    print(tau_opt.shape)

    # plot the first two positions
    plt.figure()
    plt.plot(times, q_opt[:, 0], label="Cart Position")
    plt.plot(times, q_opt[:, 1], label="Pole Angle")

    plt.figure()
    plt.plot(times, v_opt[:, 0], label="Cart Vel")
    plt.plot(times, v_opt[:, 1], label="Pole velocity")

    plt.figure()
    plt.plot(times[:-1], tau_opt[:, 0], label="Cart Force")

    plt.show()
