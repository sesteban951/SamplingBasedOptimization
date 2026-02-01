##
#
# Cross Entropy Method (CEM) for trajectory optimization.
#
##

# standard imports
import numpy as np
from dataclasses import dataclass

# jax imports
import jax
import jax.numpy as jnp

# cusotm imports
from utils.simulation import *
from utils.spline import *

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
    spline_type: str = "ZOH"   # type of spline to use

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

        # empty dsitribution
        self.mu = None
        self.Sigma = None

        print("CEM Optimizer initialized.")

    # check that the input params make sense
    def _check_valid_params(self):

        # check that number of elite samples is less than batch size
        if self.cem_config.N_elite > self.sim.B:
            raise ValueError(f"Number of elite samples [N_elite = {self.cem_config.N_elite}]" 
                             f" must be less than parallel sim envs [B = {self.sim.B}].")
        
        # check positive interval
        if self.cem_config.T <= 0.0:
            raise ValueError(f"Total time [T = {self.cem_config.T}] must be positive.")
        
        # check number of knots
        if self.cem_config.N_knots <= 1:
            raise ValueError(f"Number of knots [N_knots = {self.cem_config.N_knots}]"
                             f" must be greater than 1.")

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
        else:
            raise NotImplementedError(f"Spline type [{self.cem_config.spline_type}] not implemented.")
        
        # update the spline knot points
        self.spline.update_knots(Y0)

    # update the distribution based on elite samples
    def _update_distribution(self, Y_elite):
        """
        Update the disitrbution based on elite samples.
        
        :param self: Description
        :param Y_elite: Description
        """
        pass

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
        w_pos = 1.0
        w_vel = 0.1
        w_tau = 0.001
        wf_pos = 10.0 * w_pos 
        wf_vel = 10.0 * w_vel

        # desired states
        q_des = jnp.array([0.0, 0.0])  # shape (nq,)
        v_des = jnp.array([0.0, 0.0])  # shape (nv,)

        # running costs over t=0..N-1 (exclude terminal state)
        q_t = q[:, :-1, :]   # (B, N, nq)
        v_t = v[:, :-1, :]   # (B, N, nv)

        running_pos = w_pos * jnp.sum((q_t - q_des) ** 2, axis=(1, 2))
        running_vel = w_vel * jnp.sum((v_t - v_des) ** 2, axis=(1, 2))
        running_tau = w_tau * jnp.sum(tau ** 2, axis=(1, 2))
        running_cost = self.sim.dt * (running_pos + running_vel + running_tau)

        # terminal costs at t=N
        q_T = q[:, -1, :]
        v_T = v[:, -1, :]
        terminal_cost = (wf_pos * jnp.sum((q_T - q_des) ** 2, axis=1)
                       + wf_vel * jnp.sum((v_T - v_des) ** 2, axis=1)
        )

        # total cost
        J = running_cost + terminal_cost

        return J

    # perform CEM optimization
    def optimize(self, q0, v0):

        # construct the time array for simulation
        t_sim = jnp.arange(0.0, self.cem_config.T, self.sim.dt)  # shape (N,)

        # perform CEM iterations
        for itr in range(self.cem_config.iterations):

            # evaluate the spline at simulation times
            y_val = self.spline.evaluate(t_sim)  # shape (B, N, nu)

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
            # ...

#############################################################
# EXAMPLE USAGE
#############################################################

if __name__ == "__main__":

    # print deivce that we will use
    print(f"Using device: {jax.default_backend()}")
    if jax.default_backend() == "gpu":
        gpu_info = jax.devices("gpu")[0]
        print(f"GPU device: {gpu_info}")

    # fix the random seed
    np.random.seed(0)

    # model config
    model_config = ModelConfig(
        xml_path="./models/cartpole.xml",
        Kp=[400.0], 
        Kd=[50.0],  
        q_actuated_idx=[0],
        v_actuated_idx=[0],
        action_mode="pos"
    )

    # parallel sim config
    sim_config = ParallelSimConfig(
        batch_size = 16,
    )

    # cem config
    cem_rng = jax.random.PRNGKey(42)
    cem_config = CEM_Config(
        rng=cem_rng,
        T=5.0,
        N_knots=40,
        iterations=10,
        N_elite=4,
        spline_type="ZOH",
    )

    # create the CEM optimizer
    cem_optimizer = CEM_Optimizer(
        model_config=model_config,
        sim_config=sim_config,
        cem_config=cem_config
    )

    cem_optimizer.optimize(
        q0=jnp.array([0.0, np.pi + 0.1]),   # initial position
        v0=jnp.array([0.0, 0.0])            # initial velocity
    )
