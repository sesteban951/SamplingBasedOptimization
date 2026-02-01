##
#
#  Perform Parallel MJX Rollouts
#  
##

# standard imports
from dataclasses import dataclass
from typing import List

# jax imports
import jax
import jax.numpy as jnp
from jax import lax

# mujoco imports
import mujoco
import mujoco.mjx as mjx

#############################################################
# PARALLEL DYNAMICS ROLLOUT CLASS
#############################################################

# model configuration
@dataclass
class ModelConfig:

    # model parameters
    xml_path: str            # path to the mujoco xml model

    # PD control parameters
    Kp: List[float]          # proportional gains for each joint
    Kd: List[float]          # derivative gain for each joint

    # max and minimum torques
    tau_limit: bool           # whether to use torque limits
    tau_lb: List[float]       # minimum torques
    tau_ub: List[float]       # maximum torques

    # actuated state indices
    q_actuated_idx: List[int]   # indices of actuated positions
    v_actuated_idx: List[int]   # indices of actuated velocities

    # action mode
    action_mode: str = "tau"   # action mode: "tau" or "pos"


# parallel sim config
@dataclass
class ParallelSimConfig:

    # simulation parameters
    batch_size: int           # batch size for parallel rollout
    rng: jax.random.PRNGKey   # random number generator key


# MJX Rollout class
class ParallelSim():
    """
    Class to perform parallel rollouts using mujoco mjx on GPU.

    Args:
        model_config: ModelConfig, configuration for the mujoco model
        sim_config: ParallelSimConfig, configuration for the parallel sim
    """

    # initialize the class
    def __init__(self, model_config: ModelConfig,
                       sim_config: ParallelSimConfig,):

        # set some config params for the class
        self.rng = sim_config.rng
        self.B = sim_config.batch_size

        # load the model from XML
        self._initialize_model(model_config)

        # initialize the jit functions
        self._initialize_jit_functions()

    ####################################### INITIALIZATION #######################################

    # initialize the mujoco model
    def _initialize_model(self, model_config: ModelConfig):
        """
        Initialize the mujoco model and data for parallel rollout on GPU.
        
        Args:
            model_config: ModelConfig, configuration for the mujoco model
        """

        # mujoco model
        mj_model = mujoco.MjModel.from_xml_path(model_config.xml_path)
        mj_data = mujoco.MjData(mj_model)

        # put the model on GPU
        self.mjx_model = mjx.put_model(mj_model)
        self.mjx_data = mjx.put_data(mj_model, mj_data)

        # load sizes
        self.nq = self.mjx_model.nq
        self.nv = self.mjx_model.nv
        self.nu = self.mjx_model.nu

        # action mode
        self.use_pd = (model_config.action_mode == "pos")
        self.q_actuated_idx = jnp.asarray(model_config.q_actuated_idx, dtype=jnp.int32)
        self.v_actuated_idx = jnp.asarray(model_config.v_actuated_idx, dtype=jnp.int32)
        assert model_config.action_mode in ["tau", "pos"], "Invalid action mode."
        assert len(self.q_actuated_idx) == self.nu, "q_actuated_idx length does not match nu."
        assert len(self.v_actuated_idx) == self.nu, "v_actuated_idx length does not match nu."

        # load control parameters
        assert len(model_config.Kp) == self.nu, "Kp length does not match nu."
        assert len(model_config.Kd) == self.nu, "Kd length does not match nu."
        assert len(model_config.tau_lb) == self.nu, "tau_lb length does not match nu."
        assert len(model_config.tau_ub) == self.nu, "tau_ub length does not match nu."
        self.Kp = jnp.array(model_config.Kp)
        self.Kd = jnp.array(model_config.Kd)
        self.tau_limit = model_config.tau_limit
        self.tau_lb = jnp.array(model_config.tau_lb)
        self.tau_ub = jnp.array(model_config.tau_ub)

        # create the batched step function
        self.step_fn_batched = jax.jit(jax.vmap(lambda d: mjx.step(self.mjx_model, d), in_axes=0))
        # self.step_fn_batched = jax.vmap(lambda d: mjx.step(self.mjx_model, d))

        # print message
        print(f"Initialized batched MJX model from [{model_config.xml_path}].")
        print(f"   [nq: {self.nq}]")
        print(f"   [nv: {self.nv}]")
        print(f"   [nu: {self.nu}]")

    # initialize the jit functions
    def _initialize_jit_functions(self):
        """
        Initialize the functions with jit for rollout for speed.
        """
        
        # jit the rollout function
        self.rollout = jax.jit(self.rollout)

        # jit the pd torque function
        # self._compute_pd_torque = jax.jit(self._compute_pd_torque)

        print("JIT compilation of parallel sim function complete.")

    ####################################### ROLLOUTS #######################################

    # rollout with given inputs
    def rollout(self, q0, v0, U):
        """
        Perform parallel rollouts with a given initial state and action sequences.
        
        Args:
            q0: jnp.array, shape (nq, ), initial position state
            v0: jnp.array, shape (nv, ), initial velocity state
            U:  jnp.array, shape (B, N, nu),  batch of  action sequences, either torques or desired positions
                                              
        Returns:
            q_log: jnp.array, logged positions,  shape (B, N+1, nq)
            v_log: jnp.array, logged velocities, shape (B, N+1, nv)
            tau_log: jnp.array, logged torques, shape (N, B, nu)
        """

        # get sizes
        B, N, _ = U.shape

        # batch the initial states
        q0_batch = jnp.broadcast_to(q0, (B, self.nq))    # (B, nq)
        v0_batch = jnp.broadcast_to(v0, (B, self.nv))    # (B, nv)

        # set the initial conditions in the batched data
        data0 = jax.vmap(lambda q, v: self.mjx_data.replace(qpos=q, qvel=v))(q0_batch, v0_batch)

        # swap axes for easier indexing (B, N, nu) -> (N, B, nu)
        U = jnp.swapaxes(U, 0, 1)

        # main integration step body
        def integration_step(data, uk):
            
            # compute the control based on action mode
            tau = lax.cond(
                self.use_pd,                                                 # condition
                lambda _: self._compute_pd_torque(data.qpos, data.qvel, uk), # position target function
                lambda _: uk,                                                # torque target function
                operand=None
            )

            # step the dynamics
            data = data.replace(ctrl=tau)        # set the control
            data = self.step_fn_batched(data)    # step

            return data, (data.qpos, data.qvel, tau)

        # forward propagate
        _, (q_hist, v_hist, tau_hist) = lax.scan(integration_step, data0, U, length=N)

        # q_hist, v_hist: (N, B, nq/nv) -> (B, N, nq/nv)
        q_hist = jnp.swapaxes(q_hist, 0, 1)
        v_hist = jnp.swapaxes(v_hist, 0, 1)

        # prepend initial state so logs are N+1
        q_log = jnp.concatenate([q0_batch[:, None, :], q_hist], axis=1)
        v_log = jnp.concatenate([v0_batch[:, None, :], v_hist], axis=1)

        return q_log, v_log, tau_hist
    
    # compute PD torques
    def _compute_pd_torque(self, q, v, q_des):
        """
        Compute PD torques for given states and desired positions. Assume v_des = 0.
        
        Args:
            q:      jnp.array, shape (B, nq), current positions
            v:      jnp.array, shape (B, nv), current velocities
            q_des:  jnp.array, shape (B, nu), desired positions
        """

        # extract actuated positions and velocities
        q_act = jnp.take(q, self.q_actuated_idx, axis=1)  # (B, nu)
        v_act = jnp.take(v, self.v_actuated_idx, axis=1)  # (B, nu)

        # compute PD torques
        tau = self.Kp[None, :] * (q_des - q_act) + self.Kd[None, :] * (-v_act) # (B, nu)

        # clip the torques
        tau = lax.cond(
            self.tau_limit,  # Python bool (static)
            lambda t: jnp.clip(t, self.tau_lb[None, :], self.tau_ub[None, :]),
            lambda t: t,
            tau
        )

        return tau


#############################################################
# EXAMPLE USAGE
#############################################################

import numpy as np
import matplotlib.pyplot as plt
import time

if __name__ == "__main__":

    # model config
    model_config = ModelConfig(
        xml_path="./models/cartpole.xml",
        Kp=[300.0], 
        Kd=[20.0],  
        tau_limit=False,
        tau_lb=[-500.0],
        tau_ub=[ 500.0],
        q_actuated_idx=[0],
        v_actuated_idx=[0],
        action_mode="pos"
    )

    # parallel sim config
    sim_config = ParallelSimConfig(
        batch_size = 1024,
        rng = jax.random.PRNGKey(0)
    )

    # create the parallel sim object
    parallel_sim = ParallelSim(model_config, sim_config)

    # integration steps
    N = 2000

    # initial conditions (cartpole: usually nq=2, nv=2)
    q0 = jnp.array([0.0, jnp.pi])  # slight offset from upright
    v0 = jnp.zeros((parallel_sim.nv,))

    # random controls: (B, N, nu)
    B = sim_config.batch_size
    nu = parallel_sim.nu

    key = sim_config.rng
    key, subkey = jax.random.split(key)
    U_B = jax.random.uniform(subkey, shape=(B, nu), minval=-1.0, maxval=1.0)   # (B, nu)
    U = jnp.broadcast_to(U_B[:, None, :], (B, N, nu))

    # run rollout
    t0 = time.time()
    q_log, v_log, tau_log = parallel_sim.rollout(q0, v0, U)
    t1 = time.time()
    print(f"Rolled out {B} trajectories of length {N} in {t1 - t0:.4f} seconds.")
    q_log, v_log, tau_log = parallel_sim.rollout(q0, v0, U)
    t2 = time.time()
    print(f"Rolled out {B} trajectories of length {N} in {t2 - t1:.4f} seconds.")
    q_log, v_log, tau_log = parallel_sim.rollout(q0, v0, U)
    t3 = time.time()
    print(f"Rolled out {B} trajectories of length {N} in {t3 - t2:.4f} seconds.")
    q_log, v_log, tau_log = parallel_sim.rollout(q0, v0, U)
    t4 = time.time()
    print(f"Rolled out {B} trajectories of length {N} in {t4 - t3:.4f} seconds.")
    q_log, v_log, tau_log = parallel_sim.rollout(q0, v0, U)
    t5 = time.time()
    print(f"Rolled out {B} trajectories of length {N} in {t5 - t4:.4f} seconds.")

    # choose a few trajectories to plot
    B = q_log.shape[0]
    K = 10  # number to plot
    idx = np.random.choice(B, K, replace=False)  # or random subset

    # time axis (optional)
    dt = float(parallel_sim.mjx_model.opt.timestep)
    t = np.arange(q_log.shape[1]) * dt  # (N+1,)

    plt.figure()
    for k in idx:
        plt.plot(t, q_log[k, :, 0], alpha=0.7)
        # plt.plot(t, q_log[k, :, 1], alpha=0.7)
        # plt.plot(t[:-1], tau_log[:, k, 0], alpha=0.7)
    plt.xlabel("Time (s)")
    plt.ylabel("Positions")

    plt.show()
    