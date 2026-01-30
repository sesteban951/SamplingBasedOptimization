##
#
#  Perform Parallel MJX Rollouts
#  
##

# standard imports
from dataclasses import dataclass

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
    xml_path: str          # path to the mujoco xml model

    # PD control parameters
    Kp: jnp.array          # proportional gains for each joint
    Kd: jnp.array          # derivative gain for each joint

    # max and minimum torques
    u_lb: jnp.array       # minimum torques
    u_ub: jnp.array       # maximum torques

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
        config: ParallelSimConfig, configuration for the parallel sim
    """

    # initialize the class
    def __init__(self, sim_config: ParallelSimConfig,
                       model_config: ModelConfig):

        # set some config params for the class
        self.rng = sim_config.rng
        self.B = sim_config.batch_size

        # load the model from XML
        self._initialize_model(model_config.xml_path)

        # initialize the jit functions
        self._initialize_jit_functions()

    ####################################### INITIALIZATION #######################################

    # initialize the mujoco model
    def _initialize_model(self, xml_path: str):
        """
        Initialize the mujoco model and data for parallel rollout on GPU.
        
        Args:
            xml_path: str, path to the mujoco xml model
        """

        # mujoco model
        mj_model = mujoco.MjModel.from_xml_path(xml_path)
        mj_data = mujoco.MjData(mj_model)

        # put the model on GPU
        self.mjx_model = mjx.put_model(mj_model)
        self.mjx_data = mjx.put_data(mj_model, mj_data)

        # load sizes
        self.nq = self.mjx_model.nq
        self.nv = self.mjx_model.nv
        self.nu = self.mjx_model.nu

        # create the batched step function
        self.step_fn_batched = jax.jit(jax.vmap(lambda d: mjx.step(self.mjx_model, d), in_axes=0))
        # self.step_fn_batched = jax.vmap(lambda d: mjx.step(self.mjx_model, d))

        # print message
        print(f"Initialized batched MJX model from [{xml_path}].")
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

        print("JIT compilation of parallel sim function complete.")

    ####################################### ROLLOUTS #######################################

    # rollout with given inputs
    def rollout(self, q0, v0, U):
        """
        Perform parallel rollouts with a given initial state and control sequences
        
        Args:
            q0: jnp.array, shape (nq, ), initial position state
            v0: jnp.array, shape (nv, ), initial velocity state
            U:  jnp.array, shape (B, N, nu),  batch of control sequences
                                              
        Returns:
            q_log: jnp.array, logged positions,  shape (B, N+1, nq)
            v_log: jnp.array, logged velocities, shape (B, N+1, nv)
        """

        # get sizes
        B, N, nu = U.shape

        # batch the initial states
        q0_batch = jnp.broadcast_to(q0, (B, self.nq))    # (B, nq)
        v0_batch = jnp.broadcast_to(v0, (B, self.nv))    # (B, nv)

        # set the initial conditions in the batched data
        data0 = jax.vmap(lambda q, v: self.mjx_data.replace(qpos=q, qvel=v))(q0_batch, v0_batch)

        # swap axes for easier indexing (B, N, nu) -> (N, B, nu)
        U = jnp.swapaxes(U, 0, 1)

        # main integration step body
        def integration_step(data, uk):
            data = data.replace(ctrl=uk)      # set the control
            data = self.step_fn_batched(data) # step the dynamics
            return data, (data.qpos, data.qvel)

        # forward propagate
        data_last, (q_tail, v_tail) = lax.scan(integration_step, data0, U, length=N)

        # q_tail, v_tail: (N, B, nq/nv) -> (B, N, nq/nv)
        q_tail = jnp.swapaxes(q_tail, 0, 1)
        v_tail = jnp.swapaxes(v_tail, 0, 1)

        # prepend initial state so logs are N+1
        q_log = jnp.concatenate([q0_batch[:, None, :], q_tail], axis=1)
        v_log = jnp.concatenate([v0_batch[:, None, :], v_tail], axis=1)

        return q_log, v_log
    

#############################################################
# EXAMPLE USAGE
#############################################################

import numpy as np
import matplotlib.pyplot as plt
import time

if __name__ == "__main__":

    # xml model path
    xml_path = "./models/biped.xml"

    # parallel sim config
    config = ParallelSimConfig(
        xml_path = xml_path,
        batch_size = 1024,
        rng = jax.random.PRNGKey(0)
    )

    # create the parallel sim object
    parallel_sim = ParallelSim(config)

    # horizon
    N = 200

    # initial conditions (cartpole: usually nq=2, nv=2)
    q0 = jnp.zeros((parallel_sim.nq,))
    v0 = jnp.zeros((parallel_sim.nv,))

    # random controls: (B, N, nu)
    B = config.batch_size
    nu = parallel_sim.nu

    key = config.rng
    key, subkey = jax.random.split(key)
    U = 0.1 * jax.random.normal(subkey, (B, N, nu))  # small random torques

    # run rollout
    t0 = time.time()
    q_log, v_log = parallel_sim.rollout(q0, v0, U)
    t1 = time.time()
    print(f"Rolled out {B} trajectories of length {N} in {t1 - t0:.4f} seconds.")
    q_log, v_log = parallel_sim.rollout(q0, v0, U*0.2)
    t2 = time.time()
    print(f"Rolled out {B} trajectories of length {N} in {t2 - t1:.4f} seconds.")
    q_log, v_log = parallel_sim.rollout(q0, v0, U*0.2)
    t3 = time.time()
    print(f"Rolled out {B} trajectories of length {N} in {t3 - t2:.4f} seconds.")
    q_log, v_log = parallel_sim.rollout(q0, v0, U*0.2)
    t4 = time.time()
    print(f"Rolled out {B} trajectories of length {N} in {t4 - t3:.4f} seconds.")
    q_log, v_log = parallel_sim.rollout(q0, v0, U*0.2)
    t5 = time.time()
    print(f"Rolled out {B} trajectories of length {N} in {t5 - t4:.4f} seconds.")

    # choose a few trajectories to plot
    B = q_log.shape[0]
    K = 10  # number to plot
    idx = np.linspace(0, B-1, K, dtype=int)   # evenly spaced
    # idx = np.random.choice(B, K, replace=False)  # or random subset

    # time axis (optional)
    dt = float(parallel_sim.mjx_model.opt.timestep)
    t = np.arange(q_log.shape[1]) * dt  # (N+1,)

    # ---- Plot qpos dims (e.g., cart position and pole angle if nq >= 2) ----
    plt.figure()
    for i in idx:
        plt.plot(t, np.array(q_log[i, :, 0]), alpha=0.8)
    plt.title("Subset of trajectories: qpos[0]")
    plt.xlabel("time [s]")
    plt.ylabel("qpos[0]")
    plt.show()

    if q_log.shape[2] > 1:
        plt.figure()
        for i in idx:
            plt.plot(t, np.array(q_log[i, :, 1]), alpha=0.8)
        plt.title("Subset of trajectories: qpos[1]")
        plt.xlabel("time [s]")
        plt.ylabel("qpos[1]")
        plt.show()

    # ---- Plot a velocity dim too (optional) ----
    plt.figure()
    for i in idx:
        plt.plot(t, np.array(v_log[i, :, 0]), alpha=0.8)
    plt.title("Subset of trajectories: vvel[0]")
    plt.xlabel("time [s]")
    plt.ylabel("vvel[0]")
    plt.show()