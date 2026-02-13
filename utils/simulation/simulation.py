##
#
#  Parallel MJX Rollouts
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
class Model_Config:

    # model parameters
    xml_path: str            # path to the mujoco xml model

    # PD control parameters
    Kp: List[float]          # proportional gains for each joint
    Kd: List[float]          # derivative gain for each joint

    # actuated state indices, TODO: make this so that it is not manually entered
    q_actuated_idx: List[int]   # indices of actuated positions  (qpos)
    v_actuated_idx: List[int]   # indices of actuated velocities (qvel)

    # action mode
    action_mode: str = "tau"   # action mode:    "tau" (pure torque) 
                               #              or "pos" (PD tracking) 


# parallel sim config
@dataclass
class ParallelSim_Config:

    # simulation parameters
    batch_size: int           # batch size for parallel rollout

# MJX Rollout class
class ParallelSim():
    """
    Class to perform parallel rollouts using mujoco mjx on GPU.

    Args:
        model_config: Model_Config, configuration for the mujoco model
        sim_config: ParallelSim_Config, configuration for the parallel sim
    """

    # initialize the class
    def __init__(self, model_config: Model_Config,
                       sim_config: ParallelSim_Config,):

        # set some config params for the class
        self.B = sim_config.batch_size

        # load the model from XML
        self._initialize_model(model_config)

        # initialize the jit functions
        self._initialize_jit_functions()

        print("Parallel sim initialized.")

    ####################################### INITIALIZATION #######################################

    # initialize the mujoco model
    def _initialize_model(self, model_config: Model_Config):
        """
        Initialize the mujoco model and data for parallel rollout on GPU.
        
        Args:
            model_config: Model_Config, configuration for the mujoco model
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

        # load simulation dt (rounded) # NOTE: can change integrator and sim_dt here
        self.dt = round(float(self.mjx_model.opt.timestep), 6)

        # action mode
        self.use_pd = (model_config.action_mode == "pos")
        self.q_actuated_idx = tuple(model_config.q_actuated_idx)
        self.v_actuated_idx = tuple(model_config.v_actuated_idx)
        assert model_config.action_mode in ["tau", "pos"], "Invalid action mode."
        assert len(self.q_actuated_idx) == self.nu, "q_actuated_idx length does not match nu."
        assert len(self.v_actuated_idx) == self.nu, "v_actuated_idx length does not match nu."

        # load control parameters
        assert len(model_config.Kp) == self.nu, "Kp length does not match nu."
        assert len(model_config.Kd) == self.nu, "Kd length does not match nu."
        self.Kp = jnp.array(model_config.Kp)
        self.Kd = jnp.array(model_config.Kd)

        # grab position limits at the actuated joints, (if at nq[i] = [0., 0.] then no limits)
        joint_id_per_act = mj_model.actuator_trnid[:, 0].astype(int)  # shape (nu,)
        self.pos_limits = jnp.array(mj_model.jnt_range[joint_id_per_act, :])  # (nu, 2)

        # grab actuation limits, (if at nu[i] = [0., 0.] then no limits)
        self.ctrl_limits = jnp.array(mj_model.actuator_ctrlrange) # shape (nu, 2)

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

        # create the batched step function
        self.step_fn_batched = jax.jit(
            jax.vmap(lambda d: mjx.step(self.mjx_model, d), in_axes=0)
        )

        # create a batched template of mjx_data 
        self.data_batch0 = jax.vmap(lambda _: self.mjx_data)(jnp.arange(self.B))

        # pre-broadcast control gains
        self.Kp_batched = self.Kp[None, :]  # Pre-broadcast
        self.Kd_batched = self.Kd[None, :]
        
        # jit the rollout function
        self.rollout = jax.jit(self._rollout)

        print("JIT compilation of simulation functions complete.")

    ####################################### ROLLOUTS #######################################

    def _rollout(self, q0, v0, U):
        """
        Perform parallel rollouts with a given initial state and action sequences.
        
        Args:
            q0: jnp.array, shape (nq, ), initial position state
            v0: jnp.array, shape (nv, ), initial velocity state
            U:  jnp.array, shape (B, N, nu),  batch of  action sequences, either torques or desired positions
                                              
        Returns:
            q_log: jnp.array, logged positions,  shape (B, N+1, nq)
            v_log: jnp.array, logged velocities, shape (B, N+1, nv)
            tau_log: jnp.array, logged torques, shape (B, N, nu)
        """

        # get sizes
        B, N, _ = U.shape

        # batch the initial states
        q0_batch = jnp.broadcast_to(q0, (B, self.nq))    # (B, nq)
        v0_batch = jnp.broadcast_to(v0, (B, self.nv))    # (B, nv)

        # set the initial conditions in the batched data
        data0 = self.data_batch0.replace(qpos=q0_batch, qvel=v0_batch)

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

            return data, (data.qpos, data.qvel, data.actuator_force) # NOTE: takes torque limits into account

        # forward propagate
        _, (q_hist, v_hist, tau_hist) = lax.scan(integration_step, data0, U, length=N)

        # q_hist, v_hist: (N, B, nq/nv) -> (B, N, nq/nv)
        q_hist = jnp.swapaxes(q_hist, 0, 1)
        v_hist = jnp.swapaxes(v_hist, 0, 1)

        # prepend initial state so logs are N+1
        q_log = jnp.concatenate([q0_batch[:, None, :], q_hist], axis=1)
        v_log = jnp.concatenate([v0_batch[:, None, :], v_hist], axis=1)
        tau_log = jnp.swapaxes(tau_hist, 0, 1)

        return q_log, v_log, tau_log
    

    def _rollout_wrench(self, q0, v0, U, F_const_W, M_const_W, body_id=1):
        """
        Perform parallel rollouts with constant external wrench injected at the floating base
        via `qfrc_applied` every step.

        Args:
            q0:         (nq,) initial qpos
            v0:         (nv,) initial qvel
            U:          (B, N, nu) action sequence (torques or desired joint positions)
            F_const_W:  (3,) or (B,3) constant world force
            M_const_W:  (3,) or (B,3) constant world torque (couple), world
            body_id:    base/pelvis body id (used only for world->body torque conversion)

        Returns:
            q_log:   (B, N+1, nq)
            v_log:   (B, N+1, nv)
            tau_log: (B, N, nu)
        """
        B, N, _ = U.shape

        # Batch initial states
        q0_batch = jnp.broadcast_to(q0, (B, self.nq))
        v0_batch = jnp.broadcast_to(v0, (B, self.nv))
        data0 = self.data_batch0.replace(qpos=q0_batch, qvel=v0_batch)

        # Make constant wrench batched
        F_const_W = jnp.asarray(F_const_W)
        M_const_W = jnp.asarray(M_const_W)

        F_const_W = lax.cond(
            F_const_W.ndim == 1,
            lambda _: jnp.broadcast_to(F_const_W[None, :], (B, 3)),
            lambda _: F_const_W,
            operand=None,
        )
        M_const_W = lax.cond(
            M_const_W.ndim == 1,
            lambda _: jnp.broadcast_to(M_const_W[None, :], (B, 3)),
            lambda _: M_const_W,
            operand=None,
        )

        # (B, N, nu) -> (N, B, nu)
        U = jnp.swapaxes(U, 0, 1)

        def integration_step(data, uk):
            # Compute joint-space actuation
            tau = lax.cond(
                self.use_pd,
                lambda _: self._compute_pd_torque(data.qpos, data.qvel, uk),
                lambda _: uk,
                operand=None
            )

            # Set control
            data = data.replace(ctrl=tau)

            # Inject constant wrench at base via qfrc_applied
            data = self.inject_base_wrench_qfrc(data, F_const_W, M_const_W, body_id=body_id)

            # Step dynamics
            data = self.step_fn_batched(data)

            return data, (data.qpos, data.qvel, data.actuator_force)

        _, (q_hist, v_hist, tau_hist) = lax.scan(integration_step, data0, U, length=N)

        q_hist = jnp.swapaxes(q_hist, 0, 1)  # (B,N,nq)
        v_hist = jnp.swapaxes(v_hist, 0, 1)  # (B,N,nv)

        q_log = jnp.concatenate([q0_batch[:, None, :], q_hist], axis=1)  # (B,N+1,nq)
        v_log = jnp.concatenate([v0_batch[:, None, :], v_hist], axis=1)  # (B,N+1,nv)
        tau_log = jnp.swapaxes(tau_hist, 0, 1)  # (B,N,nu)

        return q_log, v_log, tau_log

    ####################################### AUXILLARY #######################################
    
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
        q_act = q[:, self.q_actuated_idx]  # (B, nu)
        v_act = v[:, self.v_actuated_idx]  # (B, nu)

        # compute PD torques
        tau = self.Kp_batched * (q_des - q_act) + self.Kd_batched * (-v_act) # (B, nu)

        return tau
    

    def inject_base_wrench_qfrc(self, data, F_W, M_W, body_id=1):
        """
        Inject a wrench at the floating base via `qfrc_applied`.

        Assumed convention for freejoint generalized forces:
        qfrc_applied[..., 0:3] = world linear force
        qfrc_applied[..., 3:6] = body-frame torque

        Args:
            data:   mjx.Data with batch dimension (B, ...)
            F_W:    (B,3) world force
            M_W:    (B,3) world torque (couple)
            body_id: int, body id for the base/pelvis (used to get orientation)
        """
        # data.xmat is (B, nbody, 9) row-major
        B = data.xmat.shape[0]
        R_BW = data.xmat[:, body_id, :].reshape((B, 3, 3))  # (B,3,3)

        # Convert world torque -> body torque: M_B = R_WB * M_W = R_BW^T * M_W
        M_B = jnp.einsum('bij,bj->bi', jnp.swapaxes(R_BW, 1, 2), M_W)  # (B,3)

        qfrc = jnp.zeros_like(data.qfrc_applied)  # (B,nv)
        qfrc = qfrc.at[:, 0:3].set(F_W)
        qfrc = qfrc.at[:, 3:6].set(M_B)

        # (Optional) clear xfrc_applied so you're only using qfrc injection
        data = data.replace(qfrc_applied=qfrc, xfrc_applied=jnp.zeros_like(data.xfrc_applied))
        return data

#############################################################
# EXAMPLE USAGE
#############################################################

if __name__ == "__main__":

    import numpy as np
    import matplotlib.pyplot as plt
    import time

    # print deivce that we will use
    print(f"Using device: {jax.default_backend()}")
    if jax.default_backend() == "gpu":
        gpu_info = jax.devices("gpu")[0]
        print(f"GPU device: {gpu_info}")

    # fix the random seed
    # np.random.seed(0)

    # model config
    # model_config = Model_Config(
    #     xml_path="./models/cartpole/cartpole.xml",
    #     Kp=[400.0], 
    #     Kd=[50.0],  
    #     q_actuated_idx=[0], # cart position
    #     v_actuated_idx=[0], # cart velocity
    #     action_mode="pos"
    # )
    # q0 = jnp.array([0.0, jnp.pi])  # slight offset from upright
    # model_config = Model_Config(
    #     xml_path="./models/hopper/hopper.xml",
    #     Kp=[100.0, 500.0], 
    #     Kd=[5.0, 50.0],  
    #     q_actuated_idx=[2, 3], # theta 
    #     v_actuated_idx=[2, 3], # theta dot
    #     action_mode="pos"
    # )
    # q0 = jnp.array([0.0, 1.0, 0.0, 0.0])  # in the air, leg at zero pos
    # model_config = Model_Config(
    #     xml_path="./models/biped/biped.xml",
    #     Kp=[100.0, 100.0, 100.0, 100.0], 
    #     Kd=[5.0, 5.0, 5.0, 5.0],  
    #     q_actuated_idx=[3, 4, 5, 6],
    #     v_actuated_idx=[3, 4, 5, 6],
    #     action_mode="pos"
    # )
    # q0 = jnp.array([0, 0.83, 0, 0.22, -0.415, 0.22, -0.415])  # bent knees
    # model_config = Model_Config(
    #     xml_path="./models/biped/biped.xml",
    #     Kp=[100.0, 100.0, 100.0, 100.0], 
    #     Kd=[5.0, 5.0, 5.0, 5.0],  
    #     q_actuated_idx=[3, 4, 5, 6],
    #     v_actuated_idx=[3, 4, 5, 6],
    #     action_mode="pos"
    # )
    # q0 = jnp.array([0, 0.83, 0, 0.22, -0.415, 0.22, -0.415])  # bent knees
    # model_config = Model_Config(
    #     xml_path="./models/g1/g1_planar.xml",
    #     Kp=[250, 250, 50, 250, 250, 50, # legs
    #         150, 150, 150, 150],        # arms
    #     Kd=[3.0, ] * 10,  
    #     q_actuated_idx=list(range(3,13)),
    #     v_actuated_idx=list(range(3,13)),
    #     action_mode="pos"
    # )
    # q0 = jnp.array([
    #     0.0, 0.0, 0.0,
    #     0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    #     0.0, 0.0, 0.0, 0.0
    # ])

    # load mujoco model
    xml_path = "./models/g1/g1_21dof.xml"
    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_data = mujoco.MjData(mj_model)
    keyframe = "standing"
    key_id = mj_model.key(keyframe).id
    q0 = jnp.array(mj_model.key_qpos[key_id])
    v0 = jnp.array(mj_model.key_qvel[key_id])
    model_config = Model_Config(
        xml_path=xml_path,
        Kp=[300, 300, 300, 300, 100, 100, # left leg
            300, 300, 300, 300, 100, 100, # right leg
            100,                          # waist
            150, 150, 150, 150,           # left arm
            150, 150, 150, 150],          # right arm
        Kd=[3.0, ] * 21,  
        q_actuated_idx=list(range(7,7+21)),
        v_actuated_idx=list(range(6,6+21)),
        action_mode="pos"
    )

    # parallel sim config
    sim_config = ParallelSim_Config(
        batch_size = 512,
    )

    # create the parallel sim object
    parallel_sim = ParallelSim(model_config, sim_config)

    # integration steps
    N = 300
    dt = float(parallel_sim.mjx_model.opt.timestep)

    # initial velocity conditions
    v0 = jnp.zeros((parallel_sim.nv,))

    # random controls: (B, N, nu)
    B = sim_config.batch_size
    nu = parallel_sim.nu

    key = jax.random.PRNGKey(int(time.time()))
    key, subkey = jax.random.split(key)
    U_B = jax.random.uniform(subkey, shape=(B, nu), minval=-1.0, maxval=1.0)   # (B, nu)
    U = jnp.broadcast_to(U_B[:, None, :], (B, N, nu))

    # run rollout
    t0 = time.time()
    q_log, v_log, tau_log = parallel_sim.rollout(q0, v0, U)
    q_log.block_until_ready()
    v_log.block_until_ready()
    tau_log.block_until_ready()
    t1 = time.time()
    print(f"Rolled out {B} trajectories of length {N} in {t1 - t0:.4f} seconds.")
    q_log, v_log, tau_log = parallel_sim.rollout(q0, v0, U)
    q_log.block_until_ready()
    v_log.block_until_ready()
    tau_log.block_until_ready()
    t2 = time.time()
    print(f"Rolled out {B} trajectories of length {N} in {t2 - t1:.4f} seconds.")
    q_log, v_log, tau_log = parallel_sim.rollout(q0, v0, U)
    q_log.block_until_ready()
    v_log.block_until_ready()
    tau_log.block_until_ready()
    t3 = time.time()
    print(f"Rolled out {B} trajectories of length {N} in {t3 - t2:.4f} seconds.")

    # convert to numpy for plotting
    q_log = np.array(q_log)
    v_log = np.array(v_log)
    tau_log = np.array(tau_log)

    print(f"q_log shape: {q_log.shape}")
    print(f"v_log shape: {v_log.shape}")
    print(f"tau_log shape: {tau_log.shape}")

    # choose a few trajectories to plot
    B = q_log.shape[0]
    K = 10  # number to plot
    idx = np.random.choice(B, K, replace=False)  # or random subset

    # time axis (optional)
    t = np.arange(q_log.shape[1]) * dt  # (N+1,)

    # plt.figure()
    # for k in idx:
    #     plt.plot(t, q_log[k, :, 0], alpha=0.7)
    #     plt.plot(t, q_log[k, :, 1], alpha=0.7)
    #     # plt.plot(t[:-1], tau_log[k, :, 0], alpha=0.7)

    # plt.xlabel("Time (s)")
    # plt.ylabel("Positions")

    # plt.show()
