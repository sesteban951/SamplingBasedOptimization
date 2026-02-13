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

# custom imports
from utils.simulation import dynamics
from utils.interpolation import interp
from utils.kinematics import kin


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

    # some options on whether or not to use external wrench
    use_external_wrench: bool = False
    srb_traj_dir: str = None  


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
                       sim_config: ParallelSim_Config):

        # set some config params for the class
        self.B = sim_config.batch_size

        # load the model from XML
        self._initialize_model(model_config)

        # external wrench
        self.use_external_wrench = sim_config.use_external_wrench
        if (self.use_external_wrench == True) and (self.has_3D_floating_base == True):
            self._initialize_dynamics(model_config)
            self._initialize_SRB_trajectories(sim_config)

            print("External wrench injection enabled.")

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

        # check if there is a floating base system
        self.has_3D_floating_base = bool((mj_model.jnt_type == mujoco.mjtJoint.mjJNT_FREE).any())

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
        print(f"   [3D Floating Base: {self.has_3D_floating_base}]")
        print(f"   [dt: {self.dt:.4f} seconds]")
        print(f"   [nq: {self.nq}]")
        print(f"   [nv: {self.nv}]")
        print(f"   [nu: {self.nu}]")


    def _initialize_dynamics(self, model_config: Model_Config):
        """
        Initialize the dynamics object for computing external wrenches.
        """
        # build the dynamics config
        dyn_config = dynamics.Dynamics_Config(
            xml_path=model_config.xml_path,
            num_envs=self.B,
        )
        # instantiate the dynamics object
        self.dyn = dynamics.Dynamics(dyn_config)


    def _initialize_SRB_trajectories(self, sim_config: ParallelSim_Config):
        """
        Initialize the SRB trajectories for external wrench injection.
        """
        
        # directory that has the SRB trajectories
        dir = sim_config.srb_traj_dir

        # which data to load
        time_file = dir + "time.csv"
        q_file    = dir + "q_opt.csv"
        v_file    = dir + "v_opt.csv"
        a_file    = dir + "a_opt.csv"
        tau_file  = dir + "tau_opt.csv"

        # load data from csv files
        times    = np.loadtxt(time_file, delimiter=",")
        q_traj   = np.loadtxt(q_file, delimiter=",")
        v_traj   = np.loadtxt(v_file, delimiter=",")
        a_traj   = np.loadtxt(a_file, delimiter=",")
        tau_traj = np.loadtxt(tau_file, delimiter=",")

        # extract all the data
        p_com_traj = q_traj[:, :3]    # world position
        v_com_traj = v_traj[:, :3]    # world linear velocity
        quat_traj  = q_traj[:, 3:]    # world orientation (quaternion)
        omega_traj = v_traj[:, 3:]    # body frame angular velocity
        F_W_traj   = tau_traj[:, :3]  # world forces
        M_W_traj   = tau_traj[:, 3:]  # world moments

        # add the last elemnent to the forces and moments to make sure we can interpolate all the way to the end of the trajectory
        F_W_traj = np.vstack([F_W_traj, F_W_traj[-1]])
        M_W_traj = np.vstack([M_W_traj, M_W_traj[-1]])

        # extract some other info 
        Nx_traj = q_traj.shape[0]
        Nu_traj = tau_traj.shape[0]
        T_traj = times[-1]

        # setup the simulation time axis for the SRB trajectory
        dt_traj = times[1] - times[0]
        dt_sim = self.dt
        t0 = times[0]
        tf = times[-1]

        # time array 
        t_sim = np.arange(t0, tf, dt_sim, dtype=np.float64)
        N_sim = t_sim.shape[0]

        # allocate the SRB trajectory in the class
        p_com_ref = np.zeros((N_sim, 3), dtype=np.float32)
        v_com_ref = np.zeros((N_sim, 3), dtype=np.float32)
        quat_ref  = np.zeros((N_sim, 4), dtype=np.float32)
        omega_ref = np.zeros((N_sim, 3), dtype=np.float32)
        F_W_ref = np.zeros((N_sim, 3), dtype=np.float32)
        M_W_ref = np.zeros((N_sim, 3), dtype=np.float32)

        for k in range(N_sim):
            t = float(t_sim[k])

            idx_2 = int(np.searchsorted(times, t, side="right"))
            idx_1 = idx_2 - 1

            if idx_2 >= len(times):
                idx_1 = idx_2 = len(times) - 1
                alpha = 0.0
            elif idx_1 < 0:
                idx_1 = idx_2 = 0
                alpha = 0.0
            else:
                t1_ = float(times[idx_1])
                t2_ = float(times[idx_2])
                denom = (t2_ - t1_)
                if abs(denom) < 1e-12:
                    alpha = 0.0
                else:
                    alpha = (t - t1_) / denom
                    alpha = float(np.clip(alpha, 0.0, 1.0))

            # Optional: normalize quats before slerp if your data may drift
            q1 = quat_traj[idx_1]
            q2 = quat_traj[idx_2]
            q1 = q1 / (np.linalg.norm(q1) + 1e-12)
            q2 = q2 / (np.linalg.norm(q2) + 1e-12)

            p_com_ref[k] = interp.lerp(p_com_traj[idx_1], p_com_traj[idx_2], alpha)
            v_com_ref[k] = interp.lerp(v_com_traj[idx_1], v_com_traj[idx_2], alpha)
            omega_ref[k] = interp.lerp(omega_traj[idx_1], omega_traj[idx_2], alpha)
            quat_ref[k]  = interp.slerp(q1, q2, alpha)

            F_W_ref[k] = interp.lerp(F_W_traj[idx_1], F_W_traj[idx_2], alpha)
            M_W_ref[k] = interp.lerp(M_W_traj[idx_1], M_W_traj[idx_2], alpha)

        self.p_com_ref = jnp.asarray(p_com_ref)
        self.v_com_ref = jnp.asarray(v_com_ref)
        self.quat_ref  = jnp.asarray(quat_ref)
        self.omega_ref = jnp.asarray(omega_ref)
        self.F_W_ref   = jnp.asarray(F_W_ref)
        self.M_W_ref   = jnp.asarray(M_W_ref)

        print(f"Loaded SRB trajectory from [{dir}].")
        print(f"   [Nx_traj: {Nx_traj}]")
        print(f"   [Nu_traj: {Nu_traj}]")
        print(f"   [dt_traj: {dt_traj}]")
        print(f"   [T_traj: {T_traj:.4f} seconds]")

        
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

        # fixed wrench (WORLD force, BODY torque)
        fixed_F_W = jnp.array([30.0, 0.0, 35.0 * 9.81], dtype=jnp.float32)  # example: +Z world force
        fixed_M_B = jnp.array([0.0, 5.0, 5.0], dtype=jnp.float32)          # example: no body torque
        F_W_b = jnp.broadcast_to(fixed_F_W[None, :], (B, 3))
        M_B_b = jnp.broadcast_to(fixed_M_B[None, :], (B, 3))

        # main integration step body
        def integration_step(data, uk):
            
            # compute the control based on action mode and apply
            tau = lax.cond(
                self.use_pd,                                                 # condition
                lambda _: self._compute_pd_torque(data.qpos, data.qvel, uk), # position target function
                lambda _: uk,                                                # torque target function
                operand=None
            )
            data = data.replace(ctrl=tau)   

            # inject external wrench
            if self.use_external_wrench:
                data = self._base_wrench_qfrc(data, F_W_b, M_B_b)

            # step the simulation forward
            data = self.step_fn_batched(data)    

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

    def _base_wrench_qfrc(self, data, F_W, M_B):
        """
        Inject a wrench at the floating base via `qfrc_applied`.

        NOTE: understanding how to apply wrenches to the robot
        https://github.com/google-deepmind/mujoco/discussions/2350
        https://mujoco.readthedocs.io/en/stable/mjx_api.html#mujoco.mjx.Data.qfrc_applied

        Assumed (MuJoCo freejoint) convention:
        qfrc_applied[..., 0:3] = world linear force
        qfrc_applied[..., 3:6] = body-frame torque

        Inputs:
        F_W: (B,3) world force
        M_B: (B,3) body frame torque
        """
        # build qfrc_applied (overwrite each step)
        qfrc = jnp.zeros_like(data.qfrc_applied)  # (B,nv)
        qfrc = qfrc.at[:, 0:3].set(F_W)
        qfrc = qfrc.at[:, 3:6].set(M_B)

        return data.replace(qfrc_applied=qfrc)


#############################################################
# EXAMPLE USAGE
#############################################################

if __name__ == "__main__":

    import numpy as np
    import matplotlib.pyplot as plt
    import time
    import os

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
        # use_external_wrench=True,
        # srb_traj_dir="./results/srb_jump/"
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
    # U_B = jax.random.uniform(subkey, shape=(B, nu), minval=-1.0, maxval=1.0)   # (B, nu)
    U_B = jnp.zeros((B, nu))  # zero controls
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

    # pick one of the trajectories to save as csv
    k = idx[0]
    times = t
    q_opt = q_log[k]      # (N+1, nq)
    v_opt = v_log[k]      # (N+1, nv)
    tau_opt = tau_log[k]  # (N, nu)
    # save as csv files in the results folder
    save_dir = "./results/parallel_sim/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created directory: {save_dir}")

    time_file = save_dir + "time.csv"
    q_file = save_dir + "q_opt.csv"
    v_file = save_dir + "v_opt.csv"
    tau_file = save_dir + "tau_opt.csv"
    np.savetxt(time_file, times, delimiter=",")
    np.savetxt(q_file, q_opt, delimiter=",")
    np.savetxt(v_file, v_opt, delimiter=",")
    np.savetxt(tau_file, tau_opt, delimiter=",")
    print(f"Saved results to {save_dir}")
    
    plt.figure()
    for k in idx:
        plt.plot(t, q_log[k, :, 0], alpha=0.7)
        plt.plot(t, q_log[k, :, 1], alpha=0.7)
        # plt.plot(t, q_log[k, :, 2], alpha=0.7)
        # plt.plot(t[:-1], tau_log[k, :, 0], alpha=0.7)

    plt.xlabel("Time (s)")
    plt.ylabel("Positions")

    plt.show()
