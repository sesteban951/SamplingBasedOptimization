##
#
#  Parallel MJX Rollouts
#  
##

# standard imports
import numpy as np
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
    action_mode: str = "pos"   # action mode:    "pos" (PD tracking)
                               #              or "tau" (pure torque) 


# parallel sim config
@dataclass
class ParallelSim_Config:

    # simulation parameters
    batch_size: int           # batch size for parallel rollout

    # some options on whether or not to use external wrench
    use_external_wrench: bool = False
    srb_traj_dir: str = None  

    # viertual wrench gains
    kp_lin: float = 250.0     # linear position gain
    kd_lin: float = 10.0      # linear velocity gain
    kp_ang: float = 100.0     # angular position gain
    kd_ang: float = 5.0      # angular velocity gain


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
        if (self.use_external_wrench == True) and (self.base_type in ["planar", "free"]):
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
        self.base_type = self._detect_base_type(mj_model)

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
        print(f"   [Floating Base: {self.base_type}]")
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
        p_com_traj = q_traj[:, :3]    # world com position
        v_com_traj = v_traj[:, :3]    # world com linear velocity
        a_com_traj = a_traj[:, :3]    # world com linear acceleration
        quat_traj  = q_traj[:, 3:]    # world orientation (quaternion)
        omega_traj = v_traj[:, 3:]    # body frame angular velocity
        alpha_traj = a_traj[:, 3:]    # body frame angular acceleration
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
        a_com_ref = np.zeros((N_sim, 3), dtype=np.float32)
        quat_ref  = np.zeros((N_sim, 4), dtype=np.float32)
        omega_ref = np.zeros((N_sim, 3), dtype=np.float32)
        alpha_ref = np.zeros((N_sim, 3), dtype=np.float32)
        F_W_ref = np.zeros((N_sim, 3), dtype=np.float32)
        M_W_ref = np.zeros((N_sim, 3), dtype=np.float32)

        for k in range(N_sim):
            t = float(t_sim[k])

            idx_2 = int(np.searchsorted(times, t, side="right"))
            idx_1 = idx_2 - 1

            if idx_2 >= len(times):
                idx_1 = idx_2 = len(times) - 1
                coeff = 0.0
            elif idx_1 < 0:
                idx_1 = idx_2 = 0
                coeff = 0.0
            else:
                t1_ = float(times[idx_1])
                t2_ = float(times[idx_2])
                denom = (t2_ - t1_)
                if abs(denom) < 1e-12:
                    coeff = 0.0
                else:
                    coeff = (t - t1_) / denom
                    coeff = float(np.clip(coeff, 0.0, 1.0))

            # normalize quats before slerp to guard against drift
            q1 = quat_traj[idx_1]
            q2 = quat_traj[idx_2]
            q1 = q1 / (np.linalg.norm(q1) + 1e-12)
            q2 = q2 / (np.linalg.norm(q2) + 1e-12)

            p_com_ref[k] = interp.lerp(p_com_traj[idx_1],  p_com_traj[idx_2],  coeff)
            v_com_ref[k] = interp.lerp(v_com_traj[idx_1],  v_com_traj[idx_2],  coeff)
            omega_ref[k] = interp.lerp(omega_traj[idx_1],  omega_traj[idx_2],  coeff)
            quat_ref[k]  = interp.slerp(q1, q2, coeff)
            a_com_ref[k] = interp.lerp(a_com_traj[idx_1],  a_com_traj[idx_2],  coeff)
            alpha_ref[k] = interp.lerp(alpha_traj[idx_1],  alpha_traj[idx_2],  coeff)
            F_W_ref[k]   = interp.lerp(F_W_traj[idx_1],    F_W_traj[idx_2],    coeff)
            M_W_ref[k]   = interp.lerp(M_W_traj[idx_1],    M_W_traj[idx_2],    coeff)

        self.p_com_ref  = jnp.asarray(p_com_ref)
        self.v_com_ref  = jnp.asarray(v_com_ref)
        self.a_com_ref  = jnp.asarray(a_com_ref)
        self.quat_ref   = jnp.asarray(quat_ref)
        self.omega_ref  = jnp.asarray(omega_ref)
        self.alpha_ref  = jnp.asarray(alpha_ref)
        self.F_W_ref    = jnp.asarray(F_W_ref)
        self.M_W_ref    = jnp.asarray(M_W_ref)

        # broadcast gravity and inertia
        self.gravity_batch = jnp.broadcast_to(self.dyn.gravity, (self.B, 3))
        self.I_base_batch = jnp.broadcast_to(self.dyn.I_base_nom, (self.B, 3, 3))

        # PD gains for the external wrnech 
        self.kp_lin = sim_config.kp_lin
        self.kd_lin = sim_config.kd_lin
        self.kp_ang = sim_config.kp_ang
        self.kd_ang = sim_config.kd_ang

        print(f"Loaded SRB trajectory from [{dir}].")
        print(f"   [Nx_traj: {Nx_traj}]")
        print(f"   [Nu_traj: {Nu_traj}]")
        print(f"   [dt_traj: {dt_traj}]")
        print(f"   [T_traj: {T_traj:.4f} seconds]")
        print(f"   [N_sim:   {N_sim} steps at dt={dt_sim:.4f}s]")

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
        self.Kp_batched = self.Kp[None, :]
        self.Kd_batched = self.Kd[None, :]
        
        # two compiled rollout functions:
        # - no wrench: wrench block never compiled into graph
        # - with wrench: wrench block compiled in, w_scale is dynamic
        self._jit_rollout_no_wrench = jax.jit(self._rollout_no_wrench)
        self._jit_rollout_wrench    = jax.jit(self._rollout_wrench)

        print("JIT compilation of simulation functions complete.")


    ####################################### ROLLOUTS #######################################

    def rollout(self, q0, v0, U, w_scale=0.0):
        """
        Public API. Routes to the correct compiled function based on w_scale.
        - w_scale == 0.0 or use_external_wrench == False: wrench block never compiled into graph (fastest)
        - w_scale  > 0.0 and use_external_wrench == True: wrench block compiled in, w_scale is dynamic
        
        Args:
            q0:      jnp.array, shape (nq, ), initial position state
            v0:      jnp.array, shape (nv, ), initial velocity state
            U:       jnp.array, shape (B, N, nu), batch of action sequences, either torques or desired positions
            w_scale: float, scale factor for external wrenches (0.0 = no wrench, 1.0 = full wrench)
    
        Returns:
            q_log:   jnp.array, logged positions,  shape (B, N+1, nq)
            v_log:   jnp.array, logged velocities, shape (B, N+1, nv)
            tau_log: jnp.array, logged torques,    shape (B, N, nu)
        """
        if (not self.use_external_wrench) or (w_scale == 0.0):
            return self._jit_rollout_no_wrench(q0, v0, U)
        else:
            return self._jit_rollout_wrench(q0, v0, U, jnp.float32(w_scale))


    def _rollout_no_wrench(self, q0, v0, U):
        """
        Perform parallel rollouts with no external wrench injection.
        The wrench block is never compiled into the graph — fastest path.
        
        Args:
            q0: jnp.array, shape (nq, ), initial position state
            v0: jnp.array, shape (nv, ), initial velocity state
            U:  jnp.array, shape (B, N, nu), batch of action sequences, either torques or desired positions
    
        Returns:
            q_log:   jnp.array, logged positions,  shape (B, N+1, nq)
            v_log:   jnp.array, logged velocities, shape (B, N+1, nv)
            tau_log: jnp.array, logged torques,    shape (B, N, nu)
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
        def integration_step(carry, uk):
            # unpack the carry
            data, step_idx = carry

            # compute the control based on action mode and apply (NOTE: decided once at compile time)
            if self.use_pd:
                tau = self._compute_pd_torque(data.qpos, data.qvel, uk)
            else:
                tau = uk
            data = data.replace(ctrl=tau)

            # step the simulation forward
            data = self.step_fn_batched(data)

            return (data, step_idx + 1), (data.qpos, data.qvel, data.actuator_force) # NOTE: takes torque limits into account

        # forward propagate
        (_, _), (q_hist, v_hist, tau_hist) = lax.scan(integration_step, (data0, jnp.int32(0)), U, length=N)

        # q_hist, v_hist: (N, B, nq/nv) -> (B, N, nq/nv)
        q_hist = jnp.swapaxes(q_hist, 0, 1)
        v_hist = jnp.swapaxes(v_hist, 0, 1)

        # prepend initial state so logs are N+1
        q_log   = jnp.concatenate([q0_batch[:, None, :], q_hist], axis=1)
        v_log   = jnp.concatenate([v0_batch[:, None, :], v_hist], axis=1)
        tau_log = jnp.swapaxes(tau_hist, 0, 1)

        return q_log, v_log, tau_log


    def _rollout_wrench(self, q0, v0, U, w_scale):
        """
        Perform parallel rollouts with external wrench injection.
        w_scale is a dynamic GPU float32 — any value in (0, 1] reuses the same compiled graph.
        
        Args:
            q0:      jnp.array, shape (nq, ), initial position state
            v0:      jnp.array, shape (nv, ), initial velocity state
            U:       jnp.array, shape (B, N, nu), batch of action sequences, either torques or desired positions
            w_scale: jnp.float32, scale factor for external wrenches (0.0 = no wrench, 1.0 = full wrench)
    
        Returns:
            q_log:   jnp.array, logged positions,  shape (B, N+1, nq)
            v_log:   jnp.array, logged velocities, shape (B, N+1, nv)
            tau_log: jnp.array, logged torques,    shape (B, N, nu)
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
        def integration_step(carry, uk):
            # unpack the carry
            data, step_idx = carry

            # compute the control based on action mode and apply (NOTE: decided once at compile time)
            if self.use_pd:
                tau = self._compute_pd_torque(data.qpos, data.qvel, uk)
            else:
                tau = uk
            data = data.replace(ctrl=tau)

            # inject external wrench (NOTE: decided once at compile time)
            F_W, M_B = self._compute_virtual_wrench_3D(data, step_idx, w_scale)
            data = self._base_wrench_qfrc(data, F_W, M_B)

            # step the simulation forward
            data = self.step_fn_batched(data)

            return (data, step_idx + 1), (data.qpos, data.qvel, data.actuator_force) # NOTE: takes torque limits into account

        # forward propagate
        (_, _), (q_hist, v_hist, tau_hist) = lax.scan(integration_step, (data0, jnp.int32(0)), U, length=N)

        # q_hist, v_hist: (N, B, nq/nv) -> (B, N, nq/nv)
        q_hist = jnp.swapaxes(q_hist, 0, 1)
        v_hist = jnp.swapaxes(v_hist, 0, 1)

        # prepend initial state so logs are N+1
        q_log   = jnp.concatenate([q0_batch[:, None, :], q_hist], axis=1)
        v_log   = jnp.concatenate([v0_batch[:, None, :], v_hist], axis=1)
        tau_log = jnp.swapaxes(tau_hist, 0, 1)

        return q_log, v_log, tau_log
    

    ####################################### AUXILLARY #######################################
    
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

    def _compute_virtual_wrench_3D(self, data, step_idx, w_scale):
        """
        Compute the virtual wrench that tracks the SRB trajectory
        Fᵂ = m  (â + kₚᵛ (p̂ - p) + k_dᵛ (v̂ - v) - g)
        Mᴮ = I α̂ + kₚ^ω I (q̂ - q) + k_d^ω I (ω̂ - ω) + ω × (I ω) − r × mg

        W = w_scale * [Fᵂ, Mᴮ], w_scale ∈ [0, 1]

        Args: 
            data:     mjx.Data, the current state of the simulation
            step_idx: int, the current step in the rollout
            w_scale:  jnp.float32, scale factor for the virtual wrench (0.0 = no wrench, 1.0 = full wrench)
        Returns:
            F_W: (B, 3) world frame force to apply at the base
            M_B: (B, 3) body frame torque to apply at the base
        """
        # clip the wrench scale to be between 0 and 1
        w_scale = jnp.clip(w_scale, 0.0, 1.0)

        # current states of the mujoco 
        p_base = data.qpos[:, :3]     # (B, 3)
        # v_base = data.qvel[:, :3]     # (B, 3)
        quat    = data.qpos[:, 3:7]   # (B, 4)
        omega   = data.qvel[:, 3:6]   # (B, 3)

        # current com state
        p_com_batch, v_com_batch = self.dyn.com_state_in_world(data.qpos, data.qvel)  # (B, 3), (B, 3)

        # ----------------------- references -----------------------

        # reference states from the SRB trajectory
        p_com_ref = self.p_com_ref[step_idx]  # (3,)
        v_com_ref = self.v_com_ref[step_idx]  # (3,)
        a_com_ref = self.a_com_ref[step_idx]  # (3,)
        quat_ref  = self.quat_ref[step_idx]   # (4,)
        omega_ref = self.omega_ref[step_idx]  # (3,)
        alpha_ref = self.alpha_ref[step_idx]  # (3,)

        p_com_ref_batch = jnp.broadcast_to(p_com_ref, (self.B, 3))   # (B, 3)
        v_com_ref_batch = jnp.broadcast_to(v_com_ref, (self.B, 3))   # (B, 3)
        a_com_ref_batch = jnp.broadcast_to(a_com_ref, (self.B, 3))   # (B, 3)
        quat_ref_batch  = jnp.broadcast_to(quat_ref, (self.B, 4))    # (B, 4)
        omega_ref_batch = jnp.broadcast_to(omega_ref, (self.B, 3))   # (B, 3)
        alpha_ref_batch = jnp.broadcast_to(alpha_ref, (self.B, 3))   # (B, 3)

        # ----------------------- Force in World -----------------------

        # force to track the COM state trajectory (with scaling)
        F_W = self.dyn.mass * (
              a_com_ref_batch
            + self.kp_lin * (p_com_ref_batch - p_com_batch)
            + self.kd_lin * (v_com_ref_batch - v_com_batch)
            - self.gravity_batch
        ) * w_scale  # (B, 3)

        # ----------------------- Moment in Base -----------------------

        # inertia helper: I @ vec for batched (B, 3)
        I_B = self.I_base_batch                                          # (B, 3, 3)
        Iv  = lambda vec: jnp.einsum('ij,bj->bi', self.dyn.I_base_nom, vec)  # (B, 3)

        # feedforward moment
        M_ff = Iv(alpha_ref_batch)  # (B, 3)

        # the coriolis term
        C = dynamics.Dynamics._omega_cross_Iomega(omega, I_B)      # (B, 3)

        # orientation moment
        orient_err = self.dyn.quat_log_diff(quat, quat_ref_batch)  # (B, 3)
        pd_orient  = self.kp_ang * Iv(orient_err)                  # (B, 3)
        pd_angvel = self.kd_ang * Iv(omega_ref_batch - omega)      # (B, 3)
        M_ori = pd_orient + pd_angvel                              # (B, 3)

        # gravity force in world frame → rotate into body frame
        F_grav_W = self.dyn.mass * self.gravity_batch          # (B, 3)
        F_grav_B = self.dyn.vec_world_to_body(F_grav_W, quat)  # (B, 3)

        # compensation of gravity
        r_com_W  = p_com_batch - p_base                       # (B, 3)
        r_com_B  = self.dyn.vec_world_to_body(r_com_W, quat)  # (B, 3)
        M_grav = jnp.cross(r_com_B, F_grav_B)                 # (B, 3)

        # Moment that tracks the orientation trajectory (with scaling)
        M_B = (M_ff + C + M_ori - M_grav) * w_scale  # (B, 3)

        return F_W, M_B
    

    def _compute_virtual_wrench_2D(self, data, step_idx, w_scale):
        """
        Compute the virtual wrench that tracks the planar SRB trajectory.
        Fˣ = m (âₓ + kₚ(p̂ₓ - pₓ) + kd(v̂ₓ - vₓ))
        Fᶻ = m (âᶻ + kₚ(p̂ᶻ - pᶻ) + kd(v̂ᶻ - vᶻ) - g)
        Mʸ = Iyy(α̂ + kₚ(θ̂ - θ) + kd(ω̂ - ω)) - r × mg

        W = w_scale * [Fˣ, Fᶻ, Mʸ], w_scale ∈ [0, 1]

        Args:
            data:      mjx.Data, current simulation state
            step_idx:  int, current rollout step
            w_scale:   jnp.float32, wrench scale factor
        Returns:
            Fx:  (B,) world-frame force in x
            Fz:  (B,) world-frame force in z
            My:  (B,) body-frame torque about y
        """
        w_scale = jnp.clip(w_scale, 0.0, 1.0)

        # ----------------------- current state -----------------------

        # planar qpos = [x, z, theta, ...joint angles...]
        # planar qvel = [vx, vz, omega_y, ...joint velocities...]
        p_base_x = data.qpos[:, 0]  # (B,)
        theta    = data.qpos[:, 2]  # (B,)
        omega_y  = data.qvel[:, 2]  # (B,)

        # current COM state in world frame
        p_com_batch, v_com_batch = self.dyn.com_state_in_world(data.qpos, data.qvel)  # (B, 3)

        # ----------------------- references -----------------------

        # reference states from the SRB trajectory
        p_com_ref = self.p_com_ref[step_idx]  # (2,) [px, pz]
        v_com_ref = self.v_com_ref[step_idx]  # (2,) [vx, vz]
        a_com_ref = self.a_com_ref[step_idx]  # (2,) [ax, az]
        theta_ref = self.theta_ref[step_idx]  # scalar
        omega_ref = self.omega_ref[step_idx]  # scalar
        alpha_ref = self.alpha_ref[step_idx]  # scalar

        # broadcast to batch
        p_com_ref_batch = jnp.broadcast_to(p_com_ref, (self.B, 2))  # (B, 2)
        v_com_ref_batch = jnp.broadcast_to(v_com_ref, (self.B, 2))  # (B, 2)
        a_com_ref_batch = jnp.broadcast_to(a_com_ref, (self.B, 2))  # (B, 2)
        theta_ref_batch = jnp.broadcast_to(theta_ref, (self.B,))    # (B,)
        omega_ref_batch = jnp.broadcast_to(omega_ref, (self.B,))    # (B,)
        alpha_ref_batch = jnp.broadcast_to(alpha_ref, (self.B,))    # (B,)

        # ----------------------- Force in World -----------------------

        # p_com_batch and v_com_batch xz components: (B, 2)
        p_com_xz = jnp.stack([p_com_batch[:, 0], p_com_batch[:, 2]], axis=-1)  # (B, 2)
        v_com_xz = jnp.stack([v_com_batch[:, 0], v_com_batch[:, 2]], axis=-1)  # (B, 2)

        F_W = self.dyn.mass * (
            a_com_ref_batch
            + self.kp_lin * (p_com_ref_batch - p_com_xz)
            + self.kd_lin * (v_com_ref_batch - v_com_xz)
            - jnp.stack([jnp.zeros(self.B), self.gravity_batch[:, 2]], axis=-1)
        ) * w_scale  # (B, 2)

        # ----------------------- Moment in Body -----------------------

        Iyy = self.dyn.I_base_nom[1, 1]  # scalar

        # feedforward + PD on pitch angle and angular velocity
        My = Iyy * (
            alpha_ref_batch
            + self.kp_ang * (theta_ref_batch - theta)
            + self.kd_ang * (omega_ref_batch - omega_y)
        )  # (B,)

        # gravity compensation: (r_com × F_grav)_y = r_com_x * F_grav_z
        r_com_x  = p_com_batch[:, 0] - p_base_x             # (B,)
        F_grav_z = self.dyn.mass * self.gravity_batch[:, 2]  # (B,) — negative (-mg)
        M_grav_y = r_com_x * F_grav_z                        # (B,)

        M_B = (My - M_grav_y) * w_scale  # (B,)

        return F_W, M_B


    def _base_wrench_qfrc(self, data, F_W, M_B):
        """
        Inject a wrench at the floating base via `qfrc_applied`.

        NOTE: understanding how to apply wrenches to the robot
        https://github.com/google-deepmind/mujoco/discussions/2350
        https://mujoco.readthedocs.io/en/stable/mjx_api.html#mujoco.mjx.Data.qfrc_applied

        Args:
            F_W: (B,3) or (B,2) world force
            M_B: (B,3) or (B,1) body frame torque
        Returns:
            data: mjx.Data, with the wrench applied in qfrc_applied
        """
        qfrc = jnp.zeros_like(data.qfrc_applied)  # (B, nv) — default: no wrench

        # 3D floating base wrench
        if self.base_type == "free":
            # freejoint: [0:3] = world force, [3:6] = body torque
            qfrc = qfrc.at[:, 0:3].set(F_W)
            qfrc = qfrc.at[:, 3:6].set(M_B)

        # planar wrench 
        elif self.base_type == "planar":
            # slide_x, slide_z, hinge_y -> Fx, Fz, torque_y
            qfrc = qfrc.at[:, 0].set(F_W[:, 0])  # Fx
            qfrc = qfrc.at[:, 1].set(F_W[:, 2])  # Fz
            qfrc = qfrc.at[:, 2].set(M_B[:, 1])  # torque_y

        return data.replace(qfrc_applied=qfrc)
    

    def _detect_base_type(self, mj_model):
        """
        Detects root DOF type by inspecting only joints on the root body.
        
        Args:
            mj_model: mujoco.MjModel, the mujoco model to inspect
        Returns: 'free' (3D floating), 'planar' (2D planar floating), or 'none'.
            'none'   = no unactuated root DOFs (e.g. cartpole, fixed-base arm)
            'planar' = slide_x + slide_z + hinge_y at root (e.g. biped, hopper, g1_planar)
            'free'   = freejoint at root (e.g. g1_21dof)
        """

        # iterate over joints that are attached to the root body and check their types
        root_body_id = 1
        root_joint_ids = []
        for j in range(mj_model.njnt):
            if mj_model.jnt_bodyid[j] == root_body_id:
                root_joint_ids.append(j)
        
        # no joints attached to the root body, so we assume it's not floating base type
        if not root_joint_ids:
            return "none"

        # label the root joint types
        root_joint_types = []
        for j in root_joint_ids:
            root_joint_types.append(mj_model.jnt_type[j])

        # 3D floating base 
        if mujoco.mjtJoint.mjJNT_FREE in root_joint_types:
            return "free"

        # 2D planar base
        n_slide = sum(1 for t in root_joint_types if t == mujoco.mjtJoint.mjJNT_SLIDE)
        n_hinge = sum(1 for t in root_joint_types if t == mujoco.mjtJoint.mjJNT_HINGE)
        if n_slide == 2 and n_hinge == 1:
            return "planar"

        # catch all, return none
        return "none"


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
    # model_config = Model_Config(
    #     xml_path="./models/cube/scene.xml",
    #     Kp=[10.0]*16,   # Reduced from 30 (lower = more stable)
    #     Kd=[1.0]*16,    # Reduced from 2 (lower = more stable)
    #     q_actuated_idx=list(range(16)),  # Hand joints
    #     v_actuated_idx=list(range(16)),  # Hand joint velocities
    #     action_mode="pos"
    # )
    # q0 = jnp.zeros(23)
    # v0 = jnp.zeros(22)
    # q0 = q0.at[16:19].set(jnp.array([0.11, 0.0, 0.10]))      # Position (x, y, z)
    # q0 = q0.at[19:23].set(jnp.array([1.0, 0.0, 0.0, 0.0]))   # Identity quaternion (w, x, y, z)

    # load mujoco model
    xml_path = "./models/g1/g1_21dof.xml"
    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_data = mujoco.MjData(mj_model)
    keyframe = "standing"
    key_id = mj_model.key(keyframe).id
    q0 = jnp.array(mj_model.key_qpos[key_id])
    v0 = jnp.array(mj_model.key_qvel[key_id])
    q_joints = q0[7:]
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
        use_external_wrench=True,
        srb_traj_dir="./results/srb/srb_jump/"
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
    # U_B = jnp.zeros((B, nu))  # zero controls
    U_B = jnp.broadcast_to(q_joints, (B, nu))  # hold joints at initial position
    U = jnp.broadcast_to(U_B[:, None, :], (B, N, nu))

    # run rollout
    t0 = time.time()
    q_log, v_log, tau_log = parallel_sim.rollout(q0, v0, U, jnp.float32(0.01))
    q_log.block_until_ready()
    v_log.block_until_ready()
    tau_log.block_until_ready()
    t1 = time.time()
    print(f"Rolled out {B} trajectories of length {N} in {t1 - t0:.4f} seconds.")
    
    t0 = time.time()
    q_log, v_log, tau_log = parallel_sim.rollout(q0, v0, U, jnp.float32(0.5))
    q_log.block_until_ready()
    v_log.block_until_ready()
    tau_log.block_until_ready()
    t1 = time.time()
    print(f"Rolled out {B} trajectories of length {N} in {t1 - t0:.4f} seconds.")

    t0 = time.time()
    q_log, v_log, tau_log = parallel_sim.rollout(q0, v0, U)
    q_log.block_until_ready()
    v_log.block_until_ready()
    tau_log.block_until_ready()
    t1 = time.time()
    print(f"Rolled out {B} trajectories of length {N} in {t1 - t0:.4f} seconds.")

    t0 = time.time()
    q_log, v_log, tau_log = parallel_sim.rollout(q0, v0, U, jnp.float32(0.99))
    q_log.block_until_ready()
    v_log.block_until_ready()
    tau_log.block_until_ready()
    t1 = time.time()
    print(f"Rolled out {B} trajectories of length {N} in {t1 - t0:.4f} seconds.")

    t0 = time.time()
    q_log, v_log, tau_log = parallel_sim.rollout(q0, v0, U)
    q_log.block_until_ready()
    v_log.block_until_ready()
    tau_log.block_until_ready()
    t1 = time.time()
    print(f"Rolled out {B} trajectories of length {N} in {t1 - t0:.4f} seconds.")

    t0 = time.time()
    q_log, v_log, tau_log = parallel_sim.rollout(q0, v0, U, jnp.float32(1.0))
    q_log.block_until_ready()
    v_log.block_until_ready()
    tau_log.block_until_ready()
    t1 = time.time()
    print(f"Rolled out {B} trajectories of length {N} in {t1 - t0:.4f} seconds.")

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
    save_dir = "./results/parallel_sim/parallel_sim/"
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
