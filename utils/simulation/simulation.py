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

    # viertual wrench gains
    kp_lin: float = 100.0   # linear position gain
    kd_lin: float = 5.0     # linear velocity gain
    kp_ang: float = 100.0   # angular position gain
    kd_ang: float = 5.0     # angular velocity gain


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
        if self.use_external_wrench and self.base_type in ["planar", "free"]:
            self._initialize_dynamics(model_config, sim_config)
            print("External wrench injection enabled.")

        # initialize the jit functions
        self._initialize_jit_functions()

        print("Parallel simulation initialized.")

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


    def _initialize_dynamics(self, model_config: Model_Config, sim_config: ParallelSim_Config):
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

        # pre-broadcast model constants
        self.gravity_batch = jnp.broadcast_to(self.dyn.gravity,    (self.B, 3))
        self.I_base_batch  = jnp.broadcast_to(self.dyn.I_base_nom, (self.B, 3, 3))

        # PD gains
        self.kp_lin = sim_config.kp_lin
        self.kd_lin = sim_config.kd_lin
        self.kp_ang = sim_config.kp_ang
        self.kd_ang = sim_config.kd_ang


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

    def rollout(self, q0, v0, U, 
                      q_srb_ref=None, 
                      v_srb_ref=None, 
                      a_srb_ref=None, 
                      w_scale=0.0):
        """
        Public API. Routes to the correct compiled function based on w_scale.
        - w_scale == 0.0 or use_external_wrench == False: wrench block never compiled into graph (fastest)
        - w_scale  > 0.0 and use_external_wrench == True: wrench block compiled in, w_scale is dynamic
        
        Args:
            q0:      jnp.array, shape (nq, ), initial position state
            v0:      jnp.array, shape (nv, ), initial velocity state
            U:       jnp.array, shape (B, N, nu), batch of action sequences, either torques or desired positions
            q_srb_ref: (N, 3) or (N, 7) SRB position reference
            v_srb_ref: (N, 3) or (N, 6) SRB velocity reference
            a_srb_ref: (N, 3) or (N, 6) SRB acceleration reference
            w_scale:   float, wrench scale in [0, 1]
    
        Returns:
            q_log:   jnp.array, logged positions,  shape (B, N+1, nq)
            v_log:   jnp.array, logged velocities, shape (B, N+1, nv)
            tau_log: jnp.array, logged torques,    shape (B, N, nu)
        """        
        if (not self.use_external_wrench) or (w_scale == 0.0):
            return self._jit_rollout_no_wrench(q0, v0, U)
        else:
            assert q_srb_ref is not None and v_srb_ref is not None and a_srb_ref is not None, \
                  "q_srb_ref, v_srb_ref, a_srb_ref must be provided when w_scale > 0"
            return self._jit_rollout_wrench(q0, v0, U, q_srb_ref, v_srb_ref, a_srb_ref, jnp.float32(w_scale))


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
            data = carry

            # compute the control based on action mode and apply (NOTE: decided once at compile time)
            if self.use_pd:
                tau = self._compute_pd_torque(data.qpos, data.qvel, uk)
            else:
                tau = uk
            data = data.replace(ctrl=tau)

            # step the simulation forward
            data = self.step_fn_batched(data)

            return data, (data.qpos, data.qvel, data.actuator_force)  # NOTE: takes torque limits into account

        # forward propagate
        _, (q_hist, v_hist, tau_hist) = lax.scan(integration_step, data0, U, length=N)

        # q_hist, v_hist: (N, B, nq/nv) -> (B, N, nq/nv)
        q_hist = jnp.swapaxes(q_hist, 0, 1)
        v_hist = jnp.swapaxes(v_hist, 0, 1)

        # prepend initial state so logs are N+1
        q_log   = jnp.concatenate([q0_batch[:, None, :], q_hist], axis=1)
        v_log   = jnp.concatenate([v0_batch[:, None, :], v_hist], axis=1)
        tau_log = jnp.swapaxes(tau_hist, 0, 1)

        return q_log, v_log, tau_log


    def _rollout_wrench(self, q0, v0, U, 
                              q_srb_ref, 
                              v_srb_ref, 
                              a_srb_ref, 
                              w_scale):
        """
        Perform parallel rollouts with external wrench injection.
        
        Args:
            q0:        jnp.array, shape (nq, ), initial position state
            v0:        jnp.array, shape (nv, ), initial velocity state
            U:         jnp.array, shape (B, N, nu), batch of action sequences, either torques or desired positions
            q_srb_ref: jnp.array, shape (N, 3) or (N, 7) SRB position reference
            v_srb_ref: jnp.array, shape (N, 3) or (N, 6) SRB velocity reference
            a_srb_ref: jnp.array, shape (N, 3) or (N, 6) SRB acceleration reference
            w_scale:   jnp.float32, wrench scale in [0, 1]
        Returns:
            q_log:   jnp.array, logged positions,  shape (B, N+1, nq)
            v_log:   jnp.array, logged velocities, shape (B, N+1, nv)
            tau_log: jnp.array, logged torques,    shape (B, N, nu)
            w_log:   jnp.array, logged wrenches,   shape (B, N, nw) where nw = 6 for 3D base, 3 for planar base
        """
        B, N, _ = U.shape

        # batch the initial states
        q0_batch = jnp.broadcast_to(q0, (B, self.nq))    # (B, nq)
        v0_batch = jnp.broadcast_to(v0, (B, self.nv))    # (B, nv)

        # set the initial conditions in the batched data
        data0 = self.data_batch0.replace(qpos=q0_batch, qvel=v0_batch)

        # swap axes for easier indexing (B, N, nu) -> (N, B, nu)
        U = jnp.swapaxes(U, 0, 1)

        # scan inputs: (uk, q_ref_k, v_ref_k, a_ref_k) at each step
        scan_inputs = (U, q_srb_ref, v_srb_ref, a_srb_ref)

        # main integration step body
        def integration_step(carry, inputs):
            
            # unpack the carry and inputs
            data = carry
            uk, q_ref_k, v_ref_k, a_ref_k = inputs

            # compute the control based on action mode and apply (NOTE: decided once at compile time)
            if self.use_pd:
                tau = self._compute_pd_torque(data.qpos, data.qvel, uk)
            else:
                tau = uk
            data = data.replace(ctrl=tau)

            # inject external wrench (NOTE: decided once at compile time)
            if self.base_type == "free":
                F_W, M_B = self._compute_virtual_wrench_3D(data, q_ref_k, v_ref_k, a_ref_k, w_scale)
            elif self.base_type == "planar":
                F_W, M_B = self._compute_virtual_wrench_2D(data, q_ref_k, v_ref_k, a_ref_k, w_scale)
            else:
                raise ValueError(f"Cannot inject wrench for [base_type = '{self.base_type}']")

            # apply wrench to base
            data = self._base_wrench_qfrc(data, F_W, M_B)

            # step the simulation forward
            data = self.step_fn_batched(data)

            # pack wrench into a single vector per step
            if self.base_type == "free":
                w_k = jnp.concatenate([F_W, M_B], axis=-1)          # (B, 6)
            else:  # planar
                w_k = jnp.concatenate([F_W, M_B[:, None]], axis=-1)  # (B, 3)

            return data, (data.qpos, data.qvel, data.actuator_force, w_k) # NOTE: takes torque limits into account

        # forward propagate
        _, (q_hist, v_hist, tau_hist, w_hist) = lax.scan(integration_step, data0, scan_inputs, length=N)

        # swap axes for easier indexing (N, B, ...) -> (B, N, ...)
        q_hist   = jnp.swapaxes(q_hist,   0, 1)  # (B, N, nq)
        v_hist   = jnp.swapaxes(v_hist,   0, 1)  # (B, N, nv)
        tau_hist = jnp.swapaxes(tau_hist, 0, 1)  # (B, N, nu)
        w_hist   = jnp.swapaxes(w_hist,   0, 1)  # (B, N, 6) or (B, N, 3)

        # prepend initial state so logs are N+1
        q_log   = jnp.concatenate([q0_batch[:, None, :], q_hist], axis=1)
        v_log   = jnp.concatenate([v0_batch[:, None, :], v_hist], axis=1)
        tau_log = tau_hist
        w_log   = w_hist

        return q_log, v_log, tau_log, w_log


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


    # WARNING: / TODO: there seems to be singularity at pi/2, GIMBAL LOCK?
    def _compute_virtual_wrench_3D(self, data, q_ref_k, v_ref_k, a_ref_k, w_scale):
        """
        Compute the virtual wrench that tracks the SRB trajectory
        Fᵂ = m  (â + kₚᵛ (p̂ - p) + k_dᵛ (v̂ - v) - g)
        Mᴮ = I α̂ + kₚ^ω I (q̂ - q) + k_d^ω I (ω̂ - ω) + ω × (I ω) − r × mg

        W = w_scale * [Fᵂ, Mᴮ], w_scale ∈ [0, 1]

        Args: 
            data:    mjx.Data, current simulation state
            q_ref_k: jnp.array, shape (7,) [px, py, pz, qw, qx, qy, qz]
            v_ref_k: jnp.array, shape (6,) [vx, vy, vz, wx, wy, wz]
            a_ref_k: jnp.array, shape (6,) [ax, ay, az, alphax, alphay, alphaz]
            w_scale: jnp.float32, wrench scale factor
        Returns:
            F_W: (B, 3) world frame force to apply at the base
            M_B: (B, 3) body frame torque to apply at the base
        """
        # clip the wrench scale to be between 0 and 1
        w_scale = jnp.clip(w_scale, 0.0, 1.0)

        # current states of the mujoco 
        p_base = data.qpos[:, :3]     # (B, 3)
        quat    = data.qpos[:, 3:7]   # (B, 4)
        omega   = data.qvel[:, 3:6]   # (B, 3)

        # current com state
        p_com_batch, v_com_batch = self.dyn.com_state_in_world(data.qpos, data.qvel)  # (B, 3), (B, 3)

        # ----------------------- references -----------------------

        # reference states from the SRB trajectory
        p_com_ref = q_ref_k[:3]   # (3,)
        quat_ref  = q_ref_k[3:7]  # (4,)
        v_com_ref = v_ref_k[:3]   # (3,)
        omega_ref = v_ref_k[3:6]  # (3,)
        a_com_ref = a_ref_k[:3]   # (3,)
        alpha_ref = a_ref_k[3:6]  # (3,)
        p_com_ref_batch = jnp.broadcast_to(p_com_ref, (self.B, 3))   # (B, 3)
        v_com_ref_batch = jnp.broadcast_to(v_com_ref, (self.B, 3))   # (B, 3)
        a_com_ref_batch = jnp.broadcast_to(a_com_ref, (self.B, 3))   # (B, 3)
        quat_ref_batch  = jnp.broadcast_to(quat_ref,  (self.B, 4))   # (B, 4)
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
    

    def _compute_virtual_wrench_2D(self, data, q_ref_k, v_ref_k, a_ref_k, w_scale):
        """
        Compute the virtual wrench that tracks the planar SRB trajectory.
        Fˣ = m (âₓ + kₚ(p̂ₓ - pₓ) + kd(v̂ₓ - vₓ))
        Fᶻ = m (âᶻ + kₚ(p̂ᶻ - pᶻ) + kd(v̂ᶻ - vᶻ) - g)
        Mʸ = Iyy(α̂ + kₚ(θ̂ - θ) + kd(ω̂ - ω)) - r × mg

        W = w_scale * [Fˣ, Fᶻ, Mʸ], w_scale ∈ [0, 1]

        Args:
            data:      mjx.Data, current simulation state
            q_ref_k: jnp.array, shape (3,) [px, pz, theta]
            v_ref_k: jnp.array, shape (3,) [vx, vz, omega]
            a_ref_k: jnp.array, shape (3,) [ax, az, alpha]
            w_scale: jnp.float32, wrench scale factor
        Returns:
            F_W: (B, 2) world-frame force  [Fx, Fz]
            M_B: (B,)  body-frame torque   [My]
        """
        # clip the wrench scale to be between 0 and 1
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
        p_com_ref = q_ref_k[:2]  # (2,) [px, pz]
        v_com_ref = v_ref_k[:2]  # (2,) [vx, vz]
        a_com_ref = a_ref_k[:2]  # (2,) [ax, az]
        theta_ref = q_ref_k[2]   # scalar
        omega_ref = v_ref_k[2]   # scalar
        alpha_ref = a_ref_k[2]   # scalar

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
        r_com_x  = p_com_batch[:, 0] - p_base_x              # (B,)
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
            F_W: (B, 3) world force [3D] or (B, 2) world force [planar]
            M_B: (B, 3) body torque [3D] or (B,)  body torque [planar]
        Returns:
            data: mjx.Data, with the wrench applied in qfrc_applied
        """
        qfrc = jnp.zeros_like(data.qfrc_applied)  # (B, nv) — default: no wrench

        # 3D floating base wrench
        if self.base_type == "free":
            qfrc = qfrc.at[:, 0:3].set(F_W) # [Fx, Fy, Fz] in world frame
            qfrc = qfrc.at[:, 3:6].set(M_B) # [Mx, My, Mz] in body frame

        # planar wrench 
        elif self.base_type == "planar":
            qfrc = qfrc.at[:, 0].set(F_W[:, 0])  # Fx in world frame
            qfrc = qfrc.at[:, 1].set(F_W[:, 1])  # Fz in world frame
            qfrc = qfrc.at[:, 2].set(M_B)        # My in body frame

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
    #     0.0, 0.79, 0.0,
    #     0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    #     0.25, 1.0, 0.25, 1.0
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
    )

    # create the parallel sim object
    parallel_sim = ParallelSim(model_config, sim_config)

    # integration steps
    N = 600
    dt = float(parallel_sim.mjx_model.opt.timestep)

    #  =================== 3D SRB trajectory references ===================

    # q_srb_ref = jnp.zeros((N, 3))
    # v_srb_ref = jnp.zeros((N, 3))
    # a_srb_ref = jnp.zeros((N, 3))
    # sin_t = lambda t: jnp.sin(2 * jnp.pi * 0.5 * t)  # small oscillation at 0.5 Hz
    # cos_t = lambda t: jnp.cos(2 * jnp.pi * 0.5 * t)
    # q_srb_ref = q_srb_ref.at[:, 0].set(0.25 * sin_t(jnp.arange(N) * dt) * 0.0)  # x oscillation
    # q_srb_ref = q_srb_ref.at[:, 1].set(0.25 * cos_t(jnp.arange(N) * dt) * 0.0 + 1.0)  # z oscillation
    # q_srb_ref = q_srb_ref.at[:, 2].set(jnp.pi/2.0 * cos_t(jnp.arange(N) * dt))  # theta oscillation
    
    #  =================== 3D SRB trajectory references ===================

    q_srb_ref = jnp.zeros((N, 7))
    v_srb_ref = jnp.zeros((N, 6))
    a_srb_ref = jnp.zeros((N, 6))

    sin_t = lambda t: jnp.sin(2 * jnp.pi * 0.5 * t)  # small oscillation at 0.5 Hz
    cos_t = lambda t: jnp.cos(2 * jnp.pi * 0.5 * t)
    # q_srb_ref = q_srb_ref.at[:, 0].set(0.25 * sin_t(jnp.arange(N) * dt))  # x oscillation
    # q_srb_ref = q_srb_ref.at[:, 1].set(0.25 * cos_t(jnp.arange(N) * dt))  # y oscillation
    q_srb_ref = q_srb_ref.at[:, 2].set(0.25 * cos_t(jnp.arange(N) * dt)*0.0 + 1.0)  # z oscillation

    theta = (jnp.pi/3.0) * cos_t(jnp.arange(N) * dt)  # oscillation in pitch (N,)
    cos_theta = jnp.cos(theta / 2.0)  # (N,)
    sin_theta = jnp.sin(theta / 2.0)  # (N,)
    cos_idx = 3  # w component of quaternion
    sin_idx = 5  # x component of quaternion (small oscillation in orientation)
    q_srb_ref = q_srb_ref.at[:, cos_idx].set(cos_theta)  # quaternion w
    q_srb_ref = q_srb_ref.at[:, sin_idx].set(sin_theta)  # quaternion x (small oscillation in orientation)

    #  ====================================================================

    # initial velocity conditions
    v0 = jnp.zeros((parallel_sim.nv,))

    # random controls: (B, N, nu)
    B = sim_config.batch_size
    nu = parallel_sim.nu

    key = jax.random.PRNGKey(int(time.time()))
    key, subkey = jax.random.split(key)
    # U_B = jax.random.uniform(subkey, shape=(B, nu), minval=-1.0, maxval=1.0)   # (B, nu)
    U_B = jnp.zeros((B, nu))  # zero controls
    # U_B = jnp.broadcast_to(q_joints, (B, nu))  # hold joints at initial position
    U = jnp.broadcast_to(U_B[:, None, :], (B, N, nu))

    # test 1: no wrench — fastest path
    t0 = time.time()
    q_log, v_log, tau_log = parallel_sim.rollout(q0, v0, U)
    q_log.block_until_ready()
    print(f"No wrench: {time.time() - t0:.4f}s")

    # test 2: first wrench call (triggers JIT compilation — slow)
    t0 = time.time()
    q_log, v_log, tau_log = parallel_sim.rollout(q0, v0, U, q_srb_ref, v_srb_ref, a_srb_ref, w_scale=0.5)
    q_log.block_until_ready()
    print(f"Wrench (JIT compile): {time.time() - t0:.4f}s")

    # test 3: second wrench call — reuses compiled graph, should be fast
    t0 = time.time()
    q_log, v_log, tau_log = parallel_sim.rollout(q0, v0, U, q_srb_ref, v_srb_ref, a_srb_ref, w_scale=0.5)
    q_log.block_until_ready()
    print(f"Wrench (cached): {time.time() - t0:.4f}s")

    # test 4: different w_scale — reuses same compiled graph
    t0 = time.time()
    q_log, v_log, tau_log = parallel_sim.rollout(q0, v0, U, q_srb_ref, v_srb_ref, a_srb_ref, w_scale=1.0)
    q_log.block_until_ready()
    print(f"Wrench w_scale=1.0 (cached): {time.time() - t0:.4f}s")

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
