##
#
# Class that computes a bunch of usefuls dynamics quantities
# for a given Mujoco MJX model.
# 
##

# standard imports
import numpy as np
from dataclasses import dataclass

# jax imports
import jax
import jax.numpy as jnp
jax.config.update("jax_default_matmul_precision", "highest")

# mujoco imports
import mujoco 
import mujoco.mjx as mjx


#############################################################
# DYNAMICS PROPERTIES CLASS
#############################################################

# model config
@dataclass
class Dynamics_Config:

    # model parameters
    xml_path: str

    # number of parallerl environments
    num_envs: int

    # optional list of site names to track
    pos_sensor_names: list[str] = None
    ori_sensor_names: list[str] = None


class Dynamics:
    """
    Helper class to perform parallel computation of useful
    dynamics quantities for a given Mujoco MJX model.

    Args:
        dynamics_config (Dynamics_Config): configuration dataclass for dynamics properties
    """

    def __init__(self, dynamics_config: Dynamics_Config):

        # load the model
        self._initialize_model(dynamics_config)

        # initialize jit functions
        self._initialize_jit_functions()

        print("Initialized the dynamics class.")

    
    ################################## INITIALIZATION ##################################

    def _initialize_model(self, dynamics_config: Dynamics_Config):
        """
        Initializes the parallel Mujoco environments.

        Args: 
            dynamics_config (Dynamics_Config): configuration dataclass for dynamics properties
        """

        # mujoco model
        mj_model = mujoco.MjModel.from_xml_path(dynamics_config.xml_path)
        mj_data = mujoco.MjData(mj_model)

        # put the model on GPU
        self.mjx_model = mjx.put_model(mj_model)

        # load sizes
        self.nq = self.mjx_model.nq
        self.nv = self.mjx_model.nv
        self.nu = self.mjx_model.nu

        # batch dimension
        self.B = dynamics_config.num_envs

        # replicate data0 across batch dimension
        self.data0 = jax.vmap(
            lambda _: mjx.put_data(mj_model, mj_data)
        )(jnp.arange(self.B))

        # get the default state
        keyframe = "default"
        key_id = mj_model.key(keyframe).id
        qpos_standing = jnp.array(mj_model.key_qpos[key_id])
        qvel_standing = jnp.array(mj_model.key_qvel[key_id])
        mj_data.qpos[:] = qpos_standing
        mj_data.qvel[:] = qvel_standing
        
        # move forward one step to compute derived quantities like com_pos
        mujoco.mj_forward(mj_model, mj_data)

        # compute the total mass of the robot
        self.mass = 0.0
        for i in range(mj_model.nbody):
            self.mass += mj_model.body_mass[i]

        # body name is the first body in the model XML, which is usually the pelvis or torso; 
        # we will express inertia about this body frame
        base_body_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, 1)
        base_id = mj_model.body(base_body_name).id

        p_base_w = mj_data.xpos[base_id].copy()
        R_wb = mj_data.xmat[base_id].reshape(3, 3)  # base->world
        R_bw = R_wb.T

        # position sensor addresses
        self.pos_sensor_addr = None
        self.ns_pos = 0
        if dynamics_config.pos_sensor_names is not None:
            self.pos_sensor_addr = self._get_sensor_addresses(
                mj_model, 
                dynamics_config.pos_sensor_names)
            self.ns_pos = len(self.pos_sensor_addr)
            print(f"Initialized position sensors: {dynamics_config.pos_sensor_names}, "
                  f"total {self.ns_pos} sensors.")

        # orientation sensor addresses
        self.ori_sensor_addr = None
        self.ns_ori = 0
        if dynamics_config.ori_sensor_names is not None:
            self.ori_sensor_addr = self._get_sensor_addresses(
                mj_model, 
                dynamics_config.ori_sensor_names)
            self.ns_ori = len(self.ori_sensor_addr)
            print(f"Initialized orientation sensors: {dynamics_config.ori_sensor_names}, " 
                  f"total {self.ns_ori} sensors.")

        # compute the nominal inertia about the base body frame
        I_w = np.zeros((3, 3))
        for i in range(mj_model.nbody):
            m = mj_model.body_mass[i]
            if m == 0.0:
                continue

            # inertia about body COM, expressed in world
            I_principal = np.diag(mj_model.body_inertia[i])        # principal moments
            R_wi = mj_data.ximat[i].reshape(3, 3)                  # inertia frame -> world
            I_w_com = R_wi @ I_principal @ R_wi.T

            # shift to base origin (parallel axis)
            r = mj_data.xipos[i] - p_base_w                        # base origin -> body COM (world)
            I_w_shift = m * ((r @ r) * np.eye(3) - np.outer(r, r))

            I_w += I_w_com + I_w_shift

        # express in base frame
        I_base_ = R_bw @ I_w @ R_bw.T
        self.I_base_nom = jnp.array(I_base_, dtype=jnp.float32)  # shape (3, 3)

        # get gravity vector from model
        self.gravity = jnp.array(mj_model.opt.gravity, dtype=jnp.float32) # shape (3,)

        print(f"Total mass: {self.mass:.3f} kg")
        print(f"Inertia about {base_body_name} frame:\n{I_base_}")

    
    def _initialize_jit_functions(self):
        """
        Initializes the jit functions for computing dynamics properties.
        """

        # dynamics quantities
        self.inertia_world = jax.jit(self._inertia_world)
        self.com_state_in_world = jax.jit(self._com_state_in_world)
        self.com_state_in_world_trajectory = jax.jit(self._com_state_in_world_trajectory)
        self.centroidal_ang_mom_in_world = jax.jit(self._centroidal_ang_mom_in_world)
        self.centroidal_ang_mom_in_world_trajectory = jax.jit(self._centroidal_ang_mom_in_world_trajectory)
        self.omega_cross_Iomega = jax.jit(self._omega_cross_Iomega)
        self.omega_cross_Iomega_via_skew = jax.jit(self._omega_cross_Iomega_via_skew)

        # rotations / SO(3) maps (batched)
        self.quat_to_rot_matrix = jax.jit(self._quat_to_rot_matrix)
        self.quat_log_diff = jax.jit(self._quat_log_diff)
        self.quat_rotate = jax.jit(self._quat_rotate)
        self.vec_body_to_world = jax.jit(self._vec_body_to_world)
        self.vec_world_to_body = jax.jit(self._vec_world_to_body)
        self.vec_body_to_world_trajectory = jax.jit(self._vec_body_to_world_trajectory)
        self.vec_world_to_body_trajectory = jax.jit(self._vec_world_to_body_trajectory)

        # sensor readings
        if self.pos_sensor_addr is not None:
            self.sensor_pos_in_world = jax.jit(self._sensor_pos_in_world)
            self.sensor_pos_in_world_trajectory = jax.jit(self._sensor_pos_in_world_trajectory)
        if self.ori_sensor_addr is not None:
            self.sensor_ori_in_world = jax.jit(self._sensor_ori_in_world)
            self.sensor_ori_in_world_trajectory = jax.jit(self._sensor_ori_in_world_trajectory)

        # helpful vector operations
        self.skew = jax.jit(self._skew)

        # # ---- Optional but useful if used directly in rollouts ----
        # self.quat_normalize     = jax.jit(self._quat_normalize)
        # self.quat_conj          = jax.jit(self._quat_conj)
        # self.quat_mult          = jax.jit(self._quat_mult)
        # self.quat_diff          = jax.jit(self._quat_diff)
        # self.quat_log           = jax.jit(self._quat_log)

        print("JIT compilation of dynamics functions complete.")


    ################################## DYNAMICS ##################################

    # WARNING: assumes a fixed base inertia, no accurate, OK, aligns with ZEST
    def _inertia_world(self, quat):
        """
        Computes the inertia matrix expressed in the world frame for all parallel environments.
        Assumes quat is in [qw, qx, qy, qz] format and describes the orientation of the
        body frame relative to the world frame.

        Args:
            quat (jnp.ndarray): (B,4) array of quaternions representing body orientation
        Returns:
            I_w (jnp.ndarray): (B,3,3) array of inertia matrices in world frame
        """
        quat = self._quat_normalize(quat)          # (B,4)
        R_wb = self._quat_to_rot_matrix(quat)      # (B,3,3)
        I_world  = R_wb @ self.I_base_nom @ jnp.swapaxes(R_wb, -1, -2)
        I_world = 0.5 * (I_world + jnp.swapaxes(I_world, -1, -2))  # NOTE: enforce symmetry
        
        return I_world


    def _com_state_in_world_single_env(self, model , data):
        """
        Computes the center of mass position and velocity in the 
        world frame for a single environment.
        """
        data = mjx.kinematics(model, data)
        data = mjx.com_pos(model, data)
        com_pos = data.subtree_com[0]

        data = mjx.com_vel(model, data)
        data = mjx.subtree_vel(model, data)  
        com_vel = data._impl.subtree_linvel[0]

        return com_pos, com_vel

    def _com_state_in_world(self, q, v):
        """
        Computes the center of mass position and velocity in the world frame 
        for all parallel environments.

        Args:
            q (jnp.ndarray): (B, nq) array of generalized positions 
            v (jnp.ndarray): (B, nv) array of generalized velocities
        Returns:
            p_com (jnp.ndarray): (B, 3) array of COM positions
            v_com (jnp.ndarray): (B, 3) array of COM velocities
        """
        # build batched data from q and v, no mutation
        data = self.data0.replace(qpos=q, qvel=v)

        single = lambda d: self._com_state_in_world_single_env(self.mjx_model, d)
        p_com, v_com = jax.vmap(single)(data)

        return p_com, v_com
    
    def _com_state_in_world_trajectory(self, q_traj, v_traj):
        """
        Computes COM position and velocity over a full trajectory.

        Args:
            q_traj: (B, T, nq)
            v_traj: (B, T, nv)
        Returns:
            p_com: (B, T, 3)
            v_com: (B, T, 3)
        """
        # transpose to (T, B, nq) so we can vmap over time
        q_t = jnp.swapaxes(q_traj, 0, 1)  # (T, B, nq)
        v_t = jnp.swapaxes(v_traj, 0, 1)  # (T, B, nv)

        p_com_t, v_com_t = jax.vmap(self._com_state_in_world)(q_t, v_t)  # (T, B, 3) each

        return jnp.swapaxes(p_com_t, 0, 1), jnp.swapaxes(v_com_t, 0, 1)  # (B, T, 3)
    

    def _centroidal_ang_mom_in_world_single_env(self, model, data):
        """
        Compute the centroidal angular momentum about the COM in the world frame 
        for a single environment.
        """
        data = mjx.kinematics(model, data)
        data = mjx.com_pos(model, data)
        data = mjx.com_vel(model, data)
        data = mjx.subtree_vel(model, data)

        L = data._impl.subtree_angmom[0]  # (3,) — no batch dim inside single env
        return L

    def _centroidal_ang_mom_in_world(self, q, v):
        """
        Computes the centroidal angular momentum in the world frame
        for all parallel environments.

        Args:
            q (jnp.ndarray): (B, nq) array of generalized positions
            v (jnp.ndarray): (B, nv) array of generalized velocities
        Returns:
            L (jnp.ndarray): (B, 3) array of centroidal angular momenta [kg·m²/s]
        """
        data = self.data0.replace(qpos=q, qvel=v)

        single = lambda d: self._centroidal_ang_mom_in_world_single_env(self.mjx_model, d)
        L = jax.vmap(single)(data)

        return L
    
    def _centroidal_ang_mom_in_world_trajectory(self, q_traj, v_traj):
        """
        Computes centroidal angular momentum over a full trajectory.

        Args:
            q_traj: (B, T, nq)
            v_traj: (B, T, nv)
        Returns:
            L: (B, T, 3)
        """
        q_t = jnp.swapaxes(q_traj, 0, 1)  # (T, B, nq)
        v_t = jnp.swapaxes(v_traj, 0, 1)  # (T, B, nv)

        L_t = jax.vmap(self._centroidal_ang_mom_in_world)(q_t, v_t)  # (T, B, 3)

        return jnp.swapaxes(L_t, 0, 1)  # (B, T, 3)
    

    def _sensor_pos_in_world_single_env(self, model, data):
        """
        Reads position sensor values for a single environment.
        """
        data = mjx.forward(model, data)

        # stack all positions into (ns_pos, 3)
        positions = jnp.stack([
            data.sensordata[addr:addr+3] 
            for addr in self.pos_sensor_addr.values()
        ])  # (ns_pos, 3)

        return positions
    
    def _sensor_pos_in_world(self, q, v):
        """
        Computes positions from framepos sensors in the world frame
        for all parallel environments.

        Args:
            q (jnp.ndarray): (B, nq) array of generalized positions
            v (jnp.ndarray): (B, nv) array of generalized velocities
        Returns:
            positions (jnp.ndarray): (B, ns_pos, 3) array of sensor positions
        """
        data = self.data0.replace(qpos=q, qvel=v)

        single = lambda d: self._sensor_pos_in_world_single_env(self.mjx_model, d)
        positions = jax.vmap(single)(data)  # (B, ns_pos, 3)

        return positions
    
    def _sensor_pos_in_world_trajectory(self, q_traj, v_traj):
        """
        Computes sensor positions over a full trajectory.

        Args:
            q_traj: (B, T, nq)
            v_traj: (B, T, nv)
        Returns:
            positions: (B, T, ns_pos, 3)
        """
        q_t = jnp.swapaxes(q_traj, 0, 1)  # (T, B, nq)
        v_t = jnp.swapaxes(v_traj, 0, 1)  # (T, B, nv)

        positions_t = jax.vmap(self._sensor_pos_in_world)(q_t, v_t)  # (T, B, ns_pos, 3)

        return jnp.swapaxes(positions_t, 0, 1)  # (B, T, ns_pos, 3)

    
    def _sensor_ori_in_world_single_env(self, model, data):
        """
        Reads orientation sensor values for a single environment.
        """
        data = mjx.forward(model, data)

        # stack all quaternions into (ns_ori, 4)
        orientations = jnp.stack([
            data.sensordata[addr:addr+4]
            for addr in self.ori_sensor_addr.values()
        ])  # (ns_ori, 4)

        return orientations

    def _sensor_ori_in_world(self, q, v):
        """
        Computes orientations from frameori sensors in the world frame
        for all parallel environments.

        Args:
            q (jnp.ndarray): (B, nq) array of generalized positions
            v (jnp.ndarray): (B, nv) array of generalized velocities
        Returns:
            orientations (jnp.ndarray): (B, ns_ori, 4) array of quaternions [qw, qx, qy, qz]
        """
        data = self.data0.replace(qpos=q, qvel=v)

        single = lambda d: self._sensor_ori_in_world_single_env(self.mjx_model, d)
        orientations = jax.vmap(single)(data)  # (B, ns_ori, 4)

        return orientations
    
    def _sensor_ori_in_world_trajectory(self, q_traj, v_traj):
        """
        Computes sensor orientations over a full trajectory.

        Args:
            q_traj: (B, T, nq)
            v_traj: (B, T, nv)
        Returns:
            orientations: (B, T, ns_ori, 4)
        """
        q_t = jnp.swapaxes(q_traj, 0, 1)  # (T, B, nq)
        v_t = jnp.swapaxes(v_traj, 0, 1)  # (T, B, nv)

        orientations_t = jax.vmap(self._sensor_ori_in_world)(q_t, v_t)  # (T, B, ns_ori, 4)

        return jnp.swapaxes(orientations_t, 0, 1)  # (B, T, ns_ori, 4)


    ################################## HELPERS ##################################

    @staticmethod
    def _get_sensor_addresses(mj_model, sensor_names):
        """
        Get the address of a sensor by name.

        Args:
            mj_model: The MuJoCo model.
            sensor_names: A list of sensor names to get addresses for.

        Returns:
            sensor_addr: A dictionary mapping sensor names to their addresses.
        """
        sensor_addr = {}
        for name in sensor_names:
            # get the sensor ID and address
            sensor_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            if sensor_id == -1:
                raise ValueError(f"Sensor '{name}' not found in model.")
            sensor_addr[name] = mj_model.sensor_adr[sensor_id]

        return sensor_addr

    @staticmethod
    def _skew_single_env(w: jnp.ndarray) -> jnp.ndarray:
        """
        Vector to skew-symmetric matrix for a single environment, [ ⋅ ]ₓ

        Args:
            w: (3,) vector
        Returns:
            W: (3,3) skew-symmetric matrix such that W @ v == w x v
        """
        wx, wy, wz = w[0], w[1], w[2]
        return jnp.array([
            [0.0, -wz,  wy],
            [wz,  0.0, -wx],
            [-wy, wx,  0.0]
        ], dtype=w.dtype)

    @staticmethod
    def _skew(w: jnp.ndarray) -> jnp.ndarray:
        """
        Batched skew operator.

        Args:
            w: (B,3) vectors
        Returns:
            W: (B,3,3) skew matrices
        """
        return jax.vmap(Dynamics._skew_single_env)(w)

    @staticmethod
    def _omega_cross_Iomega(omega: jnp.ndarray, I: jnp.ndarray) -> jnp.ndarray:
        """
        Compute omega x (I omega), batched.

        Args:
            omega: (B,3)
            I:     (B,3,3) or (3,3) broadcastable
        Returns:
            term:  (B,3)
        """
        Iomega = jnp.einsum('bij,bj->bi', I, omega)  # (B,3)
        return jnp.cross(omega, Iomega)              # (B,3)

    @staticmethod
    def _omega_cross_Iomega_via_skew(omega: jnp.ndarray, I: jnp.ndarray) -> jnp.ndarray:
        """
        Same as above but via skew matrix: hat(omega) @ (I omega).

        Args:
            omega: (B,3)
            I:     (B,3,3) or (3,3) broadcastable
        Returns:
            term:  (B,3)
        """
        Iomega = jnp.einsum('bij,bj->bi', I, omega)         # (B,3)
        Omega_hat = Dynamics._skew(omega)                   # (B,3,3)
        return jnp.einsum('bij,bj->bi', Omega_hat, Iomega)  # (B,3)


    ################################## ROTATIONS ##################################

    def _quat_normalize_single_env(self, q):
        """
        Stable normalization of a single quaternion, 
        with a small epsilon to avoid division by zero.
        """
        eps = 1e-12
        q_normalized = q / (jnp.linalg.norm(q) + eps)
        return q_normalized

    def _quat_conj_single_env(self, q):
        """
        Compute the quaternion conjugate of a single quaternion.
        """
        return jnp.array([q[0], -q[1], -q[2], -q[3]])
    
    def _quat_mult_single_env(self, a, b):
        """
        Compute the Hamilton product of two quaternions.
        """
        aw, ax, ay, az = a
        bw, bx, by, bz = b
        c = jnp.array([
            aw*bw - ax*bx - ay*by - az*bz,
            aw*bx + ax*bw + ay*bz - az*by,
            aw*by - ax*bz + ay*bw + az*bx,
            aw*bz + ax*by - ay*bx + az*bw
        ])
        return c
    
    def _quat_diff_single_env(self, q1, q2):
        """
        Compute the difference between two quaternions in [qw, qx, qy, qz] format.
        Equivalent of v_diff = v2 - v1 for quaternions. This is the rotation that
        when applied to q1 gives q2. In other words, q_diff ⊗ q1 = q2.
        """
        q1 = self._quat_normalize_single_env(q1)
        q2 = self._quat_normalize_single_env(q2)
        
        # compute the difference quaternion
        q_diff = self._quat_mult_single_env(
            q2, self._quat_conj_single_env(q1)
        )
        q_diff = self._quat_normalize_single_env(q_diff)

        # enforce same hemisphere convention (qw >= 0)
        q_diff = jnp.where(q_diff[0] < 0, -q_diff, q_diff)

        return q_diff

    def _quat_log_single_env(self, q):
        """
        Log map SO(3) to so(3) for a single quaternion. 
        Returns a 3D vector representing the rotation axis scaled by the rotation angle.
        Micro Lie Theory eq.(133): https://arxiv.org/pdf/1812.01537
        """
        q = self._quat_normalize_single_env(q)

        # enforce same hemisphere convention
        q = jnp.where(q[0] < 0, -q, q)

        # extract the scalar and vector parts
        w = q[0]
        v = q[1:]
        v_norm = jnp.linalg.norm(v)

        # compute the log map
        def small_angle():
            q_log = 2.0 * v
            return q_log
        def large_angle():
            angle = 2.0 * jnp.arctan2(v_norm, w)
            u = v / (v_norm + 1e-12)
            q_log = angle * u
            return q_log

        # use a conditional to handle small angles and avoid division by zero
        q_log = jax.lax.cond(v_norm < 1e-6, small_angle, large_angle)

        return q_log
        

    def _quat_normalize(self, q):
        """
        Batched quaternion normalization.
        Args:
            q: (B,4)
        Returns:
            q_normalized: (B,4)
        """
        return jax.vmap(self._quat_normalize_single_env)(q)

    def _quat_conj(self, q):
        """
        Batched quaternion conjugate.
        Args:
            q: (B,4)
        Returns:
            q_conj: (B,4)
        """
        return jax.vmap(self._quat_conj_single_env)(q)

    def _quat_mult(self, a, b):
        """
        Batched Hamilton product.
        Args:
            a: (B,4)
            b: (B,4)
        Returns:
            c: (B,4)
        """
        return jax.vmap(self._quat_mult_single_env)(a, b)

    def _quat_diff(self, q1, q2):
        """
        Batched quaternion difference.
        Your convention: q_diff ⊗ q1 = q2, so q_diff = q2 ⊗ conj(q1).
        Args:
            q1: (B,4)
            q2: (B,4)
        Returns:
            q_diff: (B,4)
        """
        return jax.vmap(self._quat_diff_single_env)(q1, q2)

    def _quat_log(self, q):
        """
        Batched quaternion log map.
        Args:
            q: (B,4)
        Returns:
            q_log: (B,3)
        """
        return jax.vmap(self._quat_log_single_env)(q)

    def _quat_log_diff(self, q1, q2):
        """
        Batched: log( q_diff ), with your diff convention.
        Args:
            q1: (B,4)
            q2: (B,4)
        Returns:
            q_log_err: (B,3)
        """
        qd = self._quat_diff(q1, q2)
        return self._quat_log(qd)


    def _quat_to_rot_matrix_single_env(self, quat):
        """
        Assumes the quaternion is in the format [qw, qx, qy, qz].
        Here, quat describes orientation of the body frame relative to the world frame,
        vec_W = R @ vec_B, so R transforms vectors from body frame to world frame.
        """
        # normalize the quaternion
        quat = self._quat_normalize_single_env(quat)

        # rotation matrix
        w, x, y, z = quat
        R = jnp.array([
            [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
            [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
            [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)]
        ])

        return R

    def _quat_to_rot_matrix(self, quat):
        """
        Batched version of the quaternion to rotation matrix function.

        Args:
            quat (jnp.ndarray): (B, 4) array of quaternions
        Returns:
            R (jnp.ndarray): (B, 3, 3) array of rotation matrices
        """
        R = jax.vmap(self._quat_to_rot_matrix_single_env)(quat)
        return R


    def _quat_rotate_single_env(self, q, v):
        """
        Rotate a single 3D vector by a single quaternion using
        v' = q ⊗ v ⊗ q*, with v as a pure quaternion [0, v].

        Args:
            q: (4,) quaternion [qw,qx,qy,qz], rotation from frame A → frame B
            v: (3,) vector expressed in frame A

        Returns:
            v_rot: (3,) vector expressed in frame B
        """
        # Normalize for safety
        q = self._quat_normalize_single_env(q)  # (4,)

        # Lift v to a pure quaternion
        v_quat = jnp.concatenate(
            [jnp.array([0.0], dtype=v.dtype), v], axis=0
        )  # (4,)

        # q ⊗ v ⊗ q*
        q_conj = self._quat_conj_single_env(q)         # (4,)
        tmp = self._quat_mult_single_env(q, v_quat)    # (4,)
        res = self._quat_mult_single_env(tmp, q_conj)  # (4,)

        # return only the vector part
        v_rot = res[1:]  # (3,)

        return v_rot
    
    def _quat_rotate(self, quat, vec):
        """
        Batched vector rotation by quaternions.
        Args:
            quat: (B,4) [qw,qx,qy,qz], rotation from frame A → frame B
            vec : (B,3) vectors expressed in frame A
        Returns:
            vec_rot: (B,3) vectors expressed in frame B
        """
        return jax.vmap(self._quat_rotate_single_env)(quat, vec)


    def _vec_body_to_world(self, vec_B, quat):
        """
        Convert vectors from body frame to world frame (batched).
        vec_W = R_wb(quat) @ vec_B = q ⊗ v_B ⊗ q*

        Args:
            vec_B: (B,3) vectors expressed in body frame
            quat : (B,4) quaternions [qw,qx,qy,qz] describing body orientation in world.
        Returns:
            vec_W: (B,3) vectors expressed in world frame
        """
        vec_W = self._quat_rotate(quat, vec_B)
        return vec_W
    
    def _vec_world_to_body(self, vec_W, quat):
        """
        Convert vectors from world frame to body frame (batched).
        vec_B = R_wb(quat)^T @ vec_W = q* ⊗ v_W ⊗ q

        Args:
            vec_W: (B,3) vectors expressed in world frame
            quat : (B,4) quaternions [qw,qx,qy,qz] describing body orientation in world.
        Returns:
            vec_B: (B,3) vectors expressed in body frame
        """
        quat_conj = self._quat_conj(quat) 
        vec_B = self._quat_rotate(quat_conj, vec_W)
        return vec_B
    

    def _vec_body_to_world_trajectory(self, vec_B_traj, quat_traj):
        """
        Convert vectors from body frame to world frame over a full trajectory.

        Args:
            vec_B_traj : (B, T, 3) vectors expressed in body frame
            quat_traj  : (B, T, 4) quaternions [qw,qx,qy,qz] describing body orientation in world
        Returns:
            vec_W_traj : (B, T, 3) vectors expressed in world frame
        """
        vec_B_t = jnp.swapaxes(vec_B_traj, 0, 1)  # (T, B, 3)
        quat_t  = jnp.swapaxes(quat_traj,  0, 1)  # (T, B, 4)

        vec_W_t = jax.vmap(self._vec_body_to_world)(vec_B_t, quat_t)  # (T, B, 3)

        return jnp.swapaxes(vec_W_t, 0, 1)  # (B, T, 3)


    def _vec_world_to_body_trajectory(self, vec_W_traj, quat_traj):
        """
        Convert vectors from world frame to body frame over a full trajectory.

        Args:
            vec_W_traj : (B, T, 3) vectors expressed in world frame
            quat_traj  : (B, T, 4) quaternions [qw,qx,qy,qz] describing body orientation in world
        Returns:
            vec_B_traj : (B, T, 3) vectors expressed in body frame
        """
        vec_W_t = jnp.swapaxes(vec_W_traj, 0, 1)  # (T, B, 3)
        quat_t  = jnp.swapaxes(quat_traj,  0, 1)  # (T, B, 4)

        vec_B_t = jax.vmap(self._vec_world_to_body)(vec_W_t, quat_t)  # (T, B, 3)

        return jnp.swapaxes(vec_B_t, 0, 1)  # (B, T, 3)


#############################################################
# EXAMPLE USAGE
#############################################################

if __name__ == "__main__":

    import time

    # configure with a built-in mujoco model
    config = Dynamics_Config(
        # xml_path="./models/g1/g1_29dof_rev_1_0.xml",  
        xml_path="./models/g1/g1_21dof.xml",  
        # xml_path="./models/biped/biped.xml",  
        num_envs=4,
        pos_sensor_names=["left_foot_position", 
                          "right_foot_position"],
        ori_sensor_names=["left_foot_orientation", 
                          "right_foot_orientation"]
    )

    # initialize
    dyn = Dynamics(config)

    # create dummy q and v
    q = jnp.zeros((dyn.B, dyn.nq))
    v = jnp.zeros((dyn.B, dyn.nv))

    q = q.at[:, 3].set(1.0)  # quat
    q = q.at[:, 4].set(0.0)  # 
    q = q.at[:, 5].set(0.0)  # 
    q = q.at[:, 6].set(0.0)  # 

    q = q.at[0, 0].set(0.1)  # env 1 pos
    q = q.at[0, 1].set(0.2)  # 
    q = q.at[0, 2].set(0.3)  # 
    q = q.at[0, 7].set(0.1)  # 
    q = q.at[0, 8].set(0.1)  # 

    q = q.at[1, 0].set(0.4)  # env 2 pos
    q = q.at[1, 1].set(0.5)  # 
    q = q.at[1, 2].set(0.6)  # 

    v = v.at[0, 0].set(4.0)  # env 1 vel
    v = v.at[0, 1].set(5.0)  # 
    v = v.at[0, 2].set(6.0)  # 

    v = v.at[0, 6].set(1.0)  #

    v = v.at[1, 0].set(4.0)  # env 2 vel
    v = v.at[1, 1].set(5.0)  # 
    v = v.at[1, 2].set(6.0)  # 

    # first call will trigger jit compilation (slow)
    t0 = time.time()
    p_com, v_com = dyn.com_state_in_world(q, v)
    t1 = time.time()
    print("p_com shape:", p_com.shape)  # expect (4, 3)
    print("p_com:\n", p_com)  # expect (4, 3)
    print("v_com shape:", v_com.shape)  # expect (4, 3)
    print("v_com:\n", v_com)  # expect (4, 3)
    print("time: ", t1 - t0, "seconds")

    _, vc1 = dyn.com_state_in_world(q, v)
    v_shift = v.at[:, 0].add(2.0)
    _, vc2 = dyn.com_state_in_world(q, v_shift)
    print(vc2[:,0] - vc1[:,0])  # should be ~2.0 for envs with same pose

    # ---- centroidal angular momentum tests ----
    print("\n================ CENTROIDAL TESTS ================\n")

    q = jnp.zeros((dyn.B, dyn.nq))
    v = jnp.zeros((dyn.B, dyn.nv))
    q = q.at[:, 3].set(1.0)  # identity quat

    # test 1: zero velocity -> zero angular momentum
    L = dyn.centroidal_ang_mom_in_world(q, v)
    print("L (zero vel):\n", L)  # expect (B, 3) all zeros

    # test 2: pure linear COM velocity -> zero angular momentum
    # translating the whole body shouldn't contribute angular momentum about COM
    v_lin = v.at[:, 0].set(1.0).at[:, 1].set(2.0).at[:, 2].set(3.0)
    L_lin = dyn.centroidal_ang_mom_in_world(q, v_lin)
    print("L (pure linear vel, expect ~0):\n", L_lin)

    # test 3: spinning about z-axis -> L should be aligned with z
    v_spin = v.at[:, 5].set(1.0)  # wz = 1 rad/s in body frame
    L_spin = dyn.centroidal_ang_mom_in_world(q, v_spin)
    print("L (spin about z, expect L_z >> L_x, L_y):\n", L_spin)
    print("L_z / L_x ratio:", L_spin[0, 2] / (jnp.abs(L_spin[0, 0]) + 1e-9))

    # test 4: validate against manual implementation
    # compare subtree_angmom path vs our body-wise summation
    q_test = jnp.zeros((dyn.B, dyn.nq)).at[:, 3].set(1.0)
    v_test = jnp.zeros((dyn.B, dyn.nv)).at[:, 3].set(0.5).at[:, 4].set(0.3).at[:, 5].set(0.7)
    L_ours = dyn.centroidal_ang_mom_in_world(q_test, v_test)
    print("L (manual summation):\n", L_ours)

    # test 5: linearity — doubling angular velocity should double L
    L1 = dyn.centroidal_ang_mom_in_world(q, v_spin)
    v_spin2 = v.at[:, 5].set(2.0)
    L2 = dyn.centroidal_ang_mom_in_world(q, v_spin2)
    print("L2 / L1 (expect ~2.0):", L2[0] / (L1[0] + 1e-9))

    # test 6: L should be identical across envs with same q, v
    q_same = jnp.zeros((dyn.B, dyn.nq)).at[:, 3].set(1.0)
    v_same = jnp.zeros((dyn.B, dyn.nv)).at[:, 3].set(1.0)
    L_same = dyn.centroidal_ang_mom_in_world(q_same, v_same)
    print("max spread across envs (expect ~0):", jnp.max(jnp.abs(L_same - L_same[0])))


    quat = jnp.array([[1.0, 0.0, 0.0, 0.0]])  # (B=1,4)
    R = dyn.quat_to_rot_matrix(quat)[0]
    print("R(identity):\n", R)
    print("||R - I||:", jnp.linalg.norm(R - jnp.eye(3)))

    key = jax.random.PRNGKey(0)
    q = jax.random.normal(key, (10, 4))
    q = q / jnp.linalg.norm(q, axis=1, keepdims=True)

    R = dyn.quat_to_rot_matrix(q)  # (10,3,3)

    RtR = jnp.einsum("bij,bkj->bik", R, R)  # R^T R
    err_orth = jnp.max(jnp.linalg.norm(RtR - jnp.eye(3), axis=(1,2)))
    detR = jnp.linalg.det(R)

    print("max ||R^T R - I||:", err_orth)
    print("det(R) min/max:", detR.min(), detR.max())

    q = jnp.array([
        [0.9238795, 0.0, 0.0, 0.3826834],  # 45 deg about z
        [0.7071068, 0.7071068, 0.0, 0.0],  # 90 deg about x
    ])
    R1 = dyn.quat_to_rot_matrix(q)
    R2 = dyn.quat_to_rot_matrix(-q)
    print("max ||R(q)-R(-q)||:", jnp.max(jnp.abs(R1 - R2)))

    theta = jnp.pi/2
    qz = jnp.array([[jnp.cos(theta/2), 0., 0., jnp.sin(theta/2)]])
    Rz = dyn.quat_to_rot_matrix(qz)[0]
    print(Rz @ jnp.array([1.,0.,0.]))  # should be ~[0,1,0]

    def assert_close(name, x, y, atol=1e-6, rtol=1e-6):
        err = jnp.max(jnp.abs(x - y))
        ok = jnp.allclose(x, y, atol=atol, rtol=rtol)
        print(f"{name}: max|diff|={float(err):.3e}  ok={bool(ok)}")
        if not ok:
            raise AssertionError(f"{name} failed (max|diff|={float(err)})")

    def random_unit_quat(key, B):
        q = jax.random.normal(key, (B, 4))
        q = q / jnp.linalg.norm(q, axis=1, keepdims=True)
        # enforce hemisphere to avoid random sign flips in tests
        q = jnp.where(q[:, :1] < 0, -q, q)
        return q

    def quat_conj(q):
        return q * jnp.array([1., -1., -1., -1.], dtype=q.dtype)

    def quat_mul(a, b):
        aw, ax, ay, az = a.T
        bw, bx, by, bz = b.T
        return jnp.stack([
            aw*bw - ax*bx - ay*by - az*bz,
            aw*bx + ax*bw + ay*bz - az*by,
            aw*by - ax*bz + ay*bw + az*bx,
            aw*bz + ax*by - ay*bx + az*bw,
        ], axis=1)

    def skew(w):
        wx, wy, wz = w
        return jnp.array([
            [0., -wz,  wy],
            [wz,  0., -wx],
            [-wy, wx,  0.]
        ], dtype=w.dtype)

    def exp_so3(phi):
        # Rodrigues: exp([phi]x)
        theta = jnp.linalg.norm(phi)
        K = skew(phi / (theta + 1e-12))
        I = jnp.eye(3, dtype=phi.dtype)
        return jax.lax.cond(
            theta < 1e-6,
            lambda: I + skew(phi),  # first-order
            lambda: I + jnp.sin(theta)*K + (1 - jnp.cos(theta))*(K @ K)
        )

    def quat_from_axis_angle(axis, angle):
        axis = axis / (jnp.linalg.norm(axis) + 1e-12)
        return jnp.array([jnp.cos(angle/2), *(axis*jnp.sin(angle/2))])

    print("\n================ ROTATION TESTS ================\n")

    # ---------- 1) rotmat basic properties ----------
    key = jax.random.PRNGKey(42)
    B = 32
    q = random_unit_quat(key, B)
    R = dyn.quat_to_rot_matrix(q)

    # orthonormality
    RtR = jnp.einsum("bij,bkj->bik", R, R)
    assert_close("R^T R == I", RtR, jnp.tile(jnp.eye(3)[None, :, :], (B,1,1)), atol=2e-6, rtol=2e-6)

    # det = +1
    detR = jnp.linalg.det(R)
    det_err = jnp.max(jnp.abs(detR - 1.0))
    print(f"det(R) max|det-1|={float(det_err):.3e}  min={float(detR.min()):.6f} max={float(detR.max()):.6f}")
    if not jnp.all(detR > 0.0):
        raise AssertionError("det(R) not positive")

    # sign invariance
    Rneg = dyn.quat_to_rot_matrix(-q)
    assert_close("R(q) == R(-q)", R, Rneg, atol=0.0, rtol=0.0)

    # ---------- 2) known rotations / convention ----------
    theta = jnp.pi/2
    qz = jnp.array([[jnp.cos(theta/2), 0., 0., jnp.sin(theta/2)]])  # +90 about z
    Rz = dyn.quat_to_rot_matrix(qz)[0]
    assert_close("Rz * ex == ey", Rz @ jnp.array([1.,0.,0.]), jnp.array([0.,1.,0.]), atol=1e-6, rtol=1e-6)

    qx = jnp.array([[jnp.cos(theta/2), jnp.sin(theta/2), 0., 0.]])  # +90 about x
    Rx = dyn.quat_to_rot_matrix(qx)[0]
    assert_close("Rx * ey == ez", Rx @ jnp.array([0.,1.,0.]), jnp.array([0.,0.,1.]), atol=1e-6, rtol=1e-6)

    # ---------- 3) multiplication consistency: R(a⊗b) = R(a) R(b) ----------
    key, k1, k2 = jax.random.split(key, 3)
    a = random_unit_quat(k1, B)
    b = random_unit_quat(k2, B)
    ab = dyn._quat_mult(a, b)         # uses your implementation
    Rab = dyn.quat_to_rot_matrix(ab)
    Ra = dyn.quat_to_rot_matrix(a)
    Rb = dyn.quat_to_rot_matrix(b)
    assert_close("R(a⊗b) == R(a)R(b)", Rab, Ra @ Rb, atol=2e-6, rtol=2e-6)

    # ---------- 4) diff convention: qd ⊗ q1 = q2 ----------
    key, k1, k2 = jax.random.split(key, 3)
    q1 = random_unit_quat(k1, B)
    q2 = random_unit_quat(k2, B)
    qd = dyn._quat_diff(q1, q2)
    recon = dyn._quat_mult(qd, q1)
    # since you enforce hemisphere in diff, recon might differ by sign; compare rotation matrices
    assert_close("qd ⊗ q1 matches q2 (as rotation)", dyn.quat_to_rot_matrix(recon), dyn.quat_to_rot_matrix(q2), atol=2e-6, rtol=2e-6)

    # also: if q2 == q1, diff should be identity
    qd_id = dyn._quat_diff(q1, q1)
    assert_close("diff(q,q) == identity", qd_id, jnp.tile(jnp.array([1.,0.,0.,0.])[None,:], (B,1)), atol=2e-6, rtol=2e-6)

    # ---------- 5) log map sanity ----------
    # (a) log(identity) = 0
    qI = jnp.tile(jnp.array([1.,0.,0.,0.])[None,:], (B,1))
    logI = dyn._quat_log(qI)
    assert_close("log(I) == 0", logI, jnp.zeros((B,3)), atol=1e-7, rtol=0.0)

    # (b) for small angles, log(q) ≈ angle*axis
    axis = jnp.array([0., 0., 1.])
    angle = 1e-4
    qs = quat_from_axis_angle(axis, angle)[None,:]  # (1,4)
    logs = dyn._quat_log(qs)[0]
    assert_close("small-angle log ≈ angle*axis", logs, angle*axis, atol=1e-7, rtol=1e-3)

    # (c) log_diff matches relative rotation matrix for moderate angles
    axis = jnp.array([0.3, -0.4, 0.5])
    axis = axis / jnp.linalg.norm(axis)
    angle = 0.7
    qd_true = quat_from_axis_angle(axis, angle)
    q1 = random_unit_quat(k1, B)
    q2 = dyn._quat_mult(jnp.tile(qd_true[None,:], (B,1)), q1)  # q2 = qd_true ⊗ q1
    phi = dyn.quat_log_diff(q1, q2)  # should be ~ angle*axis for all envs
    assert_close("log_diff recovers axis-angle", phi, jnp.tile((angle*axis)[None,:], (B,1)), atol=2e-5, rtol=2e-4)

    # (d) exp(log(q)) consistency via rotation matrices
    # build qd from random axis-angle, take log, exponentiate to R, compare
    axis = jnp.array([1., 2., 3.]); axis = axis / jnp.linalg.norm(axis)
    angle = 1.2
    qd = quat_from_axis_angle(axis, angle)[None,:]
    phi = dyn._quat_log(qd)[0]
    R_from_exp = exp_so3(phi)
    R_from_quat = dyn.quat_to_rot_matrix(qd)[0]
    assert_close("exp(log(q)) matches R(q)", R_from_exp, R_from_quat, atol=2e-5, rtol=2e-5)

    print("\n✅ All rotation tests passed.\n")

    print("\n================ VEC + INERTIA TESTS ================\n")

    # ---------- Vec body->world tests ----------
    key = jax.random.PRNGKey(123)
    B = 32

    quat = random_unit_quat(key, B)                 # (B,4)  qw qx qy qz
    vec_B = jax.random.normal(key, (B, 3))          # (B,3)

    # (a) vec_W == R(q) @ vec_B
    R = dyn.quat_to_rot_matrix(quat)                # (B,3,3)
    vec_W_ref = jnp.einsum("bij,bj->bi", R, vec_B)
    vec_W = dyn.vec_body_to_world(vec_B, quat)
    assert_close("vec_body_to_world matches R@v", vec_W, vec_W_ref, atol=2e-6, rtol=2e-6)

    # (b) identity quaternion leaves vectors unchanged
    qI = jnp.tile(jnp.array([1., 0., 0., 0.], dtype=quat.dtype)[None, :], (B, 1))
    vec_W_id = dyn.vec_body_to_world(vec_B, qI)
    assert_close("vec_body_to_world identity", vec_W_id, vec_B, atol=1e-6, rtol=1e-6)

    # (c) round-trip: v_B == R^T (R v_B)
    vec_B_rec = jnp.einsum("bji,bj->bi", R, vec_W)  # R^T @ vec_W
    assert_close("vec round-trip R^T(Rv)=v", vec_B_rec, vec_B, atol=2e-6, rtol=2e-6)

    # ---------- inertia_world tests ----------
    I_base_nom = dyn.I_base_nom

    # (a) inertia_world == R I_base R^T
    I_w_ref = R @ I_base_nom @ jnp.swapaxes(R, -1, -2)  # (B,3,3)
    I_w = dyn.inertia_world(quat)
    assert_close("inertia_world matches R I R^T", I_w, I_w_ref, atol=2e-6, rtol=2e-6)

    # (b) "identity" test done robustly: use R(qI) instead of assuming R=I
    R_I = dyn.quat_to_rot_matrix(qI)  # (B,3,3)
    err_RI = jnp.max(jnp.linalg.norm(R_I - jnp.tile(jnp.eye(3, dtype=R_I.dtype)[None,:,:], (B,1,1)), axis=(1,2)))
    print(f"||R(qI) - I|| max: {float(err_RI):.6e}")

    I_expected_id = R_I @ I_base_nom @ jnp.swapaxes(R_I, -1, -2)
    I_w_id = dyn.inertia_world(qI)
    assert_close("inertia_world at qI matches R(qI) I R(qI)^T", I_w_id, I_expected_id, atol=2e-6, rtol=2e-6)

    # (c) symmetry (numerical)
    sym_err = jnp.max(jnp.abs(I_w - jnp.swapaxes(I_w, -1, -2)))
    print(f"I_world symmetry max|I-I^T|={float(sym_err):.3e}")
    if sym_err > 5e-6:
        raise AssertionError("I_world not symmetric enough")

    # (d) eigenvalues invariant (principal moments)
    evals_base = jnp.sort(jnp.linalg.eigvalsh(0.5 * (I_base_nom + I_base_nom.T)))
    evals_w = jnp.sort(
        jnp.linalg.eigvalsh(0.5 * (I_w + jnp.swapaxes(I_w, -1, -2))),
        axis=-1
    )
    assert_close(
        "eig(I_world) == eig(I_base)",
        evals_w,
        jnp.tile(evals_base[None, :], (B, 1)).astype(evals_w.dtype),
        atol=5e-5,
        rtol=5e-5
    )

    print("\n✅ Vec + inertia tests passed.\n")


    print("\n================ SENSOR + TRAJECTORY TESTS ================\n")

    # reset q and v to match dyn.B
    q = jnp.zeros((dyn.B, dyn.nq)).at[:, 3].set(1.0)
    v = jnp.zeros((dyn.B, dyn.nv))

    # (a) output shape
    pos = dyn.sensor_pos_in_world(q, v)
    assert pos.shape == (dyn.B, dyn.ns_pos, 3), f"Expected ({dyn.B}, {dyn.ns_pos}, 3), got {pos.shape}"
    print(f"sensor_pos_in_world shape: {pos.shape}  ✅")

    # (b) identical states -> identical sensor readings across envs
    q_same = jnp.zeros((dyn.B, dyn.nq)).at[:, 3].set(1.0)
    v_same = jnp.zeros((dyn.B, dyn.nv))
    pos_same = dyn.sensor_pos_in_world(q_same, v_same)
    spread = jnp.max(jnp.abs(pos_same - pos_same[0]))
    print(f"sensor_pos identical envs max spread (expect ~0): {float(spread):.3e}")
    assert spread < 1e-6, f"Sensor positions differ across identical envs: {spread}"

    # (c) different base positions -> sensor readings should differ
    q_shifted = q_same.at[0, 0].set(1.0)  # shift env 0 by 1m in x
    pos_shifted = dyn.sensor_pos_in_world(q_shifted, v_same)
    diff_x = pos_shifted[0, :, 0] - pos_shifted[1, :, 0]  # x-component difference
    print(f"sensor_pos x-shift diff (expect ~1.0): {diff_x}")
    assert jnp.all(jnp.abs(diff_x - 1.0) < 1e-4), f"Sensor x-shift not reflected correctly: {diff_x}"

    # ---- sensor orientation tests ----

    # (d) output shape
    ori = dyn.sensor_ori_in_world(q, v)
    assert ori.shape == (dyn.B, dyn.ns_ori, 4), f"Expected ({dyn.B}, {dyn.ns_ori}, 4), got {ori.shape}"
    print(f"sensor_ori_in_world shape: {ori.shape}  ✅")

    # (e) output quaternions should be unit norm
    ori_norms = jnp.linalg.norm(ori, axis=-1)  # (B, ns_ori)
    norm_err = jnp.max(jnp.abs(ori_norms - 1.0))
    print(f"sensor_ori quaternion norm error (expect ~0): {float(norm_err):.3e}")
    assert norm_err < 1e-5, f"Sensor orientations not unit quaternions: {norm_err}"

    # (f) identical states -> identical orientations across envs
    ori_same = dyn.sensor_ori_in_world(q_same, v_same)
    ori_spread = jnp.max(jnp.abs(ori_same - ori_same[0]))
    print(f"sensor_ori identical envs max spread (expect ~0): {float(ori_spread):.3e}")
    assert ori_spread < 1e-6, f"Sensor orientations differ across identical envs: {ori_spread}"

    # ---- trajectory tests ----

    T = 10
    q_traj = jnp.tile(q_same[:, None, :], (1, T, 1))  # (B, T, nq) constant trajectory
    v_traj = jnp.zeros((dyn.B, T, dyn.nv))

    # (g) com trajectory shape
    p_com_traj, v_com_traj = dyn.com_state_in_world_trajectory(q_traj, v_traj)
    assert p_com_traj.shape == (dyn.B, T, 3), f"Expected ({dyn.B}, {T}, 3), got {p_com_traj.shape}"
    assert v_com_traj.shape == (dyn.B, T, 3), f"Expected ({dyn.B}, {T}, 3), got {v_com_traj.shape}"
    print(f"com_state_in_world_trajectory shapes: {p_com_traj.shape}, {v_com_traj.shape}  ✅")

    # (h) trajectory matches single-step for each timestep
    p_com_single, _ = dyn.com_state_in_world(q_same, v_same)
    traj_err = jnp.max(jnp.abs(p_com_traj - p_com_single[:, None, :]))
    print(f"com_traj vs single-step max err (expect ~0): {float(traj_err):.3e}")
    assert traj_err < 1e-5, f"COM trajectory inconsistent with single-step: {traj_err}"

    # (i) sensor pos trajectory shape
    pos_traj = dyn.sensor_pos_in_world_trajectory(q_traj, v_traj)
    assert pos_traj.shape == (dyn.B, T, dyn.ns_pos, 3), \
        f"Expected ({dyn.B}, {T}, {dyn.ns_pos}, 3), got {pos_traj.shape}"
    print(f"sensor_pos_in_world_trajectory shape: {pos_traj.shape}  ✅")

    # (j) sensor pos trajectory matches single-step
    pos_single = dyn.sensor_pos_in_world(q_same, v_same)  # (B, ns_pos, 3)
    pos_traj_err = jnp.max(jnp.abs(pos_traj - pos_single[:, None, :, :]))
    print(f"sensor_pos_traj vs single-step max err (expect ~0): {float(pos_traj_err):.3e}")
    assert pos_traj_err < 1e-5, f"Sensor pos trajectory inconsistent with single-step: {pos_traj_err}"

    # (k) sensor ori trajectory shape
    ori_traj = dyn.sensor_ori_in_world_trajectory(q_traj, v_traj)
    assert ori_traj.shape == (dyn.B, T, dyn.ns_ori, 4), \
        f"Expected ({dyn.B}, {T}, {dyn.ns_ori}, 4), got {ori_traj.shape}"
    print(f"sensor_ori_in_world_trajectory shape: {ori_traj.shape}  ✅")

    # (l) sensor ori trajectory matches single-step
    ori_single = dyn.sensor_ori_in_world(q_same, v_same)  # (B, ns_ori, 4)
    ori_traj_err = jnp.max(jnp.abs(ori_traj - ori_single[:, None, :, :]))
    print(f"sensor_ori_traj vs single-step max err (expect ~0): {float(ori_traj_err):.3e}")
    assert ori_traj_err < 1e-5, f"Sensor ori trajectory inconsistent with single-step: {ori_traj_err}"

    # (m) centroidal angular momentum trajectory shape
    L_traj = dyn.centroidal_ang_mom_in_world_trajectory(q_traj, v_traj)
    assert L_traj.shape == (dyn.B, T, 3), f"Expected ({dyn.B}, {T}, 3), got {L_traj.shape}"
    print(f"centroidal_ang_mom_in_world_trajectory shape: {L_traj.shape}  ✅")

    # (n) centroidal trajectory matches single-step
    L_single = dyn.centroidal_ang_mom_in_world(q_same, v_same)  # (B, 3)
    L_traj_err = jnp.max(jnp.abs(L_traj - L_single[:, None, :]))
    print(f"L_traj vs single-step max err (expect ~0): {float(L_traj_err):.3e}")
    assert L_traj_err < 1e-5, f"L trajectory inconsistent with single-step: {L_traj_err}"

    print("\n✅ All sensor + trajectory tests passed.\n")


    print("\n================ VEC TRAJECTORY TESTS ================\n")

    T = 10
    key = jax.random.PRNGKey(456)

    # constant trajectory: tile a single (B, nq) state across T timesteps
    q_same = jnp.zeros((dyn.B, dyn.nq)).at[:, 3].set(1.0)
    quat_same = random_unit_quat(key, dyn.B)  # (B, 4)
    vec_B_same = jax.random.normal(key, (dyn.B, 3))  # (B, 3)

    quat_traj = jnp.tile(quat_same[:, None, :], (1, T, 1))  # (B, T, 4)
    vec_B_traj = jnp.tile(vec_B_same[:, None, :], (1, T, 1))  # (B, T, 3)

    # (a) output shape — body to world
    vec_W_traj = dyn.vec_body_to_world_trajectory(vec_B_traj, quat_traj)
    assert vec_W_traj.shape == (dyn.B, T, 3), \
        f"Expected ({dyn.B}, {T}, 3), got {vec_W_traj.shape}"
    print(f"vec_body_to_world_trajectory shape: {vec_W_traj.shape}  ✅")

    # (b) trajectory matches single-step for each timestep
    vec_W_single = dyn.vec_body_to_world(vec_B_same, quat_same)  # (B, 3)
    traj_err = jnp.max(jnp.abs(vec_W_traj - vec_W_single[:, None, :]))
    print(f"vec_body_to_world_traj vs single-step max err (expect ~0): {float(traj_err):.3e}")
    assert traj_err < 1e-6, f"vec_body_to_world trajectory inconsistent with single-step: {traj_err}"

    # (c) output shape — world to body
    vec_W_same = jax.random.normal(key, (dyn.B, 3))  # (B, 3)
    vec_W_traj2 = jnp.tile(vec_W_same[:, None, :], (1, T, 1))  # (B, T, 3)

    vec_B_traj_out = dyn.vec_world_to_body_trajectory(vec_W_traj2, quat_traj)
    assert vec_B_traj_out.shape == (dyn.B, T, 3), \
        f"Expected ({dyn.B}, {T}, 3), got {vec_B_traj_out.shape}"
    print(f"vec_world_to_body_trajectory shape: {vec_B_traj_out.shape}  ✅")

    # (d) trajectory matches single-step for each timestep
    vec_B_single = dyn.vec_world_to_body(vec_W_same, quat_same)  # (B, 3)
    traj_err2 = jnp.max(jnp.abs(vec_B_traj_out - vec_B_single[:, None, :]))
    print(f"vec_world_to_body_traj vs single-step max err (expect ~0): {float(traj_err2):.3e}")
    assert traj_err2 < 1e-6, f"vec_world_to_body trajectory inconsistent with single-step: {traj_err2}"

    # (e) round-trip: body->world->body should recover original
    vec_W_rt = dyn.vec_body_to_world_trajectory(vec_B_traj, quat_traj)   # (B, T, 3)
    vec_B_rt = dyn.vec_world_to_body_trajectory(vec_W_rt, quat_traj)      # (B, T, 3)
    rt_err = jnp.max(jnp.abs(vec_B_rt - vec_B_traj))
    print(f"round-trip body->world->body max err (expect ~0): {float(rt_err):.3e}")
    assert rt_err < 2e-6, f"Round-trip trajectory failed: {rt_err}"

    # (f) identity quaternion trajectory leaves vectors unchanged
    qI_traj = jnp.tile(
        jnp.array([1., 0., 0., 0.], dtype=quat_traj.dtype)[None, None, :],
        (dyn.B, T, 1)
    )  # (B, T, 4)
    vec_W_id = dyn.vec_body_to_world_trajectory(vec_B_traj, qI_traj)
    assert_close("vec_body_to_world_traj identity quat", vec_W_id, vec_B_traj, atol=1e-6, rtol=1e-6)

    print("\n✅ All vec trajectory tests passed.\n")