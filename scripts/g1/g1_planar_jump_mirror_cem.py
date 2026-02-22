##
#
# G1 Walk CEM
#
##

import config

# standard imports
import numpy as np

# jax imports
import jax
import jax.numpy as jnp

# custom imports
from utils.algorithms.cem import *
from utils.simulation.simulation import *
from utils.interpolation import interp
from utils.spline.bezier import *
from utils.spline.zoh import *
from utils.spline.linear import *
from utils.spline.cubic import *
from utils.spline.fourier import *

# load the a model config
xml_path = "./models/g1/g1_planar.xml"
mj_model = mujoco.MjModel.from_xml_path(xml_path)
mj_data = mujoco.MjData(mj_model)

# standing
keyframe = "standing"
key_id = mj_model.key(keyframe).id
qpos_standing = jnp.array(mj_model.key_qpos[key_id])
qvel_standing = jnp.array(mj_model.key_qvel[key_id])

# load SRB data
srb_dir = "./results/srb/srb_jump_2d/"
t_SRB = np.loadtxt(srb_dir + "time.csv", delimiter=",")      # (T, )

#############################################################
# G1 Walk CEM
#############################################################

# joint indices
base_idx = jnp.array([0, 1])
ori_idx = jnp.array([2])
hip_idx = jnp.array([3, 6])
knee_idx = jnp.array([4, 7])
ankle_idx = jnp.array([5, 8])
shoulder_idx = jnp.array([9, 11])
elbow_idx = jnp.array([10, 12])
joints_idx = jnp.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

# joint mirroring index (mirror left with right corresponding joints)
state_mirror_dict   = {3:6, 4:7, 5:8, 9:11, 10:12}  # hip, knee, ankle, shoulder, elbow
control_mirror_dict = {0:3, 1:4, 2:5, 6:8, 7:9}     # hip, knee, ankle, shoulder, elbow

# CEM optimizer class
class G1_Walk_Mirrored_CEM(CrossEntropyMethod):

    def __init__(self, model_config: Model_Config,
                       sim_config:   ParallelSim_Config,
                       cem_config:   CrossEntropyMethod_Config):
        
        # initialize the parent class
        super().__init__(model_config, sim_config, cem_config)

        # reinitialize the initial spline sampling
        self._initialize_spline_knots()

        # Create reference trajectory (simple linear interpolation)
        self._make_SRB_reference()
        self._make_joint_reference()

        # Running cost weights (per timestep)
        self.w_px      = 5.0     # horizontal position tracking
        self.w_pz      = 20.0    # vertical position tracking (keep at default height)
        self.w_theta   = 5.0    # pitch angle (stay upright)
        
        self.w_vx      = 1.0     # forward velocity tracking
        self.w_vz      = 1.0     # vertical velocity tracking
        self.w_omega   = 0.1     # pitch velocity tracking

        self.w_p_hip   = 0.1     # hip joint tracking
        self.w_p_knee  = 0.05     # knee joint tracking
        self.w_p_ankle = 0.05     # ankle joint tracking
        self.w_p_shoulder = 0.01  # shoulder joint tracking
        self.w_p_elbow = 0.01     # elbow joint tracking
        
        self.w_v_hip   = 0.01     # hip joint velocity tracking
        self.w_v_knee  = 0.01     # knee joint velocity tracking
        self.w_v_ankle = 0.01     # ankle joint velocity tracking
        self.w_v_shoulder = 0.01  # shoulder joint velocity tracking
        self.w_v_elbow = 0.01     # elbow joint velocity tracking
        self.w_control = 0.00001  # control effort

        terminal_scale = 10.0

        self.wf_px = terminal_scale * self.w_px
        self.wf_pz = terminal_scale * self.w_pz
        self.wf_theta = terminal_scale * self.w_theta

        self.wf_vx = terminal_scale * self.w_vx
        self.wf_vz = terminal_scale * self.w_vz
        self.wf_omega = terminal_scale * self.w_omega
        
        self.wf_p_hip = terminal_scale * self.w_p_hip
        self.wf_p_knee = terminal_scale * self.w_p_knee
        self.wf_p_ankle = terminal_scale * self.w_p_ankle
        self.wf_p_shoulder = terminal_scale * self.w_p_shoulder
        self.wf_p_elbow = terminal_scale * self.w_p_elbow
        
        self.wf_v_hip = terminal_scale * self.w_v_hip
        self.wf_v_knee = terminal_scale * self.w_v_knee
        self.wf_v_ankle = terminal_scale * self.w_v_ankle
        self.wf_v_shoulder = terminal_scale * self.w_v_shoulder
        self.wf_v_elbow = terminal_scale * self.w_v_elbow


    def _initialize_spline_knots(self):
        """ 
        Depending on the actuation mode, initialize the spline knot points
        within the position or control limits. NOTE: does not use a
        pre-defined mu and Sigma.
        """

        # size of the mirror
        nj = len(control_mirror_dict)

        # initialize an empty array for the knot points
        y_size = (self.sim.B, self.cem_config.N_knots, nj)
        Y0 = jnp.zeros(y_size)

        # knot points represent desired positions
        if self.sim.use_pd == True:
            # get the position limits at actuated joints
            pos_limits = self.sim.pos_limits  # shape (nu, 2)

            # create initial knot points within the position limits
            for i, key in enumerate(control_mirror_dict):

                # get low and high limits
                pos_lo = pos_limits[key, 0] * self.cem_config.initial_action_range_scale
                pos_hi = pos_limits[key, 1] * self.cem_config.initial_action_range_scale

                # error out if no position limits are defined
                if abs(pos_hi) <1e-6 and abs(pos_lo) <1e-6:
                    raise ValueError(f"No position limits defined at actuated actuator index {key}.")

                # split the key
                self.rng, subkey = jax.random.split(self.rng)

                # replace the ith dimension of all knot points
                y_rand = jax.random.uniform(
                        subkey,
                        shape=(self.sim.B, self.cem_config.N_knots),
                        minval=pos_lo,
                        maxval=pos_hi
                )
                Y0 = Y0.at[:, :, i].set(y_rand)

        # knot points represent direct torques
        else:
            # get the control limits at actuated joints
            ctrl_limits = self.sim.ctrl_limits  # shape (nu, 2)

            # create initial knot points within the control limits
            for _, key in enumerate(control_mirror_dict):
                
                # get low and high limits
                tau_lo = ctrl_limits[key, 0] * self.cem_config.initial_action_range_scale
                tau_hi = ctrl_limits[key, 1] * self.cem_config.initial_action_range_scale

                # error out if no control limits are defined
                if abs(tau_hi) <1e-6 and abs(tau_lo) <1e-6:
                    raise ValueError(f"No control limits defined at actuated actuator index {key}.")

                # split the key
                self.rng, subkey = jax.random.split(self.rng)

                # replace the ith dimension of all knot points
                y_rand = jax.random.uniform(
                        subkey,
                        shape=(self.sim.B, self.cem_config.N_knots),
                        minval=pos_lo,
                        maxval=pos_hi
                )
                Y0 = Y0.at[:, :, i].set(y_rand)

        # create the spline object
        if self.cem_config.spline_type == "ZOH":
            self.spline = ZOH_Spline(Y0, self.T_eff)
        elif self.cem_config.spline_type == "Linear":
            self.spline = Linear_Spline(Y0, self.T_eff)
        elif self.cem_config.spline_type == "Cubic":
            self.spline = Cubic_Spline(Y0, self.T_eff)
        elif self.cem_config.spline_type == "Bezier":
            self.spline = Bezier_Spline(Y0, self.T_eff)
        elif self.cem_config.spline_type == "Fourier":
            self.spline = Fourier_Spline(Y0, self.T_eff, periodic=False)
        else:
            raise NotImplementedError(f"Spline type [{self.cem_config.spline_type}] not implemented.")

        # update the spline knot points
        self.spline.update_knots(Y0)

        # update the distribution with the initial knot points
        self._update_distribution(Y0)

    def _make_SRB_reference(self):
        
        # load the rest of the SRB reference data
        t_SRB = np.loadtxt(srb_dir + "time.csv", delimiter=",")      # (T, )
        q_SRB = np.loadtxt(srb_dir + "q_opt.csv", delimiter=",")     # (T, nq)
        v_SRB = np.loadtxt(srb_dir + "v_opt.csv", delimiter=",")     # (T, nv)
        a_SRB = np.loadtxt(srb_dir + "a_opt.csv", delimiter=",")     # (T, na)
        tau_SRB = np.loadtxt(srb_dir + "tau_opt.csv", delimiter=",") # (T-1, nu)

        # extract all the data
        p_com_traj = q_SRB[:, :2]    # world com position
        v_com_traj = v_SRB[:, :2]    # world com linear velocity
        a_com_traj = a_SRB[:, :2]    # world com linear acceleration
        theta_traj = q_SRB[:, 2]     # body frame pitch angle
        omega_traj = v_SRB[:, 2]     # body frame angular velocity
        alpha_traj = a_SRB[:, 2]     # body frame angular acceleration
        F_W_traj   = tau_SRB[:, :2]  # world forces
        M_W_traj   = tau_SRB[:, 2]   # world moments

        # add the last element to make it length T (for reference tracking)
        F_W_traj = np.vstack([F_W_traj, F_W_traj[-1]])
        M_W_traj = np.hstack([M_W_traj, M_W_traj[-1]])

        # trajectory length and time
        dt_sim = self.sim.dt
        t0 = t_SRB[0]
        tf = t_SRB[-1]

        # time array 
        t_sim = np.arange(t0, tf, dt_sim, dtype=np.float64)
        N_sim = t_sim.shape[0]

        # allocate the SRB trajectory in the class
        p_com_ref = np.zeros((N_sim, 2), dtype=np.float32)
        v_com_ref = np.zeros((N_sim, 2), dtype=np.float32)
        a_com_ref = np.zeros((N_sim, 2), dtype=np.float32)
        theta_ref = np.zeros((N_sim, 1), dtype=np.float32)
        omega_ref = np.zeros((N_sim, 1), dtype=np.float32)
        alpha_ref = np.zeros((N_sim, 1), dtype=np.float32)
        F_W_ref = np.zeros((N_sim, 2), dtype=np.float32)
        M_W_ref = np.zeros((N_sim, 1), dtype=np.float32)

        for k in range(N_sim):
            t = float(t_sim[k])

            idx_2 = int(np.searchsorted(t_SRB, t, side="right"))
            idx_1 = idx_2 - 1

            if idx_2 >= len(t_SRB):
                idx_1 = idx_2 = len(t_SRB) - 1
                coeff = 0.0
            elif idx_1 < 0:
                idx_1 = idx_2 = 0
                coeff = 0.0
            else:
                t1_ = float(t_SRB[idx_1])
                t2_ = float(t_SRB[idx_2])
                denom = (t2_ - t1_)
                if abs(denom) < 1e-12:
                    coeff = 0.0
                else:
                    coeff = (t - t1_) / denom
                    coeff = float(np.clip(coeff, 0.0, 1.0))

            p_com_ref[k] = interp.lerp(p_com_traj[idx_1], p_com_traj[idx_2], coeff)
            v_com_ref[k] = interp.lerp(v_com_traj[idx_1], v_com_traj[idx_2], coeff)
            a_com_ref[k] = interp.lerp(a_com_traj[idx_1], a_com_traj[idx_2], coeff)
            theta_ref[k] = interp.lerp(theta_traj[idx_1], theta_traj[idx_2], coeff)
            omega_ref[k] = interp.lerp(omega_traj[idx_1], omega_traj[idx_2], coeff)
            alpha_ref[k] = interp.lerp(alpha_traj[idx_1], alpha_traj[idx_2], coeff)
            F_W_ref[k] = interp.lerp(F_W_traj[idx_1], F_W_traj[idx_2], coeff)
            M_W_ref[k] = interp.lerp(M_W_traj[idx_1], M_W_traj[idx_2], coeff)

        # store the reference trajectory in the class (as jnp for use in cost)
        self.p_com_ref = jnp.array(p_com_ref)  # (N_sim, 2)
        self.v_com_ref = jnp.array(v_com_ref)  # (N_sim, 2)
        self.a_com_ref = jnp.array(a_com_ref)  # (N_sim, 2)
        self.F_W_ref = jnp.array(F_W_ref)      # (N_sim, 2)

        # WARNING: the angular references are s.t. positive is CCW. I.e., y+ is going out of the page
        #          you must account for this when computing costs.
        self.theta_ref = -jnp.array(theta_ref)  # (N_sim, 1)
        self.omega_ref = -jnp.array(omega_ref)  # (N_sim, 1)
        self.alpha_ref = -jnp.array(alpha_ref)  # (N_sim, 1)
        self.M_W_ref = -jnp.array(M_W_ref)      # (N_sim, 1)

        # precompute convenience views for cost (strip trailing dim where needed)
        px_ref = self.p_com_ref[:, 0]        # (N_sim,)
        pz_ref = self.p_com_ref[:, 1]        # (N_sim,)
        vx_ref = self.v_com_ref[:, 0]        # (N_sim,)
        vz_ref = self.v_com_ref[:, 1]        # (N_sim,)
        thetadot_ref = self.omega_ref[:, 0]  # (N_sim,)
        theta_ref = self.theta_ref[:, 0]     # (N_sim,)
        
        # pad all refs by 1 to handle N+1 vs N_sim edge cases
        self.px_ref       = jnp.concatenate([px_ref,       px_ref[-1:]])
        self.pz_ref       = jnp.concatenate([pz_ref,       pz_ref[-1:]])
        self.vx_ref       = jnp.concatenate([vx_ref,       vx_ref[-1:]])
        self.vz_ref       = jnp.concatenate([vz_ref,       vz_ref[-1:]])
        self.theta_ref    = jnp.concatenate([theta_ref,    theta_ref[-1:]])
        self.thetadot_ref = jnp.concatenate([thetadot_ref, thetadot_ref[-1:]])

    # make the joint reference
    def _make_joint_reference(self):

        # take joint state references
        self.p_hip_ref = qpos_standing[3]      # scalar
        self.p_knee_ref = qpos_standing[4]   
        self.p_ankle_ref = qpos_standing[5]
        self.p_shoulder_ref = qpos_standing[9]
        self.p_elbow_ref = qpos_standing[10]

        # take the joint velocity references
        self.v_hip_ref = qvel_standing[3]      # scalar
        self.v_knee_ref = qvel_standing[4]
        self.v_ankle_ref = qvel_standing[5]
        self.v_shoulder_ref = qvel_standing[9]
        self.v_elbow_ref = qvel_standing[10]

    def cost(self, q, v, tau):
        """
        Args:
            q:   (B, N+1, nq)
            v:   (B, N+1, nv)
            tau: (B, N, nu)
        Returns:
            J: (B,)
        """
        B, N_plus_1, _ = q.shape
        N = N_plus_1 - 1

        # ===== BASE STATE =====
        px    = q[:, :, 0]     # (B, N+1)
        pz    = q[:, :, 1]     # (B, N+1)
        theta = q[:, :, 2]     # (B, N+1)
        vx       = v[:, :, 0]  # (B, N+1)
        vz       = v[:, :, 1]  # (B, N+1)
        thetadot = v[:, :, 2]  # (B, N+1)

        px_ref = (self.px_ref[:N_plus_1] - self.px_ref[0])[None, :] + px[:, 0:1]  # (B, N+1)
        pz_ref = self.pz_ref[:N_plus_1][None, :]
        theta_ref = self.theta_ref[:N_plus_1][None, :]
        vx_ref = self.vx_ref[:N_plus_1][None, :]
        vz_ref = self.vz_ref[:N_plus_1][None, :]
        thetadot_ref = self.thetadot_ref[:N_plus_1][None, :]

        # error state
        err_px = px_ref - px
        err_pz = pz_ref - pz
        err_theta = theta_ref - theta
        err_vx = vx_ref - vx
        err_vz = vz_ref - vz
        err_thetadot = thetadot_ref - thetadot

        # ===== JOINT STATE =====
        
        # only the left joint state
        p_hip      = q[:, :, 3]  # (B, N+1)
        p_knee     = q[:, :, 4]  # (B, N+1)
        p_ankle    = q[:, :, 5]  # (B, N+1)
        p_shoulder = q[:, :, 9]  # (B, N+1)
        p_elbow    = q[:, :, 10] # (B, N+1)
        
        v_hip      = v[:, :, 3]
        v_knee     = v[:, :, 4]
        v_ankle    = v[:, :, 5]
        v_shoulder = v[:, :, 9]
        v_elbow    = v[:, :, 10]

        # joint references
        err_p_hip = self.p_hip_ref - p_hip
        err_p_knee = self.p_knee_ref - p_knee
        err_p_ankle = self.p_ankle_ref - p_ankle
        err_p_shoulder = self.p_shoulder_ref - p_shoulder
        err_p_elbow = self.p_elbow_ref - p_elbow 

        err_v_hip = self.v_hip_ref - v_hip
        err_v_knee = self.v_knee_ref - v_knee
        err_v_ankle = self.v_ankle_ref - v_ankle
        err_v_shoulder = self.v_shoulder_ref - v_shoulder
        err_v_elbow = self.v_elbow_ref - v_elbow

        def sq(x):
            """Element-wise square, summed over all non-batch dims."""
            return jnp.sum(x ** 2, axis=tuple(range(1, x.ndim)))

        # ===== RUNNING COST (t = 0..N-1) =====
        r = slice(None, -1)  # excludes terminal step

        cost_running = (
            # --- base position / orientation ---
            self.w_px    * sq(err_px      [:, r])
            + self.w_pz    * sq(err_pz      [:, r])
            + self.w_theta * sq(err_theta   [:, r])
            # --- base velocity ---
            + self.w_vx    * sq(err_vx      [:, r])
            + self.w_vz    * sq(err_vz      [:, r])
            + self.w_omega * sq(err_thetadot[:, r])
            # --- joint positions ---
            + self.w_p_hip      * sq(err_p_hip     [:, r])
            + self.w_p_knee     * sq(err_p_knee    [:, r])
            + self.w_p_ankle    * sq(err_p_ankle   [:, r])
            + self.w_p_shoulder * sq(err_p_shoulder[:, r])
            + self.w_p_elbow    * sq(err_p_elbow   [:, r])
            # --- joint velocities ---
            + self.w_v_hip      * sq(err_v_hip     [:, r])
            + self.w_v_knee     * sq(err_v_knee    [:, r])
            + self.w_v_ankle    * sq(err_v_ankle   [:, r])
            + self.w_v_shoulder * sq(err_v_shoulder[:, r])
            + self.w_v_elbow    * sq(err_v_elbow   [:, r])
            # --- control effort ---
            + self.w_control * sq(tau)
        ) * self.sim.dt   # (B,)
        

        # ===== TERMINAL COST (t = N) =====
        cost_terminal = (
            # --- base position / orientation ---
            self.wf_px    * sq(err_px      [:, -1:])
            + self.wf_pz    * sq(err_pz      [:, -1:])
            + self.wf_theta * sq(err_theta   [:, -1:])
            # --- base velocity ---
            + self.wf_vx    * sq(err_vx      [:, -1:])
            + self.wf_vz    * sq(err_vz      [:, -1:])
            + self.wf_omega * sq(err_thetadot[:, -1:])
            # --- joint positions ---
            + self.wf_p_hip      * sq(err_p_hip     [:, -1:])
            + self.wf_p_knee     * sq(err_p_knee    [:, -1:])
            + self.wf_p_ankle    * sq(err_p_ankle   [:, -1:])
            + self.wf_p_shoulder * sq(err_p_shoulder[:, -1:])
            + self.wf_p_elbow    * sq(err_p_elbow   [:, -1:])
            # --- joint velocities ---
            + self.wf_v_hip      * sq(err_v_hip     [:, -1:])
            + self.wf_v_knee     * sq(err_v_knee    [:, -1:])
            + self.wf_v_ankle    * sq(err_v_ankle   [:, -1:])
            + self.wf_v_shoulder * sq(err_v_shoulder[:, -1:])
            + self.wf_v_elbow    * sq(err_v_elbow   [:, -1:])
        )  # (B,)

        J = cost_running + cost_terminal  # (B,)
        
        return J

    def optimize(self, q0, v0):
        """
        Perform CEM optimization.

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

        # perform CEM iterations
        for itr in range(self.cem_config.iterations):

            # evaluate the spline at simulation times
            y_val = self.spline.evaluate(self.t_sim[:-1])  # shape (B, N, nu)

            # mirror the spline values according to the mirror dict
            full_size = (self.sim.B, y_val.shape[1], self.sim.nu)
            y_val_full = jnp.zeros(full_size)
            for left_idx, right_idx in control_mirror_dict.items():
                y_joint = y_val[:, :, left_idx]  # shape (B, N)
                y_val_full = y_val_full.at[:, :, left_idx].set(y_joint)
                y_val_full = y_val_full.at[:, :, right_idx].set(y_joint)

            # do forward rollout
            q_log, v_log, tau_log = self.sim.rollout(q0, v0, y_val_full)
            q_log.block_until_ready()

            # compute costs
            J = self.cost(q_log, v_log, tau_log)  # shape (B,)
            J.block_until_ready()

            # select elite samples
            J_elite_neg, elite_idx = jax.lax.top_k(-J, self.cem_config.N_elite)
            J_elite = -J_elite_neg  # shape (N_elite,)

            # select the elite splines
            Y_elite = jnp.take(self.spline.Y, elite_idx, axis=0)  # shape (N_elite, N_knots, nu)

            # update the distribution
            self._update_distribution(Y_elite)

            # sample new knot points from the updated distribution
            Y_samples = self._sample_knot_points()  # shape (B, N_knots, nu)
            self.spline.update_knots(Y_samples)

            # largest singular value
            cov_norm = jnp.linalg.norm(self.Sigma, ord=2)

            # record the best solution found so far
            J_min = J_elite.min()
            if J_min < J_opt:

                # set best
                J_opt = J_min
                idx_in_elite = jnp.argmin(J_elite)  # Find best within elites
                idx_opt = elite_idx[idx_in_elite]   # Map to actual batch index

                # set optimal
                q_opt = q_log[idx_opt, :, :]
                v_opt = v_log[idx_opt, :, :]
                tau_opt = tau_log[idx_opt, :, :]

            # compute the average elite cost for monitoring
            J_elite_avg = jnp.mean(J_elite)
            J_elite_best = J_elite.min()

            # print iteration info
            itr_width = len(str(self.cem_config.iterations))  # e.g., 400 → width=3
            print(f"Iteration {itr+1:0{itr_width}d}/{self.cem_config.iterations} | "
                  f"J_elite_avg: {J_elite_avg:.4f} | "
                  f"J_elite_best: {J_elite_best:.4f} | "
                  f"J_best: {J_opt:.4f} | "
                  f"‖Σ‖₂: {cov_norm:.4f}")
            
        return q_opt, v_opt, tau_opt

#############################################################
# EXAMPLE USAGE
#############################################################


if __name__ == "__main__":

    import time
    import os

    # print device that we will use
    print(f"Using device: {jax.default_backend()}")
    if jax.default_backend() == "gpu":
        gpu_info = jax.devices("gpu")[0]
        print(f"GPU device: {gpu_info}")

    # model config
    model_config = Model_Config(
        xml_path="./models/g1/g1_planar.xml",
        Kp=[400, 450, 500, 
            400, 450, 500, 
            150, 50, 
            150, 50],
        Kd=[2, 4, 2, 
            2, 4, 2, 
            2, 2, 
            2, 2],
        q_actuated_idx=list(range(3,13)),
        v_actuated_idx=list(range(3,13)),
        action_mode="pos"
    )

    # parallel sim config
    sim_config = ParallelSim_Config(
        batch_size = 4096,
    )

    # cem config
    cem_rng = jax.random.PRNGKey(42)
    cem_config = CrossEntropyMethod_Config(
        rng=cem_rng,
        T=t_SRB[-1],  
        iterations=30,
        N_elite=512,
        N_knots=10,
        spline_type="Bezier",
        # N_knots=20,
        # spline_type="Linear",
        # N_knots=10,
        # spline_type="Cubic",
    )

    # create the CEM optimizer
    cem_optimizer = G1_Walk_Mirrored_CEM(
        model_config=model_config,
        sim_config=sim_config,
        cem_config=cem_config
    )

    # optimize from an initial state
    t0 = time.time()
    q_opt, v_opt, tau_opt = cem_optimizer.optimize(
        q0=qpos_standing,
        v0=qvel_standing
    )
    times = cem_optimizer.t_sim
    tf = time.time()
    print(f"Optimization took {tf - t0:.2f} seconds.")

    # convert to numpy for plotting
    times = np.array(times)
    q_opt = np.array(q_opt)
    v_opt = np.array(v_opt)
    tau_opt = np.array(tau_opt)

    # save as csv files in the results folder
    save_dir = "./results/g1/g1_planar_jump_mirror_cem/"
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
