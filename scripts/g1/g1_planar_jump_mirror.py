##
#
# G1 Walk CEM
#
##

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
srb_dir = "./results/srb_jump_2d/"
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
        self._make_reference()

        # Running cost weights (per timestep)
        self.w_px      = 1.0     # horizontal position tracking
        self.w_pz      = 20.0    # vertical position tracking (keep at default height)
        self.w_theta   = 10.0    # pitch angle (stay upright)
        
        self.w_vx      = 0.1     # forward velocity tracking
        self.w_vz      = 0.1     # vertical velocity tracking
        self.w_omega   = 0.1     # pitch velocity tracking

        self.w_p_hip   = 5.0     # hip joint tracking
        self.w_p_knee  = 5.0     # knee joint tracking
        self.w_p_ankle = 5.0     # ankle joint tracking
        self.w_p_shoulder = 0.5  # shoulder joint tracking
        self.w_p_elbow = 0.5     # elbow joint tracking
        
        self.w_v_hip   = 0.1     # hip joint velocity tracking
        self.w_v_knee  = 0.1     # knee joint velocity tracking
        self.w_v_ankle = 0.1     # ankle joint velocity tracking
        self.w_v_shoulder = 0.05 # shoulder joint velocity tracking
        self.w_v_elbow = 0.05    # elbow joint velocity tracking
        self.w_control = 0.0001    # control effort

        terminal_scale = 1.0

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

    def _make_reference(self):
        
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
        dt_traj = t_SRB[1] - t_SRB[0]
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

        # store the reference trajectory in the class
        self.p_com_ref = p_com_ref
        self.v_com_ref = v_com_ref
        self.a_com_ref = a_com_ref
        self.F_W_ref = F_W_ref

        # WARNING: the angular references are s.t. positive is CCW. I.e., y+ is going out of hte page
        #          you must account for this when computing costs.
        self.theta_ref = theta_ref 
        self.omega_ref = omega_ref 
        self.alpha_ref = alpha_ref
        self.M_W_ref = M_W_ref


    def cost(self, q, v, tau):
        """
        Args:
            q: jnp.array, shape (B, N+1, nq) - positions
            v: jnp.array, shape (B, N+1, nv) - velocities
            tau: jnp.array, shape (B, N, nu) - controls
        
        Returns:
            J: jnp.array, shape (B,) - cost per trajectory
        """
        B, N_plus_1, nq = q.shape
        N = N_plus_1 - 1
        
        # ===== STATE COMPONENTS =====
        
        # base state
        px    = q[:, :, 0]     # (B, N+1) - horizontal position
        pz    = q[:, :, 1]     # (B, N+1) - vertical position
        theta = q[:, :, 2]     # (B, N+1) - pitch angle
        
        vx       = v[:, :, 0]  # (B, N+1) - forward velocity
        vz       = v[:, :, 1]  # (B, N+1) - vertical velocity
        thetadot = v[:, :, 2]  # (B, N+1) - angular velocity
        
        # Joint states
        p_hip      = q[:, :, hip_idx]      # (B, N+1, 2)
        p_knee     = q[:, :, knee_idx]     # (B, N+1, 2)
        p_ankle    = q[:, :, ankle_idx]    # (B, N+1, 2)
        p_shoulder = q[:, :, shoulder_idx] # (B, N+1, 2)
        p_elbow    = q[:, :, elbow_idx]    # (B, N+1, 2)

        v_hip      = v[:, :, hip_idx]       # (B, N+1, 2)
        v_knee     = v[:, :, knee_idx]      # (B, N+1, 2)
        v_ankle    = v[:, :, ankle_idx]     # (B, N+1, 2)
        v_shoulder = v[:, :, shoulder_idx]  # (B, N+1, 2)
        v_elbow    = v[:, :, elbow_idx]     # (B, N+1, 2)
        
        # ===== REFERENCE TRAJECTORIES =====

        # px_ref: integrate forward from each trajectory's initial x position
        # shape: (B, N+1)  —  anchored to initial position so cost is purely about progress
        px_ref    = px[:, 0:1] + self.target_velocity * self.t_ref[None, :]  # (B, N+1)

        # scalar references broadcast to (B, N+1)
        pz_ref       = self.pz_ref_val
        theta_ref    = self.theta_ref_val
        vx_ref       = self.target_velocity
        vz_ref       = 0.0
        thetadot_ref = 0.0

        # joint references broadcast to (B, N+1, 2)
        p_hip_ref      = self.p_hip_ref_val       # (2,) broadcasts fine
        p_knee_ref     = self.p_knee_ref_val
        p_ankle_ref    = self.p_ankle_ref_val
        p_shoulder_ref = self.p_shoulder_ref_val
        p_elbow_ref    = self.p_elbow_ref_val

        v_hip_ref      = 0.0
        v_knee_ref     = 0.0
        v_ankle_ref    = 0.0
        v_shoulder_ref = 0.0
        v_elbow_ref    = 0.0

        # ===== RUNNING COSTS  (sum over t = 0 .. N-1) =====
        # Use index [:-1] for running (exclude terminal step), [:, -1] for terminal.

        def sq(x):
            """Sum of squares over all axes except batch."""
            return jnp.sum(x ** 2, axis=tuple(range(1, x.ndim)))

        r = slice(None, -1)   # 0 .. N-1  (running steps)

        cost_running = (
            # --- base position / orientation ---
            self.w_px    * sq(px   [:, r] - px_ref  [:, r])
            + self.w_pz    * sq(pz   [:, r] - pz_ref)
            + self.w_theta * sq(theta[:, r] - theta_ref)
            # --- base velocity ---
            + self.w_vx    * sq(vx      [:, r] - vx_ref)
            + self.w_vz    * sq(vz      [:, r] - vz_ref)
            + self.w_omega * sq(thetadot[:, r] - thetadot_ref)
            # --- joint positions ---
            + self.w_p_hip      * sq(p_hip     [:, r] - p_hip_ref)
            + self.w_p_knee     * sq(p_knee    [:, r] - p_knee_ref)
            + self.w_p_ankle    * sq(p_ankle   [:, r] - p_ankle_ref)
            + self.w_p_shoulder * sq(p_shoulder[:, r] - p_shoulder_ref)
            + self.w_p_elbow    * sq(p_elbow   [:, r] - p_elbow_ref)
            # --- joint velocities ---
            + self.w_v_hip      * sq(v_hip     [:, r] - v_hip_ref)
            + self.w_v_knee     * sq(v_knee    [:, r] - v_knee_ref)
            + self.w_v_ankle    * sq(v_ankle   [:, r] - v_ankle_ref)
            + self.w_v_shoulder * sq(v_shoulder[:, r] - v_shoulder_ref)
            + self.w_v_elbow    * sq(v_elbow   [:, r] - v_elbow_ref)
            # --- control effort ---
            + self.w_control * sq(tau)
        ) * self.sim.dt

        # ===== TERMINAL COSTS  (t = N) =====

        cost_terminal = (
            self.wf_px    * sq(px   [:, -1:] - px_ref  [:, -1:])
            + self.wf_pz    * sq(pz   [:, -1] - pz_ref)
            + self.wf_theta * sq(theta[:, -1] - theta_ref)
            + self.wf_vx    * sq(vx      [:, -1] - vx_ref)
            + self.wf_vz    * sq(vz      [:, -1] - vz_ref)
            + self.wf_omega * sq(thetadot[:, -1] - thetadot_ref)
            + self.wf_p_hip      * sq(p_hip     [:, -1] - p_hip_ref)
            + self.wf_p_knee     * sq(p_knee    [:, -1] - p_knee_ref)
            + self.wf_p_ankle    * sq(p_ankle   [:, -1] - p_ankle_ref)
            + self.wf_p_shoulder * sq(p_shoulder[:, -1] - p_shoulder_ref)
            + self.wf_p_elbow    * sq(p_elbow   [:, -1] - p_elbow_ref)
            + self.wf_v_hip      * sq(v_hip     [:, -1] - v_hip_ref)
            + self.wf_v_knee     * sq(v_knee    [:, -1] - v_knee_ref)
            + self.wf_v_ankle    * sq(v_ankle   [:, -1] - v_ankle_ref)
            + self.wf_v_shoulder * sq(v_shoulder[:, -1] - v_shoulder_ref)
            + self.wf_v_elbow    * sq(v_elbow   [:, -1] - v_elbow_ref)
        )

        # ===== TOTAL COST =====

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

            exit(0)
            # TODO: need to implement new cost

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

            # compute the norm of the covariance for monitoring
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
    save_dir = "./results/g1_planar_jump/"
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
