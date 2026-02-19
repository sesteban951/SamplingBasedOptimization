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
state_mirror_idx   = {3:6, 4:7, 5:8, 9:11, 10:12}  # hip, knee, ankle, shoulder, elbow
control_mirror_idx = {0:3, 1:4, 2:5, 6:8, 7:9}     # hip, knee, ankle, shoulder, elbow

# CEM optimizer class
class G1_Walk_Mirrored_CEM(CrossEntropyMethod):

    def __init__(self, model_config: Model_Config,
                       sim_config:   ParallelSim_Config,
                       cem_config:   CrossEntropyMethod_Config):
        
        # initialize the parent class
        super().__init__(model_config, sim_config, cem_config)

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
        self.theta_ref = theta_ref
        self.omega_ref = omega_ref
        self.alpha_ref = alpha_ref
        self.F_W_ref = F_W_ref
        self.M_W_ref = M_W_ref

        # plot the reference trajectory
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 8))
        plt.plot(t_sim, p_com_ref[:, 0], label="p_com_x_ref")
        plt.show()

        print(f"Loaded SRB trajectory from [{dir}].")
        print(f"   [Nx_traj: {q_SRB.shape[0]}]")
        print(f"   [Nu_traj: {tau_SRB.shape[1]}]")
        print(f"   [dt_traj: {dt_traj}]")
        print(f"   [T_traj: {times[-1]:.4f} seconds]")
        print(f"   [N_sim:   {N_sim} steps at dt={dt_sim:.4f}s]")

        exit(0)


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

    # fix the random seed
    s = int(time.time())
    np.random.seed(s)

    # model config
    model_config = Model_Config(
        xml_path="./models/g1/g1_planar.xml",
        Kp=[100, 150, 40, 
            100, 150, 40, 
            100, 50, 
            100, 50],
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
    save_dir = "./results/g1_walk_mirrored_cem/"
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
