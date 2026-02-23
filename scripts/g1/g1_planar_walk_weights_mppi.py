##
#
# G1 Walk MPPI
#
##

import config

# standard imports
import numpy as np

# jax imports
import jax
import jax.numpy as jnp

# custom imports
from utils.algorithms.mppi import *
from utils.simulation.simulation import *
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

#############################################################
# G1 Walk MPPI
#############################################################

# joint indices
base_idx     = jnp.array([0, 1])
ori_idx      = jnp.array([2])
hip_idx      = jnp.array([3, 6])
knee_idx     = jnp.array([4, 7])
ankle_idx    = jnp.array([5, 8])
shoulder_idx = jnp.array([9, 11])
elbow_idx    = jnp.array([10, 12])
joints_idx   = jnp.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12])


class G1_Walk_MPPI(MPPI):

    def __init__(self, model_config:  Model_Config,
                       sim_config:    ParallelSim_Config,
                       mppi_config:   MPPI_Config):

        # initialize the parent class
        super().__init__(model_config, sim_config, mppi_config)

        # Create reference trajectory (simple linear interpolation)
        self._make_reference()

        # Running cost weights (per timestep)
        self.w_px      = 0.1     # horizontal position tracking
        self.w_pz      = 40.0    # vertical position tracking (keep at default height)
        self.w_theta   = 40.0    # pitch angle (stay upright)

        self.w_vx      = 0.01     # forward velocity tracking
        self.w_vz      = 0.1     # vertical velocity tracking
        self.w_omega   = 0.1     # pitch velocity tracking

        self.w_p_hip      = 1.0  # hip joint tracking
        self.w_p_knee     = 1.0  # knee joint tracking
        self.w_p_ankle    = 1.0  # ankle joint tracking
        self.w_p_shoulder = 1.0  # shoulder joint tracking
        self.w_p_elbow    = 1.0  # elbow joint tracking

        self.w_v_hip      = 0.1   # hip joint velocity tracking
        self.w_v_knee     = 0.1   # knee joint velocity tracking
        self.w_v_ankle    = 0.1   # ankle joint velocity tracking
        self.w_v_shoulder = 0.1  # shoulder joint velocity tracking
        self.w_v_elbow    = 0.1  # elbow joint velocity tracking
        self.w_control    = 0.001  # control effort

        terminal_scale = 10.0

        self.wf_px        = terminal_scale * self.w_px
        self.wf_pz        = terminal_scale * self.w_pz
        self.wf_theta     = terminal_scale * self.w_theta
        self.wf_vx        = terminal_scale * self.w_vx
        self.wf_vz        = terminal_scale * self.w_vz
        self.wf_omega     = terminal_scale * self.w_omega
        self.wf_p_hip     = terminal_scale * self.w_p_hip
        self.wf_p_knee    = terminal_scale * self.w_p_knee
        self.wf_p_ankle   = terminal_scale * self.w_p_ankle
        self.wf_p_shoulder = terminal_scale * self.w_p_shoulder
        self.wf_p_elbow   = terminal_scale * self.w_p_elbow
        self.wf_v_hip     = terminal_scale * self.w_v_hip
        self.wf_v_knee    = terminal_scale * self.w_v_knee
        self.wf_v_ankle   = terminal_scale * self.w_v_ankle
        self.wf_v_shoulder = terminal_scale * self.w_v_shoulder
        self.wf_v_elbow   = terminal_scale * self.w_v_elbow


    def _make_reference(self):
        """
        Create reference trajectory for forward walking.

        Reference includes:
        - Linear forward motion at constant velocity
        - Constant standing height
        - Upright orientation (zero pitch)
        """
        T = self.T_eff
        N = self.N

        # Walking parameters
        self.target_velocity = 1.0  # m/s forward velocity

        # Time vector for N+1 points (0 to T)
        self.t_ref = jnp.linspace(0.0, T, N + 1)  # (N+1,)

        # Standing height and orientation references
        self.pz_ref_val    = qpos_standing[1]  # standing height
        self.theta_ref_val = qpos_standing[2]  # upright pitch (0.0)

        # Joint reference positions from standing keyframe
        self.p_hip_ref_val      = qpos_standing[hip_idx]       # (2,)
        self.p_knee_ref_val     = qpos_standing[knee_idx]      # (2,)
        self.p_ankle_ref_val    = qpos_standing[ankle_idx]     # (2,)
        self.p_shoulder_ref_val = qpos_standing[shoulder_idx]  # (2,)
        self.p_elbow_ref_val    = qpos_standing[elbow_idx]     # (2,)

        # ── Horizon weight schedule ───────────────────────────────────────
        # w1: front-loaded exponential decay  (used at iteration 0)
        # w2: uniform flat weights            (used at final iteration)
        # At iteration k the blend is:  w = (1 - alpha) * w1 + alpha * w2
        # where alpha = k / (K - 1)  ramps linearly from 0 → 1.
        self.horizon_lambda = 10.0  # steepness of w1; higher = more front-loaded

        n = jnp.arange(N)                   # horizon indices 0 .. N-1
        x = n / jnp.maximum(N - 1, 1)       # normalise to [0, 1]      (N,)

        lam = self.horizon_lambda
        self.w1_horizon = (
            (jnp.exp(-lam * x) - jnp.exp(-lam)) / (1.0 - jnp.exp(-lam))
        )                                    # (N,)  — 1 at n=0, 0 at n=N-1
        self.w2_horizon = jnp.ones(N)        # (N,)  — uniform


    def cost(self, q, v, tau):
        """
        Args:
            q:   jnp.array, shape (B, N+1, nq) - positions
            v:   jnp.array, shape (B, N+1, nv) - velocities
            tau: jnp.array, shape (B, N,   nu) - controls

        Returns:
            J: jnp.array, shape (B,) - cost per trajectory
        """
        B, N_plus_1, nq = q.shape
        N = N_plus_1 - 1

        # ===== STATE COMPONENTS =====

        px    = q[:, :, 0]  # (B, N+1)
        pz    = q[:, :, 1]  # (B, N+1)
        theta = q[:, :, 2]  # (B, N+1)

        vx       = v[:, :, 0]  # (B, N+1)
        vz       = v[:, :, 1]  # (B, N+1)
        thetadot = v[:, :, 2]  # (B, N+1)

        p_hip      = q[:, :, hip_idx]       # (B, N+1, 2)
        p_knee     = q[:, :, knee_idx]      # (B, N+1, 2)
        p_ankle    = q[:, :, ankle_idx]     # (B, N+1, 2)
        p_shoulder = q[:, :, shoulder_idx]  # (B, N+1, 2)
        p_elbow    = q[:, :, elbow_idx]     # (B, N+1, 2)

        v_hip      = v[:, :, hip_idx]       # (B, N+1, 2)
        v_knee     = v[:, :, knee_idx]      # (B, N+1, 2)
        v_ankle    = v[:, :, ankle_idx]     # (B, N+1, 2)
        v_shoulder = v[:, :, shoulder_idx]  # (B, N+1, 2)
        v_elbow    = v[:, :, elbow_idx]     # (B, N+1, 2)

        # ===== REFERENCE TRAJECTORIES =====

        # px_ref anchored to each trajectory's initial position
        px_ref = px[:, 0:1] + self.target_velocity * self.t_ref[None, :]  # (B, N+1)

        pz_ref       = self.pz_ref_val
        theta_ref    = self.theta_ref_val
        vx_ref       = self.target_velocity
        vz_ref       = 0.0
        thetadot_ref = 0.0

        p_hip_ref      = self.p_hip_ref_val       # (2,) broadcasts
        p_knee_ref     = self.p_knee_ref_val
        p_ankle_ref    = self.p_ankle_ref_val
        p_shoulder_ref = self.p_shoulder_ref_val
        p_elbow_ref    = self.p_elbow_ref_val

        v_hip_ref = v_knee_ref = v_ankle_ref = v_shoulder_ref = v_elbow_ref = 0.0

        # ===== RUNNING COSTS — per-timestep, shape (B, N) =====

        r = slice(None, -1)  # 0 .. N-1

        cost_t = (
            # base position / orientation
            self.w_px    * (px   [:, r] - px_ref  [:, r]) ** 2
            + self.w_pz    * (pz   [:, r] - pz_ref)          ** 2
            + self.w_theta * (theta[:, r] - theta_ref)        ** 2
            # base velocity
            + self.w_vx    * (vx      [:, r] - vx_ref)        ** 2
            + self.w_vz    * (vz      [:, r] - vz_ref)        ** 2
            + self.w_omega * (thetadot[:, r] - thetadot_ref)  ** 2
            # joint positions  (sum over the 2 joints per group)
            + self.w_p_hip      * jnp.sum((p_hip     [:, r] - p_hip_ref)      ** 2, axis=-1)
            + self.w_p_knee     * jnp.sum((p_knee    [:, r] - p_knee_ref)     ** 2, axis=-1)
            + self.w_p_ankle    * jnp.sum((p_ankle   [:, r] - p_ankle_ref)    ** 2, axis=-1)
            + self.w_p_shoulder * jnp.sum((p_shoulder[:, r] - p_shoulder_ref) ** 2, axis=-1)
            + self.w_p_elbow    * jnp.sum((p_elbow   [:, r] - p_elbow_ref)    ** 2, axis=-1)
            # joint velocities
            + self.w_v_hip      * jnp.sum((v_hip     [:, r] - v_hip_ref)      ** 2, axis=-1)
            + self.w_v_knee     * jnp.sum((v_knee    [:, r] - v_knee_ref)     ** 2, axis=-1)
            + self.w_v_ankle    * jnp.sum((v_ankle   [:, r] - v_ankle_ref)    ** 2, axis=-1)
            + self.w_v_shoulder * jnp.sum((v_shoulder[:, r] - v_shoulder_ref) ** 2, axis=-1)
            + self.w_v_elbow    * jnp.sum((v_elbow   [:, r] - v_elbow_ref)    ** 2, axis=-1)
            # control effort
            + self.w_control * jnp.sum(tau ** 2, axis=-1)
        ) * self.sim.dt  # (B, N)

        # ===== HORIZON WEIGHT SCHEDULE =====
        # alpha: 0 at iteration 0  →  1 at final iteration
        # weight profile: front-loaded w1  →  uniform w2
        alpha         = self.itr / jnp.maximum(self.mppi_config.iterations - 1, 1)
        horizon_w     = (1.0 - alpha) * self.w1_horizon + alpha * self.w2_horizon  # (N,)
        cost_running  = jnp.sum(cost_t * horizon_w[None, :], axis=-1)  # (B,)

        # ===== TERMINAL COSTS — always full weight, shape (B,) =====

        def sq(x):
            """Sum of squares over all axes except batch."""
            return jnp.sum(x ** 2, axis=tuple(range(1, x.ndim)))

        cost_terminal = (
            self.wf_px        * sq(px   [:, -1:] - px_ref  [:, -1:])
            + self.wf_pz        * sq(pz   [:, -1] - pz_ref)
            + self.wf_theta     * sq(theta[:, -1] - theta_ref)
            + self.wf_vx        * sq(vx      [:, -1] - vx_ref)
            + self.wf_vz        * sq(vz      [:, -1] - vz_ref)
            + self.wf_omega     * sq(thetadot[:, -1] - thetadot_ref)
            + self.wf_p_hip     * sq(p_hip     [:, -1] - p_hip_ref)
            + self.wf_p_knee    * sq(p_knee    [:, -1] - p_knee_ref)
            + self.wf_p_ankle   * sq(p_ankle   [:, -1] - p_ankle_ref)
            + self.wf_p_shoulder * sq(p_shoulder[:, -1] - p_shoulder_ref)
            + self.wf_p_elbow   * sq(p_elbow   [:, -1] - p_elbow_ref)
            + self.wf_v_hip     * sq(v_hip     [:, -1] - v_hip_ref)
            + self.wf_v_knee    * sq(v_knee    [:, -1] - v_knee_ref)
            + self.wf_v_ankle   * sq(v_ankle   [:, -1] - v_ankle_ref)
            + self.wf_v_shoulder * sq(v_shoulder[:, -1] - v_shoulder_ref)
            + self.wf_v_elbow   * sq(v_elbow   [:, -1] - v_elbow_ref)
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

    print(f"Using device: {jax.default_backend()}")
    if jax.default_backend() == "gpu":
        gpu_info = jax.devices("gpu")[0]
        print(f"GPU device: {gpu_info}")

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
        q_actuated_idx=list(range(3, 13)),
        v_actuated_idx=list(range(3, 13)),
        action_mode="pos"
    )

    # parallel sim config
    sim_config = ParallelSim_Config(
        batch_size=4096,
    )

    # mppi config
    mppi_rng = jax.random.PRNGKey(s)
    mppi_config = MPPI_Config(
        rng=mppi_rng,
        T=2.0,
        N_knots=5,
        iterations=50,
        lam=5.0,       # temperature: lower = greedier exploitation
        sigma=0.25,     # noise std for sampling
        spline_type="Bezier",
        use_cov_contraction=True,
        sigma_min=0.01
    )

    # create the MPPI optimizer
    mppi_optimizer = G1_Walk_MPPI(
        model_config=model_config,
        sim_config=sim_config,
        mppi_config=mppi_config,
    )

    # optimize from standing
    t0 = time.time()
    q_opt, v_opt, tau_opt = mppi_optimizer.optimize(
        q0=qpos_standing,
        v0=qvel_standing,
    )
    times = mppi_optimizer.t_sim
    tf = time.time()
    print(f"Optimization took {tf - t0:.2f} seconds.")

    # convert to numpy for saving
    times   = np.array(times)
    q_opt   = np.array(q_opt)
    v_opt   = np.array(v_opt)
    tau_opt = np.array(tau_opt)

    # save results
    save_dir = "./results/g1/g1_planar_walk_weights_mppi/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created directory: {save_dir}")

    np.savetxt(save_dir + "time.csv",    times,   delimiter=",")
    np.savetxt(save_dir + "q_opt.csv",   q_opt,   delimiter=",")
    np.savetxt(save_dir + "v_opt.csv",   v_opt,   delimiter=",")
    np.savetxt(save_dir + "tau_opt.csv", tau_opt, delimiter=",")
    print(f"Saved results to {save_dir}")