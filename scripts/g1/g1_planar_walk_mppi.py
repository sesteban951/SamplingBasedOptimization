##
#
# G1 Walk MPPI
#
##

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
base_idx      = jnp.array([0, 1])
ori_idx       = jnp.array([2])
hip_idx       = jnp.array([3, 6])
knee_idx      = jnp.array([4, 7])
ankle_idx     = jnp.array([5, 8])
shoulder_idx  = jnp.array([9, 11])
elbow_idx     = jnp.array([10, 12])
joints_idx    = jnp.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12])


class G1_Walk_MPPI(MPPI):

    def __init__(self, model_config:  Model_Config,
                       sim_config:    ParallelSim_Config,
                       mppi_config:   MPPI_Config):

        # initialize the parent class
        super().__init__(model_config, sim_config, mppi_config)

        # Create reference trajectory
        self._make_reference()

        # Running cost weights (per timestep)
        self.w_px         = 1.0
        self.w_pz         = 20.0
        self.w_theta      = 10.0

        self.w_vx         = 0.1
        self.w_vz         = 0.1
        self.w_omega      = 0.1

        self.w_p_hip      = 5.0
        self.w_p_knee     = 5.0
        self.w_p_ankle    = 5.0
        self.w_p_shoulder = 0.5
        self.w_p_elbow    = 0.5

        self.w_v_hip      = 0.1
        self.w_v_knee     = 0.1
        self.w_v_ankle    = 0.1
        self.w_v_shoulder = 0.05
        self.w_v_elbow    = 0.05
        self.w_control    = 0.0001

        terminal_scale = 1.0

        self.wf_px         = terminal_scale * self.w_px
        self.wf_pz         = terminal_scale * self.w_pz
        self.wf_theta      = terminal_scale * self.w_theta

        self.wf_vx         = terminal_scale * self.w_vx
        self.wf_vz         = terminal_scale * self.w_vz
        self.wf_omega      = terminal_scale * self.w_omega

        self.wf_p_hip      = terminal_scale * self.w_p_hip
        self.wf_p_knee     = terminal_scale * self.w_p_knee
        self.wf_p_ankle    = terminal_scale * self.w_p_ankle
        self.wf_p_shoulder = terminal_scale * self.w_p_shoulder
        self.wf_p_elbow    = terminal_scale * self.w_p_elbow

        self.wf_v_hip      = terminal_scale * self.w_v_hip
        self.wf_v_knee     = terminal_scale * self.w_v_knee
        self.wf_v_ankle    = terminal_scale * self.w_v_ankle
        self.wf_v_shoulder = terminal_scale * self.w_v_shoulder
        self.wf_v_elbow    = terminal_scale * self.w_v_elbow


    def _make_reference(self):
        """
        Create reference trajectory for forward walking.
        """
        T = self.T_eff
        N = self.N

        self.target_velocity = 1.0  # m/s

        self.t_ref = jnp.linspace(0.0, T, N + 1)  # (N+1,)

        self.pz_ref_val    = qpos_standing[1]
        self.theta_ref_val = qpos_standing[2]

        self.p_hip_ref_val      = qpos_standing[hip_idx]
        self.p_knee_ref_val     = qpos_standing[knee_idx]
        self.p_ankle_ref_val    = qpos_standing[ankle_idx]
        self.p_shoulder_ref_val = qpos_standing[shoulder_idx]
        self.p_elbow_ref_val    = qpos_standing[elbow_idx]


    def cost(self, q, v, tau):
        """
        Args:
            q:   jnp.array, shape (B, N+1, nq)
            v:   jnp.array, shape (B, N+1, nv)
            tau: jnp.array, shape (B, N, nu)
        Returns:
            J: jnp.array, shape (B,)
        """
        # ===== STATE COMPONENTS =====

        px    = q[:, :, 0]
        pz    = q[:, :, 1]
        theta = q[:, :, 2]

        vx       = v[:, :, 0]
        vz       = v[:, :, 1]
        thetadot = v[:, :, 2]

        p_hip      = q[:, :, hip_idx]
        p_knee     = q[:, :, knee_idx]
        p_ankle    = q[:, :, ankle_idx]
        p_shoulder = q[:, :, shoulder_idx]
        p_elbow    = q[:, :, elbow_idx]

        v_hip      = v[:, :, hip_idx]
        v_knee     = v[:, :, knee_idx]
        v_ankle    = v[:, :, ankle_idx]
        v_shoulder = v[:, :, shoulder_idx]
        v_elbow    = v[:, :, elbow_idx]

        # ===== REFERENCE TRAJECTORIES =====

        px_ref = px[:, 0:1] + self.target_velocity * self.t_ref[None, :]  # (B, N+1)

        pz_ref       = self.pz_ref_val
        theta_ref    = self.theta_ref_val
        vx_ref       = self.target_velocity
        vz_ref       = 0.0
        thetadot_ref = 0.0

        p_hip_ref      = self.p_hip_ref_val
        p_knee_ref     = self.p_knee_ref_val
        p_ankle_ref    = self.p_ankle_ref_val
        p_shoulder_ref = self.p_shoulder_ref_val
        p_elbow_ref    = self.p_elbow_ref_val

        # ===== RUNNING COSTS =====

        def sq(x):
            return jnp.sum(x ** 2, axis=tuple(range(1, x.ndim)))

        r = slice(None, -1)

        cost_running = (
            self.w_px         * sq(px   [:, r] - px_ref  [:, r])
            + self.w_pz       * sq(pz   [:, r] - pz_ref)
            + self.w_theta    * sq(theta[:, r] - theta_ref)
            + self.w_vx       * sq(vx      [:, r] - vx_ref)
            + self.w_vz       * sq(vz      [:, r] - vz_ref)
            + self.w_omega    * sq(thetadot[:, r] - thetadot_ref)
            + self.w_p_hip      * sq(p_hip     [:, r] - p_hip_ref)
            + self.w_p_knee     * sq(p_knee    [:, r] - p_knee_ref)
            + self.w_p_ankle    * sq(p_ankle   [:, r] - p_ankle_ref)
            + self.w_p_shoulder * sq(p_shoulder[:, r] - p_shoulder_ref)
            + self.w_p_elbow    * sq(p_elbow   [:, r] - p_elbow_ref)
            + self.w_v_hip      * sq(v_hip     [:, r] - 0.0)
            + self.w_v_knee     * sq(v_knee    [:, r] - 0.0)
            + self.w_v_ankle    * sq(v_ankle   [:, r] - 0.0)
            + self.w_v_shoulder * sq(v_shoulder[:, r] - 0.0)
            + self.w_v_elbow    * sq(v_elbow   [:, r] - 0.0)
            + self.w_control    * sq(tau)
        ) * self.sim.dt

        # ===== TERMINAL COSTS =====

        cost_terminal = (
            self.wf_px        * sq(px   [:, -1:] - px_ref  [:, -1:])
            + self.wf_pz      * sq(pz   [:, -1] - pz_ref)
            + self.wf_theta   * sq(theta[:, -1] - theta_ref)
            + self.wf_vx      * sq(vx      [:, -1] - vx_ref)
            + self.wf_vz      * sq(vz      [:, -1] - vz_ref)
            + self.wf_omega   * sq(thetadot[:, -1] - thetadot_ref)
            + self.wf_p_hip      * sq(p_hip     [:, -1] - p_hip_ref)
            + self.wf_p_knee     * sq(p_knee    [:, -1] - p_knee_ref)
            + self.wf_p_ankle    * sq(p_ankle   [:, -1] - p_ankle_ref)
            + self.wf_p_shoulder * sq(p_shoulder[:, -1] - p_shoulder_ref)
            + self.wf_p_elbow    * sq(p_elbow   [:, -1] - p_elbow_ref)
            + self.wf_v_hip      * sq(v_hip     [:, -1] - 0.0)
            + self.wf_v_knee     * sq(v_knee    [:, -1] - 0.0)
            + self.wf_v_ankle    * sq(v_ankle   [:, -1] - 0.0)
            + self.wf_v_shoulder * sq(v_shoulder[:, -1] - 0.0)
            + self.wf_v_elbow    * sq(v_elbow   [:, -1] - 0.0)
        )

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
    mppi_rng = jax.random.PRNGKey(42)
    mppi_config = MPPI_Config(
        rng=mppi_rng,
        T=2.0,
        N_knots=10,
        iterations=30,
        lam=100.0,       # temperature: lower = greedier exploitation
        sigma=0.5,     # noise std for sampling
        spline_type="Bezier",
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
    save_dir = "./results/g1_walk_mppi/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created directory: {save_dir}")

    np.savetxt(save_dir + "time.csv",    times,   delimiter=",")
    np.savetxt(save_dir + "q_opt.csv",   q_opt,   delimiter=",")
    np.savetxt(save_dir + "v_opt.csv",   v_opt,   delimiter=",")
    np.savetxt(save_dir + "tau_opt.csv", tau_opt, delimiter=",")
    print(f"Saved results to {save_dir}")
