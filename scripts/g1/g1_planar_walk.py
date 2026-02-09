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
from utils.spline.bezier import *
from utils.spline.zoh import *

"""

Generalized Positions (q):
    [0]     root_x                    - Horizontal position (m)
    [1]     root_z                    - Vertical position (m)
    [2]     root_y_rotation           - Pitch angle (rad)
    [3]     left_hip_pitch_joint      - Left hip angle (rad)
    [4]     left_knee_joint           - Left knee angle (rad)
    [5]     left_ankle_pitch_joint    - Left ankle angle (rad)
    [6]     right_hip_pitch_joint     - Right hip angle (rad)
    [7]     right_knee_joint          - Right knee angle (rad)
    [8]     right_ankle_pitch_joint   - Right ankle angle (rad)
    [9]     left_shoulder_pitch_joint - Left shoulder angle (rad)
    [10]    left_elbow_joint          - Left elbow angle (rad)
    [11]    right_shoulder_pitch_joint- Right shoulder angle (rad)
    [12]    right_elbow_joint         - Right elbow angle (rad)

Generalized Velocities (v):
    [0-12]  Same ordering as q (all 1-DOF joints)

Control Structure (nu=10):
    [0]     left_hip_pitch_joint
    [1]     left_knee_joint
    [2]     left_ankle_pitch_joint
    [3]     right_hip_pitch_joint
    [4]     right_knee_joint
    [5]     right_ankle_pitch_joint
    [6]     left_shoulder_pitch_joint
    [7]     left_elbow_joint
    [8]     right_shoulder_pitch_joint
    [9]     right_elbow_joint

"""

#############################################################
# G1 Walk CEM
#############################################################

# CEM optimizer class
class G1_Walk_CEM(CrossEntropyMethod):

    def __init__(self, model_config: Model_Config,
                       sim_config:   ParallelSim_Config,
                       cem_config:   CrossEntropyMethod_Config):
        
        # initialize the parent class
        super().__init__(model_config, sim_config, cem_config)

        # Create reference trajectory (simple linear interpolation)
        self._make_reference()


    def _make_reference(self):
        """
        Create reference trajectory for forward walking at 1.0 m/s.
        
        Reference includes:
        - Linear forward motion at constant velocity
        - Constant standing height
        - Upright orientation (zero pitch)
        """
        # Extract time parameters
        T = self.T_eff
        N = self.N
        
        # Walking parameters
        self.target_velocity = 1.0      # m/s forward velocity
        self.target_height = 0.0        # m standing height (displacement from default)
        
        # Create time array
        t_ref = jnp.linspace(0, T, N+1)
        
        # Reference base positions (first 3 elements of q)
        x_ref = self.target_velocity * t_ref           # linear forward motion
        z_ref = jnp.ones(N+1) * self.target_height     # constant height
        theta_ref = jnp.zeros(N+1)                     # upright (zero pitch)
        
        # Stack base reference: shape (N+1, 3)
        self.q_base_ref = jnp.stack([x_ref, z_ref, theta_ref], axis=1)
        
        # Reference base velocities
        vx_ref = jnp.ones(N+1) * self.target_velocity  # constant forward velocity
        vz_ref = jnp.zeros(N+1)                        # no vertical motion
        vtheta_ref = jnp.zeros(N+1)                    # no rotation
        
        # Stack velocity reference: shape (N+1, 3)
        self.v_base_ref = jnp.stack([vx_ref, vz_ref, vtheta_ref], axis=1)
        
        # Nominal joint configuration
        self.q_joint_nominal = jnp.array([
            0.0, 0.0, 0.0,  # left leg (hip, knee, ankle)
            0.0, 0.0, 0.0,  # right leg (hip, knee, ankle)
            0.2, 1.0,       # left arm (shoulder, elbow)
            0.2, 1.0        # right arm (shoulder, elbow)
        ])
        
        # ===== COST WEIGHTS =====

        # Running cost weights (per timestep)
        self.w_px_running      = 1.0     # horizontal position tracking
        self.w_pz_running      = 10.0    # vertical position tracking (keep at default height)
        self.w_theta_running   = 10.0    # pitch angle (stay upright)
        self.w_qjoint_running  = 1.0      # joint position tracking

        self.w_vx_running      = 0.1      # forward velocity tracking
        self.w_vz_running      = 0.1     # vertical velocity (minimize bouncing)
        self.w_thetadot_running = 0.1    # angular velocity (minimize spinning)
        self.w_vjoint_running  = 0.1     # joint velocity regularization
        
        self.w_control_running = 0.01    # control effort

        # Terminal cost weights (10x running weights)
        self.w_px_terminal      = 10.0 * self.w_px_running       
        self.w_pz_terminal      = 100.0 * self.w_pz_running       
        self.w_theta_terminal   = 100.0 * self.w_theta_running    
        self.w_qjoint_terminal  = 10.0 * self.w_qjoint_running   

        self.w_vx_terminal      = 10.0 * self.w_vx_running       
        self.w_vz_terminal      = 10.0 * self.w_vz_running       
        self.w_thetadot_terminal = 10.0 * self.w_thetadot_running
        self.w_vjoint_terminal  = 10.0 * self.w_vjoint_running   
        
        print(f"Reference: {self.target_velocity} m/s forward, {self.target_height} m height")


    def cost(self, q, v, tau):
        """
        Quadratic cost with explicit weights for each state component.
        
        State components:
            - px, pz, theta (base position)
            - vx, vz, thetadot (base velocity)
            - q_joints (joint positions)
            - v_joints (joint velocities)
            - tau (controls)
        
        Args:
            q: jnp.array, shape (B, N+1, nq) - positions
            v: jnp.array, shape (B, N+1, nv) - velocities
            tau: jnp.array, shape (B, N, nu) - controls
        
        Returns:
            J: jnp.array, shape (B,) - cost per trajectory
        """
        B, N_plus_1, nq = q.shape
        N = N_plus_1 - 1
        
        # ===== EXTRACT STATE COMPONENTS =====
        
        # Base position components
        px = q[:, :, 0]        # (B, N+1) - horizontal position
        pz = q[:, :, 1]        # (B, N+1) - vertical position
        theta = q[:, :, 2]     # (B, N+1) - pitch angle
        
        # Base velocity components
        vx = v[:, :, 0]        # (B, N+1) - forward velocity
        vz = v[:, :, 1]        # (B, N+1) - vertical velocity
        thetadot = v[:, :, 2]  # (B, N+1) - angular velocity
        
        # Joint states
        q_joints = q[:, :, 3:]  # (B, N+1, 10)
        v_joints = v[:, :, 3:]  # (B, N+1, 10)
        
        # ===== REFERENCE TRAJECTORIES =====
        
        px_ref = self.q_base_ref[None, :, 0]       # (1, N+1)
        pz_ref = self.q_base_ref[None, :, 1]       # (1, N+1)
        theta_ref = self.q_base_ref[None, :, 2]    # (1, N+1)
        
        vx_ref = self.v_base_ref[None, :, 0]       # (1, N+1)
        vz_ref = self.v_base_ref[None, :, 1]       # (1, N+1)
        thetadot_ref = self.v_base_ref[None, :, 2] # (1, N+1)
        
        q_joints_ref = self.q_joint_nominal[None, None, :]  # (1, 1, 10)
        v_joints_ref = jnp.zeros((1, 1, 10))                # (1, 1, 10)
        
        # ===== RUNNING COSTS =====
        
        # Base position errors (sum over time)
        cost_px = jnp.sum((px - px_ref)**2, axis=1)           # (B,)
        cost_pz = jnp.sum((pz - pz_ref)**2, axis=1)           # (B,)
        cost_theta = jnp.sum((theta - theta_ref)**2, axis=1)  # (B,)
        
        # Base velocity errors (sum over time)
        cost_vx = jnp.sum((vx - vx_ref)**2, axis=1)                # (B,)
        cost_vz = jnp.sum((vz - vz_ref)**2, axis=1)                # (B,)
        cost_thetadot = jnp.sum((thetadot - thetadot_ref)**2, axis=1)  # (B,)
        
        # Joint errors (sum over time and joints)
        cost_qjoint = jnp.sum((q_joints - q_joints_ref)**2, axis=(1, 2))  # (B,)
        cost_vjoint = jnp.sum((v_joints - v_joints_ref)**2, axis=(1, 2))  # (B,)
        
        # Control effort (sum over time and actuators)
        cost_control = jnp.sum(tau**2, axis=(1, 2))  # (B,)
        
        # ===== TERMINAL COSTS =====
        
        # Final base position errors
        cost_px_final = (px[:, -1] - px_ref[0, -1])**2              # (B,)
        cost_pz_final = (pz[:, -1] - pz_ref[0, -1])**2              # (B,)
        cost_theta_final = (theta[:, -1] - theta_ref[0, -1])**2     # (B,)
        
        # Final base velocity errors
        cost_vx_final = (vx[:, -1] - vx_ref[0, -1])**2                   # (B,)
        cost_vz_final = (vz[:, -1] - vz_ref[0, -1])**2                   # (B,)
        cost_thetadot_final = (thetadot[:, -1] - thetadot_ref[0, -1])**2 # (B,)
        
        # Final joint errors
        cost_qjoint_final = jnp.sum((q_joints[:, -1, :] - q_joints_ref[0, 0, :])**2, axis=1)  # (B,)
        cost_vjoint_final = jnp.sum((v_joints[:, -1, :] - v_joints_ref[0, 0, :])**2, axis=1)  # (B,)
        
        # ===== TOTAL COST =====
        
        # Running costs (integrated over time)
        cost_running = (
            self.w_px_running * cost_px +
            self.w_pz_running * cost_pz +
            self.w_theta_running * cost_theta +
            self.w_vx_running * cost_vx +
            self.w_vz_running * cost_vz +
            self.w_thetadot_running * cost_thetadot +
            self.w_qjoint_running * cost_qjoint +
            self.w_vjoint_running * cost_vjoint +
            self.w_control_running * cost_control
        ) * self.sim.dt
        
        # Terminal costs (final timestep only)
        cost_terminal = (
            self.w_px_terminal * cost_px_final +
            self.w_pz_terminal * cost_pz_final +
            self.w_theta_terminal * cost_theta_final +
            self.w_vx_terminal * cost_vx_final +
            self.w_vz_terminal * cost_vz_final +
            self.w_thetadot_terminal * cost_thetadot_final +
            self.w_qjoint_terminal * cost_qjoint_final +
            self.w_vjoint_terminal * cost_vjoint_final
        )
        
        J = cost_running + cost_terminal
        
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
        Kp=[300, 300, 100, 300, 300, 100, # legs
            150, 150, 150, 150],        # arms
        Kd=[3.0, ] * 10,  
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
        T=3.0,
        iterations=100,
        N_elite=2048,
        N_knots=30,
        spline_type="Bezier",
    )

    # create the CEM optimizer
    cem_optimizer = G1_Walk_CEM(
        model_config=model_config,
        sim_config=sim_config,
        cem_config=cem_config
    )

    # initial state
    q0 = jnp.array([
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.2, 1.0, 0.2, 1.0
    ])
    v0 = jnp.zeros(cem_optimizer.sim.nv)

    # optimize from an initial state
    t0 = time.time()
    q_opt, v_opt, tau_opt = cem_optimizer.optimize(
        q0=q0,
        v0=v0
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
    save_dir = "./results/g1_walk/"
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
