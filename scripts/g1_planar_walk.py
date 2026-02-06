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
        Create reference trajectory for walking forward.
        """
        # Extract time parameters
        T = self.T_eff
        dt = self.sim.dt
        N = self.N
        
        # Define walking parameters
        target_forward_velocity = 0.5  # m/s forward velocity
        target_height = 0.75  # desired height (adjust based on your robot)
        
        # Create time array
        t_ref = jnp.linspace(0, T, N+1)
        
        # Reference trajectory
        # x position: linear forward motion
        x_ref = target_forward_velocity * t_ref
        
        # y position: constant height
        y_ref = jnp.ones(N+1) * target_height
        
        # theta: upright orientation
        theta_ref = jnp.zeros(N+1)
        
        # Store reference (shape: N+1, 3 for x, y, theta)
        self.q_ref = jnp.stack([x_ref, y_ref, theta_ref], axis=1)
        
        # Reference velocities
        vx_ref = jnp.ones(N+1) * target_forward_velocity
        vy_ref = jnp.zeros(N+1)
        vtheta_ref = jnp.zeros(N+1)
        self.v_ref = jnp.stack([vx_ref, vy_ref, vtheta_ref], axis=1)

    def cost(self, q, v, tau):
        """
        Cost function for walking forward trajectory optimization.
        
        Args:
            q: jnp.array, shape (B, N+1, nq) - generalized position trajectory.
            v: jnp.array, shape (B, N+1, nv) - generalized velocity trajectory.
            tau: jnp.array, shape (B, N, nu) - control input trajectory.
        
        Returns:
            J: jnp.array, shape (B,) - cost for each batch.
        """
        B = q.shape[0]
        N = q.shape[1] - 1
        
        # Extract base states (x, y, theta are first 3 elements)
        x = q[:, :, 0]       # shape (B, N+1)
        y = q[:, :, 1]       # shape (B, N+1)
        theta = q[:, :, 2]   # shape (B, N+1)
        
        vx = v[:, :, 0]      # shape (B, N+1)
        vy = v[:, :, 1]      # shape (B, N+1)
        vtheta = v[:, :, 2]  # shape (B, N+1)
        
        # Joint positions (indices 3 onwards)
        q_joints = q[:, :, 3:]  # shape (B, N+1, nq-3)
        v_joints = v[:, :, 3:]  # shape (B, N+1, nv-3)
        
        # ===== 1. FORWARD PROGRESS REWARD =====
        # Maximize forward distance traveled (negative cost)
        forward_distance = x[:, -1] - x[:, 0]  # final - initial x position
        cost_forward = -100.0 * forward_distance
        
        # Encourage consistent forward velocity
        target_vx = 0.5  # m/s
        cost_velocity = 10.0 * jnp.sum((vx - target_vx)**2, axis=1) / (N+1)
        
        # ===== 2. HEIGHT MAINTENANCE =====
        # Keep robot at desired height
        target_height = 0.75  # adjust for your robot
        cost_height = 50.0 * jnp.sum((y - target_height)**2, axis=1) / (N+1)
        
        # ===== 3. UPRIGHT POSTURE =====
        # Minimize deviation from upright orientation
        cost_orientation = 100.0 * jnp.sum(theta**2, axis=1) / (N+1)
        
        # Penalize angular velocity (avoid spinning)
        cost_angular_vel = 20.0 * jnp.sum(vtheta**2, axis=1) / (N+1)
        
        # ===== 4. CONTROL EFFORT =====
        # Penalize large control inputs
        cost_control = 0.01 * jnp.sum(tau**2, axis=(1, 2)) / N
        
        # ===== 5. SMOOTHNESS =====
        # Penalize large accelerations (differences in velocity)
        dv = v[:, 1:, :] - v[:, :-1, :]  # shape (B, N, nv)
        cost_smoothness = 1.0 * jnp.sum(dv**2, axis=(1, 2)) / N
        
        # Penalize control input changes (smooth control)
        dtau = tau[:, 1:, :] - tau[:, :-1, :]  # shape (B, N-1, nu)
        cost_control_smoothness = 0.1 * jnp.sum(dtau**2, axis=(1, 2)) / (N-1)
        
        # ===== 6. JOINT LIMITS =====
        # Soft penalty for approaching joint limits (adjust limits for your robot)
        q_joint_nominal = jnp.zeros_like(q_joints)  # nominal pose
        cost_joint_deviation = 1.0 * jnp.sum((q_joints - q_joint_nominal)**2, axis=(1, 2)) / (N+1)
        
        # Penalize large joint velocities
        cost_joint_vel = 0.5 * jnp.sum(v_joints**2, axis=(1, 2)) / (N+1)
        
        # ===== 7. STABILITY METRICS =====
        # Penalize excessive vertical velocity (avoid bouncing)
        cost_vy = 10.0 * jnp.sum(vy**2, axis=1) / (N+1)
        
        # ===== 8. TERMINAL COSTS =====
        # Encourage good final state
        final_x = x[:, -1]
        final_vx = vx[:, -1]
        final_theta = theta[:, -1]
        
        cost_final_velocity = 20.0 * (final_vx - target_vx)**2
        cost_final_orientation = 50.0 * final_theta**2
        
        # ===== TOTAL COST =====
        J = (cost_forward + 
            cost_velocity +
            cost_height + 
            cost_orientation + 
            cost_angular_vel +
            cost_control + 
            cost_smoothness +
            cost_control_smoothness +
            cost_joint_deviation +
            cost_joint_vel +
            cost_vy +
            cost_final_velocity +
            cost_final_orientation)
        
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
        Kp=[250, 250, 50, 250, 250, 50, # legs
            150, 150, 150, 150],        # arms
        Kd=[3.0, ] * 10,  
        q_actuated_idx=list(range(10)),
        v_actuated_idx=list(range(10)),
        action_mode="pos"
    )

    # parallel sim config
    sim_config = ParallelSim_Config(
        batch_size = 2048,
    )

    # cem config
    cem_rng = jax.random.PRNGKey(42)
    cem_config = CrossEntropyMethod_Config(
        rng=cem_rng,
        T=2.0,
        iterations=200,
        N_elite=1024,
        N_knots=2*10,
        spline_type="ZOH",
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
        0.0, 0.0, 0.0, 0.0
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
