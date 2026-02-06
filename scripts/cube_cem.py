##
#
# Complete Cube Reorientation CEM - Ultra Stable Version
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
Leap Hand + Cube state info:

Generalized positions (nq=23):
    qpos[0-15]  = hand joint angles (16 hinge joints)
    qpos[16]    = cube_x      (position x)
    qpos[17]    = cube_y      (position y)
    qpos[18]    = cube_z      (position z)
    qpos[19]    = cube_qw     (quaternion w)
    qpos[20]    = cube_qx     (quaternion x)
    qpos[21]    = cube_qy     (quaternion y)
    qpos[22]    = cube_qz     (quaternion z)

Generalized velocities (nv=22):
    qvel[0-15]  = hand joint velocities (16 values)
    qvel[16]    = cube_vx     (linear velocity x)
    qvel[17]    = cube_vy     (linear velocity y)
    qvel[18]    = cube_vz     (linear velocity z)
    qvel[19]    = cube_wx     (angular velocity x)
    qvel[20]    = cube_wy     (angular velocity y)
    qvel[21]    = cube_wz     (angular velocity z)

Actuators (nu=16):
    ctrl[0-15]  = hand motor torques (16 motors)
"""


#############################################################
# Cube Reorientation CEM
#############################################################
class CubeReorientation_CEM(CrossEntropyMethod):

    def __init__(self, model_config: Model_Config,
                       sim_config:   ParallelSim_Config,
                       cem_config:   CrossEntropyMethod_Config,
                       goal_quat:    jnp.array = None):
        
        # initialize the parent class
        super().__init__(model_config, sim_config, cem_config)
        
        # Goal quaternion: 180-degree pitch around y-axis
        # This represents flipping the cube upside down
        if goal_quat is None:
            # 180-degree rotation around y-axis: quat = [0, 0, 1, 0] (w, x, y, z)
            self.goal_quat = jnp.array([0.0, 0.0, 1.0, 0.0])
        else:
            self.goal_quat = goal_quat
        
        # Cost weights - balanced for holding + reorienting
        self.w_orientation = 0.05          # Track orientation during trajectory
        self.w_position = 5.0              # Keep cube near hand (INCREASED!)
        self.w_control = 0.0001          # Control effort (tiny!)
        self.w_velocity = 0.1            # Velocity regularization
        self.w_terminal_orient = 10.0      # Final orientation (INCREASED!)
        self.w_terminal_vel = 0.1          # Final velocity (want stable)
        self.w_terminal_pos = 50.0         # Final position near hand (MUCH HIGHER!)
        
        # NEW: Hand finger costs
        self.w_hand_pos = 1.0             # Keep hand joints at zero position
        self.w_hand_vel = 0.1            # Keep hand joint velocities at zero
        self.w_terminal_hand_pos = 0.1     # Terminal hand position at zero
        self.w_terminal_hand_vel = 0.01    # Terminal hand velocity at zero
        
        # Target position (near grasp site, higher to avoid floor)
        self.target_pos = jnp.array([0.11, 0.0, 0.10])

    
    def quaternion_distance(self, q1, q2):
        """
        Compute geodesic distance between two quaternions.
        Ultra-stable version with heavy epsilon protection.
        
        Args:
            q1: jnp.array, shape (..., 4) - quaternion (w, x, y, z)
            q2: jnp.array, shape (..., 4) - quaternion (w, x, y, z)
        Returns:
            dist: jnp.array, shape (...,) - angular distance in radians [0, π]
        """
        epsilon = 1e-6
        
        # Normalize quaternions with protection against zero norm
        q1_norm = jnp.linalg.norm(q1, axis=-1, keepdims=True)
        q2_norm = jnp.linalg.norm(q2, axis=-1, keepdims=True)
        
        q1 = q1 / (q1_norm + epsilon)
        q2 = q2 / (q2_norm + epsilon)
        
        # Compute dot product (accounts for quaternion double cover)
        dot = jnp.sum(q1 * q2, axis=-1)
        dot = jnp.abs(dot)  # Handle double cover
        dot = jnp.clip(dot, 0.0, 0.9999)  # Stay away from 1.0 to avoid arccos issues
        
        # Angular distance (safe arccos)
        dist = 2.0 * jnp.arccos(dot)
        
        # Clip to valid range
        dist = jnp.clip(dist, 0.0, jnp.pi)
        
        return dist


    def cost(self, q, v, tau):
        """
        Cost function for cube reorientation.
        Ultra-stable version using only L1 norms and aggressive clipping.

        Args:
            q: jnp.array, shape (B, N+1, nq) - generalized position trajectory
            v: jnp.array, shape (B, N+1, nv) - generalized velocity trajectory
            tau: jnp.array, shape (B, N, nu) - control input trajectory
        Returns:
            J: jnp.array, shape (B,) - cost for each batch
        """

        B, N_plus_1, nq = q.shape
        N = N_plus_1 - 1
        
        epsilon = 1e-8

        # ---------------------------------------------------
        # RUNNING COST (t = 0 to N-1)
        # ---------------------------------------------------

        # Extract cube states over trajectory
        cube_pos = q[:, :-1, 16:19]      # (B, N, 3) - position
        cube_quat = q[:, :-1, 19:23]     # (B, N, 4) - quaternion
        cube_linvel = v[:, :-1, 16:19]   # (B, N, 3) - linear velocity
        cube_angvel = v[:, :-1, 19:22]   # (B, N, 3) - angular velocity

        # Extract hand states over trajectory
        hand_pos = q[:, :-1, 0:16]       # (B, N, 16) - hand joint positions
        hand_vel = v[:, :-1, 0:16]       # (B, N, 16) - hand joint velocities

        # 1. Orientation tracking cost
        goal_quat_expanded = jnp.tile(self.goal_quat[None, None, :], (B, N, 1))  # (B, N, 4)
        
        # Vectorized quaternion distance computation
        def batch_quat_distance(q_batch, goal_batch):
            # q_batch: (N, 4), goal_batch: (N, 4)
            return jax.vmap(self.quaternion_distance, in_axes=(0, 0))(q_batch, goal_batch)
        
        orientation_error = jax.vmap(batch_quat_distance, in_axes=(0, 0))(cube_quat, goal_quat_expanded)  # (B, N)
        
        # Clip errors to prevent overflow
        orientation_error = jnp.clip(orientation_error, 0.0, 100.0)
        
        # Use L1 norm (sum of absolute values, no squaring to avoid overflow)
        orientation_cost = self.w_orientation * jnp.sum(orientation_error, axis=1)  # (B,)

        # 2. Position cost - keep cube near target position
        pos_error = cube_pos - self.target_pos[None, None, :]  # (B, N, 3)
        pos_dist = jnp.sqrt(jnp.sum(pos_error**2, axis=2) + epsilon)  # (B, N)
        pos_dist = jnp.clip(pos_dist, 0.0, 10.0)  # Clip at 10 meters max
        position_cost = self.w_position * jnp.sum(pos_dist, axis=1)  # (B,)

        # 3. Control effort cost
        control_mag = jnp.sqrt(jnp.sum(tau**2, axis=2) + epsilon)  # (B, N)
        control_mag = jnp.clip(control_mag, 0.0, 100.0)
        control_cost = self.w_control * jnp.sum(control_mag, axis=1)  # (B,)

        # 4. Velocity regularization (discourage excessive motion)
        linvel_mag = jnp.sqrt(jnp.sum(cube_linvel**2, axis=2) + epsilon)  # (B, N)
        angvel_mag = jnp.sqrt(jnp.sum(cube_angvel**2, axis=2) + epsilon)  # (B, N)
        linvel_mag = jnp.clip(linvel_mag, 0.0, 100.0)
        angvel_mag = jnp.clip(angvel_mag, 0.0, 100.0)
        vel_cost = self.w_velocity * (jnp.sum(linvel_mag, axis=1) + jnp.sum(angvel_mag, axis=1))  # (B,)

        # 5. NEW: Hand position cost - keep hand joints at zero
        hand_pos_mag = jnp.sqrt(jnp.sum(hand_pos**2, axis=2) + epsilon)  # (B, N)
        hand_pos_mag = jnp.clip(hand_pos_mag, 0.0, 100.0)
        hand_pos_cost = self.w_hand_pos * jnp.sum(hand_pos_mag, axis=1)  # (B,)

        # 6. NEW: Hand velocity cost - keep hand joint velocities at zero
        hand_vel_mag = jnp.sqrt(jnp.sum(hand_vel**2, axis=2) + epsilon)  # (B, N)
        hand_vel_mag = jnp.clip(hand_vel_mag, 0.0, 100.0)
        hand_vel_cost = self.w_hand_vel * jnp.sum(hand_vel_mag, axis=1)  # (B,)

        # Total running cost (scaled by timestep)
        running_cost = (orientation_cost + position_cost + control_cost + vel_cost + 
                       hand_pos_cost + hand_vel_cost) * self.sim.dt
        running_cost = jnp.clip(running_cost, 0.0, 1e4)  # Prevent overflow

        # ---------------------------------------------------
        # TERMINAL COST (t = N)
        # ---------------------------------------------------

        # Terminal cube state
        cube_pos_T = q[:, -1, 16:19]       # (B, 3)
        cube_quat_T = q[:, -1, 19:23]      # (B, 4)
        cube_linvel_T = v[:, -1, 16:19]    # (B, 3)
        cube_angvel_T = v[:, -1, 19:22]    # (B, 3)

        # Terminal hand state
        hand_pos_T = q[:, -1, 0:16]        # (B, 16)
        hand_vel_T = v[:, -1, 0:16]        # (B, 16)

        # 1. Terminal orientation cost (most important!)
        goal_quat_T = jnp.tile(self.goal_quat[None, :], (B, 1))  # (B, 4)
        terminal_orient_error = jax.vmap(self.quaternion_distance)(cube_quat_T, goal_quat_T)  # (B,)
        terminal_orient_error = jnp.clip(terminal_orient_error, 0.0, 100.0)
        terminal_orient_cost = self.w_terminal_orient * terminal_orient_error  # L1 norm (no squaring)

        # 2. Terminal velocity cost (want cube stable/stationary)
        linvel_T_mag = jnp.sqrt(jnp.sum(cube_linvel_T**2, axis=1) + epsilon)  # (B,)
        angvel_T_mag = jnp.sqrt(jnp.sum(cube_angvel_T**2, axis=1) + epsilon)  # (B,)
        linvel_T_mag = jnp.clip(linvel_T_mag, 0.0, 100.0)
        angvel_T_mag = jnp.clip(angvel_T_mag, 0.0, 100.0)
        terminal_vel_cost = self.w_terminal_vel * (linvel_T_mag + angvel_T_mag)  # (B,)

        # 3. Terminal position cost (keep cube near hand)
        pos_error_T = cube_pos_T - self.target_pos[None, :]  # (B, 3)
        pos_T_dist = jnp.sqrt(jnp.sum(pos_error_T**2, axis=1) + epsilon)  # (B,)
        pos_T_dist = jnp.clip(pos_T_dist, 0.0, 10.0)
        terminal_pos_cost = self.w_terminal_pos * pos_T_dist  # (B,)

        # 4. NEW: Terminal hand position cost
        hand_pos_T_mag = jnp.sqrt(jnp.sum(hand_pos_T**2, axis=1) + epsilon)  # (B,)
        hand_pos_T_mag = jnp.clip(hand_pos_T_mag, 0.0, 100.0)
        terminal_hand_pos_cost = self.w_terminal_hand_pos * hand_pos_T_mag  # (B,)

        # 5. NEW: Terminal hand velocity cost
        hand_vel_T_mag = jnp.sqrt(jnp.sum(hand_vel_T**2, axis=1) + epsilon)  # (B,)
        hand_vel_T_mag = jnp.clip(hand_vel_T_mag, 0.0, 100.0)
        terminal_hand_vel_cost = self.w_terminal_hand_vel * hand_vel_T_mag  # (B,)

        # Total terminal cost
        terminal_cost = (terminal_orient_cost + terminal_vel_cost + terminal_pos_cost + 
                        terminal_hand_pos_cost + terminal_hand_vel_cost)
        terminal_cost = jnp.clip(terminal_cost, 0.0, 1e4)  # Prevent overflow
        
        # ---------------------------------------------------
        # Total cost with aggressive safety measures
        # ---------------------------------------------------
        J = running_cost + terminal_cost  # (B,)
        
        # Triple safety: clip, replace NaN, replace Inf
        J = jnp.clip(J, 0.0, 1e5)
        J = jnp.where(jnp.isnan(J), 1e5, J)  # Replace NaN with large finite value
        J = jnp.where(jnp.isinf(J), 1e5, J)  # Replace Inf with large finite value
        
        return J

    
#############################################################
# MAIN EXECUTION
#############################################################


if __name__ == "__main__":

    import time
    import os

    print("="*70)
    print("CUBE REORIENTATION: 180-DEGREE PITCH (ULTRA-STABLE)")
    print("="*70)
    
    # Device info
    print(f"\nDevice: {jax.default_backend()}")
    if jax.default_backend() == "gpu":
        gpu_info = jax.devices("gpu")[0]
        print(f"GPU: {gpu_info}")

    # Random seed
    s = int(time.time())
    np.random.seed(s)
    print(f"Random seed: {s}\n")

    # ---------------------------------------------------
    # CONFIGURATION (CONSERVATIVE FOR STABILITY)
    # ---------------------------------------------------

    # Model config - REDUCED PD gains for stability
    model_config = Model_Config(
        xml_path="./models/cube/scene.xml",
        Kp=[10.0]*16,   # Reduced from 30 (lower = more stable)
        Kd=[1.0]*16,    # Reduced from 2 (lower = more stable)
        q_actuated_idx=list(range(16)),  # Hand joints
        v_actuated_idx=list(range(16)),  # Hand joint velocities
        action_mode="pos"
    )

    # Parallel sim config - SMALL batch for stability
    sim_config = ParallelSim_Config(
        batch_size=512,  # Much smaller than 2048 (reduces memory pressure)
    )

    # CEM config - CONSERVATIVE parameters
    cem_rng = jax.random.PRNGKey(42)
    cem_config = CrossEntropyMethod_Config(
        rng=cem_rng,
        T=2.0,           # Shorter horizon (2s instead of 3s)
        iterations=100,   # More iterations for convergence
        N_elite=64,      # More elite samples (25% of batch)
        N_knots=10,      # Fewer control knots
        spline_type="ZOH",
    )

    print("Configuration:")
    print(f"  XML path:    {model_config.xml_path}")
    print(f"  Batch size:  {sim_config.batch_size}")
    print(f"  Elite count: {cem_config.N_elite}")
    print(f"  Horizon:     {cem_config.T}s")
    print(f"  Iterations:  {cem_config.iterations}")
    print(f"  Control knots: {cem_config.N_knots}")
    print(f"  PD gains:    Kp={model_config.Kp[0]}, Kd={model_config.Kd[0]}")
    print()

    # Goal: 180-degree pitch (flip cube upside down)
    goal_quat = jnp.array([0.0, 0.0, 1.0, 0.0])  # 180° around y-axis
    
    print("Goal:")
    print(f"  Quaternion (w,x,y,z): {goal_quat}")
    print(f"  Description: 180° pitch (flip upside down)")
    print()

    # Create optimizer
    print("Initializing optimizer...")
    cem_optimizer = CubeReorientation_CEM(
        model_config=model_config,
        sim_config=sim_config,
        cem_config=cem_config,
        goal_quat=goal_quat
    )
    print("✓ Optimizer initialized\n")

    # ---------------------------------------------------
    # INITIAL STATE (CRITICAL FOR STABILITY)
    # ---------------------------------------------------
    
    q0 = jnp.zeros(23)
    v0 = jnp.zeros(22)
    
    # Set cube initial position - HIGHER to avoid floor collision
    q0 = q0.at[16:19].set(jnp.array([0.11, 0.0, 0.10]))      # Position (x, y, z)
    q0 = q0.at[19:23].set(jnp.array([1.0, 0.0, 0.0, 0.0]))   # Identity quaternion (w, x, y, z)
    
    # Hand joints start at zero (neutral pose)
    
    print("Initial state:")
    print(f"  Cube position:    {q0[16:19]}")
    print(f"  Cube orientation: {q0[19:23]}")
    print(f"  Hand joints:      all zeros (neutral)")
    print()

    # ---------------------------------------------------
    # RUN OPTIMIZATION
    # ---------------------------------------------------
    
    print("="*70)
    print("RUNNING OPTIMIZATION")
    print("="*70)
    print("This may take a few minutes...\n")
    
    t0 = time.time()
    q_opt, v_opt, tau_opt = cem_optimizer.optimize(q0=q0, v0=v0)
    times = cem_optimizer.t_sim
    tf = time.time()
    
    print(f"\n✓ Optimization completed in {tf - t0:.2f} seconds")

    # ---------------------------------------------------
    # ANALYZE RESULTS
    # ---------------------------------------------------
    
    # Convert to numpy
    times = np.array(times)
    q_opt = np.array(q_opt)
    v_opt = np.array(v_opt)
    tau_opt = np.array(tau_opt)

    print("\n" + "="*70)
    print("RESULTS ANALYSIS")
    print("="*70)
    
    # Check for NaN
    has_nan_q = np.any(np.isnan(q_opt))
    has_nan_v = np.any(np.isnan(v_opt))
    has_nan_tau = np.any(np.isnan(tau_opt))
    
    if has_nan_q or has_nan_v or has_nan_tau:
        print("\n⚠ WARNING: NaN values detected in solution!")
        print(f"  NaN in positions:  {has_nan_q}")
        print(f"  NaN in velocities: {has_nan_v}")
        print(f"  NaN in controls:   {has_nan_tau}")
        print("\nThis indicates numerical instability in the simulation.")
        print("Try:")
        print("  1. Increase cube height: q0[18] = 0.15")
        print("  2. Lower PD gains: Kp=[5.0]*16, Kd=[0.5]*16")
        print("  3. Smaller batch: batch_size=64")
        print("  4. Check initial state in MuJoCo viewer")
    else:
        print("\n✓ SUCCESS: No NaN values detected!\n")
        
        # Extract final state
        final_cube_pos = q_opt[-1, 16:19]
        final_cube_quat = q_opt[-1, 19:23]
        final_cube_linvel = v_opt[-1, 16:19]
        final_cube_angvel = v_opt[-1, 19:22]
        
        # Also check initial and midpoint
        initial_cube_pos = q_opt[0, 16:19]
        mid_cube_pos = q_opt[len(q_opt)//2, 16:19]
        
        print("Cube trajectory:")
        print(f"  Initial position:  [{initial_cube_pos[0]:.4f}, {initial_cube_pos[1]:.4f}, {initial_cube_pos[2]:.4f}]")
        print(f"  Midpoint position: [{mid_cube_pos[0]:.4f}, {mid_cube_pos[1]:.4f}, {mid_cube_pos[2]:.4f}]")
        print(f"  Final position:    [{final_cube_pos[0]:.4f}, {final_cube_pos[1]:.4f}, {final_cube_pos[2]:.4f}]")
        print(f"  Target position:   [0.1100, 0.0000, 0.1000]")
        
        # Check if cube fell
        if final_cube_pos[2] < 0.0:
            print("\n⚠ WARNING: Cube fell below ground (z < 0)!")
            print("  The hand is not holding the cube properly.")
            print("  Try: Increase w_position and w_terminal_pos weights")
        elif final_cube_pos[2] < 0.05:
            print("\n⚠ WARNING: Cube very close to ground")
        
        print(f"\nFinal cube state:")
        print(f"  Position:     [{final_cube_pos[0]:.4f}, {final_cube_pos[1]:.4f}, {final_cube_pos[2]:.4f}]")
        print(f"  Orientation:  [{final_cube_quat[0]:.4f}, {final_cube_quat[1]:.4f}, {final_cube_quat[2]:.4f}, {final_cube_quat[3]:.4f}]")
        print(f"  Goal orient:  [{goal_quat[0]:.4f}, {goal_quat[1]:.4f}, {goal_quat[2]:.4f}, {goal_quat[3]:.4f}]")
        print(f"  Linear vel:   [{final_cube_linvel[0]:.4f}, {final_cube_linvel[1]:.4f}, {final_cube_linvel[2]:.4f}]")
        print(f"  Angular vel:  [{final_cube_angvel[0]:.4f}, {final_cube_angvel[1]:.4f}, {final_cube_angvel[2]:.4f}]")
        
        # Calculate orientation error
        def quat_distance(q1, q2):
            q1 = q1 / np.linalg.norm(q1)
            q2 = q2 / np.linalg.norm(q2)
            dot = np.abs(np.dot(q1, q2))
            dot = np.clip(dot, 0.0, 1.0)
            return 2.0 * np.arccos(dot)
        
        final_error = quat_distance(final_cube_quat, np.array(goal_quat))
        
        print(f"\nOrientation error: {final_error:.4f} rad = {np.degrees(final_error):.2f}°")
        
        # Success check
        success_threshold = np.radians(10)  # 10 degrees
        if final_error < success_threshold:
            print(f"\n✓ EXCELLENT! Cube reoriented to within {np.degrees(success_threshold):.1f}°")
        elif final_error < np.radians(30):
            print(f"\n✓ GOOD! Orientation error < 30°")
            print("  Consider: more iterations, higher terminal weight")
        else:
            print(f"\n⚠ Orientation error > 30°")
            print("  Try: increase w_terminal_orient to 10.0")

    # ---------------------------------------------------
    # SAVE RESULTS
    # ---------------------------------------------------
    
    save_dir = "./results/cube/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"\nCreated directory: {save_dir}")
    
    time_file = save_dir + "time.csv"
    q_file = save_dir + "q_opt.csv"
    v_file = save_dir + "v_opt.csv"
    tau_file = save_dir + "tau_opt.csv"
    
    np.savetxt(time_file, times, delimiter=",")
    np.savetxt(q_file, q_opt, delimiter=",")
    np.savetxt(v_file, v_opt, delimiter=",")
    np.savetxt(tau_file, tau_opt, delimiter=",")
    
    print(f"\n✓ Results saved to: {save_dir}")
    
    # ---------------------------------------------------
    # NEXT STEPS
    # ---------------------------------------------------
    
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("1. Visualize trajectory:")
    print("   python visualize_cube_reorientation.py")
    print()
    print("2. Adjust cost weights if needed (in __init__ method):")
    print("   - Increase w_terminal_orient for better final accuracy")
    print("   - Increase w_position to keep cube closer to hand")
    print()
    print("3. Try different goals:")
    print("   - 90° rotation: [0.707, 0, 0.707, 0]")
    print("   - 180° roll: [0, 1, 0, 0]")
    print("="*70)