##
#
# Hopper Backflip CEM
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
from utils.interpolation.interp import *


#############################################################
# Hopper CEM
#############################################################

# CEM optimizer class
class Hopper_CEM(CrossEntropyMethod):

    def __init__(self, model_config: Model_Config,
                       sim_config:   ParallelSim_Config,
                       cem_config:   CrossEntropyMethod_Config):
        
        # initialize the parent class
        super().__init__(model_config, sim_config, cem_config)

        # Create reference trajectory (simple linear interpolation)
        self._make_reference()

    def _make_reference(self):
        """
        Create reference trajectory for a backflip maneuver.
        The hopper should:
        1. Jump up while starting rotation
        2. Complete a full backward rotation (2π) in the air
        3. Land with legs extended and upright
        """
        
        # Time array
        t = self.t_sim  # shape (N+1,)
        T = self.T_eff
        N = len(t) - 1
        
        # Initialize reference arrays
        q_ref = jnp.zeros((N+1, 4))
        v_ref = jnp.zeros((N+1, 4))
        
        # Normalized time [0, 1]
        tau = t / T
        
        # ========================================
        # HORIZONTAL POSITION (px)
        # ========================================
        # Slight forward motion during the flip
        px_start = 0.0
        px_end = 0.5  # move half a meter forward
        px_ref = px_start + (px_end - px_start) * tau
        
        # ========================================
        # VERTICAL POSITION (pz)
        # ========================================
        # Parabolic jump trajectory
        pz_start = 1.5   # starting height (in the air as per q0)
        pz_apex = 2.5    # peak height of the flip
        pz_end = 1.5     # land at same height
        
        # Parabolic arc: pz = a*tau^2 + b*tau + c
        # At tau=0: pz_start, at tau=0.5: pz_apex, at tau=1: pz_end
        pz_ref = pz_start + 4*(pz_apex - pz_start)*tau - 4*(pz_apex - pz_start)*tau**2
        
        # ========================================
        # ANGLE (theta) - BACKFLIP
        # ========================================
        # Complete backward rotation: 0 → -2π (or 2π depending on your sign convention)
        # Using smooth cubic interpolation for rotation
        theta_start = 0.0
        theta_end = -2.0 * jnp.pi  # full backflip (negative = backward)
        
        # Smooth rotation with cubic easing
        theta_ref = theta_start + (theta_end - theta_start) * (3*tau**2 - 2*tau**3)
        
        # ========================================
        # LEG LENGTH (pl)
        # ========================================
        # Retract during flight, extend for landing
        pl_start = 0.0     # retracted at start
        pl_mid = -0.2      # retract more during flip
        pl_end = 0.0       # extend for landing
        
        # Smooth leg trajectory
        pl_ref = jnp.where(
            tau < 0.5,
            pl_start + 2*(pl_mid - pl_start)*tau,  # retract in first half
            pl_mid + 2*(pl_end - pl_mid)*(tau - 0.5)  # extend in second half
        )
        
        # ========================================
        # VELOCITIES (numerical derivatives)
        # ========================================
        dt = self.sim.dt
        
        # Horizontal velocity
        vx_ref = jnp.gradient(px_ref, dt)
        
        # Vertical velocity
        vz_ref = jnp.gradient(pz_ref, dt)
        
        # Angular velocity
        omega_ref = jnp.gradient(theta_ref, dt)
        
        # Leg velocity
        vl_ref = jnp.gradient(pl_ref, dt)
        
        # ========================================
        # ASSEMBLE REFERENCES
        # ========================================
        q_ref = q_ref.at[:, 0].set(px_ref)
        q_ref = q_ref.at[:, 1].set(pz_ref)
        q_ref = q_ref.at[:, 2].set(theta_ref)
        q_ref = q_ref.at[:, 3].set(pl_ref)
        
        v_ref = v_ref.at[:, 0].set(vx_ref)
        v_ref = v_ref.at[:, 1].set(vz_ref)
        v_ref = v_ref.at[:, 2].set(omega_ref)
        v_ref = v_ref.at[:, 3].set(vl_ref)
        
        # Store as instance variables
        self.q_ref = q_ref
        self.v_ref = v_ref
        

    def cost(self, q, v, tau):
        """
        Cost function for backflip trajectory optimization.

        Args:
            q: jnp.array, shape (B, N+1, nq) - generalized position trajectory.
            v: jnp.array, shape (B, N+1, nv) - generalized velocity trajectory.
            tau: jnp.array, shape (B, N, nu) - control input trajectory.
        Returns:
            J: jnp.array, shape (B,) - cost for each batch.
        """

        # Cost weights
        w_px = 5.0
        w_pz = 5.0
        w_theta = 25.0        # Direct angle tracking for backflip
        w_pl = 1.0            # Increased leg tracking
        w_vx = 0.1
        w_vz = 0.1
        w_omega = 1.0         # Increased angular velocity tracking
        w_vl = 0.1
        w_tau_theta = 0.01
        w_tau_pl = 0.01

        # Terminal weights (50x running costs)
        wf_px = 50.0 * w_px
        wf_pz = 50.0 * w_pz
        wf_theta = 50.0 * w_theta
        wf_pl = 50.0 * w_pl
        wf_vx = 50.0 * w_vx
        wf_vz = 50.0 * w_vz
        wf_omega = 50.0 * w_omega
        wf_vl = 50.0 * w_vl

        # ---------------------------------------------------
        # RUNNING COST (t = 0 to N-1)
        # ---------------------------------------------------
        
        # Extract running states
        px_t = q[:, :-1, 0]      # (B, N)
        pz_t = q[:, :-1, 1]      # (B, N)
        theta_t = q[:, :-1, 2]   # (B, N) - angle
        pl_t = q[:, :-1, 3]      # (B, N) - leg length
        
        vx_t = v[:, :-1, 0]      # (B, N)
        vz_t = v[:, :-1, 1]      # (B, N)
        omega_t = v[:, :-1, 2]   # (B, N)
        vl_t = v[:, :-1, 3]      # (B, N)

        # Reference trajectory (t = 0 to N-1)
        px_ref_t = self.q_ref[:-1, 0]    # (N,)
        pz_ref_t = self.q_ref[:-1, 1]    # (N,)
        theta_ref_t = self.q_ref[:-1, 2] # (N,)
        pl_ref_t = self.q_ref[:-1, 3]    # (N,)
        
        vx_ref_t = self.v_ref[:-1, 0]      # (N,)
        vz_ref_t = self.v_ref[:-1, 1]      # (N,)
        omega_ref_t = self.v_ref[:-1, 2]   # (N,)
        vl_ref_t = self.v_ref[:-1, 3]      # (N,)

        # Compute running costs (direct tracking)
        px_running = w_px * jnp.square(px_t - px_ref_t)           # (B, N)
        pz_running = w_pz * jnp.square(pz_t - pz_ref_t)           # (B, N)
        theta_running = w_theta * jnp.square(theta_t - theta_ref_t)  # (B, N) - DIRECT ANGLE
        pl_running = w_pl * jnp.square(pl_t - pl_ref_t)           # (B, N)
        
        vx_running = w_vx * jnp.square(vx_t - vx_ref_t)           # (B, N)
        vz_running = w_vz * jnp.square(vz_t - vz_ref_t)           # (B, N)
        omega_running = w_omega * jnp.square(omega_t - omega_ref_t)  # (B, N)
        vl_running = w_vl * jnp.square(vl_t - vl_ref_t)           # (B, N)
        
        tau_running = (
            w_tau_theta * jnp.square(tau[:, :, 0]) +  # (B, N)
            w_tau_pl * jnp.square(tau[:, :, 1])      # (B, N)
        )

        running_cost = jnp.sum(
            px_running
            + pz_running
            + theta_running
            + pl_running
            + vx_running
            + vz_running
            + omega_running
            + vl_running
            + tau_running,
            axis=-1  # sum over N
        ) * self.sim.dt  # (B,)

        # ---------------------------------------------------
        # TERMINAL COST (t = N)
        # ---------------------------------------------------
        
        # Terminal states
        px_T = q[:, -1, 0]       # (B,)
        pz_T = q[:, -1, 1]       # (B,)
        theta_T = q[:, -1, 2]    # (B,)
        pl_T = q[:, -1, 3]       # (B,)
        
        vx_T = v[:, -1, 0]       # (B,)
        vz_T = v[:, -1, 1]       # (B,)
        omega_T = v[:, -1, 2]    # (B,)
        vl_T = v[:, -1, 3]       # (B,)

        # Reference terminal state
        px_ref_T = self.q_ref[-1, 0]      # scalar
        pz_ref_T = self.q_ref[-1, 1]      # scalar
        theta_ref_T = self.q_ref[-1, 2]   # scalar
        pl_ref_T = self.q_ref[-1, 3]      # scalar
        
        vx_ref_T = self.v_ref[-1, 0]      # scalar
        vz_ref_T = self.v_ref[-1, 1]      # scalar
        omega_ref_T = self.v_ref[-1, 2]   # scalar
        vl_ref_T = self.v_ref[-1, 3]      # scalar

        # Compute terminal costs (direct tracking)
        px_terminal = wf_px * jnp.square(px_T - px_ref_T)           # (B,)
        pz_terminal = wf_pz * jnp.square(pz_T - pz_ref_T)           # (B,)
        theta_terminal = wf_theta * jnp.square(theta_T - theta_ref_T)  # (B,) - DIRECT ANGLE
        pl_terminal = wf_pl * jnp.square(pl_T - pl_ref_T)           # (B,)
        
        vx_terminal = wf_vx * jnp.square(vx_T - vx_ref_T)           # (B,)
        vz_terminal = wf_vz * jnp.square(vz_T - vz_ref_T)           # (B,)
        omega_terminal = wf_omega * jnp.square(omega_T - omega_ref_T)  # (B,)
        vl_terminal = wf_vl * jnp.square(vl_T - vl_ref_T)           # (B,)

        terminal_cost = (
            px_terminal
            + pz_terminal
            + theta_terminal
            + pl_terminal
            + vx_terminal
            + vz_terminal
            + omega_terminal
            + vl_terminal
        )  # (B,)
        
        # Total cost
        J = running_cost + terminal_cost  # (B,)

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
        xml_path="./models/hopper/hopper.xml",
        Kp=[20.0, 1000.0], 
        Kd=[1.0, 10.0],  
        q_actuated_idx=[2, 3], # theta, leg
        v_actuated_idx=[2, 3], # theta ang vel, leg vel
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
        T=1.0,
        iterations=200,
        N_elite=3000,
        N_knots=2*10,
        spline_type="ZOH",
    )

    # create the CEM optimizer
    cem_optimizer = Hopper_CEM(
        model_config=model_config,
        sim_config=sim_config,
        cem_config=cem_config
    )

    # optimize from an initial state
    t0 = time.time()
    q_opt, v_opt, tau_opt = cem_optimizer.optimize(
        q0 = jnp.array([0.0, 1.5, 0.0, 0.0]),  # in the air, leg at zero pos
        v0 = jnp.array([0.0, -1.0, 0.0, 0.0])   # initial velocity
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
    save_dir = "./results/hopper_backflip/"
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
