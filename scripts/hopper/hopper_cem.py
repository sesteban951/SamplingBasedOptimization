##
#
# Hopper CEM
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
from utils.spline.bezier import *
from utils.spline.zoh import *


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
        Make a reference trajectory for the hopper.
        Simple linear interpolation from start to goal.
        """
        # Initial and final states (assuming nq=4: [px, pz, theta, leg_length])
        # WARNING: For now, this is hardcoded 
        q0 = jnp.array([0.0, 2.0, 0.0, 0.0])  # start: x=0, z=2m (in air), upright, leg at 0
        qf = jnp.array([5.0, 2.0, 0.0, 0.0])  # goal: x=1m, z=2m, upright, leg at 0
        
        # Compute constant forward velocity
        vx = (qf[0] - q0[0]) / self.T_eff
        v0 = jnp.array([vx, 0.0, 0.0, 0.0])  # constant x velocity, rest zero
        vf = jnp.array([vx, 0.0, 0.0, 0.0])
        
        # Linear interpolation
        self.q_ref = jnp.linspace(q0, qf, self.N + 1)  # (N+1, nq=4)
        self.v_ref = jnp.linspace(v0, vf, self.N + 1)  # (N+1, nv=4)

    # cost function for the trajecotry optimization
    def cost(self, q, v, tau):
        """
        Cost function.

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
        w_theta = 20.0
        w_pl = 0.1
        w_vx = 0.1
        w_vz = 0.1
        w_omega = 0.1
        w_vl = 0.1
        w_tau_theta = 0.01
        w_tau_pl = 0.01

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

        # Compute cos and sin for angle tracking (like cartpole)
        cos_t = jnp.cos(theta_t)  # (B, N)
        sin_t = jnp.sin(theta_t)  # (B, N)

        # Reference trajectory (t = 0 to N-1)
        px_ref_t = self.q_ref[:-1, 0]    # (N,)
        pz_ref_t = self.q_ref[:-1, 1]    # (N,)
        theta_ref_t = self.q_ref[:-1, 2] # (N,)
        pl_ref_t = self.q_ref[:-1, 3]    # (N,)
        
        vx_ref_t = self.v_ref[:-1, 0]      # (N,)
        vz_ref_t = self.v_ref[:-1, 1]      # (N,)
        omega_ref_t = self.v_ref[:-1, 2]   # (N,)
        vl_ref_t = self.v_ref[:-1, 3]      # (N,)

        # Desired cos/sin (upright = theta = 0)
        cos_ref_t = jnp.cos(theta_ref_t)  # (N,)
        sin_ref_t = jnp.sin(theta_ref_t)  # (N,)

        # Compute running costs (tracking reference)
        px_running = w_px * jnp.square(px_t - px_ref_t)           # (B, N)
        pz_running = w_pz * jnp.square(pz_t - pz_ref_t)           # (B, N)
        theta_running = w_theta * (
            jnp.square(cos_t - cos_ref_t) + jnp.square(sin_t - sin_ref_t)
        )  # (B, N)
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

        # Compute cos and sin for terminal angle
        cos_T = jnp.cos(theta_T)  # (B,)
        sin_T = jnp.sin(theta_T)  # (B,)

        # Reference terminal state
        px_ref_T = self.q_ref[-1, 0]      # scalar
        pz_ref_T = self.q_ref[-1, 1]      # scalar
        theta_ref_T = self.q_ref[-1, 2]   # scalar
        pl_ref_T = self.q_ref[-1, 3]      # scalar
        
        vx_ref_T = self.v_ref[-1, 0]      # scalar
        vz_ref_T = self.v_ref[-1, 1]      # scalar
        omega_ref_T = self.v_ref[-1, 2]   # scalar
        vl_ref_T = self.v_ref[-1, 3]      # scalar

        # Desired cos/sin at terminal time
        cos_ref_T = jnp.cos(theta_ref_T)  # scalar
        sin_ref_T = jnp.sin(theta_ref_T)  # scalar

        # Compute terminal costs
        px_terminal = wf_px * jnp.square(px_T - px_ref_T)           # (B,)
        pz_terminal = wf_pz * jnp.square(pz_T - pz_ref_T)           # (B,)
        theta_terminal = wf_theta * (
            jnp.square(cos_T - cos_ref_T) + jnp.square(sin_T - sin_ref_T)
        )  # (B,)
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
        batch_size = 2048,
    )

    # cem config
    cem_rng = jax.random.PRNGKey(42)
    cem_config = CrossEntropyMethod_Config(
        rng=cem_rng,
        T=3.0,
        iterations=200,
        N_elite=512,
        N_knots=3*20,
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
        q0 = jnp.array([0.0, 2.0, 0.0, 0.0]),  # in the air, leg at zero pos
        v0 = jnp.array([0.0, 0.0, 0.0, 0.0])   # initial velocity
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
    save_dir = "./results/hopper/hopper_cem/"
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
