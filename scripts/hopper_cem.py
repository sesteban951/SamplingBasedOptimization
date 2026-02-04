##
#
# Hopper CEM
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
# Hopper CEM
#############################################################


# CEM optimizer class
class Hopper_CEM(CrossEntropyMethod):

    def __init__(self, model_config: Model_Config,
                       sim_config:   ParallelSim_Config,
                       cem_config:   CrossEntropyMethod_Config):
        
        # initialize the parent class
        super().__init__(model_config, sim_config, cem_config)

        # linear interpolate based on the initial and final state
        q0 = jnp.array([0.0, 1.0, 0.0, 0.0])
        v0 = jnp.array([0.0, 0.0, 0.0, 0.0])
        qf = jnp.array([1.0, 1.0, 0.0, 0.5])
        vf = jnp.array([0.0, 0.0, 0.0, 0.0])
        self._make_reference(q0, v0, qf, vf)


    # make a reference trajectory
    def _make_reference(self, q0, v0, qf, vf):
        """
        Make a reference trajectory for the hopper.

        Args:
            q0: jnp.array, shape (nq,) - initial generalized position.
            v0: jnp.array, shape (nv,) - initial generalized velocity.
            qf: jnp.array, shape (nq,) - final generalized position.
            vf: jnp.array, shape (nv,) - final generalized velocity.
        """

        theta_des = 0.0
        cos_ = jnp.cos(theta_des)
        sin_ = jnp.sin(theta_des)

        q0 = jnp.array([0.0, 1.0, cos_, sin_ , 0.0])
        qf = jnp.array([1.0, 1.0, cos_, sin_ , 0.5])
        self.q_ref = jnp.linspace(q0, qf, self.N + 1)  # shape (N+1, nq)
        self.v_ref = jnp.linspace(v0, vf, self.N + 1)  # shape (N+1, nv)


    # running cost
    def cost(self, q, v, tau):
        """
        cost function.

        Args:
            q: jnp.array, shape (B, N+1, nq) - generalized position trajectory.
            v: jnp.array, shape (B, N+1, nv) - generalized velocity trajectory.
            tau: jnp.array, shape (B, N, nu) - control input trajectory.
        Returns:
            J: jnp.array, shape (B,) - cost for each batch.
        """

        # cost weights
        w_px = 5.0
        w_pz = 5.0
        w_theta = 15.0
        w_pl = 0.1
        w_vx = 0.1
        w_vz = 0.1
        w_omega = 1.0
        w_vl = 0.01
        w_tau = 0.01

        wf_px = 10.0 * w_px
        wf_pz = 10.0 * w_pz
        wf_theta = 10.0 * w_theta
        wf_pl = 10.0 * w_pl
        wf_vx = 10.0 * w_vx
        wf_vz = 10.0 * w_vz
        wf_omega = 10.0 * w_omega
        wf_vl = 10.0 * w_vl

        # ---------------------------------------------------
        # RUNNING COST
        # ---------------------------------------------------
        
        # running state (t = 0 to N-1)
        px_t = q[:, :-1, 0]      # (B, N)
        pz_t = q[:, :-1, 1]      # (B, N)
        cos_t = q[:, :-1, 2]     # (B, N)
        sin_t = q[:, :-1, 3]     # (B, N)
        pl_t = q[:, :-1, 4]      # (B, N)
        
        vx_t = v[:, :-1, 0]      # (B, N)
        vz_t = v[:, :-1, 1]      # (B, N)
        omega_t = v[:, :-1, 2]   # (B, N)
        vl_t = v[:, :-1, 3]      # (B, N)

        # reference trajectory (t = 0 to N-1)
        px_ref_t = self.q_ref[:-1, 0]    # (N,)
        pz_ref_t = self.q_ref[:-1, 1]    # (N,)
        cos_ref_t = self.q_ref[:-1, 2]   # (N,)
        sin_ref_t = self.q_ref[:-1, 3]   # (N,)
        pl_ref_t = self.q_ref[:-1, 4]    # (N,)
        
        vx_ref_t = self.v_ref[:-1, 0]      # (N,)
        vz_ref_t = self.v_ref[:-1, 1]      # (N,)
        omega_ref_t = self.v_ref[:-1, 2]   # (N,)
        vl_ref_t = self.v_ref[:-1, 3]      # (N,)

        # compute running costs (tracking reference)
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
        
        tau_running = w_tau * jnp.sum(jnp.square(tau), axis=-1)   # (B, N)

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
        # TERMINAL COST
        # ---------------------------------------------------
        
        # terminal states (t = N)
        px_T = q[:, -1, 0]       # (B,)
        pz_T = q[:, -1, 1]       # (B,)
        cos_T = q[:, -1, 2]      # (B,)
        sin_T = q[:, -1, 3]      # (B,)
        pl_T = q[:, -1, 4]       # (B,)
        
        vx_T = v[:, -1, 0]       # (B,)
        vz_T = v[:, -1, 1]       # (B,)
        omega_T = v[:, -1, 2]    # (B,)
        vl_T = v[:, -1, 3]       # (B,)

        # reference terminal state (t = N)
        px_ref_T = self.q_ref[-1, 0]      # scalar
        pz_ref_T = self.q_ref[-1, 1]      # scalar
        cos_ref_T = self.q_ref[-1, 2]     # scalar
        sin_ref_T = self.q_ref[-1, 3]     # scalar
        pl_ref_T = self.q_ref[-1, 4]      # scalar
        
        vx_ref_T = self.v_ref[-1, 0]      # scalar
        vz_ref_T = self.v_ref[-1, 1]      # scalar
        omega_ref_T = self.v_ref[-1, 2]   # scalar
        vl_ref_T = self.v_ref[-1, 3]      # scalar

        # compute terminal costs (tracking reference)
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
        
        # total cost
        J = running_cost + terminal_cost  # (B,)

        return J

    
#############################################################
# EXAMPLE USAGE
#############################################################


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import time

    # print deivce that we will use
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
        T=3.0,
        iterations=20,
        N_elite=2048,
        N_knots=3 * 20,
        spline_type="ZOH",
        # N_knots=20,
        # spline_type="Bezier",
    )

    # create the CEM optimizer
    cem_optimizer = Hopper_CEM(
        model_config=model_config,
        sim_config=sim_config,
        cem_config=cem_config
    )

    # optimize from an initial state
    q_opt, v_opt, tau_opt = cem_optimizer.optimize(
        q0 = jnp.array([0.0, 1.0, 0.0, 0.0]),  # in the air, leg at zero pos
        v0 = jnp.array([0.0, 0.0, 0.0, 0.0])   # initial velocity
    )
    times = cem_optimizer.t_sim

    # convert to numpy for plotting
    times = np.array(times)
    q_opt = np.array(q_opt)
    v_opt = np.array(v_opt)
    tau_opt = np.array(tau_opt)

    # save as csv files in the results folder
    time_file = "./results/hopper/times.csv"
    q_file = "./results/hopper/q_opt.csv"
    v_file = "./results/hopper/v_opt.csv"
    tau_file = "./results/hopper/tau_opt.csv"
    np.savetxt(time_file, times, delimiter=",")
    np.savetxt(q_file, q_opt, delimiter=",")
    np.savetxt(v_file, v_opt, delimiter=",")
    np.savetxt(tau_file, tau_opt, delimiter=",")

    print(times.shape)
    print(q_opt.shape)
    print(v_opt.shape)
    print(tau_opt.shape)