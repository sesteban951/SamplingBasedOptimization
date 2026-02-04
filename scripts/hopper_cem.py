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

        B, N1, nq = q.shape
        _, _, nv = v.shape
        N = N1 - 1

        # ----------------------------
        # desired targets
        # ----------------------------
        px_des   = 0.25
        pz_des   = 1.0
        theta_des = 0.0
        leg_des  = 0.0

        # indices in q (based on your usage)
        i_px, i_pz, i_theta, i_leg = 0, 1, 2, 3

        # ----------------------------
        # per-state weights (edit these)
        # ----------------------------
        # Running weights for each q state (nq,)
        wq = jnp.zeros((nq,))
        wq = wq.at[i_px].set(10.0)     # px
        wq = wq.at[i_pz].set(40.0)     # pz
        wq = wq.at[i_leg].set(0.1)     # leg position (example)
        # NOTE: we will NOT weight theta directly here if using cos/sin feature.
        # If you *also* want a direct theta penalty, set wq.at[i_theta].set(...)

        # Running weights for each v state (nv,)
        wv = jnp.zeros((nv,))
        # example: weight all velocities equally
        wv = wv.at[:].set(0.01)

        # Terminal weights (often larger)
        wqT = 10.0 * wq
        wvT = 10.0 * wv

        # Angle feature weights (cos/sin) running + terminal
        w_theta_cos = 30.0
        w_theta_sin = 30.0
        w_theta_cos_T = 1.0 * w_theta_cos
        w_theta_sin_T = 1.0 * w_theta_sin

        # Control weight
        w_tau = 0.0001

        # ----------------------------
        # slice running/terminal
        # ----------------------------
        q_t = q[:, :-1, :]      # (B, N, nq)
        v_t = v[:, :-1, :]      # (B, N, nv)
        q_T = q[:, -1, :]       # (B, nq)
        v_T = v[:, -1, :]       # (B, nv)

        # ----------------------------
        # build desired trajectories (broadcastable)
        # ----------------------------
        q_des = jnp.zeros((nq,))
        q_des = q_des.at[i_px].set(px_des)
        q_des = q_des.at[i_pz].set(pz_des)
        q_des = q_des.at[i_leg].set(leg_des)
        # theta_des handled separately via cos/sin feature (below)

        v_des = jnp.zeros((nv,))

        # ----------------------------
        # running state cost (per-state weights)
        # ----------------------------
        q_err = q_t - q_des                  # (B, N, nq)
        v_err = v_t - v_des                  # (B, N, nv)

        running_q = jnp.sum((q_err ** 2) * wq[None, None, :], axis=(1, 2))   # (B,)
        running_v = jnp.sum((v_err ** 2) * wv[None, None, :], axis=(1, 2))   # (B,)

        # angle feature running cost using cos/sin (avoids wrap issues)
        theta_t = q_t[:, :, i_theta]         # (B, N)
        cos_err = jnp.cos(theta_t) - jnp.cos(theta_des)
        sin_err = jnp.sin(theta_t) - jnp.sin(theta_des)
        running_theta = (
            w_theta_cos * jnp.sum(cos_err ** 2, axis=1) +
            w_theta_sin * jnp.sum(sin_err ** 2, axis=1)
        )  # (B,)

        # control running cost
        running_u = w_tau * jnp.sum(tau ** 2, axis=(1, 2))  # (B,)

        running_cost = self.sim.dt * (running_q + running_v + running_theta + running_u)

        # ----------------------------
        # terminal cost (per-state weights)
        # ----------------------------
        qT_err = q_T - q_des                 # (B, nq)
        vT_err = v_T - v_des                 # (B, nv)

        terminal_q = jnp.sum((qT_err ** 2) * wqT[None, :], axis=1)          # (B,)
        terminal_v = jnp.sum((vT_err ** 2) * wvT[None, :], axis=1)          # (B,)

        theta_T = q_T[:, i_theta]            # (B,)
        cos_err_T = jnp.cos(theta_T) - jnp.cos(theta_des)
        sin_err_T = jnp.sin(theta_T) - jnp.sin(theta_des)
        terminal_theta = (
            w_theta_cos_T * (cos_err_T ** 2) +
            w_theta_sin_T * (sin_err_T ** 2)
        )  # (B,)

        J = running_cost + terminal_q + terminal_v + terminal_theta
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
        T=4.0,
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