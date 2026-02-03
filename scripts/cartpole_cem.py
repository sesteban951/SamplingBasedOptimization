##
#
# Cross Entropy Method (CEM) for trajectory optimization.
#
##

# standard imports
import numpy as np
from dataclasses import dataclass

# jax imports
import jax
import jax.numpy as jnp

# custom imports
from utils.algorithms.cem import *
from utils.simulation.simulation import *
from utils.spline.bezier import *
from utils.spline.zoh import *


#############################################################
# Cartpole CEM
#############################################################

# CEM optimizer class
class Cartpole_CEM(CrossEntropyMethod):

    def __init__(self, model_config: Model_Config,
                       sim_config:   ParallelSim_Config,
                       cem_config:   CrossEntropyMethod_Config):
        
        # initialize the parent class
        super().__init__(model_config, sim_config, cem_config)

    # running cost
    def _cost(self, q, v, tau):
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
        w_pos = 10.0
        w_vel = 0.1
        w_tau = 0.01
        wf_pos = 10.0 * w_pos 
        wf_vel = 10.0 * w_vel
    
        # running costs over t=0..N-1 (exclude terminal state)
        q_t = q[:, :-1, :]   # (B, N, nq)
        v_t = v[:, :-1, :]   # (B, N, nv)

        # get pole angle positions
        theta_t = q_t[:, :, 1]  # pole angle at time t
        sin_t = jnp.sin(theta_t)
        cos_t = jnp.cos(theta_t)

        # desired angles error
        theta_des = 0.0
        sin_des = jnp.sin(theta_des)
        cos_des = jnp.cos(theta_des)
        ang_err = jnp.stack([sin_t - sin_des, cos_t - cos_des], axis=-1)  # shape (B, N, 2)

        # other desired states (cart position)
        q_des_other = jnp.array([0.0])     # (1,)
        q_other_t = q_t[:, :, 0:1]         # (B, N, 1)
        other_err = q_other_t - q_des_other  # (B, N, 1)

        # total running position penalty
        running_pos = w_pos * (
            jnp.sum(ang_err**2, axis=(1, 2)) +
            jnp.sum(other_err**2, axis=(1, 2))
        )

        # ----------------------------
        # velocity + control running cost
        # ----------------------------
        v_des = jnp.array([0.0, 0.0])  # cart vel, pole ang vel
        running_vel = w_vel * jnp.sum((v_t - v_des) ** 2, axis=(1, 2))
        running_tau = w_tau * jnp.sum(tau ** 2, axis=(1, 2))

        running_cost = self.sim.dt * (running_pos + running_vel + running_tau)

        # ----------------------------
        # terminal cost at t = N
        # ----------------------------
        q_T = q[:, -1, :]   # (B, nq)
        v_T = v[:, -1, :]   # (B, nv)

        theta_T = q_T[:, 1]  # (B,)
        ang_err_T = jnp.stack(
            [jnp.sin(theta_T) - sin_des, jnp.cos(theta_T) - cos_des],
            axis=-1
        )  # (B, 2)

        other_err_T = q_T[:, 0:1] - q_des_other  # (B, 1)

        terminal_pos = wf_pos * (
            jnp.sum(ang_err_T**2, axis=1) +
            jnp.sum(other_err_T**2, axis=1)
        )
        terminal_vel = wf_vel * jnp.sum((v_T - v_des) ** 2, axis=1)

        J = running_cost + terminal_pos + terminal_vel

        return J

#############################################################
# EXAMPLE USAGE
#############################################################

if __name__ == "__main__":

    import matplotlib.pyplot as plt

    # print deivce that we will use
    print(f"Using device: {jax.default_backend()}")
    if jax.default_backend() == "gpu":
        gpu_info = jax.devices("gpu")[0]
        print(f"GPU device: {gpu_info}")

    # fix the random seed
    np.random.seed(0)

    # model config
    model_config = Model_Config(
        xml_path="./models/cartpole/cartpole.xml",
        Kp=[500.0], 
        Kd=[50.0],  
        q_actuated_idx=[0],
        v_actuated_idx=[0],
        action_mode="tau",
        # action_mode="pos",
    )

    # parallel sim config
    sim_config = ParallelSim_Config(
        batch_size = 4096,
    )

    # cem config
    cem_rng = jax.random.PRNGKey(42)
    cem_config = CrossEntropyMethod_Config(
        rng=cem_rng,
        T=5.0,
        iterations=300,
        N_elite=2048,
        N_knots=5*5,
        spline_type="ZOH",
        # N_knots=20,
        # spline_type="Bezier",
    )

    # create the CEM optimizer
    cem_optimizer = Cartpole_CEM(
        model_config=model_config,
        sim_config=sim_config,
        cem_config=cem_config
    )

    # optimize from an initial state
    q_opt, v_opt, tau_opt = cem_optimizer.optimize(
        q0=jnp.array([0.0, np.pi + 0.1]),   # initial position
        v0=jnp.array([0.0, 0.0])            # initial velocity
    )
    times = cem_optimizer.t_sim

    # convert to numpy for plotting
    times = np.array(times)
    q_opt = np.array(q_opt)
    v_opt = np.array(v_opt)
    tau_opt = np.array(tau_opt)

    # save as csv files in the results folder
    time_file = "./results/cartpole/times.csv"
    q_file = "./results/cartpole/q_opt.csv"
    v_file = "./results/cartpole/v_opt.csv"
    tau_file = "./results/cartpole/tau_opt.csv"
    np.savetxt(time_file, times, delimiter=",")
    np.savetxt(q_file, q_opt, delimiter=",")
    np.savetxt(v_file, v_opt, delimiter=",")
    np.savetxt(tau_file, tau_opt, delimiter=",")

    print(times.shape)
    print(q_opt.shape)
    print(v_opt.shape)
    print(tau_opt.shape)

    # plot the first two positions
    plt.figure()
    plt.plot(times, q_opt[:, 0], label="Cart Position")
    plt.plot(times, q_opt[:, 1], label="Pole Angle")

    plt.figure()
    plt.plot(times, v_opt[:, 0], label="Cart Vel")
    plt.plot(times, v_opt[:, 1], label="Pole velocity")

    plt.figure()
    plt.plot(times[:-1], tau_opt[:, 0], label="Cart Force")

    plt.show()
