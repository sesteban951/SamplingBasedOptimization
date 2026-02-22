##
#
# Cartpole CEM
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
from utils.spline import *


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


    # cost function for the trajecotry optimization
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

        # goal state
        cart_pos_des = 0.0
        cos_des = 1.0
        sin_des = 0.0
        cart_vel_des = 0.0
        pole_vel_des = 0.0

        # cost weights
        w_cart_pos = 10.0
        w_pole_pos = 10.0
        w_cart_vel = 0.1
        w_pole_vel = 0.1
        w_tau = 0.01
        wf_cart_pos = 50.0 * w_cart_pos 
        wf_pole_pos = 50.0 * w_pole_pos
        wf_cart_vel = 50.0 * w_cart_vel
        wf_pole_vel = 50.0 * w_pole_vel
    
        # running state (t = 0 to N-1)
        cart_pos_t = q[:, :-1, 0]  # (B, N)
        theta_t =    q[:, :-1, 1]  # (B, N)
        cart_vel_t = v[:, :-1, 0]  # (B, N)
        pole_vel_t = v[:, :-1, 1]  # (B, N)
        cos_t = jnp.cos(theta_t)   # (B, N)
        sin_t = jnp.sin(theta_t)   # (B, N)

        # terminal states (t = N)
        cart_pos_T = q[:, -1, 0]   # (B,)
        theta_T =    q[:, -1, 1]   # (B,)
        cart_vel_T = v[:, -1, 0]   # (B,)
        pole_vel_T = v[:, -1, 1]   # (B,)
        cos_T = jnp.cos(theta_T)   # (B,)
        sin_T = jnp.sin(theta_T)   # (B,)

        # ---------------------------------------------------
        # RUNNING COST
        # ---------------------------------------------------

        # Quadratic costs
        cart_pos_running = w_cart_pos * jnp.square(cart_pos_t - cart_pos_des)  # (B, N)
        pole_pos_running = w_pole_pos * (
            jnp.square(cos_t - cos_des) + jnp.square(sin_t - sin_des)
        )  # (B, N)
        cart_vel_running = w_cart_vel * jnp.square(cart_vel_t - cart_vel_des)  # (B, N)
        pole_vel_running = w_pole_vel * jnp.square(pole_vel_t - pole_vel_des)  # (B, N)
        tau_running = w_tau * jnp.sum(jnp.square(tau), axis=-1)                # (B, N)

        running_cost = jnp.sum(
              cart_pos_running
            + pole_pos_running
            + cart_vel_running
            + pole_vel_running
            + tau_running
            , axis=-1  # sum over N
        ) * self.sim.dt  # (B, )

        # ---------------------------------------------------
        # TERMINAL COST
        # ---------------------------------------------------

        # Quadratic costs
        cart_pos_terminal = wf_cart_pos * jnp.square(cart_pos_T - cart_pos_des)  # (B,)
        pole_pos_terminal = wf_pole_pos * (
            jnp.square(cos_T - cos_des) + jnp.square(sin_T - sin_des)
        )  # (B,)
        cart_vel_terminal = wf_cart_vel * jnp.square(cart_vel_T - cart_vel_des)  # (B,)
        pole_vel_terminal = wf_pole_vel * jnp.square(pole_vel_T - pole_vel_des)  # (B,)

        terminal_cost = (
            cart_pos_terminal
            + pole_pos_terminal
            + cart_vel_terminal
            + pole_vel_terminal
        )  # (B, )

        # # Exponential costs (upside down gaussian, where bottom touches 0)
        # sigma_cart_pos = 0.5
        # sigma_pole_pos = 0.5
        # sigma_cart_vel = 0.5
        # sigma_pole_vel = 0.5
        # cart_pos_terminal = wf_cart_pos * (
        #     1 - jnp.exp(-sigma_cart_pos * jnp.square(cart_pos_T - cart_pos_des))
        # )  # (B,)

        # pole_pos_terminal = wf_pole_pos * (
        #     1 - jnp.exp(-sigma_pole_pos * (
        #         jnp.square(cos_T - cos_des) + jnp.square(sin_T - sin_des)
        #     ))
        # )  # (B,)

        # cart_vel_terminal = wf_cart_vel * (
        #     1 - jnp.exp(-sigma_cart_vel * jnp.square(cart_vel_T - cart_vel_des))
        # )  # (B,)

        # pole_vel_terminal = wf_pole_vel * (
        #     1 - jnp.exp(-sigma_pole_vel * jnp.square(pole_vel_T - pole_vel_des))
        # )  # (B,)

        terminal_cost = (
            cart_pos_terminal
            + pole_pos_terminal
            + cart_vel_terminal
            + pole_vel_terminal
        )  # (B,)
        
        # total cost
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
    np.random.seed(0)

    # model config
    model_config = Model_Config(
        xml_path="./models/cartpole/cartpole.xml",
        Kp=[500.0], 
        Kd=[50.0],  
        q_actuated_idx=[0],
        v_actuated_idx=[0],
        # action_mode="tau",
        action_mode="pos",
    )

    # parallel sim config
    sim_config = ParallelSim_Config(
        batch_size = 4096,
    )

    # cem config
    s = int(time.time())
    cem_rng = jax.random.PRNGKey(s)
    cem_config = CrossEntropyMethod_Config(
        rng=cem_rng,
        T=3.0,
        iterations=200,
        N_elite=2048,
        # N_knots=4*5,
        # spline_type="ZOH",
        # N_knots=10,
        # spline_type="Linear",
        N_knots=10,
        spline_type="Cubic",
        # N_knots=10,
        # spline_type="Bezier",
    )

    # create the CEM optimizer
    cem_optimizer = Cartpole_CEM(
        model_config=model_config,
        sim_config=sim_config,
        cem_config=cem_config
    )

    # optimize from an initial state
    t0 = time.time()
    q_opt, v_opt, tau_opt = cem_optimizer.optimize(
        q0=jnp.array([0.0, np.pi]),   # initial position
        v0=jnp.array([0.0, 0.0])      # initial velocity
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
    save_dir = "./results/cartpole/"
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
