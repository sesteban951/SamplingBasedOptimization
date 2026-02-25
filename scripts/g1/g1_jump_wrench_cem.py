##
#
# G1 SRB CEM
#
##

import config

# standard imports
import numpy as np
import time
import os

# jax imports
import jax
import jax.numpy as jnp

# mujoco imports
import mujoco

# custom imports
from utils.algorithms.cem import *
from utils.algorithms.schedule import *
from utils.simulation.simulation import *
from utils.spline import *
from utils.interpolation import interp


#################################################################
# LOAD DATA
#################################################################

# load mujoco model
xml_path = "./models/g1/g1_21dof.xml"
mj_model = mujoco.MjModel.from_xml_path(xml_path)
mj_data = mujoco.MjData(mj_model)

# standing
keyframe = "standing"
key_id = mj_model.key(keyframe).id
qpos_standing = jnp.array(mj_model.key_qpos[key_id])
qvel_standing = jnp.array(mj_model.key_qvel[key_id])

# load SRB data
srb_dir = "./results/srb/srb_jump/"
t_SRB = np.loadtxt(srb_dir + "time.csv", delimiter=",")
q_SRB = np.loadtxt(srb_dir + "q_opt.csv", delimiter=",")
v_SRB = np.loadtxt(srb_dir + "v_opt.csv", delimiter=",")
a_SRB = np.loadtxt(srb_dir + "a_opt.csv", delimiter=",")
tau_SRB = np.loadtxt(srb_dir + "tau_opt.csv", delimiter=",")


#################################################################
# CEM OPTIMIZER FOR G1 JUMPING
#################################################################

# useful joint indices
pos_base_idx        = jnp.array([0, 1, 2])
quat_base_idx       = jnp.array([3, 4, 5, 6])
pos_hip_joint_idx   = jnp.array([7, 8, 9, 13, 14, 15])
pos_knee_joint_idx  = jnp.array([10, 16])
pos_ankle_joint_idx = jnp.array([11, 12, 17, 18])
pos_waist_joint_idx = jnp.array([19])
pos_shoulder_joint_idx = jnp.array([20, 21, 22, 24, 25, 26])
pos_elbow_joint_idx = jnp.array([23, 27])
pos_joints          = jnp.arange(7, 7+21)

vel_base_idx        = jnp.array([0, 1, 2])
omega_base_idx      = jnp.array([3, 4, 5])
vel_hip_joint_idx   = jnp.array([6, 7, 8, 12, 13, 14])
vel_knee_joint_idx  = jnp.array([9, 15])
vel_ankle_joint_idx = jnp.array([10, 11, 16, 17])
vel_waist_joint_idx = jnp.array([18])
vel_shoulder_joint_idx = jnp.array([19, 20, 21, 23, 24, 25])
vel_elbow_joint_idx = jnp.array([22, 26])
vel_joints          = jnp.arange(6, 6+21)


class G1_SRB_CEM(CrossEntropyMethod):

    def __init__(self, model_config: Model_Config,
                       sim_config:   ParallelSim_Config,
                       cem_config:   CrossEntropyMethod_Config):

        super().__init__(model_config, sim_config, cem_config)

        self.sim_config = sim_config

        self._make_reference_trajectory()

        # cost values
        self.w_p_base = 10.0
        self.w_quat = 10.0
        self.w_v_base = 0.1
        self.w_omega_base = 0.1
        self.w_q_joints = 1.0
        self.w_v_joints = 0.1
        self.w_tau = 1e-6
        
        self.w_force_assist = 0.1    # external wrench penalty
        self.w_moment_assist = 0.1   # external moment penalty

        terminal_scale = 20.0

        self.wf_p_base = terminal_scale * self.w_p_base
        self.wf_quat = terminal_scale * self.w_quat
        self.wf_v_base = terminal_scale * self.w_v_base
        self.wf_omega_base = terminal_scale * self.w_omega_base
        self.wf_q_joints = terminal_scale * self.w_q_joints
        self.wf_v_joints = terminal_scale * self.w_v_joints


    def _make_reference_trajectory(self):

        # extract the trajectory components
        p_com = q_SRB[:, 0:3]   # (T_SRB, 3)
        v_com = v_SRB[:, 0:3]   # (T_SRB, 3)
        a_com = a_SRB[:, 0:3]   # (T_SRB, 3)
        quat  = q_SRB[:, 3:7]   # (T_SRB, 4)
        omega = v_SRB[:, 3:6]   # (T_SRB, 3)
        alpha = a_SRB[:, 3:6]   # (T_SRB, 3)

        # make simulation time array
        T_SRB = t_SRB[-1]         
        dt_sim = self.sim.dt
        t_sim = jnp.arange(0, T_SRB + dt_sim, dt_sim) # [0, dt_sim, 2*dt_sim, ..., T_SRB]
        nodes_sim = len(t_sim)

        # interpolate the reference trajectory at simulation time steps
        p_com_ref = np.zeros((nodes_sim, 3))
        v_com_ref = np.zeros((nodes_sim, 3))
        a_com_ref = np.zeros((nodes_sim, 3))
        quat_ref = np.zeros((nodes_sim,  4))
        omega_ref = np.zeros((nodes_sim, 3))
        alpha_ref = np.zeros((nodes_sim, 3))
        for i in range(nodes_sim):
            
            # current sim time point
            t = t_sim[i]

            # before the start of the SRB trajectory, hold the initial state
            if t < t_SRB[0]:
                p_com_ref[i] = p_com[0]
                v_com_ref[i] = v_com[0]
                a_com_ref[i] = a_com[0]
                quat_ref[i] = quat[0]
                omega_ref[i] = omega[0]
                alpha_ref[i] = alpha[0]

            # after the end of the SRB trajectory, hold the final state
            elif t > t_SRB[-1]:
                p_com_ref[i] = p_com[-1]
                v_com_ref[i] = v_com[-1]
                a_com_ref[i] = a_com[-1]
                quat_ref[i] = quat[-1]
                omega_ref[i] = omega[-1]
                alpha_ref[i] = alpha[-1]

            # within the SRB trajectory, interpolate between the two nearest points
            else: 
                # find the two SRB time points that bracket the current sim time
                idx_2 = np.searchsorted(t_SRB, t, side='right')
                idx_2 = min(idx_2, len(t_SRB) - 1)  # clamp to last valid index
                idx_1 = max(idx_2 - 1, 0)
                t1, t2 = t_SRB[idx_1], t_SRB[idx_2]

                # interpolation coeff
                coeff = (t - t1) / (t2 - t1)
                coeff = np.clip(coeff, 0.0, 1.0)  # ensure coeff is within [0, 1]

                # interpolate the reference trajectory
                p_com_ref[i]  = interp.lerp(p_com[idx_1],  p_com[idx_2],  coeff)
                v_com_ref[i]  = interp.lerp(v_com[idx_1],  v_com[idx_2],  coeff)
                a_com_ref[i]  = interp.lerp(a_com[idx_1],  a_com[idx_2],  coeff)
                quat_ref[i]   = interp.slerp(quat[idx_1],  quat[idx_2],   coeff)
                omega_ref[i]  = interp.lerp(omega[idx_1],  omega[idx_2],  coeff)
                alpha_ref[i]  = interp.lerp(alpha[idx_1],  alpha[idx_2],  coeff)

        self.p_com_ref = jnp.array(p_com_ref)  # (nodes_sim, 3)
        self.v_com_ref = jnp.array(v_com_ref)  # (nodes_sim, 3)
        self.a_com_ref = jnp.array(a_com_ref)  # (nodes_sim, 3)
        self.quat_ref  = jnp.array(quat_ref)   # (nodes_sim, 4)
        self.omega_ref = jnp.array(omega_ref)  # (nodes_sim, 3)
        self.alpha_ref = jnp.array(alpha_ref)  # (nodes_sim, 3)

        # extract nominal joint trajectories from standing position
        B = self.sim_config.batch_size
        q_joints_standing = qpos_standing[pos_joints]
        v_joints_standing = qvel_standing[vel_joints]
        self.q_joints_standing = jnp.broadcast_to(q_joints_standing, (B, q_joints_standing.shape[0]))
        self.v_joints_standing = jnp.broadcast_to(v_joints_standing, (B, v_joints_standing.shape[0]))


    def cost(self, q, v, tau, w):
        """
        Cost function to evaluate the rollouts.

        Args:
            q: jnp.array,   shape (B, N+1, nq) - generalized positions trajectory.
            v: jnp.array,   shape (B, N+1, nv) - generalized velocities trajectory.
            tau: jnp.array, shape (B, N, nu) - control inputs trajectory.
            w: jnp.array,   shape (B, N, 6) - external wrench trajectory.
        Returns:
            costs: jnp.array, shape (B,) - cost for each rollout.
        """

        B = tau.shape[0]
        N = tau.shape[1]

        # base state: full trajectory -> (B, N+1, *)
        p_base =     q[:, :N+1, pos_base_idx]
        quat_base =  q[:, :N+1, quat_base_idx]
        v_base =     v[:, :N+1, vel_base_idx]
        omega_base = v[:, :N+1, omega_base_idx]
        q_joints =   q[:, :N+1, pos_joints]
        v_joints =   v[:, :N+1, vel_joints]
        
        # errors over full trajectory -> (B, N+1, *)
        e_pos_base   = p_base    - self.p_com_ref[:N+1]                # (B, N+1, 3)
        e_quat = self.sim.dyn._quat_log_diff(
            jnp.broadcast_to(self.quat_ref[:N+1], (B, N+1, 4)).reshape(B*(N+1), 4),
            quat_base.reshape(B*(N+1), 4)
        ).reshape(B, N+1, 3)                                           # (B, N+1, 3)
        e_vel_base   = v_base    - self.v_com_ref[:N+1]                # (B, N+1, 3)
        e_omega_base = omega_base - self.omega_ref[:N+1]               # (B, N+1, 3)
        e_q_joints   = q_joints  - self.q_joints_standing[:, None, :]  # (B, N+1, 21)
        e_v_joints   = v_joints  - self.v_joints_standing[:, None, :]  # (B, N+1, 21)

        # ========================== Running cost ==========================

        # compute squared errors, slice [:, :-1] to exclude the final time step for running cost
        se_pos_base   = jnp.sum(jnp.sum(e_pos_base  [:, :-1, :]**2, axis=-1), axis=-1)
        se_quat       = jnp.sum(jnp.sum(e_quat      [:, :-1, :]**2, axis=-1), axis=-1)
        se_vel_base   = jnp.sum(jnp.sum(e_vel_base  [:, :-1, :]**2, axis=-1), axis=-1)
        se_omega_base = jnp.sum(jnp.sum(e_omega_base[:, :-1, :]**2, axis=-1), axis=-1)
        se_q_joints   = jnp.sum(jnp.sum(e_q_joints  [:, :-1, :]**2, axis=-1), axis=-1)
        se_v_joints   = jnp.sum(jnp.sum(e_v_joints  [:, :-1, :]**2, axis=-1), axis=-1)
        se_tau = jnp.sum(jnp.sum(tau**2, axis=-1), axis=-1)
        se_force_assist  = jnp.sum(jnp.sum(w[:, :, :3]**2, axis=-1), axis=-1)  # (B,)
        se_moment_assist = jnp.sum(jnp.sum(w[:, :, 3:]**2, axis=-1), axis=-1)  # (B,)

        running_cost = (
            self.w_p_base     * se_pos_base   +
            self.w_quat       * se_quat       +
            self.w_v_base     * se_vel_base   +
            self.w_omega_base * se_omega_base +
            self.w_q_joints   * se_q_joints   +
            self.w_v_joints   * se_v_joints   +
            self.w_tau        * se_tau        +
            self.w_force_assist  * se_force_assist  +
            self.w_moment_assist * se_moment_assist
        ) * self.sim.dt   # (B,)

        # ========================== Terminal cost ==========================

        # compute squared errors at the final time step [:, -1, :]
        se_pos_base_f   = jnp.sum(e_pos_base  [:, -1, :]**2, axis=-1)
        se_quat_f       = jnp.sum(e_quat      [:, -1, :]**2, axis=-1)
        se_vel_base_f   = jnp.sum(e_vel_base  [:, -1, :]**2, axis=-1)
        se_omega_base_f = jnp.sum(e_omega_base[:, -1, :]**2, axis=-1)
        se_q_joints_f   = jnp.sum(e_q_joints  [:, -1, :]**2, axis=-1)
        se_v_joints_f   = jnp.sum(e_v_joints  [:, -1, :]**2, axis=-1)

        terminal_cost = (
            self.wf_p_base     * se_pos_base_f   +
            self.wf_quat       * se_quat_f       +
            self.wf_v_base     * se_vel_base_f   +
            self.wf_omega_base * se_omega_base_f +
            self.wf_q_joints   * se_q_joints_f   +
            self.wf_v_joints   * se_v_joints_f
        )  # (B,)

        return running_cost + terminal_cost  # (B,)
    

    # Custom CEM optimization for this particular task
    def optimize(self, q0, v0):
        """
        Perform CEM optimization.

        Args:
            q0: jnp.array, shape (B, nq) - initial generalized positions.
            v0: jnp.array, shape (B, nv) - initial generalized velocities.
        Returns:
            q_opt: jnp.array, shape (N+1, nq) - optimal generalized positions trajectory.
            v_opt: jnp.array, shape (N+1, nv) - optimal generalized velocities trajectory.
            tau_opt: jnp.array, shape (N, nu) - optimal control inputs trajectory.
        """

        # initialize the optimal solution
        J_opt = jnp.inf
        q_opt = None
        v_opt = None
        tau_opt = None

        # perform CEM iterations
        for itr in range(self.cem_config.iterations):

            # evaluate the spline at simulation times
            y_val = self.spline.evaluate(self.t_sim[:-1])  # shape (B, N, nu)

            # pack SRB references for wrench injection -> (N, 7), (N, 6), (N, 6)
            N = y_val.shape[1]
            q_srb_ref = jnp.concatenate([self.p_com_ref[:N], self.quat_ref[:N]],  axis=-1)  # (N, 7)
            v_srb_ref = jnp.concatenate([self.v_com_ref[:N], self.omega_ref[:N]], axis=-1)  # (N, 6)
            a_srb_ref = jnp.concatenate([self.a_com_ref[:N], self.alpha_ref[:N]], axis=-1)  # (N, 6)
            # a = linear_schedule(itr, self.cem_config.iterations, alpha_max=1.0)
            a = exponential_schedule(itr, self.cem_config.iterations, alpha_max=1.0, lam=20.0)
            q_log, v_log, tau_log, w_log = self.sim.rollout(q0, v0, y_val,
                                                            q_srb_ref=q_srb_ref,
                                                            v_srb_ref=v_srb_ref,
                                                            a_srb_ref=a_srb_ref,
                                                            w_scale=a)
            q_log.block_until_ready()

            # compute costs
            J = self.cost(q_log, v_log, tau_log, w_log)  # shape (B,)
            J.block_until_ready()

            # select elite samples
            J_elite_neg, elite_idx = jax.lax.top_k(-J, self.cem_config.N_elite)
            J_elite = -J_elite_neg  # shape (N_elite,)

            # select the elite splines
            Y_elite = jnp.take(self.spline.Y, elite_idx, axis=0)  # shape (N_elite, N_knots, nu)

            # update the distribution
            self._update_distribution(Y_elite)

            # sample new knot points from the updated distribution
            Y_samples = self._sample_knot_points()  # shape (B, N_knots, nu)
            self.spline.update_knots(Y_samples)

            # compute the norm of the covariance for monitoring
            cov_norm = jnp.linalg.norm(self.Sigma, ord='fro')

            # record the best solution found so far
            J_min = J_elite.min()
            if J_min < J_opt:

                # set best
                J_opt = J_min
                idx_in_elite = jnp.argmin(J_elite)  # Find best within elites
                idx_opt = elite_idx[idx_in_elite]   # Map to actual batch index

                # set optimal
                q_opt = q_log[idx_opt, :, :]
                v_opt = v_log[idx_opt, :, :]
                tau_opt = tau_log[idx_opt, :, :]

            # compute the average elite cost for monitoring
            J_elite_avg = jnp.mean(J_elite)
            J_elite_best = J_elite.min()

            # print iteration info
            itr_width = len(str(self.cem_config.iterations))  # e.g., 400 → width=3
            print(f"Iteration {itr+1:0{itr_width}d}/{self.cem_config.iterations} | "
                  f"J_elite_avg: {J_elite_avg:.1f} | "
                  f"J_elite_best: {J_elite_best:.1f} | "
                  f"J_best: {J_opt:.1f} | "
                  f"‖Σ‖: {cov_norm:.4f} | "
                  f"α: {a:.3f}")
            
        return q_opt, v_opt, tau_opt

#############################################################
# EXAMPLE USAGE
#############################################################

if __name__ == "__main__":

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
        xml_path=xml_path,
        Kp=[400, 400, 400, 400, 100, 100, # left leg
            400, 400, 400, 400, 100, 100, # right leg
            100,                          # waist
            150, 150, 150, 150,           # left arm
            150, 150, 150, 150],          # right arm
        Kd=[3.0, ] * 21,  
        q_actuated_idx=list(range(7,7+21)),
        v_actuated_idx=list(range(6,6+21)),
        action_mode="pos"
    )

    # parallel sim config
    sim_config = ParallelSim_Config(
        batch_size = 4096,
        use_external_wrench=True,
    )

    # number of knots for the trajecotry
    T_SRB = t_SRB[-1]
    knots_per_sec= 5
    N_knots = int(T_SRB * knots_per_sec)

    # cem config
    cem_rng = jax.random.PRNGKey(int(time.time()))
    cem_config = CrossEntropyMethod_Config(
        rng=cem_rng,
        T=T_SRB,
        iterations=50,
        N_elite=2048,
        # N_knots=N_knots,
        # spline_type="ZOH",
        # N_knots=N_knots,
        # spline_type="Linear",
        N_knots=20,
        spline_type="Bezier",
        initial_action_range_scale=0.1
    )

    # create the CEM optimizer
    cem_optimizer = G1_SRB_CEM(
        model_config=model_config,
        sim_config=sim_config,
        cem_config=cem_config
    )

    # initial state
    q0 = qpos_standing
    v0 = qvel_standing

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
    save_dir = "./results/g1/g1_jump_wrench_cem/"
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
