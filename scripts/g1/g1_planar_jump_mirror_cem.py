##
#
# G1 Jump Mirrored CEM
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
from utils.algorithms.schedule import *
from utils.simulation.simulation import *
from utils.spline import *
from utils.interpolation import interp


#################################################################
# LOAD DATA
#################################################################

# load the a model config
xml_path = "./models/g1/g1_planar.xml"
mj_model = mujoco.MjModel.from_xml_path(xml_path)
mj_data = mujoco.MjData(mj_model)

# standing
keyframe = "standing"
key_id = mj_model.key(keyframe).id
qpos_standing = jnp.array(mj_model.key_qpos[key_id])
qvel_standing = jnp.array(mj_model.key_qvel[key_id])

# load SRB data
srb_dir = "./results/srb/srb_jump_2d/"
t_SRB = np.loadtxt(srb_dir + "time.csv", delimiter=",")
q_SRB = np.loadtxt(srb_dir + "q_opt.csv", delimiter=",")
v_SRB = np.loadtxt(srb_dir + "v_opt.csv", delimiter=",")
a_SRB = np.loadtxt(srb_dir + "a_opt.csv", delimiter=",")
tau_SRB = np.loadtxt(srb_dir + "tau_opt.csv", delimiter=",")


#############################################################
# G1 Walk CEM
#############################################################

# useful joint indices
base_idx = jnp.array([0, 1])
ori_idx = jnp.array([2])
hip_idx = jnp.array([3, 6])
knee_idx = jnp.array([4, 7])
ankle_idx = jnp.array([5, 8])
shoulder_idx = jnp.array([9, 11])
elbow_idx = jnp.array([10, 12])
joints_idx = jnp.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

# joint mirroring index (mirror left with right corresponding joints)
state_mirror_dict   = {3:6, 4:7, 5:8, 9:11, 10:12}  # hip, knee, ankle, shoulder, elbow
control_mirror_dict = {0:3, 1:4, 2:5, 6:8,  7:9}     # hip, knee, ankle, shoulder, elbow


class G1_Walk_Mirrored_CEM(CrossEntropyMethod):

    def __init__(self, model_config: Model_Config,
                       sim_config:   ParallelSim_Config,
                       cem_config:   CrossEntropyMethod_Config):
        
        # initialize the parent class
        super().__init__(model_config, sim_config, cem_config)

        self.sim_config = sim_config

        # reinitialize the initial spline sampling
        self._initialize_spline_knots()

        # make the reference trajectory
        self._make_reference_trajectory()

        # Running cost weights (per timestep)
        self.w_px      = 5.0     # horizontal position tracking
        self.w_pz      = 20.0    # vertical position tracking (keep at default height)
        self.w_theta   = 5.0    # pitch angle (stay upright)
        
        self.w_vx      = 1.0     # forward velocity tracking
        self.w_vz      = 1.0     # vertical velocity tracking
        self.w_omega   = 0.1     # pitch velocity tracking

        self.w_p_hip   = 0.5     # hip joint tracking
        self.w_p_knee  = 1.0     # knee joint tracking
        self.w_p_ankle = 1.0     # ankle joint tracking
        self.w_p_shoulder = 0.05  # shoulder joint tracking
        self.w_p_elbow = 0.05     # elbow joint tracking
        
        self.w_v_hip   = 0.01     # hip joint velocity tracking
        self.w_v_knee  = 0.01     # knee joint velocity tracking
        self.w_v_ankle = 0.01     # ankle joint velocity tracking
        self.w_v_shoulder = 0.01  # shoulder joint velocity tracking
        self.w_v_elbow = 0.01     # elbow joint velocity tracking
        self.w_control = 1e-6  # control effort

        terminal_scale = 20.0

        self.wf_px = terminal_scale * self.w_px
        self.wf_pz = terminal_scale * self.w_pz
        self.wf_theta = terminal_scale * self.w_theta

        self.wf_vx = terminal_scale * self.w_vx
        self.wf_vz = terminal_scale * self.w_vz
        self.wf_omega = terminal_scale * self.w_omega
        
        self.wf_p_hip = terminal_scale * self.w_p_hip
        self.wf_p_knee = terminal_scale * self.w_p_knee
        self.wf_p_ankle = terminal_scale * self.w_p_ankle
        self.wf_p_shoulder = terminal_scale * self.w_p_shoulder
        self.wf_p_elbow = terminal_scale * self.w_p_elbow
        
        self.wf_v_hip = terminal_scale * self.w_v_hip
        self.wf_v_knee = terminal_scale * self.w_v_knee
        self.wf_v_ankle = terminal_scale * self.w_v_ankle
        self.wf_v_shoulder = terminal_scale * self.w_v_shoulder
        self.wf_v_elbow = terminal_scale * self.w_v_elbow

    def _make_reference_trajectory(self):
        
        # extract the trajectory components
        p_com = q_SRB[:, 0:2]  # (T_SRB, 2)
        v_com = v_SRB[:, 0:2]  # (T_SRB, 2)
        a_com = a_SRB[:, 0:2]  # (T_SRB, 2)
        theta = q_SRB[:, 2]    # (T_SRB, 1)
        omega = v_SRB[:, 2]    # (T_SRB, 1)
        alpha = a_SRB[:, 2]    # (T_SRB, 1)

        # make simulation time array
        T_SRB = t_SRB[-1]         
        dt_sim = self.sim.dt
        t_sim = jnp.arange(0, T_SRB + dt_sim, dt_sim) # [0, dt_sim, 2*dt_sim, ..., T_SRB]
        nodes_sim = len(t_sim)

        # interpolate the reference trajectory at simulation time steps
        p_com_ref = np.zeros((nodes_sim, 2))
        v_com_ref = np.zeros((nodes_sim, 2))
        a_com_ref = np.zeros((nodes_sim, 2))
        theta_ref = np.zeros((nodes_sim, 1))
        omega_ref = np.zeros((nodes_sim, 1))
        alpha_ref = np.zeros((nodes_sim, 1))
        for i in range(nodes_sim):
            
            # current sim time point
            t = t_sim[i]

            # before the start of the SRB trajectory, hold the initial state
            if t < t_SRB[0]:
                p_com_ref[i] = p_com[0]
                v_com_ref[i] = v_com[0]
                a_com_ref[i] = a_com[0]
                theta_ref[i] = theta[0]
                omega_ref[i] = omega[0]
                alpha_ref[i] = alpha[0]

            # after the end of the SRB trajectory, hold the final state
            elif t > t_SRB[-1]:
                p_com_ref[i] = p_com[-1]
                v_com_ref[i] = v_com[-1]
                a_com_ref[i] = a_com[-1]
                theta_ref[i] = theta[-1]
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
                theta_ref[i]  = interp.lerp(theta[idx_1],  theta[idx_2],  coeff)
                omega_ref[i]  = interp.lerp(omega[idx_1],  omega[idx_2],  coeff)
                alpha_ref[i]  = interp.lerp(alpha[idx_1],  alpha[idx_2],  coeff)
        
        self.px_ref       = jnp.array(p_com_ref[:, 0])  # (nodes_sim,)
        self.pz_ref       = jnp.array(p_com_ref[:, 1])  # (nodes_sim,)
        self.vx_ref       = jnp.array(v_com_ref[:, 0])  # (nodes_sim,)
        self.vz_ref       = jnp.array(v_com_ref[:, 1])  # (nodes_sim,)
        self.theta_ref    = jnp.array(theta_ref[:, 0])  # (nodes_sim,)
        self.thetadot_ref = jnp.array(omega_ref[:, 0])  # (nodes_sim,)
        self.a_com_ref = jnp.array(a_com_ref)           # (nodes_sim, 2)
        self.alpha_ref = jnp.array(alpha_ref[:, 0])     # (nodes_sim,)

        # take joint state references
        self.p_hip_ref = qpos_standing[3]      # scalar
        self.p_knee_ref = qpos_standing[4]   
        self.p_ankle_ref = qpos_standing[5]
        self.p_shoulder_ref = qpos_standing[9]
        self.p_elbow_ref = qpos_standing[10]

        # take the joint velocity references
        self.v_hip_ref = qvel_standing[3]      # scalar
        self.v_knee_ref = qvel_standing[4]
        self.v_ankle_ref = qvel_standing[5]
        self.v_shoulder_ref = qvel_standing[9]
        self.v_elbow_ref = qvel_standing[10]


    def _initialize_spline_knots(self):
        """ 
        Depending on the actuation mode, initialize the spline knot points
        within the position or control limits. NOTE: does not use a
        pre-defined mu and Sigma.
        """

        # size of the mirror
        nj = len(control_mirror_dict)

        # initialize an empty array for the knot points
        y_size = (self.sim.B, self.cem_config.N_knots, nj)
        Y0 = jnp.zeros(y_size)

        # knot points represent desired positions
        if self.sim.use_pd == True:
            # get the position limits at actuated joints
            pos_limits = self.sim.pos_limits  # shape (nu, 2)

            # create initial knot points within the position limits
            for i, key in enumerate(control_mirror_dict):

                # get low and high limits
                pos_lo = pos_limits[key, 0] * self.cem_config.initial_action_range_scale
                pos_hi = pos_limits[key, 1] * self.cem_config.initial_action_range_scale

                # error out if no position limits are defined
                if abs(pos_hi) <1e-6 and abs(pos_lo) <1e-6:
                    raise ValueError(f"No position limits defined at actuated actuator index {key}.")

                # split the key
                self.rng, subkey = jax.random.split(self.rng)

                # replace the ith dimension of all knot points
                y_rand = jax.random.uniform(
                        subkey,
                        shape=(self.sim.B, self.cem_config.N_knots),
                        minval=pos_lo,
                        maxval=pos_hi
                )
                Y0 = Y0.at[:, :, i].set(y_rand)

        # knot points represent direct torques
        else:
            # get the control limits at actuated joints
            ctrl_limits = self.sim.ctrl_limits  # shape (nu, 2)

            # create initial knot points within the control limits
            for _, key in enumerate(control_mirror_dict):
                
                # get low and high limits
                tau_lo = ctrl_limits[key, 0] * self.cem_config.initial_action_range_scale
                tau_hi = ctrl_limits[key, 1] * self.cem_config.initial_action_range_scale

                # error out if no control limits are defined
                if abs(tau_hi) <1e-6 and abs(tau_lo) <1e-6:
                    raise ValueError(f"No control limits defined at actuated actuator index {key}.")

                # split the key
                self.rng, subkey = jax.random.split(self.rng)

                # replace the ith dimension of all knot points
                y_rand = jax.random.uniform(
                        subkey,
                        shape=(self.sim.B, self.cem_config.N_knots),
                        minval=tau_lo,
                        maxval=tau_hi
                )
                Y0 = Y0.at[:, :, i].set(y_rand)

        # create the spline object
        if self.cem_config.spline_type == "ZOH":
            self.spline = zoh.ZOH_Spline(Y0, self.T_eff)
        elif self.cem_config.spline_type == "Linear":
            self.spline = linear.Linear_Spline(Y0, self.T_eff)
        elif self.cem_config.spline_type == "Cubic":
            self.spline = cubic.Cubic_Spline(Y0, self.T_eff)
        elif self.cem_config.spline_type == "Bezier":
            self.spline = bezier.Bezier_Spline(Y0, self.T_eff)
        elif self.cem_config.spline_type == "Fourier":
            self.spline = fourier.Fourier_Spline(Y0, self.T_eff, periodic=False)
        else:
            raise NotImplementedError(f"Spline type [{self.cem_config.spline_type}] not implemented.")

        # update the spline knot points
        self.spline.update_knots(Y0)

        # update the distribution with the initial knot points
        self._update_distribution(Y0)


    def cost(self, q, v, tau):
            """
            Cost function to evaluate the rollouts.

            Args:
                q: jnp.array,   shape (B, N+1, nq) - generalized positions trajectory.
                v: jnp.array,   shape (B, N+1, nv) - generalized velocities trajectory.
                tau: jnp.array, shape (B, N, nu) - control inputs trajectory.
            Returns:
                costs: jnp.array, shape (B,) - cost for each rollout.
            """

            N = tau.shape[1]

            # base state: full trajectory -> (B, N+1, *)
            px       = q[:, :N+1, 0]   # (B, N+1)
            pz       = q[:, :N+1, 1]   # (B, N+1)
            theta    = q[:, :N+1, 2]   # (B, N+1)
            vx       = v[:, :N+1, 0]   # (B, N+1)
            vz       = v[:, :N+1, 1]   # (B, N+1)
            thetadot = v[:, :N+1, 2]   # (B, N+1)

            p_hip      = q[:, :N+1, 3]   # (B, N+1)
            p_knee     = q[:, :N+1, 4]   # (B, N+1)
            p_ankle    = q[:, :N+1, 5]   # (B, N+1)
            p_shoulder = q[:, :N+1, 9]   # (B, N+1)
            p_elbow    = q[:, :N+1, 10]  # (B, N+1)

            v_hip      = v[:, :N+1, 3]   # (B, N+1)
            v_knee     = v[:, :N+1, 4]   # (B, N+1)
            v_ankle    = v[:, :N+1, 5]   # (B, N+1)
            v_shoulder = v[:, :N+1, 9]   # (B, N+1)
            v_elbow    = v[:, :N+1, 10]  # (B, N+1)

            # errors over full trajectory -> (B, N+1)
            e_px       = px       - self.px_ref[:N+1]        # (B, N+1)
            e_pz       = pz       - self.pz_ref[:N+1]        # (B, N+1)
            e_theta    = theta    - self.theta_ref[:N+1]     # (B, N+1)
            e_vx       = vx       - self.vx_ref[:N+1]        # (B, N+1)
            e_vz       = vz       - self.vz_ref[:N+1]        # (B, N+1)
            e_thetadot = thetadot - self.thetadot_ref[:N+1]  # (B, N+1)

            e_p_hip      = p_hip      - self.p_hip_ref       # (B, N+1)
            e_p_knee     = p_knee     - self.p_knee_ref      # (B, N+1)
            e_p_ankle    = p_ankle    - self.p_ankle_ref     # (B, N+1)
            e_p_shoulder = p_shoulder - self.p_shoulder_ref  # (B, N+1)
            e_p_elbow    = p_elbow    - self.p_elbow_ref     # (B, N+1)

            e_v_hip      = v_hip      - self.v_hip_ref       # (B, N+1)
            e_v_knee     = v_knee     - self.v_knee_ref      # (B, N+1)
            e_v_ankle    = v_ankle    - self.v_ankle_ref     # (B, N+1)
            e_v_shoulder = v_shoulder - self.v_shoulder_ref  # (B, N+1)
            e_v_elbow    = v_elbow    - self.v_elbow_ref     # (B, N+1)

            # squared error helper: sum over time, exclude terminal step
            def se(e): return jnp.sum(e[:, :-1]**2, axis=-1)   # (B,)
            def se_f(e): return e[:, -1]**2                     # (B,)

            # ========================== Running cost ==========================

            running_cost = (
                self.w_px        * se(e_px)       +
                self.w_pz        * se(e_pz)       +
                self.w_theta     * se(e_theta)    +
                self.w_vx        * se(e_vx)       +
                self.w_vz        * se(e_vz)       +
                self.w_omega     * se(e_thetadot) +
                self.w_p_hip     * se(e_p_hip)    +
                self.w_p_knee    * se(e_p_knee)   +
                self.w_p_ankle   * se(e_p_ankle)  +
                self.w_p_shoulder* se(e_p_shoulder)+
                self.w_p_elbow   * se(e_p_elbow)  +
                self.w_v_hip     * se(e_v_hip)    +
                self.w_v_knee    * se(e_v_knee)   +
                self.w_v_ankle   * se(e_v_ankle)  +
                self.w_v_shoulder* se(e_v_shoulder)+
                self.w_v_elbow   * se(e_v_elbow)  +
                self.w_control   * jnp.sum(jnp.sum(tau**2, axis=-1), axis=-1)
            ) * self.sim.dt  # (B,)

            # ========================== Terminal cost ==========================

            terminal_cost = (
                self.wf_px        * se_f(e_px)       +
                self.wf_pz        * se_f(e_pz)       +
                self.wf_theta     * se_f(e_theta)    +
                self.wf_vx        * se_f(e_vx)       +
                self.wf_vz        * se_f(e_vz)       +
                self.wf_omega     * se_f(e_thetadot) +
                self.wf_p_hip     * se_f(e_p_hip)    +
                self.wf_p_knee    * se_f(e_p_knee)   +
                self.wf_p_ankle   * se_f(e_p_ankle)  +
                self.wf_p_shoulder* se_f(e_p_shoulder)+
                self.wf_p_elbow   * se_f(e_p_elbow)  +
                self.wf_v_hip     * se_f(e_v_hip)    +
                self.wf_v_knee    * se_f(e_v_knee)   +
                self.wf_v_ankle   * se_f(e_v_ankle)  +
                self.wf_v_shoulder* se_f(e_v_shoulder)+
                self.wf_v_elbow   * se_f(e_v_elbow)
            )  # (B,)

            return running_cost + terminal_cost  # (B,)

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

            # mirror the spline values according to the mirror dict
            full_size = (self.sim.B, y_val.shape[1], self.sim.nu)
            y_val_full = jnp.zeros(full_size)
            for left_idx, right_idx in control_mirror_dict.items():
                y_joint = y_val[:, :, left_idx]  # shape (B, N)
                y_val_full = y_val_full.at[:, :, left_idx].set(y_joint)
                y_val_full = y_val_full.at[:, :, right_idx].set(y_joint)

            # do forward rollout
            N = y_val_full.shape[1]
            q_srb_ref = jnp.stack([self.px_ref[:N], self.pz_ref[:N], self.theta_ref[:N]],    axis=-1)  # (N, 3)
            v_srb_ref = jnp.stack([self.vx_ref[:N], self.vz_ref[:N], self.thetadot_ref[:N]], axis=-1)  # (N, 3)
            a_srb_ref = jnp.concatenate([self.a_com_ref[:N], self.alpha_ref[:N, None]],      axis=-1)  # (N, 3)
            # a = linear_schedule(itr, self.cem_config.iterations, alpha_max=1.0)
            a = exponential_schedule(itr, self.cem_config.iterations, alpha_max=1.0, lam=20.0)
            q_log, v_log, tau_log = self.sim.rollout(q0, v0, y_val_full,
                                                    q_srb_ref=q_srb_ref,
                                                    v_srb_ref=v_srb_ref,
                                                    a_srb_ref=a_srb_ref,
                                                    w_scale=a)
            q_log.block_until_ready()

            # compute costs
            J = self.cost(q_log, v_log, tau_log)  # shape (B,)
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

            # largest singular value
            cov_norm = jnp.linalg.norm(self.Sigma, ord=2)

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
                  f"‖Σ‖₂: {cov_norm:.4f} | "
                  f"α: {a:.3f}")
            
        return q_opt, v_opt, tau_opt

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

    # model config
    model_config = Model_Config(
        xml_path="./models/g1/g1_planar.xml",
        Kp=[400, 450, 500, 
            400, 450, 500, 
            150, 50, 
            150, 50],
        Kd=[2, 4, 2, 
            2, 4, 2, 
            2, 2, 
            2, 2],
        q_actuated_idx=list(range(3,13)),
        v_actuated_idx=list(range(3,13)),
        action_mode="pos"
    )

    # parallel sim config
    sim_config = ParallelSim_Config(
        batch_size = 4096,
        use_external_wrench=True,
        kp_lin=10,
        kd_lin=2,
        kp_ang=10,
        kd_ang=2,
    )

    # cem config
    cem_rng = jax.random.PRNGKey(42)
    cem_config = CrossEntropyMethod_Config(
        rng=cem_rng,
        T=t_SRB[-1],  
        iterations=30,
        N_elite=512,
        # N_knots = 20,
        # spline_type="ZOH",
        # N_knots=20,
        # spline_type="Linear",
        N_knots=6,
        spline_type="Bezier",
        # N_knots=10,
        # spline_type="Cubic",
    )

    # create the CEM optimizer
    cem_optimizer = G1_Walk_Mirrored_CEM(
        model_config=model_config,
        sim_config=sim_config,
        cem_config=cem_config
    )

    # optimize from an initial state
    t0 = time.time()
    q_opt, v_opt, tau_opt = cem_optimizer.optimize(
        q0=qpos_standing,
        v0=qvel_standing
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
    save_dir = "./results/g1/g1_planar_jump_mirror_cem/"
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
