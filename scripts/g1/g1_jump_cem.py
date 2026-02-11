##
#
# G1 SRB CEM
#
##

# standard imports
import numpy as np
import time
import os

# jax imports
import jax
import jax.numpy as jnp

# mujoco imports
import mujoco
import mujoco.mjx as mjx

# custom imports
from utils.algorithms.cem import *
from utils.simulation.simulation import *
from utils.spline import *
from utils.kinematics import kin
from utils.interpolation import interp


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
experiment = "srb_jump"
time_file = f"./results/{experiment}/times.csv"
state_file = f"./results/{experiment}/states.csv"
input_file = f"./results/{experiment}/inputs.csv"

# load data from csv files
times = np.loadtxt(time_file, delimiter=",")    # shape (M+1,)
states = np.loadtxt(state_file, delimiter=",")  # shape (M+1, 13)
tau_opt = np.loadtxt(input_file, delimiter=",") # shape (M, 6) 


def quat_angle_error(q, q_ref, eps=1e-7):
    # shortest-path: flip sign if needed
    dot = jnp.sum(q * q_ref, axis=-1)
    dot = jnp.abs(dot)
    dot = jnp.clip(dot, -1.0 + eps, 1.0 - eps)
    return 2.0 * jnp.arccos(dot)

class G1_SRB_CEM(CrossEntropyMethod):

    def __init__(self, model_config: Model_Config,
                       sim_config:   ParallelSim_Config,
                       cem_config:   CrossEntropyMethod_Config):

        super().__init__(model_config, sim_config, cem_config)

        # SRB tracking costs
        self.w_p_com = 100.0
        self.w_v_com = 10.0
        self.w_ori = 100.0
        self.w_omega = 10.0

        self.wf_p_com = 20.0 * self.w_p_com
        self.wf_v_com = 20.0 * self.w_v_com
        self.wf_ori = 20.0 * self.w_ori
        self.wf_omega = 20.0 * self.w_omega

        # joint regularization
        self.w_q_joints = 20.0
        self.w_v_joints = 1.0
        self.wf_q_joints = 20.0 * self.w_q_joints
        self.wf_v_joints = 20.0 * self.w_v_joints
        self.w_tau = 0.01

        # grab the nominal joint pos and vel
        self.q_joints_ref = qpos_standing[7:]  # (21,)
        self.v_joints_ref = qvel_standing[6:]  # (21,)

        # extract postion and velocity from states
        self.t_SRB = times          # (M+1,)
        self.q_SRB = states[:, :7]  # p_com, quat   
        self.v_SRB = states[:, 7:]  # v_com, omega  
        self.F_SRB = tau_opt[:, :3] # force applied to COM in world frame
        self.M_SRB = tau_opt[:, 3:] # moment applied to COM in world frame

        # make a reference trajectory for the COM and orientation
        self.make_reference()

        # make internal MJX model and data for querying properties
        self.make_model()

    def make_reference(self):   
        # extract the COM position and velocity from the SRB data
        p_com_SRB = self.q_SRB[:, :3]   # (M+1, 3)
        v_com_SRB = self.v_SRB[:, :3]   # (M+1, 3)

        # extract the orientation and angular velocity from the SRB data
        quat_SRB = self.q_SRB[:, 3:]    # (M+1, 4)
        omega_SRB = self.v_SRB[:, 3:]   # (M+1, 3)

        # interpolate the reference trajectories to match the simulation time steps
        p_com_ref = np.zeros((len(self.t_sim), 3))
        v_com_ref = np.zeros((len(self.t_sim), 3))
        quat_ref = np.zeros((len(self.t_sim), 4))
        omega_ref = np.zeros((len(self.t_sim), 3))
        
        for k in range(len(self.t_sim)):
            # get the time
            t = self.t_sim[k]
            
            # find where t is in the SRB time array
            idx_2 = np.searchsorted(self.t_SRB, t)
            idx_1 = idx_2 - 1
            
            # handle edge cases
            if idx_2 >= len(self.t_SRB):
                # t is beyond the last time - use last values
                idx_1 = idx_2 = len(self.t_SRB) - 1
                alpha = 0.0
            elif idx_1 < 0:
                # t is before the first time - use first values
                idx_1 = idx_2 = 0
                alpha = 0.0
            else:
                # normal interpolation
                t1 = self.t_SRB[idx_1]
                t2 = self.t_SRB[idx_2]
                alpha = (t - t1) / (t2 - t1) if t2 != t1 else 0.0
                alpha = np.clip(alpha, 0.0, 1.0)  # safety clamp

            # linear interpolation for COM position and velocity
            p_com_ref[k] = interp.lerp(p_com_SRB[idx_1], p_com_SRB[idx_2], alpha)
            v_com_ref[k] = interp.lerp(v_com_SRB[idx_1], v_com_SRB[idx_2], alpha)
            omega_ref[k] = interp.lerp(omega_SRB[idx_1], omega_SRB[idx_2], alpha)

            # spherical linear interpolation for orientation
            quat_ref[k] = interp.slerp(quat_SRB[idx_1], quat_SRB[idx_2], alpha)

        # after filling numpy arrays:
        self.p_com_ref = jnp.asarray(p_com_ref)
        self.v_com_ref = jnp.asarray(v_com_ref)
        self.quat_ref  = jnp.asarray(quat_ref)
        self.omega_ref = jnp.asarray(omega_ref)

        # plot to make sure things look right
        # plt.figure()
        # plt.plot(self.t_sim, self.p_com_ref[:, 0], label='p_com_x')
        # plt.plot(self.t_sim, self.p_com_ref[:, 1], label='p_com_y')
        # plt.plot(self.t_sim, self.p_com_ref[:, 2], label='p_com_z')
        # plt.plot(self.t_sim, self.quat_ref[:, 0], label='quat_w')
        # plt.plot(self.t_sim, self.quat_ref[:, 1], label='quat_x')
        # plt.plot(self.t_sim, self.quat_ref[:, 2], label='quat_y')
        # plt.plot(self.t_sim, self.quat_ref[:, 3], label='quat_z')
        # plt.plot(self.t_sim, self.v_com_ref[:, 0], label='v_com_x')
        # plt.plot(self.t_sim, self.v_com_ref[:, 1], label='v_com_y')
        # plt.plot(self.t_sim, self.v_com_ref[:, 2], label='v_com_z')
        # plt.plot(self.t_sim, self.omega_ref[:, 0], label='omega_x')
        # plt.plot(self.t_sim, self.omega_ref[:, 1], label='omega_y')
        # plt.plot(self.t_sim, self.omega_ref[:, 2], label='omega_z')
        # plt.plot
        # plt.xlabel('Time (s)')
        # plt.title('Center of Mass Position Trajectories')
        # plt.legend()
        # plt.grid()
        # plt.show()
        # exit(0)

    # Create MJX model for COM computation
    def make_model(self):
        self.mjx_model = mjx.put_model(mj_model)
        self._data0 = mjx.make_data(self.mjx_model)   # single template


    def com_trajectory(self, q, v):
        nq = int(self.mjx_model.nq)
        nv = int(self.mjx_model.nv)

        # Make sure last dims are nq/nv
        q = self._ensure_last_dim(q, nq)
        v = self._ensure_last_dim(v, nv)

        # Optional: make sure batch dims match
        if q.shape[:-1] != v.shape[:-1]:
            raise ValueError(f"q batch shape {q.shape[:-1]} != v batch shape {v.shape[:-1]}")

        data = self._data0.replace(qpos=q, qvel=v)
        data = mjx.kinematics(self.mjx_model, data)

        p_com = data.subtree_com[..., 0, :]  # (..., 3)

        # velocity: prefer subtree_linvel if available, else FD
        if hasattr(data, "subtree_linvel"):
            v_com = data.subtree_linvel[..., 0, :]  # (..., 3)
        else:
            dt = self.sim.dt
            v_fd = (p_com[..., 1:, :] - p_com[..., :-1, :]) / dt
            v_com = jnp.concatenate([v_fd[..., :1, :], v_fd], axis=-2)

        return p_com, v_com

    def cost(self, q, v, tau):
        """
        q:   (N, T+1, nq)
        v:   (N, T+1, nv)
        tau: (N, T, nu)
        returns:
        J: (N,)
        """
        N, Tp1, nq = q.shape
        T = Tp1 - 1

        # ---- True COM from MJX (batched) ----
        p_com, v_com = self.com_trajectory(q, v)  # (N, T+1, 3), (N, T+1, 3)

        # ---- Orientation/omega choice ----
        # Recommended: track a real body (pelvis/torso) orientation in world.
        # If you haven't wired that yet, this fallback uses the free joint quaternion/omega.
        quat  = q[..., 3:7]   # (N, T+1, 4)  (fallback)
        omega = v[..., 3:6]   # (N, T+1, 3)  (fallback)

        # ---- Joint regularization ----
        q_joints = q[..., 7:]   # (N, T+1, 21)
        v_joints = v[..., 6:]   # (N, T+1, 21)

        # ---- Running errors (0..T-1) ----
        # refs: (T+1, d) so use refs[:-1]
        p_err = p_com[:, :-1, :] - self.p_com_ref[:-1][None, :, :]     # (N, T, 3)
        v_err = v_com[:, :-1, :] - self.v_com_ref[:-1][None, :, :]     # (N, T, 3)
        w_err = omega[:, :-1, :] - self.omega_ref[:-1][None, :, :]     # (N, T, 3)

        # quaternion angle error via dot product
        dot = jnp.sum(quat[:, :-1, :] * self.quat_ref[:-1][None, :, :], axis=-1)  # (N, T)
        dot = jnp.clip(jnp.abs(dot), 1e-7, 1.0 - 1e-7)
        theta = 2.0 * jnp.arccos(dot)  # (N, T)

        qj_err = q_joints[:, :-1, :] - self.q_joints_ref[None, None, :]  # (N, T, 21)
        vj_err = v_joints[:, :-1, :] - self.v_joints_ref[None, None, :]  # (N, T, 21)

        # running costs (sum over time and dimensions)
        J_p   = self.w_p_com   * jnp.sum(p_err**2, axis=(1, 2))     # (N,)
        J_v   = self.w_v_com   * jnp.sum(v_err**2, axis=(1, 2))     # (N,)
        J_ori = self.w_ori     * jnp.sum(theta**2, axis=1)          # (N,)
        J_w   = self.w_omega   * jnp.sum(w_err**2, axis=(1, 2))     # (N,)
        J_qj  = self.w_q_joints* jnp.sum(qj_err**2, axis=(1, 2))    # (N,)
        J_vj  = self.w_v_joints* jnp.sum(vj_err**2, axis=(1, 2))    # (N,)
        J_u   = self.w_tau     * jnp.sum(tau**2, axis=(1, 2))       # (N,)

        J_running = (J_p + J_v + J_ori + J_w + J_qj + J_vj + J_u) * self.sim.dt

        # ---- Terminal errors (time T) ----
        p_err_f = p_com[:, -1, :] - self.p_com_ref[-1][None, :]     # (N, 3)
        v_err_f = v_com[:, -1, :] - self.v_com_ref[-1][None, :]     # (N, 3)
        w_err_f = omega[:, -1, :] - self.omega_ref[-1][None, :]     # (N, 3)

        dot_f = jnp.sum(quat[:, -1, :] * self.quat_ref[-1][None, :], axis=-1)     # (N,)
        dot_f = jnp.clip(jnp.abs(dot_f), 1e-7, 1.0 - 1e-7)
        theta_f = 2.0 * jnp.arccos(dot_f)  # (N,)

        qj_err_f = q_joints[:, -1, :] - self.q_joints_ref[None, :]   # (N, 21)
        vj_err_f = v_joints[:, -1, :] - self.v_joints_ref[None, :]   # (N, 21)

        J_terminal = (
            self.wf_p_com    * jnp.sum(p_err_f**2, axis=1) +
            self.wf_v_com    * jnp.sum(v_err_f**2, axis=1) +
            self.wf_ori      * (theta_f**2) +
            self.wf_omega    * jnp.sum(w_err_f**2, axis=1) +
            self.wf_q_joints * jnp.sum(qj_err_f**2, axis=1) +
            self.wf_v_joints * jnp.sum(vj_err_f**2, axis=1)
        )

        return J_running + J_terminal
    
    def _ensure_last_dim(self, x, last_dim_size):
        """Move an axis of length last_dim_size to the last axis if needed."""
        x = jnp.asarray(x)
        if x.shape[-1] == last_dim_size:
            return x  # already correct

        # try to find an axis with the right size
        for ax, s in enumerate(x.shape):
            if s == last_dim_size:
                return jnp.moveaxis(x, ax, -1)

        raise ValueError(f"Could not find axis of size {last_dim_size} in shape {x.shape}")

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
        Kp=[300, 300, 300, 300, 100, 100, # left leg
            300, 300, 300, 300, 100, 100, # right leg
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
    )

    # cem config
    cem_rng = jax.random.PRNGKey(int(time.time()))
    cem_config = CrossEntropyMethod_Config(
        rng=cem_rng,
        T=times[-1],
        iterations=75,
        N_elite=2048,
        # N_knots=20,
        # spline_type="Linear",
        N_knots=20,
        spline_type="Bezier",
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
    save_dir = "./results/g1_jump/"
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
