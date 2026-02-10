##
#
# G1 Stand CEM
#
##

# standard imports
import numpy as np
import time
import os

# jax imports
import jax
import jax.numpy as jnp

import mujoco

# custom imports
from utils.algorithms.cem import *
from utils.simulation.simulation import *
from utils.spline.bezier import *
from utils.spline.zoh import *

# load mujoco model
xml_path = "./models/g1/g1_21dof.xml"
mj_model = mujoco.MjModel.from_xml_path(xml_path)
mj_data = mujoco.MjData(mj_model)

# laying_down
keyframe = "laying_down"
key_id = mj_model.key(keyframe).id
qpos_laying_down = jnp.array(mj_model.key_qpos[key_id])
qvel_laying_down = jnp.array(mj_model.key_qvel[key_id])

# standing
keyframe = "standing"
key_id = mj_model.key(keyframe).id
qpos_standing = jnp.array(mj_model.key_qpos[key_id])
qvel_standing = jnp.array(mj_model.key_qvel[key_id])

def quat_angle_error(q, q_ref, eps=1e-4):
    """
    q: (..., 4)
    q_ref: (4,) or (..., 4)
    returns: angle error in radians, shape (...) using shortest path
    """
    dot = jnp.sum(q * q_ref, axis=-1)
    dot = jnp.clip(jnp.abs(dot), -1.0 + eps, 1.0 - eps)
    theta = 2.0 * jnp.arccos(dot)
    return theta

def qslice(qx, idxs):
    return qx[..., jnp.array(idxs, dtype=jnp.int32)]

def vslice(vx, idxs):
    return vx[..., jnp.array(idxs, dtype=jnp.int32)]


class G1_Stand_CEM(CrossEntropyMethod):

    def __init__(self, model_config: Model_Config,
                       sim_config:   ParallelSim_Config,
                       cem_config:   CrossEntropyMethod_Config):

        super().__init__(model_config, sim_config, cem_config)

        # weights for running costs
        self.w_px = 0.1
        self.w_py = 0.1
        self.w_pz = 50.0
        self.w_ori = 15.0

        self.w_pos_hip   = 3.0
        self.w_pos_knee  = 0.001
        self.w_pos_ankle = 0.001
        self.w_pos_waist = 0.1
        self.w_pos_arms  = 0.001

        self.w_vx = 0.001
        self.w_vy = 0.001
        self.w_vz = 0.01
        self.w_omega = 0.001

        self.w_vel_hip   = 0.001
        self.w_vel_knee  = 0.001
        self.w_vel_ankle = 0.001
        self.w_vel_waist = 0.001
        self.w_vel_arms  = 0.001

        self.w_tau = 0.001

        # terminal cost multiplier (applied to all terminal costs)
        self.terminal_weight = 50.0

        # -------------------------
        # G1 21 DOF joint indices
        # q: [x,y,z, qw,qx,qy,qz, joints(21)]  -> nq=28
        # v: [vx,vy,vz, wx,wy,wz, joint_vels]  -> nv=27
        # -------------------------
        self.POS_X = 0
        self.POS_Y = 1
        self.POS_Z = 2
        self.QUAT  = [3, 4, 5, 6]

        self.POS_HIP   = [7, 8, 9] + [13, 14, 15]
        self.POS_KNEE  = [10] + [16]
        self.POS_ANKLE = [11, 12] + [17, 18]
        self.POS_WAIST = [19]
        self.POS_ARM   = [20, 21, 22, 23, 24, 25, 26, 27]

        self.VEL_X = 0
        self.VEL_Y = 1
        self.VEL_Z = 2
        self.OMEGA = [3, 4, 5]

        self.VEL_HIP   = [6, 7, 8] + [12, 13, 14]
        self.VEL_KNEE  = [9] + [15]
        self.VEL_ANKLE = [10, 11] + [16, 17]
        self.VEL_WAIST = [18]
        self.VEL_ARM   = [19, 20, 21, 22, 23, 24, 25, 26]

    def cost(self, q, v, tau):
        """
        q:   (B, N+1, nq)  nq=28
        v:   (B, N+1, nv)  nv=27
        tau: (B, N,   nu)  assumed aligned with steps 0..N-1
        returns:
          J: (B,)
        """

        # References (JAX arrays)
        q_ref = qpos_standing   # (nq,)
        v_ref = qvel_standing   # (nv,)

        # time slices
        q_run = q[:, :-1, :]    # (B, N, nq)
        v_run = v[:, :-1, :]    # (B, N, nv)
        q_T   = q[:, -1, :]     # (B, nq)
        v_T   = v[:, -1, :]     # (B, nv)

        # dt (adjust if your dt is stored elsewhere)
        dt = self.sim.dt

        # -------------------------
        # Base pos + quat (running)
        # -------------------------
        px_run   = q_run[..., self.POS_X]         # (B, N)
        py_run   = q_run[..., self.POS_Y]
        pz_run   = q_run[..., self.POS_Z]
        quat_run = qslice(q_run, self.QUAT)       # (B, N, 4)

        px_ref   = q_ref[self.POS_X]
        py_ref   = q_ref[self.POS_Y]
        pz_ref   = q_ref[self.POS_Z]
        quat_ref = q_ref[jnp.array(self.QUAT, dtype=jnp.int32)]  # (4,)

        c_px  = self.w_px * (px_run - px_ref) ** 2
        c_py  = self.w_py * (py_run - py_ref) ** 2
        c_pz  = self.w_pz * (pz_run - pz_ref) ** 2

        theta = quat_angle_error(quat_run, quat_ref)  # (B, N)
        c_ori = self.w_ori * (theta ** 2)

        # -------------------------
        # Base pos + quat (terminal)
        # -------------------------
        px_T   = q_T[..., self.POS_X]             # (B,)
        py_T   = q_T[..., self.POS_Y]
        pz_T   = q_T[..., self.POS_Z]
        quat_T = qslice(q_T, self.QUAT)           # (B, 4)

        cT_px = self.w_px * (px_T - px_ref) ** 2
        cT_py = self.w_py * (py_T - py_ref) ** 2
        cT_pz = self.w_pz * (pz_T - pz_ref) ** 2

        theta_T = quat_angle_error(quat_T, quat_ref)  # (B,)
        cT_ori  = self.w_ori * (theta_T ** 2)

        # -------------------------
        # Joint positions (running)
        # -------------------------
        hip_run   = qslice(q_run, self.POS_HIP)    # (B, N, nhip)
        knee_run  = qslice(q_run, self.POS_KNEE)   # (B, N, nknee)
        ankle_run = qslice(q_run, self.POS_ANKLE)  # (B, N, nankle)
        waist_run = qslice(q_run, self.POS_WAIST)  # (B, N, 1)
        arm_run   = qslice(q_run, self.POS_ARM)    # (B, N, narm)

        hip_ref   = q_ref[jnp.array(self.POS_HIP, dtype=jnp.int32)]
        knee_ref  = q_ref[jnp.array(self.POS_KNEE, dtype=jnp.int32)]
        ankle_ref = q_ref[jnp.array(self.POS_ANKLE, dtype=jnp.int32)]
        waist_ref = q_ref[jnp.array(self.POS_WAIST, dtype=jnp.int32)]
        arm_ref   = q_ref[jnp.array(self.POS_ARM, dtype=jnp.int32)]

        c_qhip   = self.w_pos_hip   * jnp.sum((hip_run   - hip_ref)   ** 2, axis=-1)  # (B, N)
        c_qknee  = self.w_pos_knee  * jnp.sum((knee_run  - knee_ref)  ** 2, axis=-1)
        c_qankle = self.w_pos_ankle * jnp.sum((ankle_run - ankle_ref) ** 2, axis=-1)
        c_qwaist = self.w_pos_waist * jnp.sum((waist_run - waist_ref) ** 2, axis=-1)
        c_qarm   = self.w_pos_arms  * jnp.sum((arm_run   - arm_ref)   ** 2, axis=-1)

        # -------------------------
        # Joint positions (terminal)
        # -------------------------
        hip_T   = qslice(q_T, self.POS_HIP)    # (B, nhip)
        knee_T  = qslice(q_T, self.POS_KNEE)
        ankle_T = qslice(q_T, self.POS_ANKLE)
        waist_T = qslice(q_T, self.POS_WAIST)  # (B, 1)
        arm_T   = qslice(q_T, self.POS_ARM)

        cT_qhip   = self.w_pos_hip   * jnp.sum((hip_T   - hip_ref)   ** 2, axis=-1)  # (B,)
        cT_qknee  = self.w_pos_knee  * jnp.sum((knee_T  - knee_ref)  ** 2, axis=-1)
        cT_qankle = self.w_pos_ankle * jnp.sum((ankle_T - ankle_ref) ** 2, axis=-1)
        cT_qwaist = self.w_pos_waist * jnp.sum((waist_T - waist_ref) ** 2, axis=-1)
        cT_qarm   = self.w_pos_arms  * jnp.sum((arm_T   - arm_ref)   ** 2, axis=-1)

        # -------------------------
        # Base velocities (running)
        # -------------------------
        vx_run    = v_run[..., self.VEL_X]        # (B, N)
        vy_run    = v_run[..., self.VEL_Y]
        vz_run    = v_run[..., self.VEL_Z]
        omega_run = vslice(v_run, self.OMEGA)     # (B, N, 3)

        vx_ref    = v_ref[self.VEL_X]
        vy_ref    = v_ref[self.VEL_Y]
        vz_ref    = v_ref[self.VEL_Z]
        omega_ref = v_ref[jnp.array(self.OMEGA, dtype=jnp.int32)]   # (3,)

        c_vx = self.w_vx * (vx_run - vx_ref) ** 2
        c_vy = self.w_vy * (vy_run - vy_ref) ** 2
        c_vz = self.w_vz * (vz_run - vz_ref) ** 2
        c_om = self.w_omega * jnp.sum((omega_run - omega_ref) ** 2, axis=-1)  # (B, N)

        # -------------------------
        # Joint velocities (running)
        # -------------------------
        vhip_run   = vslice(v_run, self.VEL_HIP)    # (B, N, nhip)
        vknee_run  = vslice(v_run, self.VEL_KNEE)
        vankle_run = vslice(v_run, self.VEL_ANKLE)
        vwaist_run = vslice(v_run, self.VEL_WAIST)  # (B, N, 1)
        varm_run   = vslice(v_run, self.VEL_ARM)

        vhip_ref   = v_ref[jnp.array(self.VEL_HIP, dtype=jnp.int32)]
        vknee_ref  = v_ref[jnp.array(self.VEL_KNEE, dtype=jnp.int32)]
        vankle_ref = v_ref[jnp.array(self.VEL_ANKLE, dtype=jnp.int32)]
        vwaist_ref = v_ref[jnp.array(self.VEL_WAIST, dtype=jnp.int32)]
        varm_ref   = v_ref[jnp.array(self.VEL_ARM, dtype=jnp.int32)]

        c_vhip   = self.w_vel_hip   * jnp.sum((vhip_run   - vhip_ref)   ** 2, axis=-1)  # (B, N)
        c_vknee  = self.w_vel_knee  * jnp.sum((vknee_run  - vknee_ref)  ** 2, axis=-1)
        c_vankle = self.w_vel_ankle * jnp.sum((vankle_run - vankle_ref) ** 2, axis=-1)
        c_vwaist = self.w_vel_waist * jnp.sum((vwaist_run - vwaist_ref) ** 2, axis=-1)
        c_varm   = self.w_vel_arms  * jnp.sum((varm_run   - varm_ref)   ** 2, axis=-1)

        # -------------------------
        # Base velocities (terminal)
        # -------------------------
        vx_T    = v_T[..., self.VEL_X]        # (B,)
        vy_T    = v_T[..., self.VEL_Y]
        vz_T    = v_T[..., self.VEL_Z]
        omega_T = vslice(v_T, self.OMEGA)     # (B, 3)

        cT_vx = self.w_vx * (vx_T - vx_ref) ** 2
        cT_vy = self.w_vy * (vy_T - vy_ref) ** 2
        cT_vz = self.w_vz * (vz_T - vz_ref) ** 2
        cT_om = self.w_omega * jnp.sum((omega_T - omega_ref) ** 2, axis=-1)  # (B,)

        # -------------------------
        # Joint velocities (terminal)
        # -------------------------
        vhip_T   = vslice(v_T, self.VEL_HIP)
        vknee_T  = vslice(v_T, self.VEL_KNEE)
        vankle_T = vslice(v_T, self.VEL_ANKLE)
        vwaist_T = vslice(v_T, self.VEL_WAIST)  # (B, 1)
        varm_T   = vslice(v_T, self.VEL_ARM)

        cT_vhip   = self.w_vel_hip   * jnp.sum((vhip_T   - vhip_ref)   ** 2, axis=-1)  # (B,)
        cT_vknee  = self.w_vel_knee  * jnp.sum((vknee_T  - vknee_ref)  ** 2, axis=-1)
        cT_vankle = self.w_vel_ankle * jnp.sum((vankle_T - vankle_ref) ** 2, axis=-1)
        cT_vwaist = self.w_vel_waist * jnp.sum((vwaist_T - vwaist_ref) ** 2, axis=-1)
        cT_varm   = self.w_vel_arms  * jnp.sum((varm_T   - varm_ref)   ** 2, axis=-1)

        # -------------------------
        # Control effort (running only)
        # -------------------------
        c_u = self.w_tau * jnp.sum(tau ** 2, axis=-1)  # (B, N)

        # -------------------------
        # Total running + terminal
        # -------------------------
        running_t = (
            c_px + c_py + c_pz + c_ori +
            c_qhip + c_qknee + c_qankle + c_qwaist + c_qarm +
            c_vx + c_vy + c_vz + c_om +
            c_vhip + c_vknee + c_vankle + c_vwaist + c_varm +
            c_u
        )  # (B, N)

        terminal = self.terminal_weight * (
            cT_px + cT_py + cT_pz + cT_ori +
            cT_qhip + cT_qknee + cT_qankle + cT_qwaist + cT_qarm +
            cT_vx + cT_vy + cT_vz + cT_om +
            cT_vhip + cT_vknee + cT_vankle + cT_vwaist + cT_varm
        )  # (B,)

        J = jnp.sum(running_t, axis=-1) * dt + terminal  # (B,)
        return J

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
        T=4.0,
        iterations=75,
        N_elite=2048,
        N_knots=20,
        spline_type="Bezier",
        # N_knots=20,
        # spline_type="ZOH",
    )

    # create the CEM optimizer
    cem_optimizer = G1_Stand_CEM(
        model_config=model_config,
        sim_config=sim_config,
        cem_config=cem_config
    )

    # initial state
    q0 = jnp.array(qpos_laying_down)
    v0 = jnp.array(qvel_laying_down)

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
    save_dir = "./results/g1_stand/"
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
