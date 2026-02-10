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
        self.w_q_joints = 2.0
        self.w_v_joints = 0.1
        self.wf_q_joints = 20.0 * self.w_q_joints
        self.wf_v_joints = 20.0 * self.w_v_joints
        self.w_tau = 0.001

        # grab the nominal joint pos and vel
        q_joints_ref = qpos_standing[7:]  # (21,)
        v_joints_ref = qvel_standing[6:]  # (21,)

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
        self.p_com_ref = np.zeros((len(self.t_sim), 3))
        self.v_com_ref = np.zeros((len(self.t_sim), 3))
        self.quat_ref = np.zeros((len(self.t_sim), 4))
        self.omega_ref = np.zeros((len(self.t_sim), 3))
        
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
            self.p_com_ref[k] = interp.lerp(p_com_SRB[idx_1], p_com_SRB[idx_2], alpha)
            self.v_com_ref[k] = interp.lerp(v_com_SRB[idx_1], v_com_SRB[idx_2], alpha)
            self.omega_ref[k] = interp.lerp(omega_SRB[idx_1], omega_SRB[idx_2], alpha)

            # spherical linear interpolation for orientation
            self.quat_ref[k] = interp.slerp(quat_SRB[idx_1], quat_SRB[idx_2], alpha)


    def make_model(self):
        
        # create MJX model and data for querying properties
        self.mjx_model = mjx.out_model(mj_model)
        self.mjx_data = mjx.out_data(mj_data)


    def cost(self, q, v, tau):

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
        Kp=[500, 500, 500, 500, 100, 100, # left leg
            500, 500, 500, 500, 100, 100, # right leg
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
        N_knots=20,
        spline_type="Linear",
        # N_knots=20,
        # spline_type="Bezier",
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
