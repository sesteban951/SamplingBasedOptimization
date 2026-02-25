##
#
# Plot data from SRB optimization results
#
##

# standard imports
import numpy as np
import matplotlib.pyplot as plt
import time
import os

# mujoco imports
import mujoco 
import mujoco.viewer

# custom imports
from utils.kinematics import kin


#################################################################
# LOAD DATA
#################################################################

# which data to load
# experiment = "srb/srb_free_wrench"
# experiment = "srb/srb_jump"
# experiment = "srb/srb_jump_2d"
experiment = "srb/srb_twist"

# which data to load
time_file = f"./results/{experiment}/time.csv"
q_file = f"./results/{experiment}/q_opt.csv"
v_file = f"./results/{experiment}/v_opt.csv"
a_file = f"./results/{experiment}/a_opt.csv"
tau_file = f"./results/{experiment}/tau_opt.csv"
feet_file = f"./results/{experiment}/feet.csv"

# load data from csv files
times = np.loadtxt(time_file, delimiter=",")
q_opt = np.loadtxt(q_file, delimiter=",")
v_opt = np.loadtxt(v_file, delimiter=",")
a_opt = np.loadtxt(a_file, delimiter=",")
tau_opt = np.loadtxt(tau_file, delimiter=",")

# optional feet trajectory: columns [pL_x, pL_y, pR_x, pR_y], shape (N, 4)
feet_opt = None
if os.path.exists(feet_file):
    feet_opt = np.loadtxt(feet_file, delimiter=",")
    if feet_opt.ndim == 1:
        feet_opt = feet_opt.reshape(1, -1)
    if feet_opt.shape[1] != 4:
        print(f"Warning: expected feet.csv with 4 columns, got {feet_opt.shape}. Ignoring feet visualization.")
        feet_opt = None

print("Loaded data:")
print(f"  times: {times.shape}")
print(f"  q_opt: {q_opt.shape}")
print(f"  v_opt: {v_opt.shape}")
print(f"  a_opt: {a_opt.shape}")
print(f"  tau_opt: {tau_opt.shape}")
if feet_opt is not None:
    print(f"  feet_opt: {feet_opt.shape}")
else:
    print("  feet_opt: not found (continuing without feet overlays)")

#################################################################
# VISUALIZATION
#################################################################

def srb_2D_to_3D(q_2d_traj):
    """
    Convert 2D SRB state trajectory to 3D SRB state for visualization.
    """
    # length of the trajectory
    N = q_2d_traj.shape[0]

    # extract states
    p_com = q_2d_traj[:, 0:2]  # (N, 2): [px, pz]
    theta = q_2d_traj[:, 2]    # (N,):

    # conver to 3D trajecotry
    q_3d_traj = np.zeros((N, 7))  # (N, 7): [px, py, pz, qw, qx, qy, qz]

    # 3D position (py = 0)
    q_3d_traj[:, 0] = p_com[:, 0]  # px
    q_3d_traj[:, 1] = 0.0          # py
    q_3d_traj[:, 2] = p_com[:, 1]  # pz

    # orientation (quaternion)
    angle_half = -theta / 2.0 # (negative, bc planar SRB y+ is out of page, Mujoco y+ into page)
    q_3d_traj[:, 3] = np.cos(angle_half)  # qw
    q_3d_traj[:, 4] = 0.0                 # qx
    q_3d_traj[:, 5] = np.sin(angle_half)  # qy
    q_3d_traj[:, 6] = 0.0                 # qz

    return q_3d_traj

# load the mujoco model
xml_path = "./models/srb/srb.xml"
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# visualize the optimal trajectory
viewer = mujoco.viewer.launch_passive(model, data)

def _add_box_geom(scene, pos, halfsizes, rgba):
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_BOX,
        np.array(halfsizes, dtype=np.float64),
        np.array(pos, dtype=np.float64),
        np.eye(3).reshape(9),
        np.array(rgba, dtype=np.float64),
    )
    geom.category = int(mujoco.mjtCatBit.mjCAT_DECOR)
    scene.ngeom += 1

def _add_connector_geom(scene, p_from, p_to, radius, rgba):
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    # Draw straight decorative connector between points.
    mujoco.mjv_connector(
        geom,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        radius,
        np.array(p_from, dtype=np.float64),
        np.array(p_to, dtype=np.float64),
    )
    geom.rgba[:] = np.array(rgba, dtype=np.float32)
    geom.category = int(mujoco.mjtCatBit.mjCAT_DECOR)
    scene.ngeom += 1

# decide if 2D or 3D trajectory
if "2d" in experiment.lower():
    q_replay = srb_2D_to_3D(q_opt)
else:
    q_replay = q_opt

# run the visualization
try:
    t0 = time.time()
    while True:

        if viewer.is_running() == False:
            break

        i = np.searchsorted(times, time.time() - t0)
        i = min(i, len(times) - 1)  # Clamp to valid range

        print(f"Time: {time.time() - t0:.2f}, Index: {i}\r", end="")

        data.qpos[:] = q_replay[i, :]
        mujoco.mj_step(model, data)

        # optional feet/legs overlays
        if viewer.user_scn is not None:
            viewer.user_scn.ngeom = 0

            if feet_opt is not None and i < feet_opt.shape[0]:
                feet_k = feet_opt[i, :]
                if np.all(np.isfinite(feet_k)):
                    p_com = q_replay[i, 0:3]
                    p_L = np.array([feet_k[0], feet_k[1], 0.0], dtype=np.float64)
                    p_R = np.array([feet_k[2], feet_k[3], 0.0], dtype=np.float64)

                    # feet as small boxes
                    _add_box_geom(
                        viewer.user_scn,
                        p_L,
                        halfsizes=[0.05, 0.03, 0.015],
                        rgba=[0.95, 0.45, 0.10, 0.9],
                    )
                    _add_box_geom(
                        viewer.user_scn,
                        p_R,
                        halfsizes=[0.05, 0.03, 0.015],
                        rgba=[0.10, 0.55, 0.95, 0.9],
                    )

                    # simple unactuated visual connectors (legs)
                    _add_connector_geom(
                        viewer.user_scn,
                        p_com,
                        p_L,
                        radius=0.01,
                        rgba=[0.85, 0.85, 0.85, 0.8],
                    )
                    _add_connector_geom(
                        viewer.user_scn,
                        p_com,
                        p_R,
                        radius=0.01,
                        rgba=[0.85, 0.85, 0.85, 0.8],
                    )

        viewer.sync()

        if time.time() - t0 > times[-1]:
            time.sleep(1.0)
            t0 = time.time()

except KeyboardInterrupt:
    print("\nClosed visualization.")

viewer.close()


#################################################################
# PLOTS
#################################################################

# 2D SRB Trajectory
if "2d" in experiment.lower():

    # ----------------------- POS / VEL -----------------------
    plt.figure(num="State")

    plt.subplot(2, 3, 1)
    plt.plot(times, q_opt[:, 0])
    plt.ylabel("Pos x (m)")
    plt.xlabel("Time (s)")
    plt.grid()

    plt.subplot(2, 3, 2)
    plt.plot(times, q_opt[:, 1])
    plt.ylabel("Pos z (m)")
    plt.xlabel("Time (s)")
    plt.grid()

    plt.subplot(2, 3, 3)
    plt.plot(times, q_opt[:, 2])
    plt.ylabel("Theta (rad)")
    plt.xlabel("Time (s)")
    plt.grid()

    plt.subplot(2, 3, 4)
    plt.plot(times, v_opt[:, 0])
    plt.ylabel("Vel x (m/s)")
    plt.xlabel("Time (s)")
    plt.grid()

    plt.subplot(2, 3, 5)
    plt.plot(times, v_opt[:, 1])
    plt.ylabel("Vel z (m/s)")
    plt.xlabel("Time (s)")
    plt.grid()

    plt.subplot(2, 3, 6)
    plt.plot(times, v_opt[:, 2])
    plt.ylabel("Omega (rad/s)")
    plt.xlabel("Time (s)")
    plt.grid()

    # ----------------------- TAU -----------------------
    plt.figure(num="Controls")

    plt.subplot(1, 3, 1)
    plt.plot(times[:-1], tau_opt[:, 0])
    plt.ylabel("Force x (N)")
    plt.xlabel("Time (s)")
    plt.grid()

    plt.subplot(1, 3, 2)
    plt.plot(times[:-1], tau_opt[:, 1])
    plt.ylabel("Force z (N)")
    plt.xlabel("Time (s)")
    plt.grid()  

    plt.subplot(1, 3, 3)
    plt.plot(times[:-1], tau_opt[:, 2])
    plt.ylabel("Torque y (Nm)")
    plt.xlabel("Time (s)")
    plt.grid()

    plt.show()


# 3D SRB Trajectory
else:

    # convert quaternions to euler angles for plotting
    euler_opt = np.zeros((q_opt.shape[0], 3))
    for i in range(q_opt.shape[0]):
        # convert quaternion to euler angles
        quat = q_opt[i, 3:7]
        euler = kin.quat_to_euler_ZYX(quat)
        euler_opt[i, :] = euler

    # ----------------------- POS -----------------------
    plt.figure(num="Positions (q)")

    plt.subplot(3, 4, 1)
    plt.plot(times, q_opt[:, 0])
    plt.ylabel("Pos x (m)")
    plt.xlabel("Time (s)")
    plt.grid()

    plt.subplot(3, 4, 2)
    plt.plot(times, q_opt[:, 1])
    plt.ylabel("Pos y (m)")
    plt.xlabel("Time (s)")
    plt.grid()

    plt.subplot(3, 4, 3)
    plt.plot(times, q_opt[:, 2])
    plt.ylabel("Pos z (m)")
    plt.xlabel("Time (s)")
    plt.grid()

    plt.subplot(3, 4, 5)
    plt.plot(times, q_opt[:, 4])
    plt.ylabel("Quat x")
    plt.xlabel("Time (s)")
    plt.grid()

    plt.subplot(3, 4, 6)
    plt.plot(times, q_opt[:, 5])
    plt.ylabel("Quat y")
    plt.xlabel("Time (s)")
    plt.grid()

    plt.subplot(3, 4, 7)
    plt.plot(times, q_opt[:, 6])
    plt.ylabel("Quat z")
    plt.xlabel("Time (s)")
    plt.grid()

    plt.subplot(3, 4, 8)
    plt.plot(times, q_opt[:, 3])
    plt.ylabel("Quat w")
    plt.xlabel("Time (s)")
    plt.grid()

    plt.subplot(3, 4, 9)
    plt.plot(times, euler_opt[:, 0])
    plt.ylabel("Roll (rad)")
    plt.xlabel("Time (s)")
    plt.grid()

    plt.subplot(3, 4, 10)
    plt.plot(times, euler_opt[:, 1])
    plt.ylabel("Pitch (rad)")
    plt.xlabel("Time (s)")
    plt.grid()

    plt.subplot(3, 4, 11)
    plt.plot(times, euler_opt[:, 2])
    plt.ylabel("Yaw (rad)")
    plt.xlabel("Time (s)")
    plt.grid()

    # ----------------------- VEL -----------------------
    plt.figure(num="Velocities (v)")

    plt.subplot(2, 3, 1)
    plt.plot(times, v_opt[:, 0])
    plt.ylabel("Vel x (m/s)")
    plt.xlabel("Time (s)")
    plt.grid()

    plt.subplot(2, 3, 2)
    plt.plot(times, v_opt[:, 1])
    plt.ylabel("Vel y (m/s)")
    plt.xlabel("Time (s)")
    plt.grid()

    plt.subplot(2, 3, 3)
    plt.plot(times, v_opt[:, 2])
    plt.ylabel("Vel z (m/s)")
    plt.xlabel("Time (s)")
    plt.grid()

    plt.subplot(2, 3, 4)
    plt.plot(times, v_opt[:, 3])
    plt.ylabel("Omega x (rad/s)")
    plt.xlabel("Time (s)")
    plt.grid()

    plt.subplot(2, 3, 5)
    plt.plot(times, v_opt[:, 4])
    plt.ylabel("Omega y (rad/s)")
    plt.xlabel("Time (s)")
    plt.grid()

    plt.subplot(2, 3, 6)
    plt.plot(times, v_opt[:, 5])
    plt.ylabel("Omega z (rad/s)")
    plt.xlabel("Time (s)")
    plt.grid()

    # ----------------------- TAU -----------------------
    plt.figure(num="Control Inputs (tau)")
    plt.subplot(2, 3, 1)
    plt.plot(times[:-1], tau_opt[:, 0])
    plt.ylabel("Force x (N)")
    plt.xlabel("Time (s)")
    plt.grid()

    plt.subplot(2, 3, 2)
    plt.plot(times[:-1], tau_opt[:, 1])
    plt.ylabel("Force y (N)")
    plt.xlabel("Time (s)")
    plt.grid()  

    plt.subplot(2, 3, 3)
    plt.plot(times[:-1], tau_opt[:, 2])
    plt.ylabel("Force z (N)")
    plt.xlabel("Time (s)")
    plt.grid()

    plt.subplot(2, 3, 4)
    plt.plot(times[:-1], tau_opt[:, 3])
    plt.ylabel("Torque x (Nm)")
    plt.xlabel("Time (s)")
    plt.grid()

    plt.subplot(2, 3, 5)
    plt.plot(times[:-1], tau_opt[:, 4])
    plt.ylabel("Torque y (Nm)")
    plt.xlabel("Time (s)")
    plt.grid()

    plt.subplot(2, 3, 6)
    plt.plot(times[:-1], tau_opt[:, 5])
    plt.ylabel("Torque z (Nm)")
    plt.xlabel("Time (s)")
    plt.grid()

    plt.show()
