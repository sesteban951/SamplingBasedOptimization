##
#
# Plot data from SRB optimization results
#
##

# standard imports
import numpy as np
import matplotlib.pyplot as plt
import time

# mujoco imports
import mujoco 
import mujoco.viewer

# custom improts
from utils.kinematics import kin


#################################################################
# LOAD DATA
#################################################################

# which data to load
time_file = "./results/srb/times.csv"
state_file = "./results/srb/states.csv"
input_file = "./results/srb/inputs.csv"

# load data from csv files
times = np.loadtxt(time_file, delimiter=",")
states = np.loadtxt(state_file, delimiter=",")
tau_opt = np.loadtxt(input_file, delimiter=",")

# extract postion and velocity from states
q_opt = states[:, :7]  # p_com, quat
v_opt = states[:, 7:]  # v_com, omega

print("Loaded data:")
print(f"  times: {times.shape}")
print(f"  q_opt: {q_opt.shape}")
print(f"  v_opt: {v_opt.shape}")
print(f"  tau_opt: {tau_opt.shape}")


#################################################################
# VISUALIZATION
#################################################################

# load the mujoco model
xml_path = "./models/srb/srb.xml"
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# visualize the optimal trajectory
viewer = mujoco.viewer.launch_passive(model, data)

# run the visualization
try:
    t0 = time.time()
    while True:

        if viewer.is_running() == False:
            break

        i = np.searchsorted(times, time.time() - t0)
        i = min(i, len(times) - 1)  # Clamp to valid range

        print(f"Time: {time.time() - t0:.2f}, Index: {i}\r", end="")

        data.qpos[:] = q_opt[i, :]
        data.qvel[:] = v_opt[i, :]
        mujoco.mj_step(model, data)
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
plt.ylabel("Quat x")
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
