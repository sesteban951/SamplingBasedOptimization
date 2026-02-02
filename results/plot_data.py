##
#
# Plot data from optimization results
#
##

# standard imports
import numpy as np
import matplotlib.pyplot as plt
import time

# mujoco imports
import mujoco 
import mujoco.viewer

# which data to load
system = "cartpole"

# load data from csv files
time_file = f"./results/{system}/times.csv"
q_file = f"./results/{system}/q_opt.csv"
v_file = f"./results/{system}/v_opt.csv"
tau_file = f"./results/{system}/tau_opt.csv"

# load data from csv files
times = np.loadtxt(time_file, delimiter=",")
q_opt = np.loadtxt(q_file, delimiter=",")
v_opt = np.loadtxt(v_file, delimiter=",")
tau_opt = np.loadtxt(tau_file, delimiter=",")

print("Loaded data:")
print(f"  times: {times.shape}")
print(f"  q_opt: {q_opt.shape}")
print(f"  v_opt: {v_opt.shape}")
print(f"  tau_opt: {tau_opt.shape}")

# load the mujoco model
model = mujoco.MjModel.from_xml_path(f"./models/{system}.xml")
data = mujoco.MjData(model)

# visualize the optimal trajectory
viewer = mujoco.viewer.launch_passive(model, data)
t0 = time.time()
while True:

    i = np.searchsorted(times, time.time() - t0)

    data.qpos[:] = q_opt[i, :]
    data.qvel[:] = v_opt[i, :]
    mujoco.mj_step(model, data)
    viewer.sync()

    if time.time() - t0 > times[-1]:
        t0 = time.time()
    
