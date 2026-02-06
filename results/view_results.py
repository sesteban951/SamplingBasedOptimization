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


#################################################################
# LOAD DATA
#################################################################

# which data to load
# experiment = "cartpole"
# xml_path = f"./models/cartpole/cartpole_walls.xml"
# experiment = "hopper"
# experiment = "hopper_backflip"
# xml_path = f"./models/hopper/hopper.xml"
experiment = "cube"
xml_path = f"./models/cube/scene.xml"
# experiment = "g1_walk"
# xml_path = f"./models/g1/g1_planar.xml"

# load data from csv files
time_file = f"./results/{experiment}/time.csv"
q_file = f"./results/{experiment}/q_opt.csv"
v_file = f"./results/{experiment}/v_opt.csv"
tau_file = f"./results/{experiment}/tau_opt.csv"

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


#################################################################
# MUJOCO VISUALIZATION
#################################################################

# load the mujoco model
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

# get the state sizes
nq = q_opt.shape[-1]
nv = v_opt.shape[-1]
nu = tau_opt.shape[-1] if tau_opt.ndim > 1 else 1

# Get joint and actuator names from MuJoCo model
q_names = [model.joint(i).name if model.joint(i).name else f'q[{i}]' for i in range(nq)]
v_names = [model.joint(i).name if model.joint(i).name else f'v[{i}]' for i in range(nv)]
u_names = [model.actuator(i).name if model.actuator(i).name else f'tau[{i}]' for i in range(nu)]

# Function to get best square tile layout
def get_tile_layout(n):
    """Get the best rows x cols layout for n subplots"""
    if n == 0:
        return 0, 0
    # Find the smallest integer >= sqrt(n)
    cols = int(np.ceil(np.sqrt(n)))
    # Find minimum rows needed
    rows = int(np.ceil(n / cols))
    return rows, cols

# Plot positions (q)
if nq > 0:
    rows_q, cols_q = get_tile_layout(nq)
    fig_q, axes_q = plt.subplots(rows_q, cols_q, figsize=(4*cols_q, 3*rows_q))
    fig_q.suptitle('Position Trajectories (q)', fontsize=16)
    
    # Flatten axes for easy indexing
    if rows_q * cols_q == 1:
        axes_q = np.array([axes_q])
    else:
        axes_q = np.atleast_1d(axes_q).flatten()
    
    for i in range(nq):
        axes_q[i].plot(times, q_opt[:, i], linewidth=2)
        axes_q[i].set_xlabel('Time (s)')
        axes_q[i].set_ylabel(q_names[i])
        axes_q[i].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(nq, rows_q * cols_q):
        axes_q[i].axis('off')
    
    plt.tight_layout()

# Plot velocities (v)
if nv > 0:
    rows_v, cols_v = get_tile_layout(nv)
    fig_v, axes_v = plt.subplots(rows_v, cols_v, figsize=(4*cols_v, 3*rows_v))
    fig_v.suptitle('Velocity Trajectories (v)', fontsize=16)
    
    # Flatten axes for easy indexing
    if rows_v * cols_v == 1:
        axes_v = np.array([axes_v])
    else:
        axes_v = np.atleast_1d(axes_v).flatten()
    
    for i in range(nv):
        axes_v[i].plot(times, v_opt[:, i], linewidth=2, color='orange')
        axes_v[i].set_xlabel('Time (s)')
        axes_v[i].set_ylabel(v_names[i])
        axes_v[i].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(nv, rows_v * cols_v):
        axes_v[i].axis('off')
    
    plt.tight_layout()

# Plot controls (tau)
if nu > 0:
    rows_u, cols_u = get_tile_layout(nu)
    fig_u, axes_u = plt.subplots(rows_u, cols_u, figsize=(4*cols_u, 3*rows_u))
    fig_u.suptitle('Control Inputs (tau)', fontsize=16)
    
    # Flatten axes for easy indexing
    if rows_u * cols_u == 1:
        axes_u = np.array([axes_u])
    else:
        axes_u = np.atleast_1d(axes_u).flatten()
    
    # Use times that match tau_opt length
    times_u = times[:tau_opt.shape[0]]
    
    # Ensure tau_opt is 2D
    if tau_opt.ndim == 1:
        tau_opt = tau_opt.reshape(-1, 1)
    
    for i in range(nu):
        axes_u[i].plot(times_u, tau_opt[:, i], linewidth=2, color='green')
        axes_u[i].set_xlabel('Time (s)')
        axes_u[i].set_ylabel(u_names[i])
        axes_u[i].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(nu, rows_u * cols_u):
        axes_u[i].axis('off')
    
    plt.tight_layout()

plt.show()