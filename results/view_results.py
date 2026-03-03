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

# jax imports 
import jax.numpy as jnp

# custom imports
from utils.simulation.dynamics import Dynamics_Config, Dynamics


#################################################################################
# SETTINGS
#################################################################################

# playback the optimal trajectory in the mujoco viewer
visualize = 1

# plot the optimal trajectory signals
plot_solution = 1

# parse extra data from the optimal trajectory (e.g. sensor trajectories)
plot_other_quantities = 0
plot_pos_sensors = 1
plot_ori_sensors = 1
plot_touch_sensors = 1


#################################################################################
# LOAD DATA
#################################################################################

# which data to load
# experiment = "cartpole/cartpole_cem"
# experiment = "cartpole/cartpole_mppi"
# experiment = "cartpole/cartpole_cmaes"
# xml_path = f"./models/cartpole/cartpole.xml"
# experiment = "cartpole/cartpole_walls_cem"
# xml_path = f"./models/cartpole/cartpole_walls.xml"

# experiment = "hopper/hopper_cem"
# experiment = "hopper/hopper_backflip_cem"
# xml_path = f"./models/hopper/hopper.xml"

# experiment = "cube/cube_cem"
# xml_path = f"./models/cube/scene.xml"

# experiment = "g1/g1_planar_walk_cem"
# experiment = "g1/g1_planar_walk_mppi"
# experiment = "g1/g1_planar_walk_weights_mppi"
experiment = "g1/g1_planar_jump_mirror_cem"
xml_path = f"./models/g1/g1_planar.xml"

# experiment = "g1/g1_stand_cem"
# experiment = "g1/g1_jump_wrench_cem"
# experiment = "parallel_sim/parallel_sim"
# xml_path = f"./models/g1/g1_21dof.xml"
# xml_path = f"./models/g1/g1_planar.xml"
# xml_path = f"./models/biped/biped.xml"

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


#################################################################################
# MUJOCO VISUALIZATION
#################################################################################

# load the mujoco model
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# visualize the optimal trajectory
if visualize == 1:

    # launch the viewer
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


#################################################################################
# PLOT SOLUTION TRAJECTORY
#################################################################################

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

if plot_solution == 1:

    # get the state sizes
    nq = q_opt.shape[-1]
    nv = v_opt.shape[-1]
    nu = tau_opt.shape[-1] if tau_opt.ndim > 1 else 1

    # Get joint and actuator names from MuJoCo model
    # q_names = [model.joint(i).name if model.joint(i).name else f'q[{i}]' for i in range(nq)]
    # v_names = [model.joint(i).name if model.joint(i).name else f'v[{i}]' for i in range(nv)]
    # use njnt for joint names, not nq/nv
    q_names = [model.joint(i).name if model.joint(i).name else f'q[{i}]' for i in range(model.njnt)]
    v_names = [model.joint(i).name if model.joint(i).name else f'v[{i}]' for i in range(model.njnt)]
    q_names += [f'q[{i}]' for i in range(model.njnt, nq)]
    v_names += [f'v[{i}]' for i in range(model.njnt, nv)]
    u_names = [model.actuator(i).name if model.actuator(i).name else f'tau[{i}]' for i in range(nu)]

    # =============================== Plot positions (q) =============================== 
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

    # =============================== Plot velocities (v) =============================== 
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

    # =============================== Plot controls (tau) =============================== 
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


#################################################################################
# PLOT OTHER QUANTITIES
#################################################################################

# parse extra data from the optimal trajectory
if plot_other_quantities == 1:

    # integer values for sensor types from mujoco
    touch_type = int(mujoco.mjtSensor.mjSENS_TOUCH)
    framepos_type = int(mujoco.mjtSensor.mjSENS_FRAMEPOS)
    framequat_type = int(mujoco.mjtSensor.mjSENS_FRAMEQUAT)

    # auto-discover supported sensor names from model
    touch_sensor_names = []
    pos_sensor_names = []
    ori_sensor_names = []
    for i in range(model.nsensor):
        sname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i)
        stype = int(model.sensor_type[i])
        if sname is None:
            continue

        if stype == touch_type:
            touch_sensor_names.append(sname)
        elif stype == framepos_type:
            pos_sensor_names.append(sname)
        elif stype == framequat_type:
            ori_sensor_names.append(sname)

    # use None (not empty list) when no sensors are present for a type
    touch_sensor_names = touch_sensor_names if len(touch_sensor_names) > 0 else None
    pos_sensor_names =   pos_sensor_names   if len(pos_sensor_names)   > 0 else None
    ori_sensor_names =   ori_sensor_names   if len(ori_sensor_names)   > 0 else None

    print("Auto-detected supported sensors:")
    print(f"  touch: {touch_sensor_names}")
    print(f"  framepos: {pos_sensor_names}")
    print(f"  framequat: {ori_sensor_names}")

    # create a dynamics object
    dyn_config = Dynamics_Config(
        xml_path=xml_path,
        num_envs = 1,
        pos_sensor_names=pos_sensor_names,
        ori_sensor_names=ori_sensor_names,
        touch_sensor_names=touch_sensor_names,
    )
    dyn = Dynamics(dyn_config)

    # broadcast state to 1 env
    q_opt_ = jnp.array(q_opt) # (N+1, nq)
    v_opt_ = jnp.array(v_opt) # (N+1, nv)
    q_opt_env = jnp.broadcast_to(q_opt_, (1, *q_opt_.shape)) # (1, N+1, nq)
    v_opt_env = jnp.broadcast_to(v_opt_, (1, *v_opt_.shape)) # (1, N+1, nv)

    # check which sensors are available
    has_pos = dyn_config.pos_sensor_names is not None
    has_ori = dyn_config.ori_sensor_names is not None
    has_touch = dyn_config.touch_sensor_names is not None

    # position sensors
    if plot_pos_sensors == 1:
        if has_pos:
            pos_t = dyn.sensor_pos_in_world_trajectory(q_opt_env, v_opt_env)  # (1, N+1, ns_pos, 3)
            print(f"sensor_pos_in_world_trajectory shape: {pos_t.shape}")
            ns_pos = len(dyn_config.pos_sensor_names)
            rows, cols = get_tile_layout(ns_pos)
            fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), sharex=True)
            fig.suptitle("Position Sensors")
            axes = np.atleast_1d(axes).flatten()

            for i, sensor_name in enumerate(dyn_config.pos_sensor_names):
                ax = axes[i]
                ax.plot(times, pos_t[0, :, i, 0], label="x")
                ax.plot(times, pos_t[0, :, i, 1], label="y")
                ax.plot(times, pos_t[0, :, i, 2], label="z")
                ax.set_title(sensor_name)
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Position [m]")
                ax.grid(True)
                ax.legend()

            for i in range(ns_pos, rows * cols):
                axes[i].axis("off")
        else:
            print("plot_pos_sensors=1, but no framepos sensors found.")

    # orientation sensors (quaternions)
    if plot_ori_sensors == 1:
        if has_ori:
            ori_t = dyn.sensor_ori_in_world_trajectory(q_opt_env, v_opt_env)  # (1, N+1, ns_ori, 4)
            print(f"sensor_ori_in_world_trajectory shape: {ori_t.shape}")
            ns_ori = len(dyn_config.ori_sensor_names)
            rows, cols = get_tile_layout(ns_ori)
            fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), sharex=True)
            fig.suptitle("Orientation Sensors (Quaternion)")
            axes = np.atleast_1d(axes).flatten()

            for i, sensor_name in enumerate(dyn_config.ori_sensor_names):
                ax = axes[i]
                ax.plot(times, ori_t[0, :, i, 0], label="qw")
                ax.plot(times, ori_t[0, :, i, 1], label="qx")
                ax.plot(times, ori_t[0, :, i, 2], label="qy")
                ax.plot(times, ori_t[0, :, i, 3], label="qz")
                ax.set_title(sensor_name)
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Quat")
                ax.grid(True)
                ax.legend()

            for i in range(ns_ori, rows * cols):
                axes[i].axis("off")
        else:
            print("plot_ori_sensors=1, but no framequat sensors found.")

    # touch sensors
    if plot_touch_sensors == 1:
        if has_touch:
            touch_t = dyn.sensor_touch_trajectory(q_opt_env, v_opt_env)  # (1, N+1, ns_touch)
            print(f"sensor_touch_trajectory shape: {touch_t.shape}")
            ns_touch = len(dyn_config.touch_sensor_names)
            rows, cols = get_tile_layout(ns_touch)
            fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), sharex=True)
            fig.suptitle("Touch Sensors")
            axes = np.atleast_1d(axes).flatten()

            for i, sensor_name in enumerate(dyn_config.touch_sensor_names):
                ax = axes[i]
                ax.plot(times, touch_t[0, :, i], label=sensor_name)
                ax.set_title(sensor_name)
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Touch [N]")
                ax.grid(True)
                ax.legend()

            for i in range(ns_touch, rows * cols):
                axes[i].axis("off")
        else:
            print("plot_touch_sensors=1, but no touch sensors found.")

    # if there are sensors to plot, show the figures
    if ((plot_pos_sensors == 1 and has_pos) or 
        (plot_ori_sensors == 1 and has_ori) or 
        (plot_touch_sensors == 1 and has_touch)):
        
        plt.show()
