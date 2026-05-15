##
#
# Visualize srb_walker 3D results as a 5-link skeleton.
#
##

import numpy as np
import time
import sys
import matplotlib.pyplot as plt

import mujoco
import mujoco.viewer

from utils.kinematics import kin
from srb_walker.g1_5link_params import (
    HIP_OFFSET_X, HIP_OFFSET_Y, HIP_OFFSET_Z,
    L_THIGH, L_SHANK, L_FOOT,
)

L2 = L_SHANK + L_FOOT   # shank + foot as single IK segment


##############################################################
# Load Data
##############################################################

result_dir = sys.argv[1] if len(sys.argv) > 1 else "./results/srb_walker/jump"

times   = np.loadtxt(f"{result_dir}/time.csv",             delimiter=",")
q_opt   = np.loadtxt(f"{result_dir}/q_opt.csv",            delimiter=",")  # (N+1, 7)
v_opt   = np.loadtxt(f"{result_dir}/v_opt.csv",            delimiter=",")
a_opt   = np.loadtxt(f"{result_dir}/a_opt.csv",            delimiter=",")
tau     = np.loadtxt(f"{result_dir}/tau_opt.csv",           delimiter=",")
feet    = np.loadtxt(f"{result_dir}/feet.csv",              delimiter=",")  # (N, 4): pL_x,pL_y,pR_x,pR_y
c_sched = np.loadtxt(f"{result_dir}/contact_schedule.csv", delimiter=",")

print(f"Loaded {result_dir}  |  {len(times)} timesteps  T={times[-1]:.2f}s")


##############################################################
# 3D Leg Kinematics
##############################################################

def hip_pos_world(p_com, R, side):
    """World-frame hip joint.  side: +1 = left, -1 = right."""
    off = np.array([HIP_OFFSET_X, side * HIP_OFFSET_Y, HIP_OFFSET_Z])
    return p_com + R @ off


def knee_pos_world(p_hip, p_foot, fwd):
    """
    3D 2-link IK placing the knee in the plane of (hip→foot, fwd).
    Picks knee-forward solution.
    """
    d = p_foot - p_hip
    L = np.linalg.norm(d)
    L = np.clip(L, abs(L_THIGH - L2) + 1e-4, L_THIGH + L2 - 1e-4)
    d_hat = d / (np.linalg.norm(d) + 1e-9)

    # preferred knee direction: body forward, projected perp to d_hat
    perp = fwd - np.dot(fwd, d_hat) * d_hat
    perp_norm = np.linalg.norm(perp)
    if perp_norm < 1e-6:
        # leg is nearly parallel to forward; fall back to world up
        up = np.array([0., 0., 1.])
        perp = up - np.dot(up, d_hat) * d_hat
        perp_norm = np.linalg.norm(perp)
    perp = perp / (perp_norm + 1e-9)

    # law of cosines: angle at hip
    cos_alpha = np.clip((L_THIGH**2 + L**2 - L2**2) / (2 * L_THIGH * L), -1.0, 1.0)
    alpha = np.arccos(cos_alpha)

    return p_hip + L_THIGH * (np.cos(alpha) * d_hat + np.sin(alpha) * perp)


def swing_knee_foot(p_hip, R, side, phase):
    """Neutral swing pose for flight phase.  phase in [0,1]."""
    q_h = 0.3 * np.sin(np.pi * phase)
    q_k = 0.6 * np.sin(np.pi * phase)
    # build leg direction from body frame
    fwd   = R[:, 0]    # body x = forward
    down  = -R[:, 2]   # body -z = down
    thigh_dir = np.cos(q_h) * down + np.sin(q_h) * fwd
    shank_dir = np.cos(q_h - q_k) * down + np.sin(q_h - q_k) * fwd
    p_knee = p_hip + L_THIGH * thigh_dir
    p_foot = p_knee + L2      * shank_dir
    return p_knee, p_foot


##############################################################
# MuJoCo Custom Geometry Helpers
##############################################################

def _capsule(scene, p0, p1, r, rgba):
    if scene.ngeom >= scene.maxgeom:
        return
    d = p1 - p0
    L = np.linalg.norm(d)
    if L < 1e-9:
        return
    z = d / L
    ref = np.array([1., 0., 0.]) if abs(z[0]) < 0.9 else np.array([0., 1., 0.])
    x = np.cross(ref, z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.column_stack((x, y, z))
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE,
                        np.array([r, 0.5 * L, 0.]),
                        0.5 * (p0 + p1), R.reshape(9),
                        np.array(rgba, np.float64))
    g.category = int(mujoco.mjtCatBit.mjCAT_DECOR)
    scene.ngeom += 1


def _sphere(scene, pos, r, rgba):
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.array([r, 0., 0.]),
                        np.array(pos, np.float64), np.eye(3).reshape(9),
                        np.array(rgba, np.float64))
    g.category = int(mujoco.mjtCatBit.mjCAT_DECOR)
    scene.ngeom += 1


##############################################################
# Colors
##############################################################

C_TORSO  = [0.30, 0.50, 0.80, 0.92]
C_TH_L   = [0.90, 0.45, 0.10, 0.92]   # left thigh  (orange)
C_SH_L   = [0.95, 0.70, 0.20, 0.92]   # left shank  (yellow)
C_TH_R   = [0.10, 0.55, 0.90, 0.92]   # right thigh (blue)
C_SH_R   = [0.20, 0.80, 0.95, 0.92]   # right shank (cyan)
C_JOINT  = [0.95, 0.95, 0.95, 1.00]
C_FOOT   = [0.85, 0.85, 0.20, 1.00]


##############################################################
# Draw one skeleton frame
##############################################################

def draw_skeleton(scene, q_row, feet_row, c):
    p_com = q_row[0:3]
    quat  = q_row[3:7]
    R     = kin.quat_to_rot_matrix(quat)   # numpy version
    fwd   = R[:, 0]                         # body forward direction

    # torso capsule along body z-axis
    torso_h = 0.18
    _capsule(scene,
             p_com - torso_h * R[:, 2] * 0.4,
             p_com + torso_h * R[:, 2] * 0.6,
             0.07, C_TORSO)

    in_contact = (c > 0.5) and np.all(np.isfinite(feet_row))

    for side, (fx_col, c_th, c_sh) in enumerate([
        ( 1, C_TH_L, C_SH_L),   # left
        (-1, C_TH_R, C_SH_R),   # right
    ]):
        p_hip = hip_pos_world(p_com, R, side)

        if in_contact:
            fx, fy = feet_row[2 - 2*side], feet_row[3 - 2*side]
            p_foot = np.array([fx, fy, 0.0])
            p_knee = knee_pos_world(p_hip, p_foot, fwd)
        else:
            p_knee, p_foot = swing_knee_foot(p_hip, R, side, phase=0.5)

        _capsule(scene, p_hip,  p_knee, 0.035, c_th)
        _capsule(scene, p_knee, p_foot, 0.025, c_sh)
        _sphere (scene, p_hip,  0.045, C_JOINT)
        _sphere (scene, p_knee, 0.035, C_JOINT)
        _sphere (scene, p_foot, 0.025, C_FOOT)


##############################################################
# Launch Viewer
##############################################################

xml_path = "./models/srb/srb.xml"
model = mujoco.MjModel.from_xml_path(xml_path)
data  = mujoco.MjData(model)
model.geom_rgba[:] = 0.0   # hide the SRB box; we draw our own skeleton

viewer = mujoco.viewer.launch_passive(model, data)

try:
    t0 = time.time()
    while viewer.is_running():
        elapsed = time.time() - t0
        i = min(int(np.searchsorted(times, elapsed)), len(times) - 1)

        q_i = q_opt[i]
        data.qpos[0:3] = q_i[0:3]     # position
        data.qpos[3:7] = q_i[3:7]     # quaternion [w, x, y, z]
        mujoco.mj_step(model, data)

        if viewer.user_scn is not None:
            viewer.user_scn.ngeom = 0
            feet_i = feet[min(i, feet.shape[0] - 1), :]
            draw_skeleton(viewer.user_scn, q_i, feet_i, c_sched[i])

        viewer.sync()

        if elapsed > times[-1]:
            time.sleep(0.5)
            t0 = time.time()

except KeyboardInterrupt:
    pass

viewer.close()


##############################################################
# Plots
##############################################################

fig, axes = plt.subplots(3, 4, figsize=(14, 8), num="5-Link Walker SRB (3D)")

# positions
for i, lbl in enumerate(["px (m)", "py (m)", "pz (m)"]):
    axes[0, i].plot(times, q_opt[:, i]); axes[0, i].set_ylabel(lbl); axes[0, i].grid()

# euler angles from quaternion
euler = np.array([kin.quat_to_euler_ZYX(q_opt[k, 3:7]) for k in range(len(times))])
for i, lbl in enumerate(["roll (rad)", "pitch (rad)", "yaw (rad)"]):
    axes[0, 3].plot(times, euler[:, i], label=lbl)
axes[0, 3].set_ylabel("orientation"); axes[0, 3].legend(fontsize=7); axes[0, 3].grid()

# velocities
for i, lbl in enumerate(["vx (m/s)", "vy (m/s)", "vz (m/s)", "wx (r/s)"]):
    axes[1, i].plot(times, v_opt[:, i]); axes[1, i].set_ylabel(lbl); axes[1, i].grid()

# controls
for i, lbl in enumerate(["Fx (N)", "Fy (N)", "Fz (N)", "My (Nm)"]):
    col = i if i < 3 else 3
    axes[2, col].plot(times[:-1], tau[:, i]); axes[2, col].set_ylabel(lbl); axes[2, col].grid()

for ax in axes.flat:
    ax.set_xlabel("t (s)")

plt.tight_layout()
plt.show()
