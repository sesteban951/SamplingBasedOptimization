##
#
# Visualize 5-link 3D articulated walker results in MuJoCo.
# Reads the joint-space trajectory produced by five_link_jump.py.
#
##

import sys
import time
import numpy as np
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt

from srb_walker.five_link_dynamics import FiveLinkDynamics

##############################################################
# Load data
##############################################################

result_dir = sys.argv[1] if len(sys.argv) > 1 else "./results/srb_walker/five_link_jump"

times   = np.loadtxt(f"{result_dir}/time.csv",    delimiter=",")
q_opt   = np.loadtxt(f"{result_dir}/q_opt.csv",   delimiter=",")  # (N+1, 13)
v_opt   = np.loadtxt(f"{result_dir}/v_opt.csv",   delimiter=",")
u_opt   = np.loadtxt(f"{result_dir}/u_opt.csv",   delimiter=",")
lam_L   = np.loadtxt(f"{result_dir}/lam_L.csv",   delimiter=",")
lam_R   = np.loadtxt(f"{result_dir}/lam_R.csv",   delimiter=",")
c_sched = np.loadtxt(f"{result_dir}/c_sched.csv", delimiter=",")

print(f"Loaded {result_dir}  |  {len(times)} steps  T={times[-1]:.2f}s")

dyn = FiveLinkDynamics()

##############################################################
# MuJoCo model from URDF
# mujoco.MjModel.from_xml_path can read URDF directly via
# MuJoCo's built-in URDF parser (MuJoCo >= 3.x)
##############################################################

xml_path = "./models/g1/g1_5link_3d.xml"
mj_model = mujoco.MjModel.from_xml_path(xml_path)
mj_data  = mujoco.MjData(mj_model)

print("MuJoCo nq:", mj_model.nq, "  nv:", mj_model.nv)

# MuJoCo qpos layout: [px,py,pz, qw,qx,qy,qz, joints(6)]
# Pinocchio qpos layout: [px,py,pz, qx,qy,qz,qw, joints(6)]
def pin_q_to_mj(q_pin):
    """Reorder base quaternion: Pinocchio (x,y,z,w) → MuJoCo (w,x,y,z)."""
    q_mj = q_pin.copy()
    q_mj[3] = q_pin[6]   # w
    q_mj[4] = q_pin[3]   # x
    q_mj[5] = q_pin[4]   # y
    q_mj[6] = q_pin[5]   # z
    return q_mj

##############################################################
# Launch viewer
##############################################################

viewer = mujoco.viewer.launch_passive(mj_model, mj_data)
viewer.cam.distance = 3.0
viewer.cam.elevation = -20.0

try:
    t0 = time.time()
    while viewer.is_running():
        elapsed = time.time() - t0
        i = min(int(np.searchsorted(times, elapsed)), len(times) - 1)

        q_mj = pin_q_to_mj(q_opt[i])
        mj_data.qpos[:] = q_mj
        mj_data.qvel[:] = v_opt[i]
        mujoco.mj_forward(mj_model, mj_data)

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

fig, axes = plt.subplots(3, 4, figsize=(15, 9), num="5-Link 3D Walker")

# base trajectory
for i, lbl in enumerate(["px (m)", "py (m)", "pz (m)"]):
    axes[0, i].plot(times, q_opt[:, i])
    axes[0, i].set_ylabel(lbl); axes[0, i].grid()
axes[0, 3].plot(times, q_opt[:, 3:7])
axes[0, 3].set_ylabel("quat xyzw"); axes[0, 3].legend(["x","y","z","w"],fontsize=7); axes[0, 3].grid()

# joint angles
joint_labels = ["l_roll","l_pitch","l_knee","r_roll","r_pitch","r_knee"]
for i in range(6):
    ax = axes[1, i % 4]
    ax.plot(times, q_opt[:, 7+i], label=joint_labels[i])
axes[1, 0].set_ylabel("joints L (rad)"); axes[1, 0].legend(fontsize=7); axes[1, 0].grid()
axes[1, 1].set_ylabel("joints R (rad)"); axes[1, 1].legend(fontsize=7); axes[1, 1].grid()

# torques
for i in range(min(4, u_opt.shape[1])):
    axes[2, i].plot(times[:-1], u_opt[:, i])
    axes[2, i].set_ylabel(f"tau_{joint_labels[i]} (Nm)"); axes[2, i].grid()

for ax in axes.flat:
    ax.set_xlabel("t (s)")

plt.tight_layout()
plt.show()
