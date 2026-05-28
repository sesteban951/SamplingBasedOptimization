##
# Print and plot the centroidal rotational inertia I_G(q) over the IK trajectory.
#
# Loads existing results from:
#   ik/results/ik_q_pin.csv   — pinocchio-convention full configuration (nq=36)
#   ik/results/ik_time.csv    — timestamps
#   results/srb/srb_aerial/feet.csv — for stance/flight/landing phase detection
#
# Run after:  python ik/pipeline_srb_ik.py --config srb.config.smalljump
##

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pinocchio as pin
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

_REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_URDF       = os.path.join(_REPO_ROOT, "models", "g1", "g1_29dof_rev_1_0.urdf")
_IK_DIR     = os.path.join(_REPO_ROOT, "ik", "results")
_SRB_DIR    = os.path.join(_REPO_ROOT, "results", "srb", "srb_aerial")

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

q_ik  = np.loadtxt(os.path.join(_IK_DIR, "ik_q_pin.csv"),  delimiter=",", comments="#")
times = np.loadtxt(os.path.join(_IK_DIR, "ik_time.csv"),   delimiter=",")
feet  = np.loadtxt(os.path.join(_SRB_DIR, "feet.csv"),     delimiter=",")

N = len(times) - 1
feet_ext    = np.vstack([feet, feet[-1:]])
is_flight   = np.isnan(feet_ext[:, 0])
stance_end  = int(np.where(is_flight)[0][0])
flight_end  = int(np.where(~is_flight[stance_end:])[0][0]) + stance_end

print(f"Frames: {N+1}  stance: 0..{stance_end-1}  "
      f"flight: {stance_end}..{flight_end-1}  landing: {flight_end}..{N}")

# ---------------------------------------------------------------------------
# Build pinocchio model and compute I_G(q) at every frame
# ---------------------------------------------------------------------------

model = pin.buildModelFromUrdf(_URDF, pin.JointModelFreeFlyer())
data  = model.createData()
nv    = model.nv
v0    = np.zeros(nv)

# I_G is the 3×3 centroidal rotational inertia; diagonal = [Ixx, Iyy, Izz]
n_frames  = q_ik.shape[0]
I_G_diag  = np.zeros((n_frames, 3))   # Ixx, Iyy, Izz
I_G_offdiag = np.zeros((n_frames, 3)) # Ixy, Ixz, Iyz
I_G_full  = np.zeros((n_frames, 3, 3))

for i, q in enumerate(q_ik):
    pin.ccrba(model, data, q, v0)
    Ig = np.array(data.Ig.inertia)   # 3×3 rotational inertia at CoM
    I_G_diag[i]    = [Ig[0,0], Ig[1,1], Ig[2,2]]
    I_G_offdiag[i] = [Ig[0,1], Ig[0,2], Ig[1,2]]
    I_G_full[i]    = Ig

# ---------------------------------------------------------------------------
# SRB reference inertia (from models/srb/srb.xml fullinertia)
# MuJoCo fullinertia: [Ixx, Iyy, Izz, Ixy, Ixz, Iyz]
# ---------------------------------------------------------------------------

I_srb_diag    = np.array([3.7475, 3.301,  0.5165])   # Ixx, Iyy, Izz
I_srb_offdiag = np.array([0.0001, 0.087, -0.0009])   # Ixy, Ixz, Iyz

# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------

labels = ["Ixx", "Iyy", "Izz"]
print("\nCentroidal inertia I_G(q)  [kg·m²]")
print(f"{'Frame':>6}  {'time':>6}  {'Ixx':>10}  {'Iyy':>10}  {'Izz':>10}  phase")
phases = (["stance"] * stance_end +
          ["flight"] * (flight_end - stance_end) +
          ["landing"] * (N + 1 - flight_end))
for i in range(0, n_frames, max(1, n_frames // 20)):   # ~20 representative rows
    d = I_G_diag[i]
    print(f"{i:>6}  {times[i]:>6.3f}  {d[0]:>10.5f}  {d[1]:>10.5f}  {d[2]:>10.5f}  {phases[i]}")

print(f"\nSRB reference  Ixx={I_srb_diag[0]}  Iyy={I_srb_diag[1]}  Izz={I_srb_diag[2]}")
print(f"               Ixy={I_srb_offdiag[0]}  Ixz={I_srb_offdiag[1]}  Iyz={I_srb_offdiag[2]}")
print(f"\nG1 range  Ixx: [{I_G_diag[:,0].min():.5f}, {I_G_diag[:,0].max():.5f}]")
print(f"          Iyy: [{I_G_diag[:,1].min():.5f}, {I_G_diag[:,1].max():.5f}]")
print(f"          Izz: [{I_G_diag[:,2].min():.5f}, {I_G_diag[:,2].max():.5f}]")

# Gap at key frames
print(f"\nGap  G1 - SRB  [kg·m²]  (diagonal only)")
print(f"{'phase':>8}  {'dIxx':>8}  {'dIyy':>8}  {'dIzz':>8}")
for label, idx in [("takeoff", stance_end - 1), ("apex", (stance_end + flight_end) // 2), ("landing", flight_end)]:
    d = I_G_diag[idx] - I_srb_diag
    print(f"{label:>8}  {d[0]:>+8.4f}  {d[1]:>+8.4f}  {d[2]:>+8.4f}")

# Print the full 3×3 at a few key frames
for label, idx in [("takeoff", stance_end - 1),
                   ("apex",    (stance_end + flight_end) // 2),
                   ("landing", flight_end)]:
    print(f"\nI_G at {label} (frame {idx}, t={times[idx]:.3f}s):")
    print(np.round(I_G_full[idx], 6))

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

t_stance_end = times[stance_end]
t_flight_end = times[flight_end]

fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
fig.suptitle("Centroidal Rotational Inertia  $I_G(q)$  over IK Trajectory", fontsize=13)

colors_diag   = ["steelblue", "darkorange", "seagreen"]
colors_offdiag = ["orchid", "sienna", "teal"]

# ── Top: diagonal (principal moments) ──────────────────────────────────────
ax = axes[0]
for j, (lbl, col) in enumerate(zip(["$I_{xx}$", "$I_{yy}$", "$I_{zz}$"], colors_diag)):
    ax.plot(times, I_G_diag[:, j], color=col, lw=1.5, label=f"G1 {lbl}")
    ax.axhline(I_srb_diag[j], color=col, lw=1.0, ls=":", alpha=0.7,
               label=f"SRB {lbl} = {I_srb_diag[j]:.4f}")
ax.set_ylabel("Inertia  [kg·m²]")
ax.set_title("Diagonal (principal moments)")
ax.legend(fontsize=8, loc="upper right", ncol=2)

# ── Bottom: off-diagonal ────────────────────────────────────────────────────
ax = axes[1]
for j, (lbl, col) in enumerate(zip(["$I_{xy}$", "$I_{xz}$", "$I_{yz}$"], colors_offdiag)):
    ax.plot(times, I_G_offdiag[:, j], color=col, lw=1.5, label=f"G1 {lbl}")
    ax.axhline(I_srb_offdiag[j], color=col, lw=1.0, ls=":", alpha=0.7,
               label=f"SRB {lbl} = {I_srb_offdiag[j]:.4f}")
ax.set_ylabel("Inertia  [kg·m²]")
ax.set_xlabel("Time  [s]")
ax.set_title("Off-diagonal (cross-products)")
ax.axhline(0, color="gray", lw=0.5)
ax.legend(fontsize=8, loc="upper right", ncol=2)

# Phase shading on both axes
for ax in axes:
    ax.axvspan(times[0],      t_stance_end, alpha=0.08, color="steelblue")
    ax.axvspan(t_flight_end,  times[-1],    alpha=0.08, color="seagreen")
    ax.axvline(t_stance_end,  color="steelblue", lw=0.9, ls="--")
    ax.axvline(t_flight_end,  color="seagreen",  lw=0.9, ls="--")

phase_patches = [
    mpatches.Patch(color="steelblue", alpha=0.3, label="Stance"),
    mpatches.Patch(color="white",     alpha=0.0, label="Flight", ec="gray"),
    mpatches.Patch(color="seagreen",  alpha=0.3, label="Landing"),
]
fig.legend(handles=phase_patches, loc="lower center", ncol=3,
           fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.01))

fig.tight_layout(rect=[0, 0.04, 1, 1])
plt.show()
