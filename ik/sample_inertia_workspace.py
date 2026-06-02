##
#
# Sample centroidal inertia bounds from random leg configurations.
#
# Samples N random leg configurations within URDF joint limits and computes
# the centroidal rotational inertia I_G(q) via pinocchio's ccrba() at each.
# Records the per-component min/max bounds over all samples.
#
# Output:
#   ik/results/inertia_bounds.csv  — 6 rows: component, min, max [kg·m²]
#
# Run once (offline) before using variable_inertia=True in the SRB:
#   conda run -n env_sbo python ik/sample_inertia_workspace.py
#
##

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pinocchio as pin

from utils.kinematics.g1_ik import G1IK

_OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(_OUT_DIR, exist_ok=True)

# ── robot model ───────────────────────────────────────────────────────────────

ik   = G1IK()
model = ik.model
data  = model.createData()

LEG_JOINTS = [
    "left_hip_pitch_joint",    "left_hip_roll_joint",    "left_hip_yaw_joint",
    "left_knee_joint",         "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint",   "right_hip_roll_joint",   "right_hip_yaw_joint",
    "right_knee_joint",        "right_ankle_pitch_joint","right_ankle_roll_joint",
]

leg_qidx = []
leg_lo   = []
leg_hi   = []
for name in LEG_JOINTS:
    jid  = model.getJointId(name)
    qi   = model.joints[jid].idx_q
    leg_qidx.append(qi)
    leg_lo.append(model.lowerPositionLimit[qi])
    leg_hi.append(model.upperPositionLimit[qi])

leg_lo = np.array(leg_lo)
leg_hi = np.array(leg_hi)

# ── base configuration (pelvis at nominal standing height, upright) ───────────

q_base = pin.neutral(model)
q_base[2] = 0.77   # nominal CoM height; sampling does not require a specific height
# pinocchio quaternion convention is [qx, qy, qz, qw] at indices 3:7
q_base[3:7] = [0.0, 0.0, 0.0, 1.0]   # identity rotation

v0 = np.zeros(model.nv)

# ── sampling ──────────────────────────────────────────────────────────────────

N_SAMPLES = 50_000
rng = np.random.default_rng(42)

COMPONENTS = ["Ixx", "Iyy", "Izz", "Ixy", "Ixz", "Iyz"]
I_min = np.full(6,  np.inf)
I_max = np.full(6, -np.inf)

print(f"Sampling {N_SAMPLES:,} random leg configurations ...")
print(f"  Leg joints: {len(LEG_JOINTS)}")
print(f"  Tracking: {COMPONENTS}\n")

for idx in range(N_SAMPLES):
    q = q_base.copy()
    q_legs = rng.uniform(leg_lo, leg_hi)
    for j, qi in enumerate(leg_qidx):
        q[qi] = q_legs[j]

    pin.ccrba(model, data, q, v0)
    I_G = np.array(data.Ig.inertia)   # 3×3 symmetric, world frame

    i6 = np.array([
        I_G[0, 0],  # Ixx
        I_G[1, 1],  # Iyy
        I_G[2, 2],  # Izz
        I_G[0, 1],  # Ixy
        I_G[0, 2],  # Ixz
        I_G[1, 2],  # Iyz
    ])
    I_min = np.minimum(I_min, i6)
    I_max = np.maximum(I_max, i6)

    if (idx + 1) % 10_000 == 0:
        pct = (idx + 1) / N_SAMPLES * 100
        print(f"  [{pct:5.1f}%]  "
              f"Ixx∈[{I_min[0]:.3f},{I_max[0]:.3f}]  "
              f"Iyy∈[{I_min[1]:.3f},{I_max[1]:.3f}]  "
              f"Izz∈[{I_min[2]:.3f},{I_max[2]:.3f}]")

# ── save ──────────────────────────────────────────────────────────────────────

out_path = os.path.join(_OUT_DIR, "inertia_bounds.csv")
with open(out_path, "w") as f:
    f.write("# component,min,max  (kg·m²)  — centroidal inertia from leg-only random sampling\n")
    for comp, lo, hi in zip(COMPONENTS, I_min, I_max):
        f.write(f"{comp},{lo:.10f},{hi:.10f}\n")

print(f"\nBounds saved → {out_path}")
print("\nSummary:")
print(f"  {'Component':<8}  {'Min (kg·m²)':>12}  {'Max (kg·m²)':>12}  {'Range':>10}")
for comp, lo, hi in zip(COMPONENTS, I_min, I_max):
    print(f"  {comp:<8}  {lo:>12.6f}  {hi:>12.6f}  {hi-lo:>10.6f}")
