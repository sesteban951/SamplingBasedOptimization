##
#
# Sample centroidal inertia workspace vs. a tuck parameter.
#
# Two modes in one script:
#
#   1) TUCK SWEEP (fast, targeted):
#      Drives the primary tuck joints (hip_pitch, knee, ankle_pitch) along a
#      1-D curve t ∈ [0,1] from fully extended to fully tucked, while randomly
#      sampling the secondary DOFs (hip_roll, hip_yaw, ankle_roll) to capture
#      the achievable inertia range at each tuck level.
#
#      Output → ik/results/inertia_tuck_curve.npz:
#          tuck              (N_TUCK,)      tuck parameter values [0, 1]
#          Iyy_mean          (N_TUCK,)      mean Iyy [kg·m²]
#          Iyy_std           (N_TUCK,)
#          Iyy_min           (N_TUCK,)      bounds for optimizer constraint
#          Iyy_max           (N_TUCK,)
#          foot_dist_mean    (N_TUCK,)      mean min(dist_L, dist_R) pelvis-frame [m]
#          foot_pos_L_mean   (N_TUCK, 3)   mean left  foot (pelvis frame) [m]
#          foot_pos_R_mean   (N_TUCK, 3)   mean right foot (pelvis frame) [m]
#          prim_angles       (N_TUCK, N_PRIM)  primary joint angles at each tuck level
#
#      Output → ik/results/inertia_tuck_scatter.npz:
#          tuck              (M,)           tuck level each sample belongs to
#          inertia           (M, 6)         [Ixx,Iyy,Izz,Ixy,Ixz,Iyz] kg·m²
#          foot_pos_L        (M, 3)         left  foot pos, pelvis frame [m]
#          foot_pos_R        (M, 3)         right foot pos, pelvis frame [m]
#
#   2) RANDOM SAMPLING (overall bounds — slower):
#      Uniform random samples over all leg joints (collision-filtered).
#
#      Output → ik/results/inertia_bounds.csv
#      Output → ik/results/inertia_samples.npz
#
# Run once before variable_inertia=True in the SRB:
#   conda run -n env_sbo python ik/sample_inertia_workspace.py
#
##

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pinocchio as pin
from tqdm import tqdm

from utils.kinematics.g1_ik import G1IK

_OUT_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_URDF_PATH = os.path.join(_REPO_ROOT, "models", "g1", "g1_29dof_rev_1_0.urdf")
_PKG_DIRS  = [os.path.join(_REPO_ROOT, "models", "g1")]
os.makedirs(_OUT_DIR, exist_ok=True)

# ── robot model ────────────────────────────────────────────────────────────────

ik    = G1IK()
model = ik.model
data  = model.createData()

l_foot_id = model.getFrameId(G1IK.L_FOOT)
r_foot_id = model.getFrameId(G1IK.R_FOOT)

# ── collision model ────────────────────────────────────────────────────────────

print("Building collision geometry from URDF ...")
geom_model = pin.GeometryModel()
pin.buildGeomFromUrdf(model, _URDF_PATH, pin.COLLISION, geom_model, _PKG_DIRS)
print(f"  {geom_model.ngeoms} collision geometries loaded.")
print("Adding non-adjacent collision pairs ...")
geom_model.addAllCollisionPairs()

def _ancestors(model, joint_id):
    ids = set()
    j = joint_id
    while j > 0:
        ids.add(j)
        j = model.parents[j]
    ids.add(0)
    return ids

pairs_to_remove = []
for k, pair in enumerate(geom_model.collisionPairs):
    j1 = geom_model.geometryObjects[pair.first].parentJoint
    j2 = geom_model.geometryObjects[pair.second].parentJoint
    if j1 == j2 or model.parents[j1] == j2 or model.parents[j2] == j1:
        pairs_to_remove.append(k)

for k in reversed(pairs_to_remove):
    geom_model.removeCollisionPair(geom_model.collisionPairs[k])

print(f"  {len(geom_model.collisionPairs)} non-adjacent collision pairs active "
      f"({len(pairs_to_remove)} adjacent pairs removed).")
geom_data = geom_model.createData()
print("Ready.\n")

# ── leg joint layout ───────────────────────────────────────────────────────────

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
    jid = model.getJointId(name)
    qi  = model.joints[jid].idx_q
    leg_qidx.append(qi)
    leg_lo.append(model.lowerPositionLimit[qi])
    leg_hi.append(model.upperPositionLimit[qi])

leg_lo = np.array(leg_lo)
leg_hi = np.array(leg_hi)

# ── tuck joint classification ──────────────────────────────────────────────────
# Primary: hip_pitch, knee, ankle_pitch — driven deterministically by tuck t.
# Secondary: hip_roll, hip_yaw, ankle_roll — randomly sampled at each tuck level.

_PRIM_KEYS = ("hip_pitch", "knee", "ankle_pitch")
_SEC_KEYS  = ("hip_roll",  "hip_yaw", "ankle_roll")

prim_leg_idx = [i for i, n in enumerate(LEG_JOINTS) if any(k in n for k in _PRIM_KEYS)]
sec_leg_idx  = [i for i, n in enumerate(LEG_JOINTS) if any(k in n for k in _SEC_KEYS)]

prim_qidx = [leg_qidx[i] for i in prim_leg_idx]
sec_qidx  = [leg_qidx[i] for i in sec_leg_idx]
sec_lo    = leg_lo[sec_leg_idx]
sec_hi    = leg_hi[sec_leg_idx]

# Tuck endpoints for each primary joint (extended → tucked):
#   hip_pitch  : 0 → min(URDF upper, 2.0 rad)   — pull legs toward chest
#   knee       : 0 → min(URDF upper, 2.5 rad)   — bend knee tight
#   ankle_pitch: 0 → max(URDF lower, -0.5 rad)  — dorsiflexion (toes up)
prim_start = np.zeros(len(prim_leg_idx))
prim_end   = np.zeros(len(prim_leg_idx))
for k, i in enumerate(prim_leg_idx):
    name = LEG_JOINTS[i]
    lo_i, hi_i = leg_lo[i], leg_hi[i]
    if "hip_pitch"   in name:
        prim_end[k] = min(hi_i,  2.0)
    elif "knee"      in name:
        prim_end[k] = min(hi_i,  2.5)
    elif "ankle_pitch" in name:
        prim_end[k] = max(lo_i, -0.5)

print("Tuck joint classification:")
print("  Primary  (driven by t):", [LEG_JOINTS[i] for i in prim_leg_idx])
print("  Secondary (random):    ", [LEG_JOINTS[i] for i in sec_leg_idx])
print()
print("  Primary tuck endpoints (t=0 → t=1):")
for k, i in enumerate(prim_leg_idx):
    print(f"    {LEG_JOINTS[i]:<35s}: {prim_start[k]:+.3f} → {prim_end[k]:+.3f} rad")
print()

# ── base configuration ─────────────────────────────────────────────────────────

q_base = pin.neutral(model)
q_base[2] = 0.77
q_base[3:7] = [0.0, 0.0, 0.0, 1.0]

v0 = np.zeros(model.nv)

COMPONENTS = ["Ixx", "Iyy", "Izz", "Ixy", "Ixz", "Iyz"]

# ══════════════════════════════════════════════════════════════════════════════
# 1. TUCK SWEEP
# ══════════════════════════════════════════════════════════════════════════════

N_TUCK = 25    # tuck levels in [0, 1]
N_SEC  = 300   # random secondary-DOF samples per tuck level

tuck_params = np.linspace(0.0, 1.0, N_TUCK)
rng_tuck    = np.random.default_rng(123)

# Per-tuck-level statistics
curve_Iyy_mean       = np.zeros(N_TUCK)
curve_Iyy_std        = np.zeros(N_TUCK)
curve_Iyy_min        = np.full(N_TUCK,  np.inf)
curve_Iyy_max        = np.full(N_TUCK, -np.inf)
curve_foot_dist_mean = np.zeros(N_TUCK)
curve_foot_pos_L     = np.zeros((N_TUCK, 3))
curve_foot_pos_R     = np.zeros((N_TUCK, 3))
curve_n_valid        = np.zeros(N_TUCK, dtype=int)
curve_prim_angles    = np.zeros((N_TUCK, len(prim_leg_idx)))

# Raw scatter for fitting / plotting
scatter_tuck    = []
scatter_inertia = []
scatter_foot_L  = []
scatter_foot_R  = []

print(f"Tuck sweep: {N_TUCK} levels × {N_SEC} secondary samples = "
      f"{N_TUCK * N_SEC:,} configs\n")

for ti, t in enumerate(tqdm(tuck_params, desc="Tuck sweep", unit="level")):

    prim_angles = prim_start + t * (prim_end - prim_start)
    curve_prim_angles[ti] = prim_angles

    sec_samples = rng_tuck.uniform(sec_lo, sec_hi, size=(N_SEC, len(sec_leg_idx)))

    i6_vals     = []
    foot_L_vals = []
    foot_R_vals = []

    for sec in sec_samples:
        q = q_base.copy()
        for qi, angle in zip(prim_qidx, prim_angles):
            q[qi] = angle
        for qi, angle in zip(sec_qidx, sec):
            q[qi] = angle

        pin.computeCollisions(model, data, geom_model, geom_data, q, True)
        if any(geom_data.collisionResults[k].isCollision()
               for k in range(len(geom_model.collisionPairs))):
            continue

        pin.ccrba(model, data, q, v0)
        pin.updateFramePlacements(model, data)

        I_G = np.array(data.Ig.inertia)
        i6  = np.array([
            I_G[0, 0], I_G[1, 1], I_G[2, 2],
            I_G[0, 1], I_G[0, 2], I_G[1, 2],
        ])

        p_pelvis = data.oMi[1].translation
        p_l = data.oMf[l_foot_id].translation - p_pelvis
        p_r = data.oMf[r_foot_id].translation - p_pelvis

        i6_vals.append(i6)
        foot_L_vals.append(p_l.copy())
        foot_R_vals.append(p_r.copy())

    n = len(i6_vals)
    curve_n_valid[ti] = n
    if n == 0:
        continue

    i6_arr     = np.array(i6_vals)
    foot_L_arr = np.array(foot_L_vals)
    foot_R_arr = np.array(foot_R_vals)
    Iyy_arr    = i6_arr[:, 1]

    curve_Iyy_mean[ti]       = Iyy_arr.mean()
    curve_Iyy_std[ti]        = Iyy_arr.std()
    curve_Iyy_min[ti]        = Iyy_arr.min()
    curve_Iyy_max[ti]        = Iyy_arr.max()
    curve_foot_pos_L[ti]     = foot_L_arr.mean(axis=0)
    curve_foot_pos_R[ti]     = foot_R_arr.mean(axis=0)
    curve_foot_dist_mean[ti] = np.minimum(
        np.linalg.norm(foot_L_arr, axis=1),
        np.linalg.norm(foot_R_arr, axis=1),
    ).mean()

    scatter_tuck.append(np.full(n, t))
    scatter_inertia.append(i6_arr)
    scatter_foot_L.append(foot_L_arr)
    scatter_foot_R.append(foot_R_arr)

# ── save tuck curve ────────────────────────────────────────────────────────────

curve_path = os.path.join(_OUT_DIR, "inertia_tuck_curve.npz")
np.savez(
    curve_path,
    tuck            = tuck_params,
    Iyy_mean        = curve_Iyy_mean,
    Iyy_std         = curve_Iyy_std,
    Iyy_min         = curve_Iyy_min,
    Iyy_max         = curve_Iyy_max,
    foot_dist_mean  = curve_foot_dist_mean,
    foot_pos_L_mean = curve_foot_pos_L,
    foot_pos_R_mean = curve_foot_pos_R,
    prim_angles     = curve_prim_angles,
    n_valid         = curve_n_valid,
)

scatter_path = os.path.join(_OUT_DIR, "inertia_tuck_scatter.npz")
np.savez(
    scatter_path,
    tuck       = np.concatenate(scatter_tuck),
    inertia    = np.vstack(scatter_inertia),
    foot_pos_L = np.vstack(scatter_foot_L),
    foot_pos_R = np.vstack(scatter_foot_R),
)

print(f"\nTuck curve  → {curve_path}")
print(f"Tuck scatter→ {scatter_path}")
print()
print(f"  {'t':>5}  {'n_valid':>7}  {'Iyy_mean':>10}  {'Iyy_min':>9}  "
      f"{'Iyy_max':>9}  {'foot_dist':>9}")
for ti, t in enumerate(tuck_params):
    print(f"  {t:5.2f}  {curve_n_valid[ti]:7d}  "
          f"{curve_Iyy_mean[ti]:10.4f}  {curve_Iyy_min[ti]:9.4f}  "
          f"{curve_Iyy_max[ti]:9.4f}  {curve_foot_dist_mean[ti]:9.4f}m")

# ══════════════════════════════════════════════════════════════════════════════
# 2. SYMMETRIC SWEEP (secondary DOFs = 0 — used by srb_aerial.py)
# ══════════════════════════════════════════════════════════════════════════════
# Deterministic single-pass sweep: primary tuck joints driven by t, all
# secondary joints (hip_roll, hip_yaw, ankle_roll) held at zero.
# Gives a clean monotone I(t) curve for the backflip configuration where
# the legs tuck symmetrically without lateral spread.  This is the curve
# consumed by the centroidal optimizer for the tuck → inertia coupling.

print("\nSymmetric sweep (secondary = 0) ...")

sym_Ixx = np.zeros(N_TUCK)
sym_Iyy = np.zeros(N_TUCK)
sym_Izz = np.zeros(N_TUCK)
sym_foot_L = np.zeros((N_TUCK, 3))
sym_foot_R = np.zeros((N_TUCK, 3))

for ti, t in enumerate(tuck_params):
    prim_angles = prim_start + t * (prim_end - prim_start)

    q = q_base.copy()
    for qi, angle in zip(prim_qidx, prim_angles):
        q[qi] = angle
    # secondary joints stay at neutral (zero) — q_base already zeros them

    pin.ccrba(model, data, q, v0)
    pin.updateFramePlacements(model, data)

    I_G = np.array(data.Ig.inertia)
    sym_Ixx[ti] = I_G[0, 0]
    sym_Iyy[ti] = I_G[1, 1]
    sym_Izz[ti] = I_G[2, 2]

    p_pelvis = data.oMi[1].translation
    sym_foot_L[ti] = data.oMf[l_foot_id].translation - p_pelvis
    sym_foot_R[ti] = data.oMf[r_foot_id].translation - p_pelvis

sym_path = os.path.join(_OUT_DIR, "inertia_tuck_sym.npz")
np.savez(
    sym_path,
    tuck          = tuck_params,
    Ixx_sym       = sym_Ixx,
    Iyy_sym       = sym_Iyy,
    Izz_sym       = sym_Izz,
    foot_pos_L_sym = sym_foot_L,
    foot_pos_R_sym = sym_foot_R,
)

print(f"Symmetric curve → {sym_path}")
print(f"  Iyy: t=0 → {sym_Iyy[0]:.4f},  t=1 → {sym_Iyy[-1]:.4f},  "
      f"delta = {sym_Iyy[-1] - sym_Iyy[0]:+.4f} kg·m²")

# ══════════════════════════════════════════════════════════════════════════════
# 3. RANDOM SAMPLING (overall inertia bounds)
# ══════════════════════════════════════════════════════════════════════════════

N_SAMPLES = 50_000
rng = np.random.default_rng(42)

I_min = np.full(6,  np.inf)
I_max = np.full(6, -np.inf)

all_inertia   = np.zeros((N_SAMPLES, 6))
all_foot_dist = np.zeros(N_SAMPLES)

print(f"\nRandom sampling: {N_SAMPLES:,} collision-free leg configurations ...")

accepted = 0
rejected = 0

pbar = tqdm(total=N_SAMPLES, unit="sample",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]  {postfix}")

while accepted < N_SAMPLES:
    q = q_base.copy()
    q_legs = rng.uniform(leg_lo, leg_hi)
    for j, qi in enumerate(leg_qidx):
        q[qi] = q_legs[j]

    pin.computeCollisions(model, data, geom_model, geom_data, q, True)
    if any(geom_data.collisionResults[k].isCollision()
           for k in range(len(geom_model.collisionPairs))):
        rejected += 1
        continue

    pin.ccrba(model, data, q, v0)
    pin.updateFramePlacements(model, data)

    I_G = np.array(data.Ig.inertia)
    i6  = np.array([
        I_G[0, 0], I_G[1, 1], I_G[2, 2],
        I_G[0, 1], I_G[0, 2], I_G[1, 2],
    ])

    p_pelvis = data.oMi[1].translation
    p_l = data.oMf[l_foot_id].translation
    p_r = data.oMf[r_foot_id].translation
    foot_dist = min(np.linalg.norm(p_l - p_pelvis),
                    np.linalg.norm(p_r - p_pelvis))

    I_min = np.minimum(I_min, i6)
    I_max = np.maximum(I_max, i6)
    all_inertia[accepted]   = i6
    all_foot_dist[accepted] = foot_dist
    accepted += 1

    pbar.update(1)
    if accepted % 5_000 == 0:
        pbar.set_postfix(
            rejected=f"{rejected}({100*rejected/(accepted+rejected):.0f}%)",
            Iyy=f"[{I_min[1]:.3f},{I_max[1]:.3f}]",
            foot=f"[{all_foot_dist[:accepted].min():.2f},{all_foot_dist[:accepted].max():.2f}]m",
        )

pbar.close()
print(f"\nDone. Accepted {accepted:,} / {accepted+rejected:,} "
      f"({100*rejected/(accepted+rejected):.1f}% collision-rejected).")

# ── save random sampling outputs ───────────────────────────────────────────────

bounds_path = os.path.join(_OUT_DIR, "inertia_bounds.csv")
with open(bounds_path, "w") as f:
    f.write("# component,min,max  (kg·m²)\n")
    for comp, lo, hi in zip(COMPONENTS, I_min, I_max):
        f.write(f"{comp},{lo:.10f},{hi:.10f}\n")

npz_path = os.path.join(_OUT_DIR, "inertia_samples.npz")
np.savez(npz_path, inertia=all_inertia, foot_dist=all_foot_dist)

print(f"\nBounds saved  → {bounds_path}")
print(f"Samples saved → {npz_path}")
print("\nSummary (random bounds):")
print(f"  {'Component':<8}  {'Min (kg·m²)':>12}  {'Max (kg·m²)':>12}  {'Range':>10}")
for comp, lo, hi in zip(COMPONENTS, I_min, I_max):
    print(f"  {comp:<8}  {lo:>12.6f}  {hi:>12.6f}  {hi-lo:>10.6f}")
print(f"\n  foot_dist  {all_foot_dist.min():>12.4f}m  {all_foot_dist.max():>12.4f}m")
