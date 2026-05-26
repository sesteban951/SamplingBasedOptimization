##
#
# Sagittal-plane CoM workspace boundary sweep for the G1 robot.
#
# Finds the minimum reachable CoM z at each CoM x, with feet fixed flat
# at x=0. Uses bisection per x-slice with multiple initial guesses and
# body pitches so deep-squat configurations (knee->165°) aren't missed.
# Saves the boundary as CSV and fits a polynomial for use in srb_aerial.py.
#
# Usage:
#   conda run -n env_sbo python ik/squat_workspace.py
#
##

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pinocchio as pin
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.kinematics.g1_ik import G1IK

_OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(_OUT_DIR, exist_ok=True)

# ── IK setup ──────────────────────────────────────────────────────────────────

ik = G1IK()

R_flat  = np.eye(3)
oMl_des = pin.SE3(R_flat, np.array([0.0,  G1IK.HIP_WIDTH, G1IK.ANKLE_HEIGHT]))
oMr_des = pin.SE3(R_flat, np.array([0.0, -G1IK.HIP_WIDTH, G1IK.ANKLE_HEIGHT]))

MAX_PITCH_RAD = 0.40   # matches smalljump stance_rotation_allow
PITCHES       = [0.0, 0.10, 0.20, 0.30, MAX_PITCH_RAD]

# ── initial-guess helpers ─────────────────────────────────────────────────────

def _set_pitch(q, pitch_rad):
    q[3] = 0.0
    q[4] = np.sin(pitch_rad / 2.0)
    q[5] = 0.0
    q[6] = np.cos(pitch_rad / 2.0)
    return q


def _guesses(x, z):
    """
    Return a list of initial configurations to try at (x, z).

    Three strategies cover the main kinematic branches:
      1. standing_config — good for moderate squats near x=0
      2. deep-squat bias — high knee, low ankle; needed at x<0 where the
         ankle stays plantarflexed and the knee goes past 90°
      3. backward-lean bias — less ankle, more hip, for negative-x poses
    Each strategy is then repeated for every body pitch in PITCHES.
    """
    NOMINAL_H = 0.79
    drop = max(0.0, NOMINAL_H - z)

    # --- strategy A: standard squat bias (standing_config logic) ---
    qA = ik.standing_config(com_height=z)
    qA[0] = x

    # --- strategy B: deep-squat bias ---
    qB = pin.neutral(ik.model)
    qB[0] = x
    qB[2] = z
    knee_B   = np.clip(drop * 4.5, 0.10, np.radians(155))
    hip_p_B  = np.clip(drop * 2.5, 0.0,  np.radians(120))
    # at negative x the shin leans back → less dorsiflexion needed
    ankle_scale = 1.0 + x / 0.25   # ~0 at x=-0.25, ~1 at x=0, ~1.8 at x=+0.20
    ankle_scale = np.clip(ankle_scale, 0.0, 2.0)
    ankle_B  = np.clip(-drop * 1.5 * ankle_scale, np.radians(-45), 0.0)
    for name, angle in [
        ("left_hip_pitch_joint",   hip_p_B),
        ("right_hip_pitch_joint",  hip_p_B),
        ("left_knee_joint",        knee_B),
        ("right_knee_joint",       knee_B),
        ("left_ankle_pitch_joint", ankle_B),
        ("right_ankle_pitch_joint",ankle_B),
    ]:
        jid = ik.model.getJointId(name)
        qB[ik.model.joints[jid].idx_q] = angle

    # --- strategy C: maximum-knee bias (thigh-to-calf) ---
    qC = pin.neutral(ik.model)
    qC[0] = x
    qC[2] = z
    for name, angle in [
        ("left_hip_pitch_joint",   np.radians(100)),
        ("right_hip_pitch_joint",  np.radians(100)),
        ("left_knee_joint",        np.radians(145)),
        ("right_knee_joint",       np.radians(145)),
        ("left_ankle_pitch_joint", np.radians(-20)),
        ("right_ankle_pitch_joint",np.radians(-20)),
    ]:
        jid = ik.model.getJointId(name)
        qC[ik.model.joints[jid].idx_q] = angle

    guesses = []
    for q_base in (qA, qB, qC):
        for p in PITCHES:
            q = q_base.copy()
            _set_pitch(q, p)
            guesses.append(q)
    return guesses


def is_reachable(x, z, warm=None):
    """
    Returns (reachable: bool, q_sol or None).
    Tries all guesses; warm-start (if given) is tried first.
    """
    candidates = []
    if warm is not None:
        w = warm.copy()
        w[0] = x
        w[2] = z
        candidates.append(w)
    candidates.extend(_guesses(x, z))

    for q0 in candidates:
        q_sol, ok, _ = ik.solve(q0, oMl_des, oMr_des,
                                max_iter=200, tol=1e-5)
        if ok:
            return True, q_sol
    return False, None

# ── bisection sweep ───────────────────────────────────────────────────────────

x_vals  = np.linspace(-0.20, 0.30, 75)
z_floor = 0.22   # absolute lower bound for search
N_BIS   = 8      # bisection depth → resolution ~2.5 mm

# Maximum z reachable at each x is set by leg-reach geometry:
#   z_reach = sqrt(L_reach^2 - x^2 - HIP_WIDTH^2) + ANKLE_HEIGHT
# Use L_reach slightly below the URDF max to leave IK headroom.
L_REACH = 0.78
def z_high_for_x(x):
    inner = L_REACH**2 - x**2 - G1IK.HIP_WIDTH**2
    if inner <= 0:
        return None
    return np.sqrt(inner) + G1IK.ANKLE_HEIGHT - 0.01  # small margin

boundary_x   = []
boundary_z   = []
warm_prev    = None   # warm-start carried across x slices

print(f"Bisection sweep: {len(x_vals)} x-slices, {N_BIS} bisection steps each")
print(f"  z_floor={z_floor:.2f}  z_high is geometry-adaptive  resolution ~{(0.85 - z_floor) / 2**N_BIS * 1000:.1f} mm\n")

for i, x in enumerate(x_vals):
    z_high = z_high_for_x(x)
    if z_high is None or z_high < z_floor + 0.05:
        print(f"  x={x:+.3f}  geometry says leg can't reach here, skipping")
        continue

    ok_high, q_high = is_reachable(x, z_high, warm=warm_prev)
    if not ok_high:
        # step z_high down until we find a reachable pose
        found = False
        for z_try in np.arange(z_high - 0.02, z_floor, -0.02):
            ok_try, q_try = is_reachable(x, z_try, warm=warm_prev)
            if ok_try:
                z_high, q_high = z_try, q_try
                found = True
                break
        if not found:
            print(f"  x={x:+.3f}  no reachable z_high found, skipping")
            continue

    # check if floor is reachable (skip bisection if so)
    ok_floor, q_floor = is_reachable(x, z_floor, warm=q_high)
    if ok_floor:
        boundary_x.append(x)
        boundary_z.append(z_floor)
        warm_prev = q_floor
        pct = (i + 1) / len(x_vals) * 100
        print(f"  x={x:+.3f}  pz_min<={z_floor:.3f} m (floor reached)  [{pct:.0f}%]")
        continue

    # bisect between z_floor (unreachable) and z_high (reachable)
    z_lo, z_hi = z_floor, z_high
    q_best = q_high
    for _ in range(N_BIS):
        z_mid = 0.5 * (z_lo + z_hi)
        ok_mid, q_mid = is_reachable(x, z_mid, warm=q_best)
        if ok_mid:
            z_hi   = z_mid
            q_best = q_mid
        else:
            z_lo = z_mid

    boundary_x.append(x)
    boundary_z.append(z_hi)
    warm_prev = q_best
    pct = (i + 1) / len(x_vals) * 100
    print(f"  x={x:+.3f}  pz_min={z_hi:.4f} m  [{pct:.0f}%]")

boundary_x = np.array(boundary_x)
boundary_z = np.array(boundary_z)

# save raw boundary
csv_path = os.path.join(_OUT_DIR, "workspace_boundary.csv")
np.savetxt(csv_path, np.column_stack([boundary_x, boundary_z]),
           delimiter=",", header="x_com,pz_min", comments="")
print(f"\nBoundary saved → {csv_path}")

# ── polynomial fit ────────────────────────────────────────────────────────────

# Fit over the region relevant to the SRB optimizer (x in [-0.25, 0.35]).
# Degree 4 balances accuracy vs. smoothness; check residuals.
deg  = 4
coeffs = np.polyfit(boundary_x, boundary_z, deg)
z_fit  = np.polyval(coeffs, boundary_x)
rmse   = np.sqrt(np.mean((z_fit - boundary_z) ** 2))
maxerr = np.max(np.abs(z_fit - boundary_z))

print(f"\nPolynomial fit (degree {deg}):")
print(f"  coefficients (highest power first): {coeffs}")
print(f"  RMSE={rmse*100:.1f} mm   max_err={maxerr*100:.1f} mm")

# save coefficients
coeff_path = os.path.join(_OUT_DIR, "workspace_poly_coeffs.csv")
np.savetxt(coeff_path, coeffs, delimiter=",",
           header=f"degree-{deg} poly coeffs for pz_min(x), highest power first")
print(f"  Coefficients saved → {coeff_path}")

# ── load SRB stance trajectory ────────────────────────────────────────────────

_RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "results", "srb", "srb_aerial")
srb_x, srb_z = None, None
if os.path.exists(os.path.join(_RESULTS, "q_opt.csv")):
    q_opt    = np.loadtxt(os.path.join(_RESULTS, "q_opt.csv"), delimiter=",")
    N_stance = 25
    srb_x    = q_opt[:N_stance + 1, 0]
    srb_z    = q_opt[:N_stance + 1, 2]

# ── plot ──────────────────────────────────────────────────────────────────────

x_plot = np.linspace(boundary_x.min(), boundary_x.max(), 300)
z_plot = np.polyval(coeffs, x_plot)

fig, ax = plt.subplots(figsize=(10, 6))

ax.fill_between(x_plot, z_plot, 0.90,
                alpha=0.25, color="green", label="Reachable (fit)")
ax.plot(boundary_x, boundary_z, "o", color="green", markersize=4,
        label="IK boundary (bisection)")
ax.plot(x_plot, z_plot, "-", color="darkgreen", linewidth=2,
        label=f"Poly fit deg={deg}  RMSE={rmse*100:.1f} mm")

if srb_x is not None:
    ax.plot(srb_x, srb_z, "o-", color="darkorange", markersize=4,
            linewidth=1.5, label="SRB stance traj (pre-constraint)")
    ax.scatter(srb_x[19], srb_z[19], color="red", s=80, zorder=6,
               label=f"peak ({srb_x[19]:.2f}, {srb_z[19]:.2f}) m")

ax.axvline(0.0, color="gray", linewidth=0.8, alpha=0.4)
ax.set_xlabel("CoM x (m)  [+ = forward of feet]")
ax.set_ylabel("CoM z (m)")
ax.set_title("G1 Sagittal Workspace Boundary\n"
             f"(feet flat at x=0, body pitch 0–{int(np.degrees(MAX_PITCH_RAD))}°, "
             f"bisection {N_BIS}-step)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.25)
ax.set_ylim(0.20, 0.92)

fig.tight_layout()
plot_path = os.path.join(_OUT_DIR, "workspace_sagittal.png")
fig.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"\nPlot saved → {plot_path}")
