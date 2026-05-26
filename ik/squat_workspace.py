##
#
# Sagittal-plane CoM workspace boundary sweep for the G1 robot.
#
# Sweeps (x_com, pitch) jointly to find pz_min(x, pitch) — the minimum
# reachable CoM z at each (x, body-pitch) combination with feet fixed flat
# at x=0.  Uses bisection per (x, pitch)-slice with multiple initial guesses.
#
# Outputs:
#   ik/results/workspace_boundary_2d.csv      — raw (x, pitch, pz_min) data
#   ik/results/workspace_poly_coeffs_2d.csv   — degree-3 surface fit
#   ik/results/workspace_sagittal_2d.png      — heatmap + slice plots
#
# Also keeps the original 1D outputs (pitch-marginalised) for comparison:
#   ik/results/workspace_boundary.csv
#   ik/results/workspace_poly_coeffs.csv
#   ik/results/workspace_sagittal.png
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

# Pitch grid: covers stance wind-up (positive) and landing backward tilt (negative)
PITCHES_SWEEP = np.linspace(-0.40, 0.40, 9)   # rad

# ── helpers ───────────────────────────────────────────────────────────────────

def _set_pitch(q, pitch_rad):
    q[3] = 0.0
    q[4] = np.sin(pitch_rad / 2.0)
    q[5] = 0.0
    q[6] = np.cos(pitch_rad / 2.0)
    return q


def _guesses_fixed_pitch(x, z, fixed_pitch):
    """
    Three kinematic strategies (standard, deep-squat, max-knee) all pinned to
    fixed_pitch.  Unlike the old helper, pitch is not varied — it is fixed by
    the caller so the IK checks feasibility at exactly that orientation.
    """
    NOMINAL_H = 0.79
    drop = max(0.0, NOMINAL_H - z)

    # Strategy A: standard squat bias
    qA = ik.standing_config(com_height=z)
    qA[0] = x

    # Strategy B: deep-squat bias
    qB = pin.neutral(ik.model)
    qB[0] = x
    qB[2] = z
    knee_B  = np.clip(drop * 4.5, 0.10, np.radians(155))
    hip_p_B = np.clip(drop * 2.5, 0.0,  np.radians(120))
    ankle_scale = np.clip(1.0 + x / 0.25, 0.0, 2.0)
    ankle_B = np.clip(-drop * 1.5 * ankle_scale, np.radians(-45), 0.0)
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

    # Strategy C: maximum-knee bias
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
        q = q_base.copy()
        _set_pitch(q, fixed_pitch)
        guesses.append(q)
    return guesses


def is_reachable_pitched(x, z, fixed_pitch, warm=None):
    """Returns (reachable: bool, q_sol or None) with body pitch fixed."""
    candidates = []
    if warm is not None:
        w = warm.copy()
        w[0] = x
        w[2] = z
        _set_pitch(w, fixed_pitch)
        candidates.append(w)
    candidates.extend(_guesses_fixed_pitch(x, z, fixed_pitch))

    for q0 in candidates:
        q_sol, ok, _ = ik.solve(q0, oMl_des, oMr_des, max_iter=200, tol=1e-5)
        if ok:
            return True, q_sol
    return False, None

# ── sweep parameters ──────────────────────────────────────────────────────────

x_vals  = np.linspace(-0.20, 0.30, 75)
z_floor = 0.22
N_BIS   = 8    # bisection depth → ~2.5 mm resolution

L_REACH = 0.78
def z_high_for_x(x):
    inner = L_REACH**2 - x**2 - G1IK.HIP_WIDTH**2
    if inner <= 0:
        return None
    return np.sqrt(inner) + G1IK.ANKLE_HEIGHT - 0.01

# ── 2D sweep ──────────────────────────────────────────────────────────────────

bx2d, bp2d, bz2d = [], [], []   # non-censored points for 2D fit

total_slices = len(PITCHES_SWEEP) * len(x_vals)
done = 0

print(f"2D bisection sweep: {len(PITCHES_SWEEP)} pitch slices × {len(x_vals)} x-slices")
print(f"  z_floor={z_floor:.2f}  N_BIS={N_BIS}  resolution ~{(0.85-z_floor)/2**N_BIS*1000:.1f} mm\n")

for pi, pitch in enumerate(PITCHES_SWEEP):
    warm_prev = None
    print(f"── pitch = {np.degrees(pitch):+.1f}° ──")

    for xi, x in enumerate(x_vals):
        done += 1
        pct = done / total_slices * 100

        z_high = z_high_for_x(x)
        if z_high is None or z_high < z_floor + 0.05:
            continue

        ok_high, q_high = is_reachable_pitched(x, z_high, pitch, warm=warm_prev)
        if not ok_high:
            found = False
            for z_try in np.arange(z_high - 0.02, z_floor, -0.02):
                ok_try, q_try = is_reachable_pitched(x, z_try, pitch, warm=warm_prev)
                if ok_try:
                    z_high, q_high = z_try, q_try
                    found = True
                    break
            if not found:
                continue

        ok_floor, q_floor = is_reachable_pitched(x, z_floor, pitch, warm=q_high)
        if ok_floor:
            # censored — robot can go below z_floor; skip for fit
            warm_prev = q_floor
            print(f"  x={x:+.3f}  pz_min<={z_floor:.3f} m (floor)  [{pct:.0f}%]")
            continue

        z_lo, z_hi = z_floor, z_high
        q_best = q_high
        for _ in range(N_BIS):
            z_mid = 0.5 * (z_lo + z_hi)
            ok_mid, q_mid = is_reachable_pitched(x, z_mid, pitch, warm=q_best)
            if ok_mid:
                z_hi   = z_mid
                q_best = q_mid
            else:
                z_lo = z_mid

        bx2d.append(x)
        bp2d.append(pitch)
        bz2d.append(z_hi)
        warm_prev = q_best
        print(f"  x={x:+.3f}  pz_min={z_hi:.4f} m  [{pct:.0f}%]")

bx2d = np.array(bx2d)
bp2d = np.array(bp2d)
bz2d = np.array(bz2d)

csv2d_path = os.path.join(_OUT_DIR, "workspace_boundary_2d.csv")
np.savetxt(csv2d_path, np.column_stack([bx2d, bp2d, bz2d]),
           delimiter=",", header="x_com,pitch_rad,pz_min", comments="")
print(f"\n2D boundary saved → {csv2d_path}  ({len(bz2d)} non-censored points)")

# ── 2D polynomial fit ─────────────────────────────────────────────────────────

deg2d = 3
monomials = [(i, j)
             for d in range(deg2d + 1)
             for i in range(d + 1)
             for j in range(d - i + 1)]

A2d = np.column_stack([bx2d**i * bp2d**j for (i, j) in monomials])
coeffs2d, res2d, _, _ = np.linalg.lstsq(A2d, bz2d, rcond=None)

z_fit2d = A2d @ coeffs2d
rmse2d  = np.sqrt(np.mean((z_fit2d - bz2d)**2))
maxe2d  = np.max(np.abs(z_fit2d - bz2d))

print(f"\n2D polynomial fit (total degree {deg2d}, {len(monomials)} terms):")
print(f"  RMSE={rmse2d*1000:.1f} mm   max_err={maxe2d*1000:.1f} mm")

# Save: header row, then i, j, coeff per term
coeff2d_path = os.path.join(_OUT_DIR, "workspace_poly_coeffs_2d.csv")
with open(coeff2d_path, "w") as f:
    f.write(f"# degree-{deg2d} polynomial in (x_com, pitch_rad); columns: i, j, coeff\n")
    for (i, j), c in zip(monomials, coeffs2d):
        f.write(f"{i},{j},{c:.10f}\n")
print(f"  2D coefficients saved → {coeff2d_path}")

# ── 1D marginalised boundary (pitch=0 slice + envelope) ──────────────────────
# Keep the 1D outputs for backward compat and plotting comparisons.

# Envelope: for each x, take the max pz_min over all pitch slices (worst case).
x_unique = np.unique(bx2d)
bx1d, bz1d = [], []
for xv in x_unique:
    mask = np.abs(bx2d - xv) < 1e-6
    if mask.sum() > 0:
        bx1d.append(xv)
        bz1d.append(np.max(bz2d[mask]))   # worst-case (max) across pitches

bx1d = np.array(bx1d)
bz1d = np.array(bz1d)

csv1d_path = os.path.join(_OUT_DIR, "workspace_boundary.csv")
np.savetxt(csv1d_path, np.column_stack([bx1d, bz1d]),
           delimiter=",", header="x_com,pz_min", comments="")

deg1d  = 4
c1d    = np.polyfit(bx1d, bz1d, deg1d)
z1d_fit = np.polyval(c1d, bx1d)
rmse1d  = np.sqrt(np.mean((z1d_fit - bz1d)**2))
maxe1d  = np.max(np.abs(z1d_fit - bz1d))
print(f"\n1D envelope poly (deg {deg1d}): RMSE={rmse1d*1000:.1f} mm  max={maxe1d*1000:.1f} mm")
print(f"  coeffs (high→low): {c1d}")

coeff1d_path = os.path.join(_OUT_DIR, "workspace_poly_coeffs.csv")
np.savetxt(coeff1d_path, c1d, delimiter=",",
           header=f"degree-{deg1d} poly coeffs for pz_min(x), highest power first")
print(f"  1D coefficients saved → {coeff1d_path}")

# ── load SRB stance trajectory ────────────────────────────────────────────────

_RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "results", "srb", "srb_aerial")
srb_x, srb_z = None, None
if os.path.exists(os.path.join(_RESULTS, "q_opt.csv")):
    q_opt = np.loadtxt(os.path.join(_RESULTS, "q_opt.csv"), delimiter=",")
    N_stance = 25
    srb_x = q_opt[:N_stance + 1, 0]
    srb_z = q_opt[:N_stance + 1, 2]

# ── plots ─────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Left: heatmap of pz_min(x, pitch) from the 2D fit
x_grid = np.linspace(-0.20, 0.30, 120)
p_grid = np.linspace(-0.40, 0.40, 80)
Xg, Pg = np.meshgrid(x_grid, p_grid)
Xgf, Pgf = Xg.ravel(), Pg.ravel()
A_grid = np.column_stack([Xgf**i * Pgf**j for (i, j) in monomials])
Zg = (A_grid @ coeffs2d).reshape(Xg.shape)

ax = axes[0]
cm = ax.contourf(x_grid, np.degrees(p_grid), Zg, levels=20, cmap="viridis")
fig.colorbar(cm, ax=ax, label="pz_min (m)")
ax.scatter(bx2d, np.degrees(bp2d), c=bz2d, cmap="viridis",
           s=8, edgecolors="k", linewidths=0.3, label="IK bisection pts")
ax.set_xlabel("CoM x (m)")
ax.set_ylabel("Body pitch (deg)")
ax.set_title(f"2D workspace boundary  RMSE={rmse2d*1000:.1f} mm")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2)

# Right: x-z slices at several pitches
ax = axes[1]
colors = plt.cm.coolwarm(np.linspace(0, 1, len(PITCHES_SWEEP)))
x_plot = np.linspace(-0.20, 0.30, 200)

for pi, (pitch, col) in enumerate(zip(PITCHES_SWEEP, colors)):
    p_arr = np.full_like(x_plot, pitch)
    A_line = np.column_stack([x_plot**i * p_arr**j for (i, j) in monomials])
    z_line = A_line @ coeffs2d
    ax.plot(x_plot, z_line, color=col, linewidth=1.5,
            label=f"{np.degrees(pitch):+.0f}°")

x_env_plot = np.linspace(bx1d.min(), bx1d.max(), 200)
ax.plot(x_env_plot, np.polyval(c1d, x_env_plot), "k--", linewidth=2,
        label="1D envelope (worst-case)")

if srb_x is not None:
    ax.plot(srb_x, srb_z, "o-", color="darkorange", markersize=4,
            linewidth=1.5, label="SRB stance traj")

ax.set_xlabel("CoM x (m)  [+ = forward of feet]")
ax.set_ylabel("pz_min (m)")
ax.set_title("pz_min(x) slices by pitch angle")
ax.legend(fontsize=8, ncol=2)
ax.grid(True, alpha=0.25)
ax.set_ylim(0.20, 0.90)

fig.tight_layout()
plot_path = os.path.join(_OUT_DIR, "workspace_sagittal_2d.png")
fig.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"\nPlot saved → {plot_path}")
