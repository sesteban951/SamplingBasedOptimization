##
#
# Upper-bound CoM workspace sweep for the G1 robot.
#
# Mirror of squat_workspace.py: sweeps (x_rel, pitch) to find pz_max(x, pitch) —
# the maximum CoM z at which the robot can still place its feet flat on the floor.
# Above this height the legs are fully extended and IK fails.
#
# This bound is needed as an upper constraint in the SRB optimizer to prevent
# planning CoM heights that the real robot kinematics cannot achieve at touchdown
# (where feet may be offset forward from the CoM).
#
# Outputs:
#   ik/results/workspace_boundary_2d_upper.csv     — raw (x, pitch, pz_max) data
#   ik/results/workspace_poly_coeffs_2d_upper.csv  — degree-3 surface fit
#   ik/results/workspace_sagittal_2d_upper.png     — heatmap + slice plots
#
# Usage:
#   conda run -n env_sbo python ik/squat_workspace_upper.py
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

ik = G1IK()

R_flat  = np.eye(3)
oMl_des = pin.SE3(R_flat, np.array([0.0,  G1IK.HIP_WIDTH, G1IK.ANKLE_HEIGHT]))
oMr_des = pin.SE3(R_flat, np.array([0.0, -G1IK.HIP_WIDTH, G1IK.ANKLE_HEIGHT]))

PITCHES_SWEEP = np.linspace(-0.40, 0.40, 9)   # rad


def _set_pitch(q, pitch_rad):
    q[3] = 0.0
    q[4] = np.sin(pitch_rad / 2.0)
    q[5] = 0.0
    q[6] = np.cos(pitch_rad / 2.0)
    return q


def _guesses(x, z, pitch):
    NOMINAL_H = 0.79
    drop = max(0.0, NOMINAL_H - z)

    qA = ik.standing_config(com_height=z)
    qA[0] = x

    qB = pin.neutral(ik.model)
    qB[0] = x; qB[2] = z
    for name, angle in [
        ("left_hip_pitch_joint",    np.clip(drop * 1.5, 0, np.radians(90))),
        ("right_hip_pitch_joint",   np.clip(drop * 1.5, 0, np.radians(90))),
        ("left_knee_joint",         np.clip(drop * 3.0, 0.05, np.radians(155))),
        ("right_knee_joint",        np.clip(drop * 3.0, 0.05, np.radians(155))),
        ("left_ankle_pitch_joint",  np.clip(-drop * 1.5, np.radians(-45), 0)),
        ("right_ankle_pitch_joint", np.clip(-drop * 1.5, np.radians(-45), 0)),
    ]:
        jid = ik.model.getJointId(name)
        qB[ik.model.joints[jid].idx_q] = angle

    # near-full-extension guess (relevant for upper bound search)
    qC = ik.standing_config(com_height=z)
    qC[0] = x
    for name, angle in [
        ("left_hip_pitch_joint",   0.05),
        ("right_hip_pitch_joint",  0.05),
        ("left_knee_joint",        0.10),
        ("right_knee_joint",       0.10),
        ("left_ankle_pitch_joint", -0.05),
        ("right_ankle_pitch_joint",-0.05),
    ]:
        jid = ik.model.getJointId(name)
        qC[ik.model.joints[jid].idx_q] = angle

    return [_set_pitch(q.copy(), pitch) for q in (qA, qB, qC)]


def is_reachable(x, z, pitch, warm=None):
    candidates = []
    if warm is not None:
        w = warm.copy(); w[0] = x; w[2] = z
        _set_pitch(w, pitch)
        candidates.append(w)
    candidates.extend(_guesses(x, z, pitch))
    for q0 in candidates:
        q_sol, ok, _ = ik.solve(q0, oMl_des, oMr_des, max_iter=200, tol=1e-5)
        if ok:
            return True, q_sol
    return False, None


# geometric upper bound on z given leg reach L_REACH
L_REACH = 0.78
def z_geom_max(x):
    inner = L_REACH**2 - x**2 - G1IK.HIP_WIDTH**2
    if inner <= 0:
        return None
    return np.sqrt(inner) + G1IK.ANKLE_HEIGHT

# A safe low height where IK always succeeds (used as bisection lower anchor)
Z_LOW_ANCHOR = 0.55

x_vals  = np.linspace(-0.20, 0.30, 75)
N_BIS   = 8    # ~2.5 mm resolution

bx2d, bp2d, bz2d = [], [], []

total = len(PITCHES_SWEEP) * len(x_vals)
done  = 0

print(f"Upper-bound bisection sweep: {len(PITCHES_SWEEP)} pitch × {len(x_vals)} x slices")
print(f"  z_low={Z_LOW_ANCHOR:.2f}  N_BIS={N_BIS}  res ~{(0.85-Z_LOW_ANCHOR)/2**N_BIS*1000:.1f} mm\n")

for pitch in PITCHES_SWEEP:
    warm_prev = None
    print(f"── pitch = {np.degrees(pitch):+.1f}° ──")

    for x in x_vals:
        done += 1
        pct = done / total * 100

        z_high = z_geom_max(x)
        if z_high is None:
            continue

        # confirm z_low is feasible
        ok_low, q_low = is_reachable(x, Z_LOW_ANCHOR, pitch, warm=warm_prev)
        if not ok_low:
            print(f"  x={x:+.3f}  anchor infeasible — skip  [{pct:.0f}%]")
            continue

        # check if z_high is already infeasible (expected most of the time)
        ok_high, q_high = is_reachable(x, z_high, pitch, warm=q_low)
        if ok_high:
            # robot can reach even the geometric ceiling — record as censored
            warm_prev = q_high
            print(f"  x={x:+.3f}  pz_max>={z_high:.3f} m (ceiling)  [{pct:.0f}%]")
            bx2d.append(x); bp2d.append(pitch); bz2d.append(z_high)
            continue

        # bisect upward: z_lo is feasible, z_hi is infeasible
        z_lo, z_hi = Z_LOW_ANCHOR, z_high
        q_best = q_low
        for _ in range(N_BIS):
            z_mid = 0.5 * (z_lo + z_hi)
            ok_mid, q_mid = is_reachable(x, z_mid, pitch, warm=q_best)
            if ok_mid:
                z_lo   = z_mid   # feasible: can go higher
                q_best = q_mid
            else:
                z_hi   = z_mid   # infeasible: max is lower

        bx2d.append(x); bp2d.append(pitch); bz2d.append(z_lo)
        warm_prev = q_best
        print(f"  x={x:+.3f}  pz_max={z_lo:.4f} m  [{pct:.0f}%]")

bx2d = np.array(bx2d)
bp2d = np.array(bp2d)
bz2d = np.array(bz2d)

csv_path = os.path.join(_OUT_DIR, "workspace_boundary_2d_upper.csv")
np.savetxt(csv_path, np.column_stack([bx2d, bp2d, bz2d]),
           delimiter=",", header="x_com,pitch_rad,pz_max", comments="")
print(f"\nUpper boundary saved → {csv_path}  ({len(bz2d)} points)")

# ── polynomial fit ────────────────────────────────────────────────────────────

deg = 3
monomials = [(i, j)
             for d in range(deg + 1)
             for i in range(d + 1)
             for j in range(d - i + 1)]

A = np.column_stack([bx2d**i * bp2d**j for (i, j) in monomials])
coeffs, _, _, _ = np.linalg.lstsq(A, bz2d, rcond=None)

z_fit  = A @ coeffs
rmse   = np.sqrt(np.mean((z_fit - bz2d)**2))
maxe   = np.max(np.abs(z_fit - bz2d))
print(f"\nPolynomial fit (total degree {deg}, {len(monomials)} terms):")
print(f"  RMSE={rmse*1000:.1f} mm   max_err={maxe*1000:.1f} mm")

coeff_path = os.path.join(_OUT_DIR, "workspace_poly_coeffs_2d_upper.csv")
with open(coeff_path, "w") as f:
    f.write(f"# degree-{deg} polynomial in (x_com, pitch_rad); columns: i, j, coeff\n")
    for (i, j), c in zip(monomials, coeffs):
        f.write(f"{i},{j},{c:.10f}\n")
print(f"  Coefficients saved → {coeff_path}")

# ── plot ──────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

x_grid = np.linspace(-0.20, 0.30, 120)
p_grid = np.linspace(-0.40, 0.40, 80)
Xg, Pg = np.meshgrid(x_grid, p_grid)
A_grid = np.column_stack([Xg.ravel()**i * Pg.ravel()**j for (i, j) in monomials])
Zg = (A_grid @ coeffs).reshape(Xg.shape)

ax = axes[0]
cm = ax.contourf(x_grid, np.degrees(p_grid), Zg, levels=20, cmap="plasma")
fig.colorbar(cm, ax=ax, label="pz_max (m)")
ax.scatter(bx2d, np.degrees(bp2d), c=bz2d, cmap="plasma",
           s=8, edgecolors="k", linewidths=0.3)
ax.set_xlabel("CoM x rel to feet (m)"); ax.set_ylabel("Body pitch (deg)")
ax.set_title(f"2D upper workspace boundary  RMSE={rmse*1000:.1f} mm")
ax.grid(True, alpha=0.2)

ax = axes[1]
colors = plt.cm.coolwarm(np.linspace(0, 1, len(PITCHES_SWEEP)))
x_plot = np.linspace(-0.20, 0.30, 200)
for pitch, col in zip(PITCHES_SWEEP, colors):
    p_arr = np.full_like(x_plot, pitch)
    A_line = np.column_stack([x_plot**i * p_arr**j for (i, j) in monomials])
    ax.plot(x_plot, A_line @ coeffs, color=col, lw=1.5,
            label=f"{np.degrees(pitch):+.0f}°")
ax.set_xlabel("CoM x rel to feet (m)"); ax.set_ylabel("pz_max (m)")
ax.set_title("pz_max(x) slices by pitch")
ax.legend(fontsize=8, ncol=2); ax.grid(True, alpha=0.25)
ax.set_ylim(0.50, 0.90)

fig.tight_layout()
plot_path = os.path.join(_OUT_DIR, "workspace_sagittal_2d_upper.png")
fig.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"Plot saved → {plot_path}")
