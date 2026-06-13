##
#
# Kino-dynamics pipeline: SRB trajectory → centroidal NLP → full-body joint trajectory.
#
# Usage:
#   conda run -n env_sbo python kino_ik/pipeline_kino_ik.py \
#       --config srb.config.backflip_varinertia \
#       --warmstart-dir results/srb/backflip/ik
#
# Steps:
#   1. Load SRB trajectory (q_opt, v_opt, feet, forces, time)
#   2. Load warm-start q_warm from a prior IK run (basic backflip, no varinertia)
#   3. Compute H_srb = [m*v_com ; R_body @ I_nom @ w_body] at every node
#   4. Compute V_warm by finite-differencing q_warm
#   5. Print warm-start residuals (momentum + CoM) before solving
#   6. Build and solve KinoNLP
#   7. Save kino_q.csv, kino_v.csv, kino_h.csv to <srb_save_dir>/kino/
#   8. Print post-solve residuals and foot sliding check
#   9. Visualize in MuJoCo (reuses pipeline_srb_ik visualizer)
#
##

import sys, os, argparse, importlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pinocchio as pin

from kino_ik.kino_nlp import KinoNLP
from utils.kinematics.kin import quat_to_rot_matrix

# Reuse visualizer and XML builder from the existing IK pipeline
from ik.pipeline_srb_ik import (
    build_combined_xml,
    visualize,
    load_srb_results,
    load_arm_defaults,
    pin_q_to_mj,
)

_REPO_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_URDF = os.path.join(_REPO_ROOT, "models", "g1", "g1_29dof_rev_1_0.urdf")
_G1_XML       = os.path.join(_REPO_ROOT, "models", "g1", "g1_29dof_rev_1_0.xml")
_SRB_XML      = os.path.join(_REPO_ROOT, "models", "srb", "srb.xml")
_COMBINED_XML = os.path.join(_REPO_ROOT, "models", "g1", "g1_srb_combined.xml")

# SRB physical constants (from srb/srb.py)
_M    = 33.34   # kg
_G    = 9.81    # m/s^2
_I_NOM = np.array([                      # nominal body-frame inertia [kg·m²]
    [3.747533,  0.000051,  0.086972],
    [0.000051,  3.300958, -0.000894],
    [0.086972, -0.000894,  0.516523],
])

ANKLE_HEIGHT = 0.0332   # metres (from G1IPOPTIK.ANKLE_HEIGHT)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mj_to_pin_quat(q_srb_row):
    """MuJoCo [qw,qx,qy,qz] → pinocchio [qx,qy,qz,qw] for one row."""
    qw, qx, qy, qz = q_srb_row[3], q_srb_row[4], q_srb_row[5], q_srb_row[6]
    return np.array([qx, qy, qz, qw])


def compute_H_srb(q_srb, v_srb, I_opt=None):
    """
    Compute centroidal momentum H_srb = [linear; angular] in world frame.

    Linear part:  k = m * v_com  (world frame)
    Angular part: L = R_body @ I(tuck) @ w_body  (body → world)

    q_srb : (N+1, 7)  [px,py,pz, qw,qx,qy,qz]   MuJoCo quat convention
    v_srb : (N+1, 6)  [vx,vy,vz, wx,wy,wz]        CoM vel + body ang vel
    I_opt : (N+1, 6)  optional [Ixx,Iyy,Izz,Ixy,Ixz,Iyz]; uses I_NOM if None
    """
    N1 = q_srb.shape[0]
    H  = np.zeros((N1, 6))

    for k in range(N1):
        # Linear momentum (world frame)
        H[k, 0:3] = _M * v_srb[k, 0:3]

        # Body rotation matrix from MuJoCo quat [qw,qx,qy,qz]
        # quat_to_rot_matrix expects [qw,qx,qy,qz]
        R = quat_to_rot_matrix(q_srb[k, 3:7])

        # Inertia tensor in body frame
        if I_opt is not None:
            I_k = np.array([
                [I_opt[k, 0], I_opt[k, 3], I_opt[k, 4]],
                [I_opt[k, 3], I_opt[k, 1], I_opt[k, 5]],
                [I_opt[k, 4], I_opt[k, 5], I_opt[k, 2]],
            ])
        else:
            I_k = _I_NOM

        # Angular momentum: rotate body-frame L to world frame
        L_body = I_k @ v_srb[k, 3:6]
        H[k, 3:6] = R @ L_body

    return H


def compute_V_warm(model, Q_warm, dt_vec):
    """
    Finite-difference velocity warm start using pin.difference (manifold-correct).

    V_warm[k] = pin.difference(q[k], q[k+1]) / dt[k]
    V_warm[N]  = V_warm[N-1]  (repeat last)
    """
    N1 = Q_warm.shape[0]
    nv = model.nv
    V_warm = np.zeros((N1, nv))
    for k in range(N1 - 1):
        dq = pin.difference(model, Q_warm[k], Q_warm[k + 1])
        V_warm[k] = dq / max(dt_vec[k], 1e-6)
    V_warm[-1] = V_warm[-2]
    return V_warm


def print_warmstart_residuals(nlp, Q_warm, V_warm):
    print("\n[kino] Warm-start residuals:")
    mom = nlp.momentum_residuals(Q_warm, V_warm)
    com = nlp.com_residuals(Q_warm, V_warm)
    foot = nlp.foot_xy_residuals(Q_warm)
    print(f"  momentum  — mean {mom.mean():.4f}  max {mom.max():.4f}")
    print(f"  CoM       — mean {com.mean():.4f}  max {com.max():.4f}")
    contact_foot = foot[~np.isnan(foot)]
    if len(contact_foot) > 0:
        print(f"  foot XY   — mean {contact_foot.mean():.4f}  max {contact_foot.max():.4f}")
    print()


def print_solve_residuals(nlp, Q_sol, V_sol):
    print("\n[kino] Post-solve residuals:")
    mom = nlp.momentum_residuals(Q_sol, V_sol)
    momdyn = nlp.momentum_dynamics_residuals(Q_sol, V_sol)
    com = nlp.com_residuals(Q_sol, V_sol)
    foot = nlp.foot_xy_residuals(Q_sol)
    print(f"  mom balance — mean {momdyn.mean():.2e}  max {momdyn.max():.2e}  "
          f"({'OK' if momdyn.max() < 1e-3 else 'VIOLATED'})")
    print(f"  momentum  — mean {mom.mean():.6f}  max {mom.max():.6f}  "
          f"({'OK' if mom.max() < 1e-2 else 'HIGH'})")
    print(f"  CoM       — mean {com.mean():.4f}  max {com.max():.4f}")
    contact_foot = foot[~np.isnan(foot)]
    if len(contact_foot) > 0:
        print(f"  foot XY   — mean {contact_foot.mean()*1e3:.2f} mm  "
              f"max {contact_foot.max()*1e3:.2f} mm  "
              f"({'OK' if contact_foot.max() < 5e-3 else 'SLIDING'})")
    print()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_kino_pipeline(config_module, warmstart_dir, skip_solve=False, speed=1.0,
                      no_visualize=False, weights=None, hessian_mode="limited-memory"):

    # ── 1. Resolve paths from config ─────────────────────────────────────────
    cfg    = importlib.import_module(config_module)
    srb_dir = os.path.join(_REPO_ROOT, cfg.config.save_dir.lstrip("./"))
    out_dir = os.path.join(srb_dir, "kino")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[kino] SRB dir  : {srb_dir}")
    print(f"[kino] Output   : {out_dir}")
    print(f"[kino] Warm-start: {warmstart_dir}")

    # ── 2. Load SRB trajectory ────────────────────────────────────────────────
    times, q_srb, feet_ext, stance_end, flight_end = load_srb_results(srb_dir)
    N  = len(times) - 1
    dt_vec = np.diff(times)   # (N,)

    v_srb = np.loadtxt(os.path.join(srb_dir, "v_opt.csv"), delimiter=",")
    assert v_srb.shape == (N + 1, 6), f"Unexpected v_srb shape {v_srb.shape}"

    # Optional: variable inertia
    I_opt = None
    I_opt_path = os.path.join(srb_dir, "I_opt.csv")
    if os.path.exists(I_opt_path):
        raw = np.loadtxt(I_opt_path, delimiter=",", comments="#")
        if raw.shape == (N + 1, 6):
            I_opt = raw
            print("[kino] Variable inertia loaded from I_opt.csv")

    # ── 3. Build pinocchio model ──────────────────────────────────────────────
    model = pin.buildModelFromUrdf(_DEFAULT_URDF, pin.JointModelFreeFlyer())
    data  = model.createData()
    q_arm_default = load_arm_defaults(model)

    # ── 4. Load warm-start q_warm ─────────────────────────────────────────────
    warmstart_path = os.path.join(_REPO_ROOT, warmstart_dir.lstrip("./"), "ik_q_pin.csv")
    if not os.path.exists(warmstart_path):
        raise FileNotFoundError(
            f"Warm-start IK not found: {warmstart_path}\n"
            "Run: python ik/pipeline_srb_ik.py --config srb.config.backflip first."
        )
    Q_warm_raw = np.loadtxt(warmstart_path, delimiter=",")
    print(f"[kino] Warm-start loaded: {Q_warm_raw.shape} from {warmstart_path}")

    # Resample if frame count differs from current SRB trajectory
    if Q_warm_raw.shape[0] != N + 1:
        print(f"[kino] Resampling warm-start from {Q_warm_raw.shape[0]} → {N+1} frames")
        idx_orig = np.linspace(0, Q_warm_raw.shape[0] - 1, Q_warm_raw.shape[0])
        idx_new  = np.linspace(0, Q_warm_raw.shape[0] - 1, N + 1)
        Q_warm = np.zeros((N + 1, model.nq))
        for col in range(model.nq):
            Q_warm[:, col] = np.interp(idx_new, idx_orig, Q_warm_raw[:, col])
        # Re-normalise quaternion columns after interpolation
        norms = np.linalg.norm(Q_warm[:, 3:7], axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        Q_warm[:, 3:7] /= norms
    else:
        Q_warm = Q_warm_raw.copy()

    # ── 5. Convert SRB quaternion to pinocchio convention ─────────────────────
    # SRB:      [qw,qx,qy,qz]  (MuJoCo, columns 3..6 of q_srb)
    # Pinocchio:[qx,qy,qz,qw]
    quat_srb_pin = np.column_stack([q_srb[:, 4:7], q_srb[:, 3:4]])  # (N+1, 4)
    w_body_srb   = v_srb[:, 3:6]                                     # (N+1, 3)

    # ── 6. Compute H_srb ─────────────────────────────────────────────────────
    print("[kino] Computing H_srb...")
    H_srb = compute_H_srb(q_srb, v_srb, I_opt=I_opt)

    # Build external wrench W_ext[k] = [F_contact - (0,0,mg) ; M_contact] (world frame).
    # This drives the centroidal momentum balance in the NLP and is the same
    # increment used to sanity-check H_srb below.
    _tau_path = os.path.join(srb_dir, "tau_opt.csv")
    if not os.path.exists(_tau_path):
        raise FileNotFoundError(f"tau_opt.csv (external wrench) not found: {_tau_path}")
    tau = np.loadtxt(_tau_path, delimiter=",")   # (N, 6): [Fx,Fy,Fz,Mx,My,Mz]
    assert tau.shape == (N, 6), f"Unexpected tau_opt shape {tau.shape}"
    W_ext = tau.copy()
    W_ext[:, 2] -= _M * _G        # subtract gravity on linear z

    # Sanity-check: forward-integrate H from W_ext and compare to H_srb
    H_check = H_srb[0].copy()
    max_h_err = 0.0
    for k in range(N):
        H_check += dt_vec[k] * W_ext[k]
        err = np.linalg.norm(H_check - H_srb[k + 1])
        max_h_err = max(max_h_err, err)
    print(f"[kino] H_srb consistency check: max integration error = {max_h_err:.6f} "
          f"({'OK' if max_h_err < 1e-2 else 'HIGH — check H_srb computation'})")

    # ── 7. Compute V_warm ────────────────────────────────────────────────────
    print("[kino] Computing V_warm from finite differences...")
    V_warm = compute_V_warm(model, Q_warm, dt_vec)

    # Override pelvis warm start with SRB CoM trajectory + velocity.
    # The old IK may have been solved for a different trajectory, causing the
    # integrated pelvis to drift far from p_com_srb.  Seeding from SRB directly
    # gives IPOPT a much better starting point; the CoM cost handles the small
    # body-frame offset between pelvis and CoM (~0.09 m).
    Q_warm[:, 0:3] = q_srb[:, 0:3]      # pelvis ≈ CoM (optimizer adjusts offset)
    V_warm[:, 0:3] = v_srb[:, 0:3]      # pelvis velocity ≈ CoM velocity

    # ── 8. Floor z (ankle height above ground) ───────────────────────────────
    floor_z = ANKLE_HEIGHT
    try:
        stance_gz = cfg.config.constraints.stance_ground_z
        floor_z   = ANKLE_HEIGHT + stance_gz
    except AttributeError:
        pass

    # ── 9. Foot positions for cost ────────────────────────────────────────────
    # feet_ext: (N+1, 4) [pLx,pLy,pRx,pRy], NaN during flight
    p_foot_srb = feet_ext.copy()

    # ── 10. Build NLP ─────────────────────────────────────────────────────────
    # CoM tracking target.  The SRB config's initial p_com (0.77) is a PELVIS-
    # height value, not the true CoM (~0.69 = srb pz_com = robot standing CoM).
    # So q_srb[0:3] rides ~0.09 m too high for the full-body CoM to reach.  Shift
    # the target by the body-frame pelvis->CoM offset, ROTATED with the body
    # (vital for the 360deg flip — a fixed-z offset would be ~0.18 m wrong when
    # inverted).  Dynamics refs (H_srb, W_ext) are velocity/force-based and so
    # unaffected; only this position reference needs correcting.
    q_neutral = pin.neutral(model)
    pin.centerOfMass(model, data, q_neutral)
    offset_body = np.array(data.com[0]) - q_neutral[0:3]    # ~[0.02, 0, -0.089]
    p_com_srb = np.zeros((N + 1, 3))
    for k in range(N + 1):
        R_k = quat_to_rot_matrix(q_srb[k, 3:7])             # expects [qw,qx,qy,qz]
        p_com_srb[k] = q_srb[k, 0:3] + R_k @ offset_body
    print(f"[kino] CoM target shifted by body-frame offset {offset_body.round(3)} "
          f"(pelvis->CoM); start z {q_srb[0,2]:.3f} -> {p_com_srb[0,2]:.3f}")

    nlp = KinoNLP(
        dt_vec       = dt_vec,
        stance_end   = stance_end,
        flight_end   = flight_end,
        quat_srb_pin = quat_srb_pin,
        w_body_srb   = w_body_srb,
        H_srb        = H_srb,
        W_ext        = W_ext,
        p_com_srb    = p_com_srb,
        p_foot_srb   = p_foot_srb,
        floor_z      = floor_z,
        q_arm_default = q_arm_default,
        Q_warm       = Q_warm,
        V_warm       = V_warm,
        weights      = weights,
        hessian_mode = hessian_mode,
    )

    # ── 11. Warm-start residuals ──────────────────────────────────────────────
    print_warmstart_residuals(nlp, Q_warm, V_warm)

    if skip_solve:
        print("[kino] --skip-solve: loading existing kino_q.csv")
        Q_sol = np.loadtxt(os.path.join(out_dir, "kino_q.csv"), delimiter=",")
        V_sol = np.loadtxt(os.path.join(out_dir, "kino_v.csv"), delimiter=",")
        success = True
    else:
        # ── 12. Solve ─────────────────────────────────────────────────────────
        Q_sol, V_sol, success = nlp.solve()
        print(f"\n[kino] Solve {'SUCCEEDED' if success else 'FAILED (returning best iterate)'}")

    # ── 13. Post-solve residuals ───────────────────────────────────────────────
    print_solve_residuals(nlp, Q_sol, V_sol)

    # ── 14. Compute solved centroidal momentum for saving ─────────────────────
    H_sol = np.zeros((N + 1, 6))
    for k in range(N + 1):
        H_sol[k] = np.array(nlp.f_h(Q_sol[k], V_sol[k])).flatten()

    # ── 15. Save outputs ──────────────────────────────────────────────────────
    np.savetxt(os.path.join(out_dir, "kino_q.csv"), Q_sol, delimiter=",",
               header="pinocchio q (nq=36): px,py,pz,qx,qy,qz,qw,joints×29")
    np.savetxt(os.path.join(out_dir, "kino_v.csv"), V_sol, delimiter=",",
               header="pinocchio v (nv=35)")
    np.savetxt(os.path.join(out_dir, "kino_h.csv"), H_sol, delimiter=",",
               header="centroidal momentum [kx,ky,kz,Lx,Ly,Lz] world frame")
    np.savetxt(os.path.join(out_dir, "kino_time.csv"), times, delimiter=",")
    print(f"[kino] Saved outputs → {out_dir}")

    # Also save MuJoCo-convention qpos for easy downstream loading
    q_mj = np.array([pin_q_to_mj(Q_sol[i]) for i in range(N + 1)])
    np.savetxt(os.path.join(out_dir, "kino_q_mujoco.csv"), q_mj, delimiter=",",
               header="MuJoCo qpos (nq=36): px,py,pz,qw,qx,qy,qz,joints×29")

    # ── 16. Visualize ─────────────────────────────────────────────────────────
    if no_visualize:
        return Q_sol, V_sol, success

    build_combined_xml(_G1_XML, _SRB_XML, _COMBINED_XML)
    print("[kino] Launching MuJoCo viewer (close to exit)...")
    visualize(Q_sol, q_srb, times, _COMBINED_XML, speed=speed, I_opt=I_opt)

    return Q_sol, V_sol, success


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Kino-dynamics pipeline: SRB → centroidal NLP → full-body trajectory"
    )
    parser.add_argument("--config", default="srb.config.backflip_varinertia",
                        help="SRB config module (default: srb.config.backflip_varinertia)")
    parser.add_argument("--warmstart-dir", default="results/srb/backflip_varinertia/ik",
                        help="Directory containing ik_q_pin.csv warm start "
                             "(default: results/srb/backflip_varinertia/ik)")
    parser.add_argument("--skip-solve", action="store_true",
                        help="Skip NLP solve, reload existing kino_q.csv")
    parser.add_argument("--speed", type=float, default=0.5,
                        help="Visualizer playback speed (default: 0.5 = half speed)")
    parser.add_argument("--no-visualize", action="store_true",
                        help="Skip MuJoCo viewer")
    # Weight overrides
    parser.add_argument("--w-mom",      type=float, default=None)
    parser.add_argument("--w-com",      type=float, default=None)
    parser.add_argument("--w-quat",     type=float, default=None)
    parser.add_argument("--w-wbase",    type=float, default=None)
    parser.add_argument("--w-sym",      type=float, default=None)
    parser.add_argument("--w-qreg",     type=float, default=None)
    parser.add_argument("--w-vsmooth",  type=float, default=None)
    parser.add_argument("--hessian", choices=["limited-memory", "exact"],
                        default="limited-memory",
                        help="IPOPT Hessian mode (L-BFGS vs exact)")

    args = parser.parse_args()

    weights = {}
    for key, val in [("w_mom",      args.w_mom),
                     ("w_com",      args.w_com),
                     ("w_quat",     args.w_quat),
                     ("w_wbase",    args.w_wbase),
                     ("w_sym",      args.w_sym),
                     ("w_qreg",     args.w_qreg),
                     ("w_vsmooth",  args.w_vsmooth)]:
        if val is not None:
            weights[key] = val

    run_kino_pipeline(
        config_module  = args.config,
        warmstart_dir  = args.warmstart_dir,
        skip_solve     = args.skip_solve,
        speed          = args.speed,
        no_visualize   = args.no_visualize,
        weights        = weights or None,
        hessian_mode   = args.hessian,
    )
