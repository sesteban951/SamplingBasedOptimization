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
#   3. Load per-foot wrench reference + CoM position/velocity references
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

from kino_ik.kino_nlp import (KinoNLP, FOOT_CORNERS, N_PT,
                              distribute_wrench_to_points, _DEFAULT_WEIGHTS)
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

ANKLE_HEIGHT = 0.0332   # metres (from G1IPOPTIK.ANKLE_HEIGHT)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mj_to_pin_quat(q_srb_row):
    """MuJoCo [qw,qx,qy,qz] → pinocchio [qx,qy,qz,qw] for one row."""
    qw, qx, qy, qz = q_srb_row[3], q_srb_row[4], q_srb_row[5], q_srb_row[6]
    return np.array([qx, qy, qz, qw])


def load_lambda_srb(srb_dir, N):
    """Per-foot contact wrench reference lambda = [F_L, M_L, F_R, M_R] (world).

    Built from the SRB per-foot force/moment outputs (each (N,3)):
        force_left.csv, moment_left.csv, force_right.csv, moment_right.csv
    Returns (N, 12).  Zero during flight (SRB writes zero contact force there).
    """
    def _load(name):
        a = np.loadtxt(os.path.join(srb_dir, name), delimiter=",")
        assert a.shape == (N, 3), f"Unexpected {name} shape {a.shape} (want {(N,3)})"
        return a
    F_L = _load("force_left.csv")
    M_L = _load("moment_left.csv")
    F_R = _load("force_right.csv")
    M_R = _load("moment_right.csv")
    return np.hstack([F_L, M_L, F_R, M_R])   # (N, 12)


def compute_com_refs(model, data, q_srb, v_srb):
    """CoM position / velocity references from the SRB trajectory.

    The SRB state position rides at ~pelvis height (init p_com ≈ 0.77), which is
    ~0.09 m above the full-body standing CoM.  Shift by the body-frame pelvis→CoM
    offset (rotated with the body — vital for the 360° flip) so the target is
    reachable.  The CoM velocity gets the matching omega × offset term so the
    position and velocity references stay kinematically consistent.

    Returns c_srb (N+1,3), cd_srb (N+1,3) in world frame.
    """
    q_neutral = pin.neutral(model)
    pin.centerOfMass(model, data, q_neutral)
    offset_body = np.array(data.com[0]) - q_neutral[0:3]     # ~[0.02, 0, -0.089]

    N1 = q_srb.shape[0]
    c_srb  = np.zeros((N1, 3))
    cd_srb = np.zeros((N1, 3))
    for k in range(N1):
        R_k   = quat_to_rot_matrix(q_srb[k, 3:7])            # expects [qw,qx,qy,qz]
        off_w = R_k @ offset_body
        w_w   = R_k @ v_srb[k, 3:6]                          # body ang vel → world
        c_srb[k]  = q_srb[k, 0:3] + off_w
        cd_srb[k] = v_srb[k, 0:3] + np.cross(w_w, off_w)
    return c_srb, cd_srb, offset_body


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
    com  = nlp.com_residuals(Q_warm, V_warm)
    cdot = nlp.cdot_residuals(Q_warm, V_warm)
    foot = nlp.foot_xy_residuals(Q_warm)
    print(f"  CoM pos   — mean {com.mean():.4f}  max {com.max():.4f}")
    print(f"  CoM vel   — mean {cdot.mean():.4f}  max {cdot.max():.4f}")
    contact_foot = foot[~np.isnan(foot)]
    if len(contact_foot) > 0:
        print(f"  foot XY   — mean {contact_foot.mean():.4f}  max {contact_foot.max():.4f}")
    print()


def print_solve_residuals(nlp, Q_sol, V_sol):
    print("\n[kino] Post-solve residuals:")
    lam = getattr(nlp, "Lam_sol", None)
    momdyn = nlp.momentum_dynamics_residuals(Q_sol, V_sol, lam=lam)
    com    = nlp.com_residuals(Q_sol, V_sol)
    cdot   = nlp.cdot_residuals(Q_sol, V_sol)
    foot   = nlp.foot_xy_residuals(Q_sol)
    tag = "solved λ" if lam is not None else "SRB ref λ"
    print(f"  mom dyn (h_dot vs wrench, {tag}) — mean {momdyn.mean():.3f}  max {momdyn.max():.3f}")
    print(f"  CoM pos   — mean {com.mean():.4f}  max {com.max():.4f}")
    print(f"  CoM vel   — mean {cdot.mean():.4f}  max {cdot.max():.4f}")
    contact_foot = foot[~np.isnan(foot)]
    if len(contact_foot) > 0:
        print(f"  foot XY   — mean {contact_foot.mean()*1e3:.2f} mm  "
              f"max {contact_foot.max()*1e3:.2f} mm  "
              f"({'OK' if contact_foot.max() < 5e-3 else 'SLIDING'})")
    print()


def print_ik_vs_ref_comparison(nlp, Q_sol, V_sol, lam_sol):
    """End-of-run comparison of IK (solved kino) vs SRB reference:
    CoM position/velocity and per-foot force/moment (and net wrench)."""
    N = nlp.N

    # CoM position/velocity: IK = com(q) / h_linear(q,v)/m ; ref = SRB c / c_dot
    com_ik = np.array([np.array(nlp.f_com(Q_sol[k])).flatten() for k in range(N + 1)])
    cd_ik  = np.array([np.array(nlp.f_h(Q_sol[k], V_sol[k])).flatten()[0:3] / nlp.mass
                       for k in range(N + 1)])
    d_com = np.linalg.norm(com_ik - nlp.c_srb, axis=1)
    d_cd  = np.linalg.norm(cd_ik  - nlp.cd_srb, axis=1)

    print("\n[kino] ===== IK vs SRB reference =====")
    print("  CoM position  |ik - ref|  (m) :  "
          f"mean {d_com.mean():.4f}  max {d_com.max():.4f}")
    print("  CoM velocity  |ik - ref|  (m/s):  "
          f"mean {d_cd.mean():.4f}  max {d_cd.max():.4f}")

    if lam_sol is None:
        print("  (force/wrench comparison unavailable — no solved lambda)")
        print()
        return

    # lambda = [fL1..fL4, fR1..fR4] (24).  Reduce point-forces to per-foot net
    # force and emergent ankle moment (sum r_i x f_i, foot frame) for comparison.
    def _net_and_moment(lam):
        Nn = lam.shape[0]
        FL = lam[:, 0:3 * N_PT].reshape(Nn, N_PT, 3).sum(axis=1)
        FR = lam[:, 3 * N_PT:6 * N_PT].reshape(Nn, N_PT, 3).sum(axis=1)
        ML = np.zeros((Nn, 3)); MR = np.zeros((Nn, 3))
        for i, r in enumerate(FOOT_CORNERS):
            ML += np.cross(r, lam[:, 3 * i:3 * i + 3])
            MR += np.cross(r, lam[:, 3 * N_PT + 3 * i:3 * N_PT + 3 * i + 3])
        return FL, FR, ML, MR

    fl_i, fr_i, ml_i, mr_i = _net_and_moment(lam_sol)
    fl_r, fr_r, ml_r, mr_r = _net_and_moment(nlp.lam_srb)   # reference reproduces SRB net

    def _stats(a, b):
        d = np.linalg.norm(a - b, axis=1)
        return d.mean(), d.max()

    fL = _stats(fl_i, fl_r); fR = _stats(fr_i, fr_r)
    mL = _stats(ml_i, ml_r); mR = _stats(mr_i, mr_r)

    print("  net force  |ik - ref|  (N)  :  "
          f"L mean {fL[0]:6.2f} max {fL[1]:6.2f}  |  R mean {fR[0]:6.2f} max {fR[1]:6.2f}")
    print("  ankle mom  |ik - ref|  (N·m):  "
          f"L mean {mL[0]:6.2f} max {mL[1]:6.2f}  |  R mean {mR[0]:6.2f} max {mR[1]:6.2f}")
    # per-point peak force magnitude (diagnostic: how much each corner carries)
    pts = lam_sol.reshape(lam_sol.shape[0], 2 * N_PT, 3)
    pk = np.linalg.norm(pts, axis=2)
    print(f"  per-point |f| (N): max {pk.max():6.1f}  mean(active) "
          f"{pk[pk > 1e-6].mean() if np.any(pk > 1e-6) else 0.0:6.1f}")

    # friction / unilateral diagnostic on the solved forces
    fz = pts[:, :, 2]
    ft = np.linalg.norm(pts[:, :, 0:2], axis=2)
    active = fz > 1e-3
    ratio = (ft[active] / fz[active]) if np.any(active) else np.array([0.0])
    print(f"  unilateral: min f_z {fz.min():+.2f} N   "
          f"friction util |f_t|/f_z: max {ratio.max():.2f} (mu={nlp.mu})")
    print()


def plot_ik_vs_ref(nlp, Q_sol, V_sol, lam_sol, times, out_dir, show=True):
    """Per-direction (x/y/z) IK-vs-SRB tracking: CoM position, CoM velocity,
    orientation error (rotation vector), body angular velocity, and total net
    contact force.  Always saves a PNG; pops a window if `show`."""
    import matplotlib
    if not show:
        matplotlib.use("Agg")          # headless — save only
    import matplotlib.pyplot as plt

    N = nlp.N
    t = np.asarray(times)
    axl = ["x", "y", "z"]

    # CoM position / velocity (ik = FK CoM, h_lin/m ; ref = SRB)
    com_ik = np.array([np.array(nlp.f_com(Q_sol[k])).flatten() for k in range(N + 1)])
    cv_ik  = np.array([np.array(nlp.f_h(Q_sol[k], V_sol[k])).flatten()[0:3] / nlp.mass
                       for k in range(N + 1)])

    # Orientation error rotvec  e = log3(R_ref^T R_ik)  (per-axis angular diff)
    def _R(qw, qx, qy, qz):
        return quat_to_rot_matrix(np.array([qw, qx, qy, qz]))   # expects [qw,qx,qy,qz]
    oerr = np.zeros((N + 1, 3))
    for k in range(N + 1):
        q = Q_sol[k]; qr = nlp.quat_srb_pin[k]            # both pinocchio [qx,qy,qz,qw]
        R_ik  = _R(q[6],  q[3],  q[4],  q[5])
        R_ref = _R(qr[3], qr[0], qr[1], qr[2])
        oerr[k] = pin.log3(R_ref.T @ R_ik)
    w_ik = V_sol[:, 3:6]                                   # body angular velocity

    if lam_sol is not None:
        nf_ik  = lam_sol.reshape(N, 2 * N_PT, 3).sum(1)    # total ground force (both feet)
        nf_ref = nlp.lam_srb.reshape(N, 2 * N_PT, 3).sum(1)

    # (row label, ik series, ref series or None, time, is-overlay)
    rows = [
        ("CoM pos [m]",      com_ik,   nlp.c_srb,        t,      True),
        ("CoM vel [m/s]",    cv_ik,    nlp.cd_srb,       t,      True),
        ("orient err [rad]", oerr,     None,             t,      False),
        ("omega [rad/s]",    w_ik,     nlp.w_body_srb,   t,      True),
        ("net force [N]",    nf_ik if lam_sol is not None else None,
                             nf_ref if lam_sol is not None else None, t[:N], True),
    ]

    fig, ax = plt.subplots(len(rows), 3, figsize=(13, 15), sharex=True)
    for r, (lbl, ik, ref, tt, overlay) in enumerate(rows):
        for c in range(3):
            a = ax[r][c]
            if ik is None:
                a.set_visible(False); continue
            if overlay:
                a.plot(tt, ik[:, c],  color="C0", label="ik")
                a.plot(tt, ref[:, c], color="C1", ls="--", label="ref")
                d = ik[:, c] - ref[:, c]
            else:                                          # error series toward 0
                a.plot(tt, ik[:, c], color="C3")
                a.axhline(0, color="k", lw=0.5, alpha=0.4)
                d = ik[:, c]
            a.set_title(f"{lbl} {axl[c]}   (max|Δ| {np.abs(d).max():.3f})", fontsize=9)
            a.grid(alpha=0.3)
            if r == 0 and c == 2 and overlay:
                a.legend(fontsize=7)
            a.axvline(t[nlp.stance_end], ls=":", color="gray", alpha=0.5)
            a.axvline(t[nlp.flight_end], ls=":", color="gray", alpha=0.5)
        ax[r][0].set_ylabel(lbl)
    for c in range(3):
        ax[-1][c].set_xlabel("time [s]")
    fig.suptitle("Kino IK vs SRB reference — per direction "
                 "(dotted = stance|flight|landing)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    png = os.path.join(out_dir, "kino_tracking.png")
    fig.savefig(png, dpi=110)
    print(f"[kino] tracking plot saved → {png}")
    if show:
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_kino_pipeline(config_module, warmstart_dir, skip_solve=False, speed=1.0,
                      no_visualize=False, weights=None, hessian_mode="exact",
                      expand=True, max_iter=10000, linear_solver="mumps",
                      mu_strategy="monotone", rebuild=False, plot=True, cache_keep=2,
                      self_collision=True, collision_margin=None):

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

    # ── 6. Contact reference: per-foot SRB wrench → 4-corner point forces ──────
    print("[kino] Loading per-foot wrench reference (force_*/moment_* .csv)...")
    wrench_foot_srb = load_lambda_srb(srb_dir, N)          # (N,12) [F_L,M_L,F_R,M_R]
    lam_srb = distribute_wrench_to_points(wrench_foot_srb)  # (N,24) corner point forces
    print(f"[kino] lambda_srb {lam_srb.shape} (4 pts/foot); per-foot |F| max "
          f"{np.linalg.norm(wrench_foot_srb[:, [0,1,2,6,7,8]], axis=1).max():.1f} N")

    # ── 7. Compute V_warm ────────────────────────────────────────────────────
    print("[kino] Computing V_warm from finite differences...")
    V_warm = compute_V_warm(model, Q_warm, dt_vec)

    # Override pelvis warm start with SRB CoM trajectory + velocity.
    # The old IK may have been solved for a different trajectory, causing the
    # integrated pelvis to drift far from the CoM target.  Seeding from SRB
    # directly gives IPOPT a much better starting point; the CoM tracking cost
    # handles the small body-frame offset between pelvis and CoM (~0.09 m).
    Q_warm[:, 0:3] = q_srb[:, 0:3]      # pelvis ≈ CoM (optimizer adjusts offset)
    V_warm[:, 0:3] = v_srb[:, 0:3]      # pelvis velocity ≈ CoM velocity

    # ── 8. Floor z (ankle height above ground) + friction coefficient ────────
    floor_z = ANKLE_HEIGHT
    try:
        stance_gz = cfg.config.constraints.stance_ground_z
        floor_z   = ANKLE_HEIGHT + stance_gz
    except AttributeError:
        pass
    mu = getattr(getattr(cfg.config, "constraints", None), "mu", 1.0)
    print(f"[kino] friction coefficient mu = {mu}")

    # ── 9. Foot positions for contact constraints ────────────────────────────
    # feet_ext: (N+1, 4) [pLx,pLy,pRx,pRy], NaN during flight
    p_foot_srb = feet_ext.copy()

    # ── 10. CoM position / velocity references (reachable, kinematically consistent)
    c_srb, cd_srb, offset_body = compute_com_refs(model, data, q_srb, v_srb)
    print(f"[kino] CoM target shifted by body-frame offset {offset_body.round(3)} "
          f"(pelvis->CoM); start z {q_srb[0,2]:.3f} -> {c_srb[0,2]:.3f}")

    # ── 11. Build NLP ─────────────────────────────────────────────────────────
    nlp = KinoNLP(
        dt_vec       = dt_vec,
        stance_end   = stance_end,
        flight_end   = flight_end,
        quat_srb_pin = quat_srb_pin,
        w_body_srb   = w_body_srb,
        c_srb        = c_srb,
        cd_srb       = cd_srb,
        lam_srb      = lam_srb,
        p_foot_srb   = p_foot_srb,
        floor_z      = floor_z,
        q_arm_default = q_arm_default,
        Q_warm       = Q_warm,
        V_warm       = V_warm,
        mu           = mu,
        weights      = weights,
        hessian_mode = hessian_mode,
        expand       = expand,
        max_iter     = max_iter,
        linear_solver = linear_solver,
        mu_strategy  = mu_strategy,
        rebuild      = rebuild,
        cache_keep   = cache_keep,
        self_collision   = self_collision,
        collision_margin = collision_margin,
    )

    # ── 11. Warm-start residuals ──────────────────────────────────────────────
    print_warmstart_residuals(nlp, Q_warm, V_warm)

    if skip_solve:
        print("[kino] --skip-solve: loading existing kino_q.csv")
        Q_sol = np.loadtxt(os.path.join(out_dir, "kino_q.csv"), delimiter=",")
        V_sol = np.loadtxt(os.path.join(out_dir, "kino_v.csv"), delimiter=",")
        lam_path = os.path.join(out_dir, "kino_lambda.csv")
        lam_sol = (np.loadtxt(lam_path, delimiter=",")
                   if os.path.exists(lam_path) else None)
        success = True
    else:
        # ── 12. Solve ─────────────────────────────────────────────────────────
        Q_sol, V_sol, success = nlp.solve()
        lam_sol = nlp.Lam_sol
        print(f"\n[kino] Solve {'SUCCEEDED' if success else 'FAILED (returning best iterate)'}")

    # ── 13. Post-solve residuals + IK-vs-reference comparison ──────────────────
    print_solve_residuals(nlp, Q_sol, V_sol)
    print_ik_vs_ref_comparison(nlp, Q_sol, V_sol, lam_sol)

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
    if lam_sol is not None:
        np.savetxt(os.path.join(out_dir, "kino_lambda.csv"), lam_sol, delimiter=",",
                   header="contact-polygon point forces (24) [fL1..fL4, fR1..fR4] world, "
                          "corners heel(-0.05,+-0.025,-0.03)/toe(0.12,+-0.03,-0.03) ankle frame")
    np.savetxt(os.path.join(out_dir, "kino_time.csv"), times, delimiter=",")
    print(f"[kino] Saved outputs → {out_dir}")

    # Also save MuJoCo-convention qpos for easy downstream loading
    q_mj = np.array([pin_q_to_mj(Q_sol[i]) for i in range(N + 1)])
    np.savetxt(os.path.join(out_dir, "kino_q_mujoco.csv"), q_mj, delimiter=",",
               header="MuJoCo qpos (nq=36): px,py,pz,qw,qx,qy,qz,joints×29")

    # ── 16. Tracking plot (CoM pos/vel + per-foot net force vs SRB) ────────────
    if plot:
        plot_ik_vs_ref(nlp, Q_sol, V_sol, lam_sol, times, out_dir,
                       show=not no_visualize)

    # ── 17. Visualize ─────────────────────────────────────────────────────────
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
    # Cost-weight overrides — one flag per weight, default shown is the current
    # value in _DEFAULT_WEIGHTS.  Changing a weight reuses the cached solver
    # (weights are runtime parameters), so it only pays the solve, not a rebuild.
    wgrp = parser.add_argument_group("cost weights (override _DEFAULT_WEIGHTS)")
    for _wk in ['w_com', 'w_cdot', 'w_quat', 'w_wbase', 'w_lam',
                'w_qreg', 'w_vsmooth', 'w_ke', 'w_config']:
        wgrp.add_argument(f"--{_wk.replace('_', '-')}", dest=_wk, type=float,
                          default=None, metavar="W",
                          help=f"{_wk} (default {_DEFAULT_WEIGHTS[_wk]:g})")
    parser.add_argument("--hessian", choices=["limited-memory", "exact"],
                        default="exact",
                        help="IPOPT Hessian mode (default: exact, ~5x faster than L-BFGS)")
    parser.add_argument("--no-expand", action="store_true",
                        help="Disable expand=True (keep MX function-call graph)")
    parser.add_argument("--max-iter", type=int, default=10000,
                        help="IPOPT max iterations (use a small value to benchmark)")
    parser.add_argument("--linear-solver", default="mumps",
                        help="IPOPT linear solver (mumps/ma27/ma57/ma97/ma86/spral/pardiso)")
    parser.add_argument("--mu-strategy", default="monotone",
                        choices=["monotone", "adaptive"],
                        help="IPOPT barrier update strategy")
    parser.add_argument("--rebuild", action="store_true",
                        help="Force rebuild of the cached solver (ignore existing cache)")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip the end-of-solve tracking plot (CoM + force error)")
    parser.add_argument("--cache-keep", type=int, default=2,
                        help="Keep this many most-recent solver caches; prune older "
                             "(caches can be multiple GB each; default 2)")
    parser.add_argument("--no-self-collision", action="store_true",
                        help="Disable capsule self-collision constraints")
    parser.add_argument("--collision-margin", type=float, default=None,
                        help="Self-collision clearance margin [m] (default collision_model.MARGIN)")

    args = parser.parse_args()

    weights = {wk: getattr(args, wk)
               for wk in ['w_com', 'w_cdot', 'w_quat', 'w_wbase', 'w_lam',
                          'w_qreg', 'w_vsmooth', 'w_ke', 'w_config']
               if getattr(args, wk) is not None}

    run_kino_pipeline(
        config_module  = args.config,
        warmstart_dir  = args.warmstart_dir,
        skip_solve     = args.skip_solve,
        speed          = args.speed,
        no_visualize   = args.no_visualize,
        weights        = weights or None,
        hessian_mode   = args.hessian,
        expand         = not args.no_expand,
        max_iter       = args.max_iter,
        linear_solver  = args.linear_solver,
        mu_strategy    = args.mu_strategy,
        rebuild        = args.rebuild,
        plot           = not args.no_plot,
        cache_keep     = args.cache_keep,
        self_collision   = not args.no_self_collision,
        collision_margin = args.collision_margin,
    )
