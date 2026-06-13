##
#
# Validation script for kino-dynamics output.
#
# Usage:
#   conda run -n env_sbo python kino_ik/validate_kino.py \
#       --config srb.config.backflip_varinertia
#
##

import sys, os, argparse, importlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pinocchio as pin
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_URDF = os.path.join(_REPO_ROOT, "models", "g1", "g1_29dof_rev_1_0.urdf")

L_FOOT = "left_ankle_roll_link"
R_FOOT = "right_ankle_roll_link"


def validate(config_module):
    cfg     = importlib.import_module(config_module)
    srb_dir = os.path.join(_REPO_ROOT, cfg.config.save_dir.lstrip("./"))
    kino_dir = os.path.join(srb_dir, "kino")

    # Load data
    q_srb   = np.loadtxt(os.path.join(srb_dir,  "q_opt.csv"),   delimiter=",")  # (N+1,7)
    v_srb   = np.loadtxt(os.path.join(srb_dir,  "v_opt.csv"),   delimiter=",")  # (N+1,6)
    times   = np.loadtxt(os.path.join(srb_dir,  "time.csv"),    delimiter=",")  # (N+1,)
    feet_raw= np.loadtxt(os.path.join(srb_dir,  "feet.csv"),    delimiter=",")  # (N,4)

    kino_q  = np.loadtxt(os.path.join(kino_dir, "kino_q.csv"),  delimiter=",")
    kino_v  = np.loadtxt(os.path.join(kino_dir, "kino_v.csv"),  delimiter=",")
    kino_h  = np.loadtxt(os.path.join(kino_dir, "kino_h.csv"),  delimiter=",")

    N = len(times) - 1
    feet = np.vstack([feet_raw, feet_raw[-1:]])

    stance_end = next(k for k in range(N) if np.isnan(feet[k, 0]))
    flight_end = next(k for k in range(N-1, -1, -1) if np.isnan(feet[k, 0])) + 1

    model = pin.buildModelFromUrdf(_DEFAULT_URDF, pin.JointModelFreeFlyer())
    data  = model.createData()
    l_id  = model.getFrameId(L_FOOT)
    r_id  = model.getFrameId(R_FOOT)

    # Compute per-node quantities
    com_kino   = np.zeros((N+1, 3))
    com_srb    = q_srb[:, 0:3]
    foot_l     = np.zeros((N+1, 3))
    foot_r     = np.zeros((N+1, 3))

    for k in range(N+1):
        pin.framesForwardKinematics(model, data, kino_q[k])
        pin.centerOfMass(model, data, kino_q[k])
        com_kino[k] = data.com[0]
        foot_l[k]   = data.oMf[l_id].translation
        foot_r[k]   = data.oMf[r_id].translation

    com_err  = np.linalg.norm(com_kino - com_srb, axis=1)
    H_srb_file = os.path.join(kino_dir, "kino_h.csv")

    # Momentum residual from saved H
    # (kino_h already contains Ag(q)v from the solution)
    from kino_ik.kino_nlp import KinoNLP
    from kino_ik.pipeline_kino_ik import compute_H_srb, _mj_to_pin_quat

    I_opt = None
    I_opt_path = os.path.join(srb_dir, "I_opt.csv")
    if os.path.exists(I_opt_path):
        raw = np.loadtxt(I_opt_path, delimiter=",", comments="#")
        if raw.shape == (N+1, 6):
            I_opt = raw
    H_srb = compute_H_srb(q_srb, v_srb, I_opt=I_opt)
    mom_err = np.linalg.norm(kino_h - H_srb, axis=1)

    # Momentum-balance (dynamics) residual: ||(h_{k+1}-h_k) - dt*W_ext||
    _M, _G = 33.34, 9.81
    tau = np.loadtxt(os.path.join(srb_dir, "tau_opt.csv"), delimiter=",")  # (N,6)
    W_ext = tau.copy(); W_ext[:, 2] -= _M * _G
    dt_vec = np.diff(times)
    momdyn_err = np.array([
        np.linalg.norm((kino_h[k+1] - kino_h[k]) - dt_vec[k] * W_ext[k])
        for k in range(N)
    ])

    # Foot XY errors (contact nodes only)
    foot_l_err = np.full(N+1, np.nan)
    foot_r_err = np.full(N+1, np.nan)
    for k in range(N+1):
        is_contact = (k < stance_end) or (k >= flight_end)
        if not is_contact or np.isnan(feet[k, 0]):
            continue
        foot_l_err[k] = np.linalg.norm(foot_l[k, :2] - feet[k, 0:2]) * 1e3
        foot_r_err[k] = np.linalg.norm(foot_r[k, :2] - feet[k, 2:4]) * 1e3

    contact_err = np.nanmax(np.stack([foot_l_err, foot_r_err], axis=1), axis=1)

    # ── Print summary ──────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("KINO-DYNAMICS VALIDATION")
    print("="*60)
    print(f"  Nodes: {N+1}  stance=[0,{stance_end})  flight=[{stance_end},{flight_end})  landing=[{flight_end},{N+1})")
    print()
    print(f"  CoM error (pinocchio vs SRB):")
    print(f"    mean = {com_err.mean()*1e3:.1f} mm   max = {com_err.max()*1e3:.1f} mm")
    for phase, sl in [("stance",  slice(0, stance_end)),
                      ("flight",  slice(stance_end, flight_end)),
                      ("landing", slice(flight_end, N+1))]:
        print(f"    {phase:7s}: mean={com_err[sl].mean()*1e3:.1f} mm  max={com_err[sl].max()*1e3:.1f} mm")
    print()
    print(f"  Momentum residual ||Ag(q)v - H_srb||:")
    print(f"    mean = {mom_err.mean():.4f}   max = {mom_err.max():.4f}")
    print(f"  Momentum-balance residual ||(h+ - h) - dt*W_ext||:")
    print(f"    mean = {momdyn_err.mean():.2e}   max = {momdyn_err.max():.2e}")
    print()
    print(f"  Foot XY error (contact nodes only):")
    valid = contact_err[~np.isnan(contact_err)]
    if len(valid):
        print(f"    mean = {valid.mean():.1f} mm   max = {valid.max():.1f} mm")
        stance_err  = foot_l_err[:stance_end]
        landing_err = foot_l_err[flight_end:]
        print(f"    stance:  mean={np.nanmean(stance_err):.1f} mm  max={np.nanmax(stance_err):.1f} mm")
        print(f"    landing: mean={np.nanmean(landing_err):.1f} mm  max={np.nanmax(landing_err):.1f} mm")
    print()

    # Per-node foot error table for worst nodes
    worst = np.argsort(np.where(np.isnan(contact_err), 0, contact_err))[-10:][::-1]
    print(f"  Worst 10 foot-XY nodes:")
    for k in worst:
        if np.isnan(contact_err[k]):
            continue
        phase = "stance" if k < stance_end else "land"
        print(f"    k={k:2d} ({phase}) L={foot_l_err[k]:.0f}mm R={foot_r_err[k]:.0f}mm  "
              f"com_err={com_err[k]*1e3:.0f}mm  mom_err={mom_err[k]:.3f}")

    # ── Plots ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    t = times

    # CoM X
    ax = axes[0, 0]
    ax.plot(t, com_srb[:, 0], 'b--', label='SRB CoM X')
    ax.plot(t, com_kino[:, 0], 'r-', label='Kino CoM X')
    ax.axvspan(t[stance_end], t[flight_end], alpha=0.1, color='gray', label='flight')
    ax.set_ylabel('CoM X (m)'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_title('CoM X')

    # CoM Z
    ax = axes[0, 1]
    ax.plot(t, com_srb[:, 2], 'b--', label='SRB CoM Z')
    ax.plot(t, com_kino[:, 2], 'r-', label='Kino CoM Z')
    ax.axvspan(t[stance_end], t[flight_end], alpha=0.1, color='gray')
    ax.set_ylabel('CoM Z (m)'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_title('CoM Z')

    # CoM error
    ax = axes[1, 0]
    ax.plot(t, com_err * 1e3, 'k-')
    ax.axvspan(t[stance_end], t[flight_end], alpha=0.1, color='gray')
    ax.set_ylabel('||CoM error|| (mm)'); ax.grid(True, alpha=0.3)
    ax.set_title('Full-body CoM error vs SRB')

    # Foot XY error
    ax = axes[1, 1]
    ax.plot(t, np.where(np.isnan(foot_l_err), np.nan, foot_l_err), 'b-', label='Left foot')
    ax.plot(t, np.where(np.isnan(foot_r_err), np.nan, foot_r_err), 'r-', label='Right foot')
    ax.axvspan(t[stance_end], t[flight_end], alpha=0.1, color='gray', label='flight')
    ax.set_ylabel('Foot XY error (mm)'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_title('Foot XY sliding (contact nodes)')

    # Momentum residual
    ax = axes[2, 0]
    ax.plot(t, mom_err, 'k-')
    ax.axvspan(t[stance_end], t[flight_end], alpha=0.1, color='gray')
    ax.set_ylabel('||h - H_srb||'); ax.grid(True, alpha=0.3)
    ax.set_title('Momentum residual')

    # Foot Z
    ax = axes[2, 1]
    ax.plot(t, foot_l[:, 2]*1e3, 'b-', label='Left foot Z')
    ax.plot(t, foot_r[:, 2]*1e3, 'r-', label='Right foot Z')
    ax.axhline(33.2, color='k', linestyle='--', linewidth=0.8, label='floor')
    ax.axvspan(t[stance_end], t[flight_end], alpha=0.1, color='gray')
    ax.set_ylabel('Foot Z (mm)'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_title('Foot Z height (should be ~33mm contact, higher in flight)')

    for ax in axes.flat:
        ax.set_xlabel('time (s)')

    fig.suptitle(f'Kino-dynamics validation: {config_module}', fontsize=12)
    fig.tight_layout()
    out_path = os.path.join(kino_dir, "validation.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n  Plot saved → {out_path}")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="srb.config.backflip_varinertia")
    args = parser.parse_args()
    validate(args.config)
