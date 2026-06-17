##
# View a SAVED kino-dynamics result — per-direction tracking plot and/or MuJoCo
# playback — WITHOUT building or loading the (large) cached NLP solver.
#
# It reloads <srb_dir>/kino/{kino_q,kino_v,kino_lambda}.csv plus the SRB
# references, builds a lightweight stub exposing only what the plot needs
# (f_com / f_h via pinocchio), and reuses plot_ik_vs_ref + the MuJoCo viewer.
#
# Usage:
#   conda run -n env_sbo python kino_ik/view_kino_results.py \
#       --config srb.config.backflip_varinertia
#   ... --no-visualize   (plot only)
#   ... --no-plot        (MuJoCo only)
##

import os, sys, argparse, importlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pinocchio as pin
from types import SimpleNamespace

from kino_ik.pipeline_kino_ik import (
    load_srb_results, compute_com_refs, load_lambda_srb, plot_ik_vs_ref,
    build_combined_xml, visualize,
    _REPO_ROOT, _DEFAULT_URDF, _G1_XML, _SRB_XML, _COMBINED_XML,
)
from kino_ik.kino_nlp import distribute_wrench_to_points


def _make_stub(model, data, q_srb, v_srb, c_srb, cd_srb, lam_srb,
               stance_end, flight_end, N):
    """Lightweight object exposing only what plot_ik_vs_ref reads."""
    mass = float(sum(model.inertias[i].mass for i in range(1, model.njoints)))

    def f_com(q):
        return pin.centerOfMass(model, data, np.asarray(q).flatten())

    def f_h(q, v):
        pin.ccrba(model, data, np.asarray(q).flatten(), np.asarray(v).flatten())
        return np.concatenate([np.array(data.hg.linear), np.array(data.hg.angular)])

    return SimpleNamespace(
        N=N, mass=mass, f_com=f_com, f_h=f_h,
        c_srb=c_srb, cd_srb=cd_srb,
        quat_srb_pin=np.column_stack([q_srb[:, 4:7], q_srb[:, 3:4]]),
        w_body_srb=v_srb[:, 3:6], lam_srb=lam_srb,
        stance_end=stance_end, flight_end=flight_end)


def main():
    ap = argparse.ArgumentParser(description="View a saved kino result (plot + MuJoCo)")
    ap.add_argument("--config", default="srb.config.backflip_varinertia",
                    help="SRB config module (locates <save_dir>/kino/)")
    ap.add_argument("--no-plot", action="store_true", help="Skip the tracking plot")
    ap.add_argument("--no-visualize", action="store_true", help="Skip MuJoCo playback")
    ap.add_argument("--speed", type=float, default=0.5, help="Playback speed")
    args = ap.parse_args()

    cfg = importlib.import_module(args.config)
    srb_dir = os.path.join(_REPO_ROOT, cfg.config.save_dir.lstrip("./"))
    out_dir = os.path.join(srb_dir, "kino")
    print(f"[view] SRB dir : {srb_dir}")

    times, q_srb, feet_ext, stance_end, flight_end = load_srb_results(srb_dir)
    N = len(times) - 1
    v_srb = np.loadtxt(os.path.join(srb_dir, "v_opt.csv"), delimiter=",")

    model = pin.buildModelFromUrdf(_DEFAULT_URDF, pin.JointModelFreeFlyer())
    data = model.createData()
    c_srb, cd_srb, _ = compute_com_refs(model, data, q_srb, v_srb)
    lam_srb = distribute_wrench_to_points(load_lambda_srb(srb_dir, N))

    Q_sol = np.loadtxt(os.path.join(out_dir, "kino_q.csv"), delimiter=",")
    V_sol = np.loadtxt(os.path.join(out_dir, "kino_v.csv"), delimiter=",")
    lam_path = os.path.join(out_dir, "kino_lambda.csv")
    lam_sol = np.loadtxt(lam_path, delimiter=",") if os.path.exists(lam_path) else None
    print(f"[view] Loaded kino_q {Q_sol.shape}, kino_v {V_sol.shape}, "
          f"lambda {None if lam_sol is None else lam_sol.shape}")

    nlp = _make_stub(model, data, q_srb, v_srb, c_srb, cd_srb, lam_srb,
                     stance_end, flight_end, N)

    if not args.no_plot:
        plot_ik_vs_ref(nlp, Q_sol, V_sol, lam_sol, times, out_dir, show=True)

    if not args.no_visualize:
        I_opt = None
        I_opt_path = os.path.join(srb_dir, "I_opt.csv")
        if os.path.exists(I_opt_path):
            raw = np.loadtxt(I_opt_path, delimiter=",", comments="#")
            if raw.shape == (N + 1, 6):
                I_opt = raw
        build_combined_xml(_G1_XML, _SRB_XML, _COMBINED_XML)
        print("[view] Launching MuJoCo viewer (close to exit)...")
        visualize(Q_sol, q_srb, times, _COMBINED_XML, speed=args.speed, I_opt=I_opt)


if __name__ == "__main__":
    main()
