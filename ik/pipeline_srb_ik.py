##
#
# Pipeline: SRB solver → IK → combined visualization.
#
# Usage:
#   python ik/pipeline_srb_ik.py [--config srb.config.smalljump] [--skip-solve]
#
# Steps:
#   1. Run the SRB trajectory optimizer
#   2. Load CSVs and detect stance / flight / landing phases
#   3. Run Newton-Raphson IK for stance & landing using SRB foot positions
#   4. Interpolate leg joints toward standing during flight
#   5. Export full G1 config (nq=36) + timestamps as CSV
#   6. Visualize: G1 humanoid overlaid with semi-transparent SRB ellipsoid
#
##

import sys, os, time, argparse, subprocess, copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pinocchio as pin
import mujoco
import mujoco.viewer
import xml.etree.ElementTree as ET

from utils.kinematics.g1_ik import G1IK

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_G1_XML     = os.path.join(_REPO_ROOT, "models", "g1", "g1_29dof_rev_1_0.xml")
_SRB_XML    = os.path.join(_REPO_ROOT, "models", "srb", "srb.xml")
_COMBINED   = os.path.join(_REPO_ROOT, "models", "g1", "g1_srb_combined.xml")
_RESULTS    = os.path.join(_REPO_ROOT, "results", "srb", "srb_aerial")
_EXPORT_DIR = os.path.join(_REPO_ROOT, "ik", "results")

# ---------------------------------------------------------------------------
# Step 1 — run SRB solver
# ---------------------------------------------------------------------------

def run_srb_solver(config_module: str):
    print(f"[pipeline] Running SRB solver: python -m srb.srb_aerial {config_module}")
    ret = subprocess.run(
        ["python", "-m", "srb.srb_aerial", config_module],
        cwd=_REPO_ROOT,
    )
    if ret.returncode != 0:
        raise RuntimeError("SRB solver failed — check output above.")
    print("[pipeline] SRB solver done.")

# ---------------------------------------------------------------------------
# Step 2 — load results and detect phases
# ---------------------------------------------------------------------------

def load_srb_results(results_dir: str):
    """
    Returns:
        times  (N+1,)
        q_srb  (N+1, 7)  [px,py,pz, qw,qx,qy,qz]   MuJoCo quat convention
        feet   (N,   4)  [pLx,pLy, pRx,pRy]  NaN during flight
        stance_end   int  — first flight timestep index  (0-based, N+1 frame)
        flight_end   int  — first landing timestep index (0-based, N+1 frame)
    """
    times = np.loadtxt(os.path.join(results_dir, "time.csv"),  delimiter=",")
    q_srb = np.loadtxt(os.path.join(results_dir, "q_opt.csv"), delimiter=",")
    feet  = np.loadtxt(os.path.join(results_dir, "feet.csv"),  delimiter=",")

    N = len(times) - 1
    assert q_srb.shape == (N + 1, 7), f"Unexpected q_srb shape {q_srb.shape}"
    assert feet.shape  == (N,     4), f"Unexpected feet shape {feet.shape}"

    # Extend feet to N+1 by repeating last row (covers the terminal state)
    feet_ext = np.vstack([feet, feet[-1:]])  # (N+1, 4)

    is_flight = np.isnan(feet_ext[:, 0])

    flight_indices = np.where(is_flight)[0]
    if len(flight_indices) == 0:
        raise ValueError("No flight phase detected in feet.csv")

    stance_end = int(flight_indices[0])
    # First non-flight frame after stance_end
    landing_indices = np.where(~is_flight[stance_end:])[0]
    if len(landing_indices) == 0:
        raise ValueError("No landing phase detected after flight in feet.csv")
    flight_end = int(landing_indices[0]) + stance_end

    print(f"[pipeline] N={N}  stance: 0..{stance_end-1}  "
          f"flight: {stance_end}..{flight_end-1}  landing: {flight_end}..{N}")

    return times, q_srb, feet_ext, stance_end, flight_end

# ---------------------------------------------------------------------------
# Step 3 — run IK trajectory
# ---------------------------------------------------------------------------

def _mj_to_pin_quat(qw, qx, qy, qz):
    """MuJoCo [qw,qx,qy,qz] → pinocchio [qx,qy,qz,qw]."""
    return np.array([qx, qy, qz, qw])


def _build_q0(ik: G1IK, q_srb_row: np.ndarray, q_warm=None,
              warmstart_min_drop: float = 0.03):
    """
    Build a pinocchio initial guess from the SRB state row.
    Base pose comes from the SRB trajectory; leg joints are either warm-started
    or set from standing_config's squat-biased heuristic.
    """
    h = q_srb_row[2]
    q0 = ik.standing_config(h)
    q0[0:3] = q_srb_row[0:3]
    q0[3:7] = _mj_to_pin_quat(q_srb_row[3], q_srb_row[4], q_srb_row[5], q_srb_row[6])
    # Only warm-start joints when meaningfully below standing (avoids wrong branch)
    drop = max(0.0, 0.79 - h)
    if q_warm is not None and drop > warmstart_min_drop:
        q0[7:] = q_warm[7:]
    return q0


def run_ik_trajectory(ik: G1IK, q_srb: np.ndarray, feet_ext: np.ndarray,
                      stance_end: int, flight_end: int):
    """
    Returns q_ik (N+1, 36) pinocchio configuration at every timestep.
    """
    N = q_srb.shape[0] - 1
    q_ik = np.zeros((N + 1, ik.model.nq))

    # ---- STANCE ----
    print("[pipeline] Running stance IK ...")
    q_warm = None
    for i in range(stance_end):
        q0 = _build_q0(ik, q_srb[i], q_warm)
        oMl = pin.SE3(np.eye(3), np.array([feet_ext[i,0], feet_ext[i,1], ik.ANKLE_HEIGHT]))
        oMr = pin.SE3(np.eye(3), np.array([feet_ext[i,2], feet_ext[i,3], ik.ANKLE_HEIGHT]))
        q_sol, ok, errs = ik.solve(q0, oMl, oMr)
        if not ok and errs[-1] > 1e-3:
            print(f"  [warn] stance frame {i}: IK did not converge (err={errs[-1]:.2e})")
        q_ik[i] = q_sol
        q_warm = q_sol

    # ---- FLIGHT — interpolate joints toward standing ----
    print("[pipeline] Interpolating flight joints ...")
    joints_takeoff  = q_ik[stance_end - 1, 7:].copy()
    joints_standing = ik.standing_config(0.79)[7:]   # all-zero joints
    n_flight = flight_end - stance_end
    for idx, i in enumerate(range(stance_end, flight_end)):
        t = idx / max(1, n_flight - 1)
        q_ik[i, 0:3] = q_srb[i, 0:3]
        q_ik[i, 3:7] = _mj_to_pin_quat(q_srb[i,3], q_srb[i,4], q_srb[i,5], q_srb[i,6])
        q_ik[i, 7:]  = (1 - t) * joints_takeoff + t * joints_standing

    # ---- LANDING ----
    print("[pipeline] Running landing IK ...")
    q_warm = None
    for i in range(flight_end, N + 1):
        # Fresh guess on first landing frame (coming from flight, nearly standing)
        q0 = _build_q0(ik, q_srb[i], q_warm)
        oMl = pin.SE3(np.eye(3), np.array([feet_ext[i,0], feet_ext[i,1], ik.ANKLE_HEIGHT]))
        oMr = pin.SE3(np.eye(3), np.array([feet_ext[i,2], feet_ext[i,3], ik.ANKLE_HEIGHT]))
        q_sol, ok, errs = ik.solve(q0, oMl, oMr)
        if not ok and errs[-1] > 1e-3:
            print(f"  [warn] landing frame {i}: IK did not converge (err={errs[-1]:.2e})")
        q_ik[i] = q_sol
        q_warm = q_sol

    return q_ik

# ---------------------------------------------------------------------------
# Step 4 — convert IK solution to MuJoCo qpos and export
# ---------------------------------------------------------------------------

def pin_q_to_mj(q_pin: np.ndarray) -> np.ndarray:
    """Pinocchio [qx,qy,qz,qw] → MuJoCo [qw,qx,qy,qz]; joints unchanged."""
    q_mj = q_pin.copy()
    q_mj[3] = q_pin[6]   # qw
    q_mj[4] = q_pin[3]   # qx
    q_mj[5] = q_pin[4]   # qy
    q_mj[6] = q_pin[5]   # qz
    return q_mj


def export_ik_solution(q_ik: np.ndarray, times: np.ndarray, export_dir: str):
    os.makedirs(export_dir, exist_ok=True)
    np.savetxt(os.path.join(export_dir, "ik_q_pin.csv"),  q_ik,  delimiter=",",
               header="pinocchio q (nq=36): px,py,pz,qx,qy,qz,qw,joints×29")
    np.savetxt(os.path.join(export_dir, "ik_time.csv"),   times, delimiter=",")
    # Also export MuJoCo-convention qpos for easy loading
    q_mj = np.array([pin_q_to_mj(r) for r in q_ik])
    np.savetxt(os.path.join(export_dir, "ik_q_mujoco.csv"), q_mj, delimiter=",",
               header="MuJoCo qpos (nq=36): px,py,pz,qw,qx,qy,qz,joints×29")
    print(f"[pipeline] Exported IK solution ({q_ik.shape[0]} frames) → {export_dir}")

# ---------------------------------------------------------------------------
# Step 5 — build combined MuJoCo XML
# ---------------------------------------------------------------------------

def build_combined_xml(g1_xml: str, srb_xml: str, output: str, srb_alpha: float = 0.35):
    """
    Appends the SRB visual body (semi-transparent) into the G1 worldbody.
    The combined qpos layout is:
        [0:7]   G1 floating-base free joint
        [7:36]  G1 revolute joints  (29)
        [36:43] SRB free joint
    """
    ET.register_namespace("", "")
    g1_tree = ET.parse(g1_xml)
    g1_root = g1_tree.getroot()

    srb_tree = ET.parse(srb_xml)
    srb_root = srb_tree.getroot()

    srb_body = copy.deepcopy(srb_root.find(".//body[@name='base']"))

    # Rename body and joint to avoid conflicts
    srb_body.set("name", "srb_body")
    fj = srb_body.find("freejoint")
    if fj is not None:
        fj.set("name", "srb_free")

    # Make all SRB geoms semi-transparent
    for geom in srb_body.findall("geom"):
        rgba = geom.get("rgba", "0.5 0.5 0.5 1.0").split()
        rgba[3] = f"{srb_alpha:.2f}"
        geom.set("rgba", " ".join(rgba))

    g1_root.find("worldbody").append(srb_body)
    g1_tree.write(output, encoding="unicode", xml_declaration=False)
    print(f"[pipeline] Combined XML written → {output}")

# ---------------------------------------------------------------------------
# Step 6 — visualize
# ---------------------------------------------------------------------------

def visualize(q_ik: np.ndarray, q_srb: np.ndarray, times: np.ndarray,
              combined_xml: str, dt_wall: float = 0.04):
    """
    Animate the combined model.  Both the G1 humanoid and the SRB ellipsoid
    are driven by their respective trajectories.
    """
    mj_model = mujoco.MjModel.from_xml_path(combined_xml)
    mj_data  = mujoco.MjData(mj_model)

    # Locate where the SRB free joint starts in qpos
    srb_joint_id  = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, "srb_free")
    srb_qpos_adr  = mj_model.jnt_qposadr[srb_joint_id]
    g1_nq         = srb_qpos_adr  # G1 occupies qpos[0:srb_qpos_adr]

    print(f"[pipeline] Combined model: nq={mj_model.nq}, "
          f"G1 qpos[0:{g1_nq}], SRB qpos[{srb_qpos_adr}:{srb_qpos_adr+7}]")
    print(f"[pipeline] {q_ik.shape[0]} frames @ {dt_wall*1000:.0f} ms/frame. "
          "Close viewer to quit.\n")

    # Precompute MuJoCo-format G1 qpos for every frame
    q_g1_mj = np.array([pin_q_to_mj(q_ik[i]) for i in range(q_ik.shape[0])])

    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        # Side view: forward pitch motion is clearly visible
        viewer.cam.lookat[:]  = [0.0, 0.0, 0.6]
        viewer.cam.distance   = 3.2
        viewer.cam.azimuth    = 90.0
        viewer.cam.elevation  = -12.0

        while viewer.is_running():
            for i in range(q_ik.shape[0]):
                if not viewer.is_running():
                    break

                # G1 humanoid (pinocchio → MuJoCo quat)
                mj_data.qpos[:g1_nq] = q_g1_mj[i]

                # SRB ellipsoid (already MuJoCo quat convention from q_opt.csv)
                mj_data.qpos[srb_qpos_adr : srb_qpos_adr + 7] = q_srb[i]

                mj_data.qvel[:] = 0
                mujoco.mj_forward(mj_model, mj_data)
                viewer.sync()
                time.sleep(dt_wall)

            time.sleep(0.5)  # pause at end before looping

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      default="srb.config.smalljump",
                        help="SRB config module (default: srb.config.smalljump)")
    parser.add_argument("--skip-solve",  action="store_true",
                        help="Skip running the SRB solver (reuse existing CSVs)")
    parser.add_argument("--skip-ik",     action="store_true",
                        help="Skip IK (reuse existing ik_q_pin.csv)")
    parser.add_argument("--frame-ms",    type=float, default=40.0,
                        help="Wall-clock ms per animation frame (default: 40)")
    args = parser.parse_args()

    # 1. SRB solver
    if not args.skip_solve:
        run_srb_solver(args.config)

    # 2. Load results
    times, q_srb, feet_ext, stance_end, flight_end = load_srb_results(_RESULTS)

    # 3. IK
    ik = G1IK()
    if not args.skip_ik:
        q_ik = run_ik_trajectory(ik, q_srb, feet_ext, stance_end, flight_end)
        # 4. Export
        export_ik_solution(q_ik, times, _EXPORT_DIR)
    else:
        q_ik_mj = np.loadtxt(os.path.join(_EXPORT_DIR, "ik_q_mujoco.csv"),
                              delimiter=",", comments="#")
        # Convert back to pinocchio for internal use (just flip quat)
        q_ik = np.array([np.concatenate([r[:3],
                                          [r[4], r[5], r[6], r[3]],
                                          r[7:]]) for r in q_ik_mj])
        print(f"[pipeline] Loaded cached IK solution ({q_ik.shape[0]} frames)")

    # 5. Build combined XML
    build_combined_xml(_G1_XML, _SRB_XML, _COMBINED)

    # 6. Visualize
    visualize(q_ik, q_srb, times, _COMBINED, dt_wall=args.frame_ms / 1000.0)
