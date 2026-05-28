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
from utils.kinematics.g1_ipopt_ik import G1IPOPTIK

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_G1_XML     = os.path.join(_REPO_ROOT, "models", "g1", "g1_29dof_rev_1_0.xml")
_SRB_XML    = os.path.join(_REPO_ROOT, "models", "srb", "srb.xml")
_COMBINED   = os.path.join(_REPO_ROOT, "models", "g1", "g1_srb_combined.xml")
_RESULTS_DEFAULT = os.path.join(_REPO_ROOT, "results", "srb", "srb_aerial")
_EXPORT_DIR      = os.path.join(_REPO_ROOT, "ik", "results")

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
# Step 3 — upper-body defaults from traj_opt reference config
# ---------------------------------------------------------------------------

# Default upper-body pose (copied from traj_opt srb_aerial reference config).
# Legs are excluded — IK overwrites those.
_ARM_DEFAULTS = {
    'waist_yaw_joint':              0.0,
    'left_shoulder_pitch_joint':    0.498,
    'left_shoulder_roll_joint':     0.3,
    'left_shoulder_yaw_joint':      0.0,
    'left_elbow_joint':             0.501,
    'right_shoulder_pitch_joint':   0.498,
    'right_shoulder_roll_joint':   -0.3,
    'right_shoulder_yaw_joint':     0.0,
    'right_elbow_joint':            0.501,
}


def load_arm_defaults(model: pin.Model) -> np.ndarray:
    """Return a full pinocchio q-length array with upper-body defaults filled in."""
    defaults = np.zeros(model.nq)
    for jname, angle in _ARM_DEFAULTS.items():
        try:
            jid   = model.getJointId(jname)
            idx_q = model.joints[jid].idx_q
            defaults[idx_q] = angle
        except Exception:
            pass
    return defaults


# Step 3 — run IK trajectory
# ---------------------------------------------------------------------------

def _mj_to_pin_quat(qw, qx, qy, qz):
    """MuJoCo [qw,qx,qy,qz] → pinocchio [qx,qy,qz,qw]."""
    return np.array([qx, qy, qz, qw])


def _build_q0(ik, q_srb_row: np.ndarray, q_warm=None,
              warmstart_min_drop: float = 0.03):
    """
    Build a pinocchio initial guess from the SRB state row.
    Base pose comes from the SRB trajectory; leg joints are either warm-started
    or set from standing_config's squat-biased heuristic.

    For IPOPT IK: q0[0:3] is used as p_com_des.  The SRB approximates CoM ≈ pelvis,
    but the actual G1 CoM is ~0.089 m below the pelvis.  We correct q0[2] to the
    actual robot CoM (evaluated at the warm-start config) so IPOPT targets a feasible
    height.  The solver will set pelvis_z ≈ SRB height to satisfy com(q) = corrected.
    """
    h = q_srb_row[2]
    q0 = ik.standing_config(h)
    q0[0:3] = q_srb_row[0:3]
    q0[3:7] = _mj_to_pin_quat(q_srb_row[3], q_srb_row[4], q_srb_row[5], q_srb_row[6])
    # Only warm-start joints when meaningfully below standing (avoids wrong branch)
    drop = max(0.0, 0.79 - h)
    if q_warm is not None and drop > warmstart_min_drop:
        q0[7:] = q_warm[7:]
    if isinstance(ik, G1IPOPTIK):
        # Correct the CoM target: compute actual robot CoM at this warm-start config
        # and use that as p_com_des instead of the raw SRB CoM height.
        pin.centerOfMass(ik.model, ik.data, q0)
        q0[2] = ik.data.com[0][2]
    return q0


def run_ik_trajectory(ik: G1IK, q_srb: np.ndarray, feet_ext: np.ndarray,
                      stance_end: int, flight_end: int,
                      arm_defaults: np.ndarray = None):
    """
    Returns q_ik (N+1, 36) pinocchio configuration at every timestep.
    arm_defaults: full nq-length array with upper-body joint angles to hold
                  throughout (from load_arm_defaults).  None = zeros.
    """
    N = q_srb.shape[0] - 1
    q_ik = np.zeros((N + 1, ik.model.nq))

    # Pre-fill upper-body defaults on every frame — IK only overwrites leg joints
    if arm_defaults is not None:
        q_ik[:] = arm_defaults[np.newaxis, :]

    # ---- STANCE ----
    print("[pipeline] Running stance IK ...")
    q_warm = None
    for i in range(stance_end):
        q0 = _build_q0(ik, q_srb[i], q_warm)
        if arm_defaults is not None:
            q0[19:] = arm_defaults[19:]   # IK only updates leg joints; these survive
        oMl = pin.SE3(np.eye(3), np.array([feet_ext[i,0], feet_ext[i,1], ik.ANKLE_HEIGHT]))
        oMr = pin.SE3(np.eye(3), np.array([feet_ext[i,2], feet_ext[i,3], ik.ANKLE_HEIGHT]))
        q_sol, ok, errs = ik.solve(q0, oMl, oMr)
        if not ok and errs[-1] > 1e-3:
            print(f"  [warn] stance frame {i}: IK did not converge (err={errs[-1]:.2e})")
        q_ik[i] = q_sol
        q_warm = q_sol

    # ---- FLIGHT — interpolate leg joints toward touchdown pose ----
    # Pre-compute IK at the first landing frame so flight interpolation ends exactly
    # at the touchdown configuration, eliminating the stance-width snap.
    print("[pipeline] Interpolating flight joints ...")
    joints_takeoff = q_ik[stance_end - 1, 7:].copy()
    # Warm-start touchdown IK from liftoff config — same standing-like pose, avoids cold-start
    q0_td = _build_q0(ik, q_srb[flight_end], q_warm=q_ik[stance_end - 1])
    if arm_defaults is not None:
        q0_td[19:] = arm_defaults[19:]
    oMl_td = pin.SE3(np.eye(3), np.array([feet_ext[flight_end, 0], feet_ext[flight_end, 1], ik.ANKLE_HEIGHT]))
    oMr_td = pin.SE3(np.eye(3), np.array([feet_ext[flight_end, 2], feet_ext[flight_end, 3], ik.ANKLE_HEIGHT]))
    q_td, ok_td, errs_td = ik.solve(q0_td, oMl_td, oMr_td)
    if not ok_td and errs_td[-1] > 1e-3:
        print(f"  [warn] touchdown frame {flight_end}: IK did not converge (err={errs_td[-1]:.2e})")
    joints_touchdown = q_td[7:].copy()

    # Pelvis position during flight — interpolate per-axis CoM→pelvis offsets so
    # the pelvis trajectory is C0-continuous at both phase boundaries.
    # IPOPT can place the pelvis offset from the SRB CoM in all three axes (not
    # just z) when the robot is pitched or the feet are spread asymmetrically.
    # Interpolating all three offsets from liftoff to touchdown values makes the
    # transition smooth without hardcoding any specific CoM–pelvis geometry.
    liftoff_x_offset   = q_ik[stance_end - 1, 0] - q_srb[stance_end - 1, 0]
    liftoff_y_offset   = q_ik[stance_end - 1, 1] - q_srb[stance_end - 1, 1]
    liftoff_z_offset   = q_ik[stance_end - 1, 2] - q_srb[stance_end - 1, 2]
    touchdown_x_offset = q_td[0] - q_srb[flight_end, 0]
    touchdown_y_offset = q_td[1] - q_srb[flight_end, 1]
    touchdown_z_offset = q_td[2] - q_srb[flight_end, 2]
    n_flight = flight_end - stance_end
    for idx, i in enumerate(range(stance_end, flight_end)):
        t = idx / max(1, n_flight - 1)
        q_ik[i, 0]   = q_srb[i, 0] + (1 - t) * liftoff_x_offset + t * touchdown_x_offset
        q_ik[i, 1]   = q_srb[i, 1] + (1 - t) * liftoff_y_offset + t * touchdown_y_offset
        q_ik[i, 2]   = q_srb[i, 2] + (1 - t) * liftoff_z_offset + t * touchdown_z_offset
        q_ik[i, 3:7] = _mj_to_pin_quat(q_srb[i,3], q_srb[i,4], q_srb[i,5], q_srb[i,6])
        q_ik[i, 7:]  = (1 - t) * joints_takeoff + t * joints_touchdown

    # ---- LANDING ----
    # w_reg penalises ||q_legs - q_prev_legs||² to keep the solver on the same
    # joint-space branch across consecutive frames.  A small weight (0.05) is
    # enough to break ties between equivalent solutions without fighting the
    # foot-placement objective (w_foot=1, foot errors are ~mm²).
    print("[pipeline] Running landing IK ...")
    q_ik[flight_end] = q_td   # already solved above
    q_warm = q_td
    for i in range(flight_end + 1, N + 1):
        q0 = _build_q0(ik, q_srb[i], q_warm)
        if arm_defaults is not None:
            q0[19:] = arm_defaults[19:]
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


# Joint order expected by downstream consumers (matches yaml_to_csv.py TARGET_JOINTS)
_TARGET_JOINTS = [
    'left_hip_pitch_joint',   'left_hip_roll_joint',    'left_hip_yaw_joint',
    'left_knee_joint',        'left_ankle_pitch_joint',  'left_ankle_roll_joint',
    'right_hip_pitch_joint',  'right_hip_roll_joint',   'right_hip_yaw_joint',
    'right_knee_joint',       'right_ankle_pitch_joint', 'right_ankle_roll_joint',
    'waist_yaw_joint',        'waist_roll_joint',        'waist_pitch_joint',
    'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint',
    'left_elbow_joint',
    'left_wrist_roll_joint',  'left_wrist_pitch_joint',  'left_wrist_yaw_joint',
    'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint',
    'right_elbow_joint',
    'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint',
]
_BASE_COLS  = ['com_x', 'com_y', 'com_z', 'quat_x', 'quat_y', 'quat_z', 'quat_w']
_OUT_HEADER = ['time'] + _BASE_COLS + _TARGET_JOINTS


def _resample_and_save(out: np.ndarray, export_dir: str, hz: float = 50.0):
    """
    Resample the full-robot trajectory (variable dt) to a uniform grid at `hz`.
    Column layout: [time, com_xyz, quat_xyzw, 29 joints]  — same as ik_fullrobot.csv.
    Quaternion columns (4:8) are resampled via Slerp; all others via linear interp.
    """
    from scipy.spatial.transform import Rotation, Slerp

    t_orig = out[:, 0]
    dt_out = 1.0 / hz
    t_new  = np.arange(t_orig[0], t_orig[-1] + 1e-9, dt_out)

    n_new  = len(t_new)
    n_cols = out.shape[1]
    resampled = np.zeros((n_new, n_cols))

    # time column
    resampled[:, 0] = t_new

    # position + joints: linear interp
    for col in list(range(1, 4)) + list(range(8, n_cols)):
        resampled[:, col] = np.interp(t_new, t_orig, out[:, col])

    # quaternion: Slerp (columns 4:8 are xyzw)
    rots  = Rotation.from_quat(out[:, 4:8])   # scipy expects xyzw
    slerp = Slerp(t_orig, rots)
    resampled[:, 4:8] = slerp(t_new).as_quat()

    path = os.path.join(export_dir, f"ik_fullrobot_{int(hz)}hz.csv")
    np.savetxt(path, resampled, delimiter=",")
    print(f"[pipeline] Resampled {int(hz)} Hz CSV → {path}  ({n_new} frames)")


def _build_joint_remap(model: pin.Model):
    """
    Returns a list of (src_q_idx, dst_col_idx) pairs that map pinocchio q
    (indices 7…nq-1) into the _TARGET_JOINTS column order.
    Joints absent from pinocchio (e.g. wrists) are left as zero.
    """
    # Pinocchio joint names in q-index order (skip 'universe' and 'root_joint')
    pin_joints = {}
    for jid in range(1, model.njoints):  # 0 = universe
        name = model.names[jid]
        if name == "root_joint":
            continue
        idx_q = model.joints[jid].idx_q
        pin_joints[name] = idx_q

    pairs = []
    for dst_col, jname in enumerate(_TARGET_JOINTS):
        if jname in pin_joints:
            pairs.append((pin_joints[jname], 8 + dst_col))  # 8 = 1 (time) + 7 (base cols)
    return pairs


def export_ik_solution(q_ik: np.ndarray, times: np.ndarray, export_dir: str,
                       model: pin.Model = None):
    os.makedirs(export_dir, exist_ok=True)
    np.savetxt(os.path.join(export_dir, "ik_q_pin.csv"),  q_ik,  delimiter=",",
               header="pinocchio q (nq=36): px,py,pz,qx,qy,qz,qw,joints×29")
    np.savetxt(os.path.join(export_dir, "ik_time.csv"),   times, delimiter=",")
    # MuJoCo-convention qpos for easy loading
    q_mj = np.array([pin_q_to_mj(r) for r in q_ik])
    np.savetxt(os.path.join(export_dir, "ik_q_mujoco.csv"), q_mj, delimiter=",",
               header="MuJoCo qpos (nq=36): px,py,pz,qw,qx,qy,qz,joints×29")

    # Full-robot CSV in yaml_to_csv.py style:
    # time, com_x, com_y, com_z, quat_x, quat_y, quat_z, quat_w, <29 joints>
    if model is not None:
        n_frames = q_ik.shape[0]
        n_cols   = 1 + len(_BASE_COLS) + len(_TARGET_JOINTS)  # time + 7 + 29
        out = np.zeros((n_frames, n_cols))
        out[:, 0]   = times            # time
        out[:, 1:4] = q_ik[:, 0:3]    # com_xyz  (pinocchio px,py,pz)
        out[:, 4:8] = q_ik[:, 3:7]    # quat_xyzw (pinocchio convention)
        # Remap joints from pinocchio q-indices into TARGET_JOINTS columns
        for src_q_idx, dst_col in _build_joint_remap(model):
            out[:, dst_col] = q_ik[:, src_q_idx]
        # Normalise quaternions
        norms = np.linalg.norm(out[:, 4:8], axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        out[:, 4:8] /= norms
        csv_path = os.path.join(export_dir, "ik_fullrobot.csv")
        np.savetxt(csv_path, out, delimiter=",")
        print(f"[pipeline] Full-robot CSV → {csv_path}  "
              f"({n_frames} frames, {len(_TARGET_JOINTS)} joints)")

        # Resample to uniform 50 Hz for downstream use
        _resample_and_save(out, export_dir, hz=50)

    print(f"[pipeline] Exported IK solution ({q_ik.shape[0]} frames) → {export_dir}")

# ---------------------------------------------------------------------------
# Step 4b — IK error report
# ---------------------------------------------------------------------------

def check_ik_errors(q_ik: np.ndarray, times: np.ndarray,
                    feet_ext: np.ndarray, stance_end: int, flight_end: int,
                    ik_model) -> None:
    """
    Print per-frame foot position errors for stance and landing phases.
    Flags any frame where either foot exceeds 1 mm from its target.
    """
    ANKLE_H  = ik_model.ANKLE_HEIGHT
    n_frames = q_ik.shape[0]
    N        = n_frames - 1

    contact_frames = list(range(stance_end)) + list(range(flight_end, N + 1))

    max_err   = 0.0
    bad_frames = []

    print("\n[pipeline] IK error report (stance + landing frames):")
    print(f"  {'frame':>5}  {'time':>6}  {'phase':<7}  {'err_L (m)':>10}  {'err_R (m)':>10}")

    for i in contact_frames:
        q = q_ik[i]
        pin.framesForwardKinematics(ik_model.model, ik_model.data, q)
        p_l = ik_model.data.oMf[ik_model.l_foot_id].translation
        p_r = ik_model.data.oMf[ik_model.r_foot_id].translation
        des_l = np.array([feet_ext[i, 0], feet_ext[i, 1], ANKLE_H])
        des_r = np.array([feet_ext[i, 2], feet_ext[i, 3], ANKLE_H])
        el = np.linalg.norm(p_l - des_l)
        er = np.linalg.norm(p_r - des_r)
        phase = "stance" if i < stance_end else "landing"
        max_err = max(max_err, el, er)
        if el > 1e-3 or er > 1e-3:
            bad_frames.append(i)
            print(f"  {i:>5}  {times[i]:>6.3f}  {phase:<7}  {el:>10.4e}  {er:>10.4e}  ← !")
        elif i % max(1, len(contact_frames) // 10) == 0:
            print(f"  {i:>5}  {times[i]:>6.3f}  {phase:<7}  {el:>10.4e}  {er:>10.4e}")

    status = "OK" if not bad_frames else f"{len(bad_frames)} frame(s) > 1 mm"
    print(f"\n  max foot error: {max_err*1e3:.3f} mm  —  {status}\n")


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
# Step 5b — plot SRB wrenches
# ---------------------------------------------------------------------------

def plot_srb_wrenches(results_dir: str, times: np.ndarray,
                      stance_end: int, flight_end: int,
                      save_path: str = None):
    """
    Plot per-foot forces and ankle moments from the SRB solution.

    Layout: 2 rows × 3 cols
      Row 0: Fx, Fy, Fz  — left (blue), right (orange), total (black dashed)
      Row 1: Mx, My, Mz  — left (blue), right (orange), total (black dashed)
    Phase regions shaded: stance=blue, flight=white, landing=green.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fl_path = os.path.join(results_dir, "force_left.csv")
    if not os.path.exists(fl_path):
        print("[pipeline] force_left.csv not found — run solver first (without --skip-solve)")
        return

    FL = np.loadtxt(fl_path, delimiter=",")
    FR = np.loadtxt(os.path.join(results_dir, "force_right.csv"),  delimiter=",")
    ML = np.loadtxt(os.path.join(results_dir, "moment_left.csv"),  delimiter=",")
    MR = np.loadtxt(os.path.join(results_dir, "moment_right.csv"), delimiter=",")


    N = FL.shape[0]
    t = times[:N]  # control nodes share time index with state[0..N-1]

    t_stance_end = times[stance_end]
    t_flight_end = times[flight_end]

    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
    fig.suptitle("SRB Wrench Trajectory", fontsize=13)

    labels_F = ["Fx (N)", "Fy (N)", "Fz (N)"]
    labels_M = ["Mx (Nm)", "My (Nm)", "Mz (Nm)"]

    for col in range(3):
        ax_f = axes[0, col]
        ax_m = axes[1, col]

        for ax in (ax_f, ax_m):
            ax.axvspan(t[0],         t_stance_end, alpha=0.08, color="steelblue",  zorder=0)
            ax.axvspan(t_flight_end, t[-1],        alpha=0.08, color="seagreen",   zorder=0)
            ax.axvline(t_stance_end, color="steelblue", lw=0.8, ls="--")
            ax.axvline(t_flight_end, color="seagreen",  lw=0.8, ls="--")
            ax.axhline(0, color="gray", lw=0.5)
            ax.set_xlabel("time (s)")

        ax_f.plot(t, FL[:, col], color="steelblue", lw=1.2, label="Left")
        ax_f.plot(t, FR[:, col], color="darkorange", lw=1.2, label="Right")
        ax_f.plot(t, FL[:, col] + FR[:, col], color="black", lw=1.0, ls="--", label="Total")
        ax_f.set_ylabel(labels_F[col])

        ax_m.plot(t, ML[:, col], color="steelblue", lw=1.2, label="Left")
        ax_m.plot(t, MR[:, col], color="darkorange", lw=1.2, label="Right")
        ax_m.plot(t, ML[:, col] + MR[:, col], color="black", lw=1.0, ls="--", label="Total")
        ax_m.set_ylabel(labels_M[col])

        if col == 2:
            ax_f.legend(fontsize=7, loc="upper right")
            ax_m.legend(fontsize=7, loc="upper right")

    phase_patches = [
        mpatches.Patch(color="steelblue", alpha=0.3, label="Stance"),
        mpatches.Patch(color="white",     alpha=0.0, label="Flight",  ec="gray"),
        mpatches.Patch(color="seagreen",  alpha=0.3, label="Landing"),
    ]
    fig.legend(handles=phase_patches, loc="lower center", ncol=3,
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.01))

    fig.tight_layout(rect=[0, 0.04, 1, 1])

    out = save_path or os.path.join(results_dir, "srb_wrenches.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[pipeline] Wrench plot saved → {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Step 6 — visualize
# ---------------------------------------------------------------------------

def visualize(q_ik: np.ndarray, q_srb: np.ndarray, times: np.ndarray,
              combined_xml: str, speed: float = 1.0):
    """
    Animate the combined model in real time.
    Each frame is displayed for its actual trajectory duration (times[i+1]-times[i])
    divided by speed, so playback is physically correct across variable-dt phases.
    speed < 1 → slow motion, speed > 1 → fast forward.
    """
    mj_model = mujoco.MjModel.from_xml_path(combined_xml)
    mj_data  = mujoco.MjData(mj_model)

    srb_joint_id  = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, "srb_free")
    srb_qpos_adr  = mj_model.jnt_qposadr[srb_joint_id]
    g1_nq         = srb_qpos_adr

    # Per-frame sleep durations from the actual trajectory timestamps
    dts = np.diff(times)           # shape (N,)
    dts = np.append(dts, dts[-1])  # repeat last dt for the terminal frame

    total_t = times[-1] - times[0]
    print(f"[pipeline] Combined model: nq={mj_model.nq}, "
          f"G1 qpos[0:{g1_nq}], SRB qpos[{srb_qpos_adr}:{srb_qpos_adr+7}]")
    print(f"[pipeline] {q_ik.shape[0]} frames, {total_t:.2f}s trajectory "
          f"(speed={speed}x). Close viewer to quit.\n")

    q_g1_mj = np.array([pin_q_to_mj(q_ik[i]) for i in range(q_ik.shape[0])])

    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        viewer.cam.lookat[:]  = [0.0, 0.0, 0.6]
        viewer.cam.distance   = 3.2
        viewer.cam.azimuth    = 90.0
        viewer.cam.elevation  = -12.0

        while viewer.is_running():
            for i in range(q_ik.shape[0]):
                if not viewer.is_running():
                    break

                mj_data.qpos[:g1_nq] = q_g1_mj[i]
                mj_data.qpos[srb_qpos_adr : srb_qpos_adr + 7] = q_srb[i]
                mj_data.qvel[:] = 0
                mujoco.mj_forward(mj_model, mj_data)
                viewer.sync()
                time.sleep(dts[i] / speed)

            time.sleep(0.5)  # pause before looping

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
    parser.add_argument("--speed",        type=float, default=1.0,
                        help="Playback speed multiplier (default: 1.0 = real-time)")
    args = parser.parse_args()

    # 1. SRB solver
    if not args.skip_solve:
        run_srb_solver(args.config)

    # 2. Load results — derive path from config's save_dir if available
    try:
        import importlib
        _cfg_mod = importlib.import_module(args.config)
        _results_dir = os.path.join(_REPO_ROOT, _cfg_mod.config.save_dir.lstrip("./"))
    except Exception:
        _results_dir = _RESULTS_DEFAULT
    times, q_srb, feet_ext, stance_end, flight_end = load_srb_results(_results_dir)

    # 3. IK — export into a config-specific subdirectory so configs don't overwrite each other
    export_dir = os.path.join(_results_dir, "ik")
    os.makedirs(export_dir, exist_ok=True)
    ik = G1IPOPTIK()
    arm_defaults = load_arm_defaults(ik.model)
    if not args.skip_ik:
        q_ik = run_ik_trajectory(ik, q_srb, feet_ext, stance_end, flight_end,
                                  arm_defaults=arm_defaults)
        # 4. Export
        export_ik_solution(q_ik, times, export_dir, model=ik.model)
    else:
        q_ik_mj = np.loadtxt(os.path.join(export_dir, "ik_q_mujoco.csv"),
                              delimiter=",", comments="#")
        # Convert back to pinocchio for internal use (just flip quat)
        q_ik = np.array([np.concatenate([r[:3],
                                          [r[4], r[5], r[6], r[3]],
                                          r[7:]]) for r in q_ik_mj])
        print(f"[pipeline] Loaded cached IK solution ({q_ik.shape[0]} frames)")

    # 4b. IK error report
    check_ik_errors(q_ik, times, feet_ext, stance_end, flight_end, ik)

    # 5. Wrench plot
    plot_srb_wrenches(_results_dir, times, stance_end, flight_end)

    # 5b. Build combined XML
    build_combined_xml(_G1_XML, _SRB_XML, _COMBINED)

    # 6. Visualize
    visualize(q_ik, q_srb, times, _COMBINED, speed=args.speed)
