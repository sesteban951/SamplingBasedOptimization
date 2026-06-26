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
_G1_URDF    = os.path.join(_REPO_ROOT, "models", "g1", "g1_29dof_rev_1_0.urdf")
_G1_PKG_DIRS = [os.path.join(_REPO_ROOT, "models", "g1")]
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


def _srb_body_yaw(q_srb_row: np.ndarray) -> float:
    """Body yaw [rad] from an SRB state row (MuJoCo quat in cols 3:7 = qw,qx,qy,qz)."""
    qw, qx, qy, qz = q_srb_row[3], q_srb_row[4], q_srb_row[5], q_srb_row[6]
    return float(np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy**2 + qz**2)))


def _ctrl_frame_R(yaw: float) -> np.ndarray:
    """Control-frame foot orientation: flat on the ground (z up), heading = body yaw.
    This is Rz(yaw) — the leveled body-heading frame the contact feet should align to."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


_COM_PELVIS_NOMINAL: float = None  # cached at first call


def _get_com_pelvis_dz(ik) -> float:
    """CoM-pelvis z offset at nominal standing pose (~-0.089 m). Computed once."""
    global _COM_PELVIS_NOMINAL
    if _COM_PELVIS_NOMINAL is None:
        q_nom = ik.standing_config(0.79)
        pin.centerOfMass(ik.model, ik.data, q_nom)
        _COM_PELVIS_NOMINAL = float(ik.data.com[0][2] - q_nom[2])
    return _COM_PELVIS_NOMINAL


def _build_q0(ik, q_srb_row: np.ndarray, q_warm=None,
              warmstart_min_drop: float = 0.03, stance_ground_z: float = 0.0):
    """
    Build a pinocchio initial guess from the SRB state row.
    Base pose comes from the SRB trajectory; leg joints are either warm-started
    or set from standing_config's squat-biased heuristic.

    For IPOPT IK: q0[0:3] is used as p_com_des.  The SRB stores pelvis position,
    not the full-body CoM, so q_srb[2] must be shifted by the nominal CoM-pelvis
    z offset (_COM_PELVIS_NOMINAL) to obtain the correct CoM target for IPOPT.
    A fixed nominal offset is used (not recomputed from q_warm) so the CoM target
    tracks the SRB smoothly across all frames and phase transitions.

    stance_ground_z: elevation of the stance surface (e.g. box height).  Used to
    compute the correct squat depth for the joint initial guess.
    """
    h = q_srb_row[2]
    # Joint angles depend on CoM height *above the feet*, not above world z=0.
    q0 = ik.standing_config(h - stance_ground_z)
    q0[0:3] = q_srb_row[0:3]
    q0[3:7] = _mj_to_pin_quat(q_srb_row[3], q_srb_row[4], q_srb_row[5], q_srb_row[6])
    # Only warm-start joints when meaningfully below standing (avoids wrong branch)
    drop = max(0.0, 0.79 - (h - stance_ground_z))
    if q_warm is not None and drop > warmstart_min_drop:
        q0[7:] = q_warm[7:]
    if isinstance(ik, G1IPOPTIK):
        # Use the nominal standing CoM-pelvis z offset (constant) so that
        # p_com_des[2] = q_srb[2] + offset tracks the SRB smoothly across all
        # frames and phase transitions.  The previous approach recomputed the
        # offset from q_warm each frame, which introduced discontinuities at
        # phase boundaries where q_warm's body orientation changed abruptly
        # (e.g. liftoff → touchdown), causing 9 cm jumps in the CoM target.
        q0[2] = q_srb_row[2] + _get_com_pelvis_dz(ik)
    return q0


def _row_to_I_mat(row: np.ndarray) -> np.ndarray:
    """[Ixx,Iyy,Izz,Ixy,Ixz,Iyz] → symmetric 3×3 numpy matrix."""
    return np.array([
        [row[0], row[3], row[4]],
        [row[3], row[1], row[5]],
        [row[4], row[5], row[2]],
    ])


# Nominal standing Iyy [kg·m²] — from srb.xml; used to compute tuck ratio.
_IYY_NOM = 3.301

# Target flight tuck configuration (leg joints only, 12-vector).
# Chosen via viz_crouch_configs.py at t=0.90 — hip=-2.25, knee=2.59, ankle=+0.45.
# Negative hip_pitch = hip flexion (thigh/foot swings forward+up toward chest).
_FLIGHT_TUCK_LEGS = np.array([
    -2.247, 0.0, 0.0,  2.588,  0.447, 0.0,   # left  hip_p/r/y, knee, ankle_p/r
    -2.247, 0.0, 0.0,  2.588,  0.447, 0.0,   # right
])

def _tuck_q0(ik, q_srb_row: np.ndarray, tuck_ratio: float,
             q_legs_base: np.ndarray,
             arm_defaults: np.ndarray = None) -> np.ndarray:
    """
    Generate a tuck-biased initial guess for flight IK.

    tuck_ratio ∈ [0, 1]: 0 = q_legs_base, 1 = _FLIGHT_TUCK_LEGS.
    q_legs_base: 12-vector of actual leg joints at the tuck=0 boundary
                 (liftoff joints for flight, touchdown joints reversed for landing).

    Blending between real robot states means the warm-start is physically
    continuous at both phase boundaries, regardless of body orientation.
    """
    q = pin.neutral(ik.model)
    q[0:3] = q_srb_row[0:3]
    q[3:7] = _mj_to_pin_quat(q_srb_row[3], q_srb_row[4], q_srb_row[5], q_srb_row[6])
    if arm_defaults is not None:
        q[19:] = arm_defaults[19:]

    tr = float(np.clip(tuck_ratio, 0.0, 1.0))
    q_legs = (1.0 - tr) * q_legs_base + tr * _FLIGHT_TUCK_LEGS
    for name, val in zip(G1IK.LEG_JOINTS, q_legs):
        jid = ik.model.getJointId(name)
        q[ik.model.joints[jid].idx_q] = val

    return q


def _build_collision_checker(model):
    """Lazily build a pinocchio collision model+data for self-collision queries."""
    geom_model = pin.GeometryModel()
    pin.buildGeomFromUrdf(model, _G1_URDF, pin.COLLISION, geom_model, _G1_PKG_DIRS)
    geom_model.addAllCollisionPairs()
    return geom_model, geom_model.createData()


def _check_self_collision(model, data, geom_model, geom_data, q) -> bool:
    pin.computeCollisions(model, data, geom_model, geom_data, q, True)
    return any(geom_data.collisionResults[k].isCollision()
               for k in range(len(geom_model.collisionPairs)))


def _quat_slerp(q0: np.ndarray, q1: np.ndarray, a: float) -> np.ndarray:
    """Shortest-path SLERP between two pinocchio-convention quaternions [qx,qy,qz,qw]."""
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:          # take the shorter arc
        q1, dot = -q1, -dot
    if dot > 0.9995:       # nearly aligned — linear blend avoids 0/0
        q = (1.0 - a) * q0 + a * q1
        return q / np.linalg.norm(q)
    theta = np.arccos(dot)
    s = np.sin(theta)
    return (np.sin((1.0 - a) * theta) / s) * q0 + (np.sin(a * theta) / s) * q1


def _solve_contact_frame(ik, q0: np.ndarray, oMl: pin.SE3, oMr: pin.SE3,
                         q_warm, solve_kwargs: dict,
                         com_step_max: float = 0.02, ang_step_max: float = 0.1,
                         n_sub_cap: int = 20, verify_step: float = 0.5):
    """
    Continuation solve for a fully-determined contact-frame IK.

    The contact IK (CoM + both feet pose, all hard) has DOF == #constraints, so the
    solution is unique but very sensitive to the warm start.  Two opposite failure
    modes appear near full leg extension (a kinematic singularity):

      * Landing squat — CoM drops fast while the body is pitched.  A single
        full-target solve from the previous frame falls out of its basin, returns a
        stale (straight-leg) iterate, and the knee snaps a frame later.  Small steps
        FIX this.
      * Stance pre-jump — the leg starts at full extension.  Here a *small* step has
        a near-zero CoM-vs-knee gradient and stalls at the straight singularity,
        whereas a single large solve has enough push to escape onto the bent branch.
        Small steps make this WORSE.

    Resolution — the determined IK also has multiple solution branches, so a solve
    can *converge* yet jump to a far branch (a snap).  We therefore judge a solve by
    BOTH convergence and step size relative to the previous config:

      1. Direct full-target solve.  If it converges with a SMALL step, accept it
         (fast path; also the stance escape when the big single step is genuinely
         needed is handled in step 3).
      2. Otherwise (failed, or a large step that may be a branch flip) run a
         continuation: walk the CoM target + pelvis orientation from the previous
         converged config to the target in small sub-steps, each warm-started from
         the last, to stay on the NEAREST branch.  Feet are planted (constant within
         a contact phase), so only CoM and orientation are interpolated.
      3. Prefer the continuation result when it converged AND its step is no larger
         than the direct solve's (the landing squat: continuation finds the smooth
         on-branch path the direct solve snapped past).  Keep the direct solve when
         continuation stalls (the stance singularity: small steps can't escape full
         extension, so the big direct step is the only valid branch).

    q_warm=None → single direct solve (cold start, e.g. first stance frame).
    solve_kwargs: forwarded to ik.solve WITHOUT q_prev (set per sub-step here).
    Returns (q_sol, ok, errs).
    """
    if q_warm is None:
        return ik.solve(q0, oMl, oMr, q_prev=None, **solve_kwargs)

    leg = ik._leg_qidx
    def _leg_step(qa, qb):
        return float(np.linalg.norm(np.array([qa[i] - qb[i] for i in leg])))

    def _continuation():
        com_to  = np.asarray(q0[0:3], dtype=float)
        quat_to = np.asarray(q0[3:7], dtype=float)
        pin.centerOfMass(ik.model, ik.data, q_warm)
        com_from  = np.array(ik.data.com[0])
        quat_from = np.asarray(q_warm[3:7], dtype=float)

        d_com = float(np.linalg.norm(com_to - com_from))
        dot   = abs(float(np.dot(quat_from / np.linalg.norm(quat_from),
                                 quat_to  / np.linalg.norm(quat_to))))
        d_ang = 2.0 * np.arccos(min(1.0, dot))
        n_sub = max(1, int(np.ceil(max(d_com / com_step_max, d_ang / ang_step_max))))
        n_sub = min(n_sub, n_sub_cap)

        q_seed = q_warm.copy()
        q_s, ok_s, errs_s = q_warm, False, [float("inf")]
        for s in range(1, n_sub + 1):
            a = s / n_sub
            q0_s = q_seed.copy()
            q0_s[0:3] = (1.0 - a) * com_from + a * com_to
            q0_s[3:7] = _quat_slerp(quat_from, quat_to, a)
            q_s, ok_s, errs_s = ik.solve(q0_s, oMl, oMr, q_prev=q_seed, **solve_kwargs)
            q_seed = q_s
        return q_s, ok_s, errs_s

    # 1) Direct full-target solve.
    q_dir, ok_dir, e_dir = ik.solve(q0, oMl, oMr, q_prev=q_warm, **solve_kwargs)
    if ok_dir and _leg_step(q_dir, q_warm) <= verify_step:
        return q_dir, ok_dir, e_dir

    # 2) Failed or large step (possible branch flip): refine on the nearest branch.
    q_con, ok_con, e_con = _continuation()

    # 3) Prefer continuation when it converged and is no worse a step than direct;
    #    otherwise keep direct (stance singularity escape).
    if ok_con and (not ok_dir or _leg_step(q_con, q_warm) <= _leg_step(q_dir, q_warm)):
        return q_con, ok_con, e_con
    return q_dir, ok_dir, e_dir


def run_ik_trajectory(ik: G1IK, q_srb: np.ndarray, feet_ext: np.ndarray,
                      stance_end: int, flight_end: int,
                      arm_defaults: np.ndarray = None,
                      stance_ground_z: float = 0.0,
                      tuck_opt: np.ndarray = None,
                      w_sym_flight: float = 2.0,
                      w_reg_flight: float = 1.0,
                      reject_self_collision: bool = False,
                      times: np.ndarray = None,
                      q_dot_max_flight: float = 10.0,
                      control_frame: bool = False,
                      free_foot_yaw: bool = False):
    """
    Returns q_ik (N+1, 36) pinocchio configuration at every timestep.

    arm_defaults:    full nq-length array with upper-body joint angles to hold
                     throughout (from load_arm_defaults).  None = zeros.
    stance_ground_z: height of the stance surface above z=0 (e.g. box top).
    tuck_opt:        (N_flight+1,) tuck scalar per flight node from the SRB solver.
                     When provided, flight IK uses tuck-interpolated warm starts
                     (liftoff joints → _FLIGHT_TUCK_LEGS → touchdown joints).
                     None → flight frames use linear joint interpolation only.
    w_sym_flight:    bilateral symmetry cost weight during flight.
    w_reg_flight:    joint regularisation weight during flight.
    times:           (N+1,) timestamps [s] for per-frame dt.  None → dt=0.02 s.
    q_dot_max_flight: max joint velocity [rad/s] box constraint during flight IK.
    control_frame:   if True, contact (stance + landing) feet are pinned RIGIDLY to the
                     leveled body-heading frame Rz(body_yaw) — a flat-sole foot that
                     points straight in the SRB body heading (the "straight 90deg feet
                     turn").  False (default) → feet pinned rigidly to world-forward
                     (identity), the original behaviour.  Orthogonal to free_foot_yaw:
                     this chooses the *target* orientation; free_foot_yaw chooses whether
                     foot yaw is held rigid or left free.
    free_foot_yaw:   if True, contact feet are held flat on the ground but free to rotate
                     about vertical (foot yaw is a free DOF in the solve), so the heading
                     is whatever the IK prefers (needed when the body yaws while planted,
                     e.g. a twist).  False (default) → foot yaw held rigid to the target
                     orientation (identity, or Rz(body_yaw) when control_frame=True).
    """
    N = q_srb.shape[0] - 1
    q_ik = np.zeros((N + 1, ik.model.nq))

    stance_foot_z  = ik.ANKLE_HEIGHT + stance_ground_z
    landing_foot_z = ik.ANKLE_HEIGHT  # landing always on ground

    use_ipopt      = isinstance(ik, G1IPOPTIK)
    run_flight_ik  = (tuck_opt is not None) and use_ipopt

    # Per-frame dt array — used for velocity box constraints in flight IK.
    _nom_dt = 0.02
    if times is not None:
        _dt_arr = np.diff(times.astype(float))
        _dt_arr = np.concatenate([[_nom_dt], _dt_arr])
    else:
        _dt_arr = np.full(N + 1, _nom_dt)

    geom_model = geom_data = None
    if reject_self_collision:
        print("[pipeline] Building self-collision model ...")
        geom_model, geom_data = _build_collision_checker(ik.model)
        print(f"  {len(geom_model.collisionPairs)} collision pairs loaded.")

    # Pre-fill upper-body defaults on every frame — IK only overwrites leg joints
    if arm_defaults is not None:
        q_ik[:] = arm_defaults[np.newaxis, :]

    # ---- STANCE ----
    print("[pipeline] Running stance IK ...")
    q_warm = None
    p_l_prev = np.array([feet_ext[0, 0], feet_ext[0, 1], stance_foot_z])
    p_r_prev = np.array([feet_ext[0, 2], feet_ext[0, 3], stance_foot_z])
    # Two orthogonal toggles govern contact-foot orientation:
    #   ctrl_align (control_frame): pick the foot TARGET orientation.  When on, align the
    #     feet to the leveled SRB body-heading frame Rz(body_yaw), evaluated at the plant
    #     frame and held across the phase (no skid) — the straight 90deg feet turn.  Stance
    #     plants at frame 0 (body yaw ~0).  Off → world-forward (identity), the original.
    #   ff_yaw (free_foot_yaw): whether foot yaw is held rigid to that target or left free
    #     (flat sole, heading chosen by the solve).  Passed through to ik.solve.
    ctrl_align = control_frame and use_ipopt
    ff_yaw     = free_foot_yaw and use_ipopt
    R_ctrl_st = _ctrl_frame_R(_srb_body_yaw(q_srb[0])) if ctrl_align else np.eye(3)
    if ctrl_align:
        print(f"  [pipeline] stance feet aligned to body yaw "
              f"{np.degrees(_srb_body_yaw(q_srb[0])):+.1f}deg (control frame)")
    if ff_yaw:
        print(f"  [pipeline] contact feet: flat sole, yaw free (free_foot_yaw=True)")
    for i in range(stance_end):
        q0 = _build_q0(ik, q_srb[i], q_warm, stance_ground_z=stance_ground_z)
        if arm_defaults is not None:
            q0[19:] = arm_defaults[19:]
        oMl = pin.SE3(R_ctrl_st, np.array([feet_ext[i,0], feet_ext[i,1], stance_foot_z]))
        oMr = pin.SE3(R_ctrl_st, np.array([feet_ext[i,2], feet_ext[i,3], stance_foot_z]))
        q_sol, ok, errs = _solve_contact_frame(
            ik, q0, oMl, oMr, q_warm,
            dict(w_reg=0.1 if q_warm is not None else 0.0,
                 p_l_prev=p_l_prev, p_r_prev=p_r_prev,
                 w_foot_vel=0.1 if use_ipopt else 0.0,
                 floor_z=stance_foot_z if use_ipopt else -1000.0,
                 foot_hard=use_ipopt,
                 free_foot_yaw=ff_yaw,   # rigid to target, or flat-sole/free-yaw
                 dt=float(_dt_arr[i])))
        if not ok and errs[-1] > 1e-3:
            print(f"  [warn] stance frame {i}: IK did not converge (err={errs[-1]:.2e})")
        if geom_model and _check_self_collision(ik.model, ik.data, geom_model, geom_data, q_sol):
            print(f"  [warn] stance frame {i}: self-collision detected")
        q_ik[i] = q_sol
        q_warm = q_sol
        pin.framesForwardKinematics(ik.model, ik.data, q_sol)
        p_l_prev = ik.data.oMf[ik.l_foot_id].translation.copy()
        p_r_prev = ik.data.oMf[ik.r_foot_id].translation.copy()

    # ---- FLIGHT — interpolate leg joints, then refine with inertia IK ----
    # Pre-compute IK at the first landing frame so flight interpolation ends exactly
    # at the touchdown configuration, eliminating the stance-width snap.
    print("[pipeline] Interpolating flight joints ...")
    joints_takeoff = q_ik[stance_end - 1, 7:].copy()
    # Warm-start touchdown IK from liftoff config — same standing-like pose, avoids cold-start
    # For touchdown, always use q_warm joints (warmstart_min_drop=0) — the liftoff
    # configuration is the best available warm-start for landing, regardless of squat depth.
    q0_td = _build_q0(ik, q_srb[flight_end], q_warm=q_ik[stance_end - 1],
                      warmstart_min_drop=-1.0)
    if arm_defaults is not None:
        q0_td[19:] = arm_defaults[19:]
    # Landing feet align to the SRB body yaw at touchdown (control frame), held across
    # the landing phase.  R_ctrl_td flat + body-heading; identity (forward) when off.
    R_ctrl_td = _ctrl_frame_R(_srb_body_yaw(q_srb[flight_end])) if ctrl_align else np.eye(3)
    if ctrl_align:
        print(f"  [pipeline] landing feet aligned to body yaw "
              f"{np.degrees(_srb_body_yaw(q_srb[flight_end])):+.1f}deg (control frame)")
    oMl_td = pin.SE3(R_ctrl_td, np.array([feet_ext[flight_end, 0], feet_ext[flight_end, 1], landing_foot_z]))
    oMr_td = pin.SE3(R_ctrl_td, np.array([feet_ext[flight_end, 2], feet_ext[flight_end, 3], landing_foot_z]))
    q_td, ok_td, errs_td = ik.solve(q0_td, oMl_td, oMr_td,
                                     q_prev=q0_td, w_reg=0.1, w_sym=2.0,
                                     floor_z=landing_foot_z if use_ipopt else -1000.0,
                                     foot_hard=use_ipopt,
                                     free_foot_yaw=ff_yaw,   # rigid to target, or flat-sole/free-yaw
                                     q_dot_max=0.0)
    if not ok_td and errs_td[-1] > 1e-3:
        print(f"  [warn] touchdown frame {flight_end}: IK did not converge (err={errs_td[-1]:.2e})")
    if geom_model and _check_self_collision(ik.model, ik.data, geom_model, geom_data, q_td):
        print(f"  [warn] touchdown frame {flight_end}: self-collision detected")
    joints_touchdown = q_td[7:].copy()

    # Pelvis position during flight — interpolate per-axis CoM→pelvis offsets so
    # the pelvis trajectory is C0-continuous at both phase boundaries.
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

    # Flight IK — run per-frame IPOPT solve when tuck_opt is available.
    # Warm-start blends from actual liftoff joints → _FLIGHT_TUCK_LEGS → touchdown
    # joints, matching the SRB tuck profile.  No inertia cost — the warm start
    # already encodes the tuck geometry; CoM equality + reg handles the rest.
    if run_flight_ik:
        print("[pipeline] Running flight tuck IK ...")
        oMl_dummy = pin.SE3(np.eye(3), np.array([0.0,  ik.HIP_WIDTH, ik.ANKLE_HEIGHT]))
        oMr_dummy = pin.SE3(np.eye(3), np.array([0.0, -ik.HIP_WIDTH, ik.ANKLE_HEIGHT]))

        q_liftoff_legs   = q_ik[stance_end - 1, 7:19].copy()
        q_touchdown_legs = joints_touchdown[:12].copy()
        q_flight_warm    = q_ik[stance_end - 1].copy()

        _flight_ik_end = max(stance_end, flight_end - 5)
        for idx, i in enumerate(range(stance_end, _flight_ik_end)):
            tuck_ratio = float(np.clip(tuck_opt[idx], 0.0, 1.0))
            alpha_fl   = idx / max(1, n_flight - 1)

            # Warm-start boundary blends from liftoff legs (start of flight) to
            # touchdown legs (end of flight) so the last flight frame naturally
            # converges to a configuration close to the first landing frame.
            q_legs_base = (1.0 - alpha_fl) * q_liftoff_legs + alpha_fl * q_touchdown_legs
            q0_fl = _tuck_q0(ik, q_srb[i], tuck_ratio, q_legs_base, arm_defaults)

            # CoM target: use offset from last converged frame (correct foot placement).
            pin.centerOfMass(ik.model, ik.data, q_flight_warm)
            com_pelvis_dz = ik.data.com[0][2] - q_flight_warm[2]
            q0_fl[0:2] = q_srb[i, 0:2]
            q0_fl[2]   = q_srb[i, 2] + com_pelvis_dz

            q_fl, ok_fl, _ = ik.solve(q0_fl, oMl_dummy, oMr_dummy,
                                       q_prev=q0_fl,        # reg pulls toward tuck target
                                       w_reg=w_reg_flight,
                                       w_foot=0.0,
                                       w_sym=w_sym_flight,
                                       q_dot_max=0.0,       # no velocity box in flight
                                       dt=float(_dt_arr[i]))
            if ok_fl:
                q_ik[i, 7:19] = q_fl[7:19]
                q_flight_warm = q_fl
            else:
                print(f"  [warn] flight frame {i} (tuck={tuck_ratio:.2f}): IK did not converge — keeping interpolated joints")
                q_flight_warm = q_ik[i].copy()

        # Re-interpolate the tail frames (_flight_ik_end..flight_end-1) from the
        # last tuck IK joints to joints_touchdown.  The original linear interpolation
        # used liftoff→touchdown, which is now discontinuous with the tuck IK result.
        if _flight_ik_end < flight_end:
            q_tuck_end = q_ik[_flight_ik_end - 1, 7:19].copy()
            n_tail = flight_end - _flight_ik_end + 1  # includes touchdown frame
            for idx2, i in enumerate(range(_flight_ik_end, flight_end)):
                t2 = (idx2 + 1) / n_tail
                q_ik[i, 7:19] = (1.0 - t2) * q_tuck_end + t2 * joints_touchdown[:12]

    # ---- LANDING ----
    print("[pipeline] Running landing IK ...")
    q_ik[flight_end] = q_td   # already solved above
    q_warm = q_td
    pin.framesForwardKinematics(ik.model, ik.data, q_td)
    p_l_prev = ik.data.oMf[ik.l_foot_id].translation.copy()
    p_r_prev = ik.data.oMf[ik.r_foot_id].translation.copy()

    # Landing feet stay pinned to the touchdown control frame (R_ctrl_td: flat + the SRB
    # body yaw at touchdown), held across the whole landing phase — the foot does not
    # re-choose or skid its heading after planting.
    for j, i in enumerate(range(flight_end + 1, N + 1)):
        # warmstart_min_drop=-1 forces q_warm joints to always be used (avoids
        # standing_config fallback which uses wrong-sign hip_pitch at near-standing height).
        q0 = _build_q0(ik, q_srb[i], q_warm, warmstart_min_drop=-1.0)
        if arm_defaults is not None:
            q0[19:] = arm_defaults[19:]
        oMl = pin.SE3(R_ctrl_td, np.array([feet_ext[i,0], feet_ext[i,1], landing_foot_z]))
        oMr = pin.SE3(R_ctrl_td, np.array([feet_ext[i,2], feet_ext[i,3], landing_foot_z]))
        q_sol, ok, errs = _solve_contact_frame(
            ik, q0, oMl, oMr, q_warm,
            dict(w_reg=0.1,
                 p_l_prev=p_l_prev, p_r_prev=p_r_prev,
                 w_foot_vel=0.1 if use_ipopt else 0.0,
                 w_sym=2.0,
                 floor_z=landing_foot_z if use_ipopt else -1000.0,
                 foot_hard=use_ipopt,
                 free_foot_yaw=ff_yaw,   # rigid to target, or flat-sole/free-yaw
                 dt=float(_dt_arr[i])))
        if not ok and errs[-1] > 1e-3:
            print(f"  [warn] landing frame {i}: IK did not converge (err={errs[-1]:.2e})")
        if geom_model and _check_self_collision(ik.model, ik.data, geom_model, geom_data, q_sol):
            print(f"  [warn] landing frame {i}: self-collision detected")
        q_ik[i] = q_sol
        q_warm = q_sol
        pin.framesForwardKinematics(ik.model, ik.data, q_sol)
        p_l_prev = ik.data.oMf[ik.l_foot_id].translation.copy()
        p_r_prev = ik.data.oMf[ik.r_foot_id].translation.copy()

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

# Columns to drop when converting 29-DOF → 23-DOF output (0-indexed from the
# no-time CSV, i.e. col 0 = com_x).  Corresponds to joint indices 13,14,20,21,27,28
# inside _TARGET_JOINTS: waist_roll, waist_pitch, left_wrist_pitch, left_wrist_yaw,
# right_wrist_pitch, right_wrist_yaw.
_DROP_23DOF = [20, 21, 27, 28, 34, 35]
_DROP_23DOF_NAMES = [_TARGET_JOINTS[c - 7] for c in _DROP_23DOF]


def _check_and_save_23dof(data_29: np.ndarray, export_dir: str, hz: float,
                          prefix: str = "ik_fullrobot") -> None:
    """Drop 6 secondary joints, save 23-DOF CSV, and report whether dropped cols are zero."""
    data_23 = np.delete(data_29, _DROP_23DOF, axis=1)
    path = os.path.join(export_dir, f"{prefix}_{int(hz)}hz_23dof.csv")
    np.savetxt(path, data_23, delimiter=",")

    dropped = data_29[:, _DROP_23DOF]
    max_vals = np.abs(dropped).max(axis=0)
    nonzero = [(col, name, val)
               for col, name, val in zip(_DROP_23DOF, _DROP_23DOF_NAMES, max_vals)
               if val > 1e-9]

    _sep = "#" * 64
    print(f"\n{_sep}")
    print(f"#  23-DOF output  |  {int(hz)} Hz  |  {data_23.shape[0]} frames  |  {data_23.shape[1]} cols")
    print(f"#  File : {path}")
    print(f"#  Dropped cols {_DROP_23DOF}:")
    for col, name in zip(_DROP_23DOF, _DROP_23DOF_NAMES):
        print(f"#    col {col:2d}  {name}")
    if nonzero:
        print(f"#")
        print(f"#  !! WARNING: dropped columns contain non-zero values !!")
        for col, name, val in nonzero:
            print(f"#    col {col:2d}  {name:<35s}  max|val| = {val:.4e}")
        print(f"#  The 23-DOF CSV may silently discard motion — verify before use.")
    else:
        print(f"#  All dropped columns are zero — 23-DOF output is safe to use.")
    print(f"{_sep}\n")


def _resample_and_save(out: np.ndarray, export_dir: str, hz: float = 50.0,
                       prefix: str = "ik_fullrobot"):
    """
    Resample the full-robot trajectory (variable dt) to a uniform grid at `hz`.
    Column layout: [time, com_xyz, quat_xyzw, 29 joints]  — same as ik_fullrobot.csv.
    Quaternion columns (4:8) are resampled via Slerp; all others via linear interp.

    prefix: output filename stem; emits <prefix>_<hz>hz.csv (29-DOF) and
            <prefix>_<hz>hz_23dof.csv.  Defaults to the IK pipeline's name.
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

    path = os.path.join(export_dir, f"{prefix}_{int(hz)}hz.csv")
    data_no_time = resampled[:, 1:]                        # drop time column; freq is fixed
    np.savetxt(path, data_no_time, delimiter=",")

    _sep = "#" * 64
    print(f"\n{_sep}")
    print(f"#  Resampled output: {int(hz)} Hz  |  dt = {dt_out*1000:.1f} ms  |  {n_new} frames")
    print(f"#  File : {path}")
    print(f"#  Cols : com_xyz (3)  quat_xyzw (4)  joints (29)  — NO time column")
    print(f"{_sep}\n")

    _check_and_save_23dof(data_no_time, export_dir, hz, prefix=prefix)


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


def _pad_trajectory(q_ik: np.ndarray, times: np.ndarray,
                    hold_seconds: float = 1.0, dt: float = 0.02):
    """
    Prepend the first frame and append the last frame, each held for hold_seconds.
    Uses a fixed dt for the padding (independent of the trajectory's variable dt).
    Returns (q_padded, times_padded).
    """
    n_pad = max(1, int(round(hold_seconds / dt)))
    pre_times  = times[0]  - dt * np.arange(n_pad, 0, -1)   # e.g. -1.0 … -0.02
    post_times = times[-1] + dt * np.arange(1, n_pad + 1)    # e.g.  T+0.02 … T+1.0
    q_pre  = np.tile(q_ik[0],  (n_pad, 1))
    q_post = np.tile(q_ik[-1], (n_pad, 1))
    q_out  = np.vstack([q_pre,  q_ik,  q_post])
    t_out  = np.concatenate([pre_times, times, post_times])
    return q_out, t_out


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
                    ik_model, stance_ground_z: float = 0.0) -> None:
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
        foot_z = ANKLE_H + (stance_ground_z if i < stance_end else 0.0)
        des_l = np.array([feet_ext[i, 0], feet_ext[i, 1], foot_z])
        des_r = np.array([feet_ext[i, 2], feet_ext[i, 3], foot_z])
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

def build_combined_xml(g1_xml: str, srb_xml: str, output: str,
                       srb_alpha: float = 0.35, box_xml: str = None):
    """
    Appends the SRB visual body (semi-transparent) into the G1 worldbody.
    The combined qpos layout is:
        [0:7]   G1 floating-base free joint
        [7:36]  G1 revolute joints  (29)
        [36:43] SRB free joint

    box_xml: optional path to a box scene XML; its worldbody geoms/bodies are
             merged into the combined scene so the platform appears in the viewer.
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

    worldbody = g1_root.find("worldbody")
    worldbody.append(srb_body)

    # Foot marker mocap bodies (left=blue, right=orange)
    for name, rgba in [("srb_foot_L", "0.2 0.55 1.0 0.9"),
                        ("srb_foot_R", "1.0 0.45 0.1 0.9")]:
        foot_body = ET.SubElement(worldbody, "body", name=name, mocap="true",
                                  pos="0 0 -100")
        ET.SubElement(foot_body, "geom", type="sphere", size="0.04", rgba=rgba,
                      contype="0", conaffinity="0")

    # Merge box geometry into the scene
    if box_xml is not None:
        box_tree = ET.parse(box_xml)
        box_wb   = box_tree.getroot().find("worldbody")
        if box_wb is not None:
            for child in list(box_wb):
                worldbody.append(copy.deepcopy(child))

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
# Step 5c — plot SRB states
# ---------------------------------------------------------------------------

def plot_srb_states(results_dir: str, times: np.ndarray,
                    stance_end: int, flight_end: int,
                    save_path: str = None):
    """
    Plot the SRB state trajectory (position, orientation, velocities).
    If I_opt.csv / tuck_opt.csv are present, an extra row shows tuck + Ixx + Iyy.

    Layout:
      Row 0: px, py, pz  (m)
      Row 1: roll, pitch, yaw  (deg, ZYX Euler from quaternion)
      Row 2: vx, vy, vz  (m/s)
      Row 3: wx, wy, wz  (rad/s, body frame)
      Row 4: tuck, Ixx, Iyy  (kg·m²)  — only when varinertia data exists
    Phase regions shaded: stance=blue, flight=white, landing=green.

    Shows the plot live (plt.show()).  If save_path is given, also saves to that path.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from scipy.spatial.transform import Rotation

    q_path = os.path.join(results_dir, "q_opt.csv")
    if not os.path.exists(q_path):
        print("[pipeline] q_opt.csv not found — run solver first")
        return

    q_srb = np.loadtxt(q_path, delimiter=",")          # (N+1, 7) [px,py,pz,qw,qx,qy,qz]
    v_srb = np.loadtxt(os.path.join(results_dir, "v_opt.csv"), delimiter=",")  # (N+1, 6)

    I_path    = os.path.join(results_dir, "I_opt.csv")
    tuck_path = os.path.join(results_dir, "tuck_opt.csv")
    has_varinertia = os.path.exists(I_path) and os.path.exists(tuck_path)

    if has_varinertia:
        I_opt    = np.loadtxt(I_path,    delimiter=",", comments="#")  # (N+1, 6)
        tuck_opt = np.loadtxt(tuck_path, delimiter=",", comments="#").flatten()

    t = times  # (N+1,)
    t_stance_end = times[stance_end]
    t_flight_end = times[flight_end]

    # Quaternion → ZYX Euler (deg)  — q_srb uses scalar-first [qw,qx,qy,qz]
    quats_xyzw = np.column_stack([q_srb[:, 4:7], q_srb[:, 3]])  # scipy xyzw
    euler_deg  = Rotation.from_quat(quats_xyzw).as_euler("ZYX", degrees=True)
    yaw_deg, pitch_deg, roll_deg = euler_deg[:, 0], euler_deg[:, 1], euler_deg[:, 2]

    n_rows = 5 if has_varinertia else 4
    fig, axes = plt.subplots(n_rows, 3, figsize=(14, 3 * n_rows), sharex=True)
    fig.suptitle("SRB State Trajectory", fontsize=13)

    def _shade(ax):
        ax.axvspan(t[0],         t_stance_end, alpha=0.08, color="steelblue", zorder=0)
        ax.axvspan(t_flight_end, t[-1],        alpha=0.08, color="seagreen",  zorder=0)
        ax.axvline(t_stance_end, color="steelblue", lw=0.8, ls="--")
        ax.axvline(t_flight_end, color="seagreen",  lw=0.8, ls="--")
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_xlabel("time (s)")

    colors = ["steelblue", "darkorange", "seagreen"]

    # Row 0 — position
    for col, (label, data) in enumerate(zip(
            ["px (m)", "py (m)", "pz (m)"],
            [q_srb[:, 0], q_srb[:, 1], q_srb[:, 2]])):
        ax = axes[0, col]
        _shade(ax)
        ax.plot(t, data, color=colors[col], lw=1.4)
        ax.set_ylabel(label)

    # Row 1 — orientation (ZYX Euler)
    for col, (label, data) in enumerate(zip(
            ["roll (deg)", "pitch (deg)", "yaw (deg)"],
            [roll_deg, pitch_deg, yaw_deg])):
        ax = axes[1, col]
        _shade(ax)
        ax.plot(t, data, color=colors[col], lw=1.4)
        ax.set_ylabel(label)

    # Row 2 — linear velocity
    for col, (label, data) in enumerate(zip(
            ["vx (m/s)", "vy (m/s)", "vz (m/s)"],
            [v_srb[:, 0], v_srb[:, 1], v_srb[:, 2]])):
        ax = axes[2, col]
        _shade(ax)
        ax.plot(t, data, color=colors[col], lw=1.4)
        ax.set_ylabel(label)

    # Row 3 — angular velocity (body frame)
    for col, (label, data) in enumerate(zip(
            ["wx (rad/s)", "wy (rad/s)", "wz (rad/s)"],
            [v_srb[:, 3], v_srb[:, 4], v_srb[:, 5]])):
        ax = axes[3, col]
        _shade(ax)
        ax.plot(t, data, color=colors[col], lw=1.4)
        ax.set_ylabel(label)

    # Row 4 — varinertia: tuck, Ixx, Iyy  (3 columns = 3 axes, no overflow)
    if has_varinertia:
        t_tuck = times[stance_end : flight_end + 1]  # (N_flight+1,)

        ax_tuck = axes[4, 0]
        _shade(ax_tuck)
        ax_tuck.plot(t_tuck, tuck_opt, color="purple", lw=1.4)
        ax_tuck.set_ylabel("tuck")
        ax_tuck.set_ylim(-0.05, 1.05)

        for col, (label, col_idx) in enumerate(
                zip(["Ixx (kg·m²)", "Iyy (kg·m²)"], [0, 1])):
            ax = axes[4, col + 1]
            _shade(ax)
            ax.plot(t, I_opt[:, col_idx], color=colors[col_idx], lw=1.4)
            ax.set_ylabel(label)

    phase_patches = [
        mpatches.Patch(color="steelblue", alpha=0.3, label="Stance"),
        mpatches.Patch(color="white",     alpha=0.0, label="Flight", ec="gray"),
        mpatches.Patch(color="seagreen",  alpha=0.3, label="Landing"),
    ]
    fig.legend(handles=phase_patches, loc="lower center", ncol=3,
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.01))

    fig.tight_layout(rect=[0, 0.03, 1, 1])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[pipeline] State plot saved → {save_path}")

    plt.show()


# ---------------------------------------------------------------------------
# Step 6 — visualize
# ---------------------------------------------------------------------------

def visualize(q_ik: np.ndarray, q_srb: np.ndarray, times: np.ndarray,
              combined_xml: str, speed: float = 1.0,
              I_opt: np.ndarray = None,
              feet_world: np.ndarray = None):
    """
    Animate the combined model in real time.
    Each frame is displayed for its actual trajectory duration (times[i+1]-times[i])
    divided by speed, so playback is physically correct across variable-dt phases.
    speed < 1 → slow motion, speed > 1 → fast forward.

    I_opt:       (N+1, 6) inertia trajectory [Ixx,Iyy,Izz,Ixy,Ixz,Iyz].  When provided,
                 the SRB ellipsoid geom is rescaled each frame.
    feet_world:  (N+1, 6) world-frame foot positions [pLx,pLy,pLz, pRx,pRy,pRz].
                 NaN rows → feet hidden below ground (flight phase).
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

    # Inertia-scaled ellipsoid setup — geom_size is (ngeom, 3)
    _srb_geom_id   = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "base_geom")
    _scale_inertia = (I_opt is not None) and (_srb_geom_id >= 0)
    if _scale_inertia:
        _nom_abc = mj_model.geom_size[_srb_geom_id].copy()
        _Ixx_nom = 3.747533
        _Iyy_nom = 3.300958
        print(f"[pipeline] SRB ellipsoid scaling active — nominal size: "
              f"a={_nom_abc[0]:.3f}, b={_nom_abc[1]:.3f}, c={_nom_abc[2]:.3f}")

    def _update_ellipsoid(i):
        if not _scale_inertia:
            return
        Ixx = I_opt[i, 0]
        Iyy = I_opt[i, 1]
        scale_ac = np.sqrt(np.clip(Iyy / _Iyy_nom, 0.01, 4.0))
        scale_b  = np.sqrt(np.clip(Ixx / _Ixx_nom, 0.01, 4.0))
        mj_model.geom_size[_srb_geom_id, 0] = _nom_abc[0] * scale_ac
        mj_model.geom_size[_srb_geom_id, 1] = _nom_abc[1] * scale_b
        mj_model.geom_size[_srb_geom_id, 2] = _nom_abc[2] * scale_ac

    # Foot marker mocap setup
    _HIDDEN = np.array([0.0, 0.0, -100.0])
    _foot_L_body = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "srb_foot_L")
    _foot_R_body = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "srb_foot_R")
    _foot_L_mocap = int(mj_model.body_mocapid[_foot_L_body]) if _foot_L_body >= 0 else -1
    _foot_R_mocap = int(mj_model.body_mocapid[_foot_R_body]) if _foot_R_body >= 0 else -1
    _show_feet = (feet_world is not None) and (_foot_L_mocap >= 0) and (_foot_R_mocap >= 0)

    def _update_feet(i):
        if not _show_feet:
            return
        row = feet_world[i]
        mj_data.mocap_pos[_foot_L_mocap] = row[0:3] if not np.any(np.isnan(row[0:3])) else _HIDDEN
        mj_data.mocap_pos[_foot_R_mocap] = row[3:6] if not np.any(np.isnan(row[3:6])) else _HIDDEN

    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        viewer.cam.lookat[:]  = [0.0, 0.0, 0.6]
        viewer.cam.distance   = 3.2
        viewer.cam.azimuth    = 90.0
        viewer.cam.elevation  = -12.0

        while viewer.is_running():
            for i in range(q_ik.shape[0]):
                if not viewer.is_running():
                    break

                _update_ellipsoid(i)
                _update_feet(i)
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
    parser.add_argument("--speed",       type=float, default=1.0,
                        help="Playback speed multiplier (default: 1.0 = real-time)")
    parser.add_argument("--box-xml",     default=None,
                        help="Path to box scene XML to include in the viewer "
                             "(e.g. models/box/box_20x30x24in.xml)")
    parser.add_argument("--no-visualize", action="store_true",
                        help="Skip the MuJoCo visualizer (useful for non-interactive runs)")
    parser.add_argument("--plot-states",  action="store_true",
                        help="Save an SRB state trajectory plot (srb_states.png)")
    parser.add_argument("--control-frame", action="store_true", default=False,
                        help="Force contact feet rigidly aligned to the SRB body heading "
                             "Rz(body_yaw) — a straight 90deg feet turn (overrides the "
                             "per-config constraints.control_frame flag).")
    parser.add_argument("--free-foot-yaw", action="store_true", default=False,
                        help="Force flat-sole/free-yaw contact feet — foot yaw is a free DOF "
                             "in the solve (overrides the per-config constraints.free_foot_yaw "
                             "flag). Use for motions where the body yaws while planted, e.g. a "
                             "twist.")

    args = parser.parse_args()

    # 1. SRB solver
    if not args.skip_solve:
        run_srb_solver(args.config)

    # 2. Load results — derive path and stance_ground_z from config if available
    try:
        import importlib
        _cfg_mod = importlib.import_module(args.config)
        _results_dir   = os.path.join(_REPO_ROOT, _cfg_mod.config.save_dir.lstrip("./"))
        _stance_ground_z = _cfg_mod.config.constraints.stance_ground_z
        _reject_self_collision = _cfg_mod.config.solver.reject_self_collision
        _control_frame = getattr(_cfg_mod.config.constraints, "control_frame", False)
        _free_foot_yaw = getattr(_cfg_mod.config.constraints, "free_foot_yaw", False)
    except Exception:
        _results_dir     = _RESULTS_DEFAULT
        _stance_ground_z = 0.0
        _reject_self_collision = False
        _control_frame = False
        _free_foot_yaw = False
    # CLI flags force each on regardless of the config default.
    _control_frame = _control_frame or args.control_frame
    _free_foot_yaw = _free_foot_yaw or args.free_foot_yaw
    if _control_frame:
        print("[pipeline] Contact feet: rigid to body heading Rz(yaw) (control_frame=True)")
    if _free_foot_yaw:
        print("[pipeline] Contact feet: flat-sole, free yaw (free_foot_yaw=True)")
    times, q_srb, feet_ext, stance_end, flight_end = load_srb_results(_results_dir)

    # 3. IK — export into a config-specific subdirectory so configs don't overwrite each other
    export_dir = os.path.join(_results_dir, "ik")
    os.makedirs(export_dir, exist_ok=True)
    ik = G1IPOPTIK()
    arm_defaults = load_arm_defaults(ik.model)

    # Load SRB tuck trajectory if available (output of variable_inertia=True SRB run).
    _tuck_opt_path = os.path.join(_results_dir, "tuck_opt.csv")
    _tuck_opt = None
    if os.path.exists(_tuck_opt_path):
        _tuck_opt = np.loadtxt(_tuck_opt_path, delimiter=",", comments="#").flatten()
        print(f"[pipeline] Loaded tuck trajectory → flight tuck IK active")
        print(f"  tuck ∈ [{_tuck_opt.min():.3f}, {_tuck_opt.max():.3f}]")

    if not args.skip_ik:
        q_ik = run_ik_trajectory(ik, q_srb, feet_ext, stance_end, flight_end,
                                  arm_defaults=arm_defaults,
                                  stance_ground_z=_stance_ground_z,
                                  tuck_opt=_tuck_opt,
                                  reject_self_collision=_reject_self_collision,
                                  times=times,
                                  control_frame=_control_frame,
                                  free_foot_yaw=_free_foot_yaw)
    else:
        q_ik_mj = np.loadtxt(os.path.join(export_dir, "ik_q_mujoco.csv"),
                              delimiter=",", comments="#")
        # Convert back to pinocchio for internal use (just flip quat)
        q_ik = np.array([np.concatenate([r[:3],
                                          [r[4], r[5], r[6], r[3]],
                                          r[7:]]) for r in q_ik_mj])
        print(f"[pipeline] Loaded cached IK solution ({q_ik.shape[0]} frames)")

    # 4b. IK error report (uses original times before padding)
    check_ik_errors(q_ik, times, feet_ext, stance_end, flight_end, ik,
                    stance_ground_z=_stance_ground_z)

    # 5. Wrench plot (uses original times)
    plot_srb_wrenches(_results_dir, times, stance_end, flight_end)

    # 5c. State plot
    if args.plot_states:
        plot_srb_states(_results_dir, times, stance_end, flight_end)

    # 4. Export — pad first/last pose for 1 s on each end before writing CSVs
    q_ik_export, times_export = _pad_trajectory(q_ik, times, hold_seconds=1.0)
    export_ik_solution(q_ik_export, times_export, export_dir, model=ik.model)

    # 5b. Build combined XML
    box_xml = os.path.join(_REPO_ROOT, args.box_xml) if args.box_xml else None
    build_combined_xml(_G1_XML, _SRB_XML, _COMBINED, box_xml=box_xml)

    # 6. Visualize
    if not args.no_visualize:
        _I_opt_vis_path = os.path.join(_results_dir, "I_opt.csv")
        _I_opt_vis = np.loadtxt(_I_opt_vis_path, delimiter=",", comments="#") if os.path.exists(_I_opt_vis_path) else None

        # Build world-frame foot positions (N+1, 6): NaN rows during flight → hidden.
        _landing_gz = getattr(getattr(_cfg_mod, 'config', None), 'constraints', None)
        _landing_gz = float(_landing_gz.landing_ground_z) if _landing_gz is not None else 0.0
        _feet_world = np.full((len(times), 6), np.nan)
        for _k in range(len(times)):
            if _k < stance_end:
                _feet_world[_k] = [feet_ext[_k, 0], feet_ext[_k, 1], _stance_ground_z,
                                   feet_ext[_k, 2], feet_ext[_k, 3], _stance_ground_z]
            elif _k >= flight_end:
                _feet_world[_k] = [feet_ext[_k, 0], feet_ext[_k, 1], _landing_gz,
                                   feet_ext[_k, 2], feet_ext[_k, 3], _landing_gz]

        visualize(q_ik, q_srb, times, _COMBINED, speed=args.speed,
                  I_opt=_I_opt_vis, feet_world=_feet_world)
    else:
        print("[pipeline] Skipping visualizer (--no-visualize).")
