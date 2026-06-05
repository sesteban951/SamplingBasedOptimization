##
#
# Visualize G1 crouch / flight-tuck configurations in MuJoCo.
#
# Single persistent viewer with keyboard controls — no close/reopen needed.
#
# Controls:
#   LEFT / RIGHT   — step tuck slider by 0.05
#   UP   / DOWN    — step tuck slider by 0.01 (fine)
#   G              — ground mode (feet planted, pelvis drops)
#   F              — flight mode (pelvis fixed at flight_z, robot floats)
#   [ / ]          — decrease / increase flight pelvis height by 0.1 m
#   R              — reset to standing (t = 0)
#   T              — print current config to terminal
#
# Run from repo root:
#   conda run -n env_sbo python ik/viz_crouch_configs.py
#
##

import os
import numpy as np
import mujoco
import mujoco.viewer
import pinocchio as pin
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_XML  = os.path.join(_REPO_ROOT, "models", "g1", "g1_29dof_rev_1_0.xml")
_URDF = os.path.join(_REPO_ROOT, "models", "g1", "g1_29dof_rev_1_0.urdf")

# ── pinocchio model ───────────────────────────────────────────────────────────

pin_model = pin.buildModelFromUrdf(_URDF, pin.JointModelFreeFlyer())
pin_data  = pin_model.createData()

LEG_JOINTS = [
    "left_hip_pitch_joint",    "left_hip_roll_joint",   "left_hip_yaw_joint",
    "left_knee_joint",         "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint",   "right_hip_roll_joint",  "right_hip_yaw_joint",
    "right_knee_joint",        "right_ankle_pitch_joint","right_ankle_roll_joint",
]
_leg_qidx = [pin_model.joints[pin_model.getJointId(n)].idx_q for n in LEG_JOINTS]
_leg_lo   = np.array([pin_model.lowerPositionLimit[qi] for qi in _leg_qidx])
_leg_hi   = np.array([pin_model.upperPositionLimit[qi] for qi in _leg_qidx])

ANKLE_HEIGHT = 0.0332
L_FOOT = pin_model.getFrameId("left_ankle_roll_link")
R_FOOT = pin_model.getFrameId("right_ankle_roll_link")

# ── Joint targets ─────────────────────────────────────────────────────────────
#
# Convention: negative hip_pitch = hip flexion (thigh/foot swings forward+up).
# MAX_CROUCH is the tightest valid flight tuck near joint limits.

STANDING = np.array([
    0.03,  0.0,  0.0,   0.05,  -0.03, 0.0,
    0.03,  0.0,  0.0,   0.05,  -0.03, 0.0,
])

MAX_CROUCH = np.clip(np.array([
    -2.5,  0.0,  0.0,   2.87,  0.5,   0.0,
    -2.5,  0.0,  0.0,   2.87,  0.5,   0.0,
]), _leg_lo, _leg_hi)

# Chosen target for flight IK — t=0.90 (hip=-2.25, knee=2.59, ankle=+0.45).
# Stored in pipeline_srb_ik.py as _FLIGHT_TUCK_LEGS.
TUCK_TARGET = (1.0 - 0.90) * STANDING + 0.90 * MAX_CROUCH

# ── MuJoCo setup ─────────────────────────────────────────────────────────────

mj_model = mujoco.MjModel.from_xml_path(_XML)
mj_data  = mujoco.MjData(mj_model)

def _mj_qidx(name):
    jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
    return mj_model.jnt_qposadr[jid]

_mj_leg_idx = [_mj_qidx(n) for n in LEG_JOINTS]

_ARM_DEFAULTS = {
    "left_shoulder_pitch_joint":  0.498,
    "left_shoulder_roll_joint":   0.3,
    "left_shoulder_yaw_joint":    0.0,
    "left_elbow_joint":           0.501,
    "right_shoulder_pitch_joint": 0.498,
    "right_shoulder_roll_joint":  -0.3,
    "right_shoulder_yaw_joint":   0.0,
    "right_elbow_joint":          0.501,
}
_mj_arm = {_mj_qidx(n): v for n, v in _ARM_DEFAULTS.items()}

# ── State ─────────────────────────────────────────────────────────────────────

state = {
    "t":         0.0,
    "mode":      "flight",   # "ground" or "flight"
    "flight_z":  1.0,
    "dirty":     True,
}

# ── Config application ────────────────────────────────────────────────────────

def _pelvis_z_ground(q_legs):
    q = pin.neutral(pin_model); q[2] = 0.79
    for pi_i, v in zip(_leg_qidx, q_legs): q[pi_i] = v
    for _ in range(30):
        pin.framesForwardKinematics(pin_model, pin_data, q)
        err = 0.5*(pin_data.oMf[L_FOOT].translation[2] +
                   pin_data.oMf[R_FOOT].translation[2]) - ANKLE_HEIGHT
        q[2] -= err
        if abs(err) < 1e-6: break
    return float(q[2])


def apply_config():
    t  = state["t"]
    ql = (1.0 - t) * STANDING + t * MAX_CROUCH

    for mj_i, v in zip(_mj_leg_idx, ql): mj_data.qpos[mj_i] = v
    for mj_i, v in _mj_arm.items():      mj_data.qpos[mj_i] = v

    pz = _pelvis_z_ground(ql) if state["mode"] == "ground" else state["flight_z"]
    mj_data.qpos[0:3] = [0.0, 0.0, pz]
    mj_data.qpos[3]   = 1.0   # quat w
    mj_data.qpos[4:7] = 0.0
    mujoco.mj_forward(mj_model, mj_data)


def print_config():
    t  = state["t"]
    ql = (1.0 - t) * STANDING + t * MAX_CROUCH
    pz = mj_data.qpos[2]
    print(f"\n── t={t:.2f}  mode={state['mode']}  pelvis_z={pz:.3f}m ───────────")
    print(f"  hip_pitch={ql[0]:+.4f}  knee={ql[3]:.4f}  ankle={ql[4]:+.4f}")

    if state["mode"] == "ground":
        q = pin.neutral(pin_model); q[2] = pz
        for pi_i, v in zip(_leg_qidx, ql): q[pi_i] = v
        pin.centerOfMass(pin_model, pin_data, q)
        print(f"  CoM z = {float(pin_data.com[0][2]):.4f} m")

    foot_z = mj_data.qpos[2]  # rough proxy; proper FK printed above
    print(f"  full 12-vec: {np.round(ql, 4).tolist()}")

# ── Keyboard callback (runs in viewer thread) ─────────────────────────────────
# GLFW key codes: LEFT=263, RIGHT=262, UP=265, DOWN=264
# ASCII uppercase: G=71, F=70, R=82, T=84, [=91, ]=93

def key_callback(keycode):
    dirty = True
    if keycode == 262:    # RIGHT — coarse +
        state["t"] = min(1.0, round(state["t"] + 0.05, 4))
    elif keycode == 263:  # LEFT — coarse -
        state["t"] = max(0.0, round(state["t"] - 0.05, 4))
    elif keycode == 265:  # UP — fine +
        state["t"] = min(1.0, round(state["t"] + 0.01, 4))
    elif keycode == 264:  # DOWN — fine -
        state["t"] = max(0.0, round(state["t"] - 0.01, 4))
    elif keycode == 71:   # G — ground mode
        state["mode"] = "ground"
    elif keycode == 70:   # F — flight mode
        state["mode"] = "flight"
    elif keycode == 91:   # [ — lower pelvis
        state["flight_z"] = round(state["flight_z"] - 0.1, 2)
    elif keycode == 93:   # ] — raise pelvis
        state["flight_z"] = round(state["flight_z"] + 0.1, 2)
    elif keycode == 82:   # R — reset
        state["t"] = 0.0
    elif keycode == 84:   # T — print
        print_config()
        dirty = False
    else:
        dirty = False

    if dirty:
        state["dirty"] = True
        t = state["t"]
        ql = (1.0 - t) * STANDING + t * MAX_CROUCH
        print(f"\rt={t:.2f}  mode={state['mode']}  flight_z={state['flight_z']:.1f}m"
              f"  hip={ql[0]:+.2f}  knee={ql[3]:.2f}  ank={ql[4]:+.2f}   ", end="", flush=True)

# ── Main loop ─────────────────────────────────────────────────────────────────

print("=" * 62)
print("G1 Crouch / Tuck Visualizer")
print("=" * 62)
print("  LEFT / RIGHT   coarse tuck  (±0.05)")
print("  UP   / DOWN    fine tuck    (±0.01)")
print("  G / F          ground / flight mode")
print("  [ / ]          pelvis height ±0.1 m  (flight mode)")
print("  R              reset to standing")
print("  T              print current joint config")
print("=" * 62)
print("Click the viewer window to capture key input.\n")

apply_config()

with mujoco.viewer.launch_passive(mj_model, mj_data,
                                   key_callback=key_callback) as viewer:
    viewer.cam.azimuth   = 160
    viewer.cam.elevation = -15
    viewer.cam.distance  = 2.5
    viewer.cam.lookat[:] = [0.0, 0.0, 0.6]
    viewer.sync()

    while viewer.is_running():
        if state["dirty"]:
            apply_config()
            viewer.sync()
            state["dirty"] = False
        time.sleep(0.02)

print("\nDone.")
