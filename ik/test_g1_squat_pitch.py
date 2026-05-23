##
#
# Animated squat with forward pelvis pitch using Newton-Raphson IK.
# The pelvis descends from 0.79 m to 0.62 m while pitching forward up to 25 deg,
# then returns to standing.  IK warm-starts from the previous frame.
#
##

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pinocchio as pin
import mujoco
import mujoco.viewer

from utils.kinematics.g1_ik import G1IK
from utils.kinematics.kin import euler_ZYX_to_quat


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pin_q_to_mj(q_pin, nq_mj):
    """Pinocchio [qx,qy,qz,qw] → MuJoCo [qw,qx,qy,qz], joints unchanged."""
    q_mj = np.zeros(nq_mj)
    q_mj[0:3] = q_pin[0:3]
    qx, qy, qz, qw = q_pin[3], q_pin[4], q_pin[5], q_pin[6]
    q_mj[3:7]  = [qw, qx, qy, qz]
    q_mj[7:]   = q_pin[7:7 + (nq_mj - 7)]
    return q_mj


def make_base_config(ik, h, pitch):
    """Build pinocchio q with pelvis at height h and pitch angle (rad)."""
    q = ik.standing_config(h)
    qw_xyzw = euler_ZYX_to_quat(0.0, pitch, 0.0)   # returns [qw,qx,qy,qz]
    q[3:7] = [qw_xyzw[1], qw_xyzw[2], qw_xyzw[3], qw_xyzw[0]]  # → pin [qx,qy,qz,qw]
    return q


# ---------------------------------------------------------------------------
# Build the IK trajectory
# ---------------------------------------------------------------------------

def build_trajectory(n_frames=40):
    ik = G1IK()

    H_START, H_END   = 0.79, 0.62    # pelvis height range (m)
    P_START, P_END   = 0.00, 0.44    # pitch range (rad) ≈ 0 → 25 deg

    # Half-cycle: descend + pitch forward, then reverse
    t_half = np.linspace(0.0, 1.0, n_frames)
    t_full = np.concatenate([t_half, t_half[::-1]])

    frames   = []
    q_warm   = None
    WARMSTART_MIN_DROP = 0.03   # below this pelvis drop, always use fresh guess

    for t in t_full:
        h     = H_START + t * (H_END - H_START)
        pitch = P_START + t * (P_END - P_START)
        drop  = H_START - h

        q0 = make_base_config(ik, h, pitch)

        # Warm-start only when meaningfully away from standing.
        # Near standing the leg is nearly straight, so deep-squat warm-start
        # angles send the IK to the wrong branch and the feet slip underground.
        if q_warm is not None and drop > WARMSTART_MIN_DROP:
            q0[7:] = q_warm[7:]

        oMl, oMr = ik.floor_targets(q0)
        q_sol, ok, errs = ik.solve(q0, oMl, oMr)

        frames.append(q_sol)
        q_warm = q_sol

        print(f"  h={h:.3f}  pitch={np.degrees(pitch):5.1f}°  "
              f"ok={ok}  err={errs[-1]:.1e}")

    return frames


# ---------------------------------------------------------------------------
# Visualize
# ---------------------------------------------------------------------------

def visualize(frames, dt=0.04):
    xml_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            "models", "g1", "g1_29dof_rev_1_0.xml")
    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_data  = mujoco.MjData(mj_model)

    print(f"\nOpening MuJoCo viewer  ({len(frames)} frames, {dt*1000:.0f} ms/frame).")
    print("Close the window to quit.\n")

    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        # Side view so the forward pitch is clearly visible
        viewer.cam.lookat[:]  = [0.0, 0.0, 0.5]
        viewer.cam.distance   = 2.8
        viewer.cam.azimuth    = 90.0    # look from robot's left side (+y)
        viewer.cam.elevation  = -10.0

        while viewer.is_running():
            for q_sol in frames:
                if not viewer.is_running():
                    break
                mj_data.qpos[:] = pin_q_to_mj(q_sol, mj_model.nq)
                mj_data.qvel[:] = 0
                mujoco.mj_forward(mj_model, mj_data)
                viewer.sync()
                time.sleep(dt)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Building IK trajectory...")
    frames = build_trajectory(n_frames=40)
    visualize(frames)
