##
#
# Test + MuJoCo visualization for G1 Newton-Raphson IK.
# Sweeps pelvis heights, solves IK, then displays the final pose.
#
##

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pinocchio as pin
import mujoco
import mujoco.viewer

from utils.kinematics.g1_ik import G1IK


# ---------------------------------------------------------------------------
# Quaternion convention conversion
# ---------------------------------------------------------------------------

def pin_quat_to_mj(q_pin):
    """Pinocchio [qx, qy, qz, qw] → MuJoCo [qw, qx, qy, qz]."""
    qx, qy, qz, qw = q_pin[3], q_pin[4], q_pin[5], q_pin[6]
    return np.array([qw, qx, qy, qz])


def pin_q_to_mj(q_pin, nq_mj):
    """
    Convert full pinocchio configuration to MuJoCo qpos.
    Layout is identical except for the free-flyer quaternion slot.
    """
    q_mj = np.zeros(nq_mj)
    q_mj[0:3] = q_pin[0:3]                   # base position
    q_mj[3:7] = pin_quat_to_mj(q_pin)        # quaternion (reordered)
    q_mj[7:]  = q_pin[7:7 + (nq_mj - 7)]    # joint angles
    return q_mj


# ---------------------------------------------------------------------------
# IK sweep
# ---------------------------------------------------------------------------

def run_ik_sweep():
    ik = G1IK()
    heights = [0.79, 0.72, 0.65, 0.58]

    results = []
    q_prev  = None

    print(f"\n{'Height':>8}  {'Conv':>5}  {'Iters':>6}  {'FinalErr':>10}  "
          f"{'L-foot z':>10}  {'R-foot z':>10}")
    print("-" * 62)

    for h in heights:
        # Warm-start from previous solution if available, else neutral
        if q_prev is None:
            q0 = ik.standing_config(h)
        else:
            q0 = q_prev.copy()
            q0[2] = h   # update pelvis height, keep joint angles as warm start

        oMl_des, oMr_des = ik.floor_targets(q0)
        q_sol, ok, errs  = ik.solve(q0, oMl_des, oMr_des)

        # Verify via FK
        d = ik.model.createData()
        pin.forwardKinematics(ik.model, d, q_sol)
        pin.updateFramePlacements(ik.model, d)
        pl = d.oMf[ik.l_foot_id].translation
        pr = d.oMf[ik.r_foot_id].translation

        print(f"{h:8.2f}  {'✓' if ok else '✗':>5}  {len(errs):6d}  "
              f"{errs[-1]:10.2e}  {pl[2]:10.4f}  {pr[2]:10.4f}")

        results.append((h, q_sol, ok))
        q_prev = q_sol

    return results


# ---------------------------------------------------------------------------
# MuJoCo visualization
# ---------------------------------------------------------------------------

def visualize(results):
    xml_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            "models", "g1", "g1_29dof_rev_1_0.xml")

    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_data  = mujoco.MjData(mj_model)

    print("\nOpening MuJoCo viewer.")
    print("  Press [N] / [B] to cycle forward / backward through heights.")
    print("  Press [ESC] or close window to quit.\n")

    idx  = [0]
    keys = {ord('n'): 1, ord('b'): -1}

    def set_pose(i):
        h, q_sol, _ = results[i % len(results)]
        mj_data.qpos[:] = pin_q_to_mj(q_sol, mj_model.nq)
        mj_data.qvel[:] = 0
        mujoco.mj_forward(mj_model, mj_data)
        print(f"  Showing pelvis height = {h:.2f} m")

    set_pose(0)

    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        while viewer.is_running():
            # poll keyboard
            for key, delta in keys.items():
                if viewer.is_running():
                    pass  # handled via key_callback below
            viewer.sync()

    # Fallback: cycle through all poses once, then hold last
    print("\nViewer closed.")


def visualize_single(q_sol, mj_nq):
    xml_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            "models", "g1", "g1_29dof_rev_1_0.xml")
    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_data  = mujoco.MjData(mj_model)

    mj_data.qpos[:] = pin_q_to_mj(q_sol, mj_nq)
    mj_data.qvel[:] = 0
    mujoco.mj_forward(mj_model, mj_data)

    print("\nOpening MuJoCo viewer — close window to exit.")
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        while viewer.is_running():
            viewer.sync()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results = run_ik_sweep()

    # Visualize the last (deepest squat) pose
    _, q_final, _ = results[-1]
    xml_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            "models", "g1", "g1_29dof_rev_1_0.xml")
    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_data  = mujoco.MjData(mj_model)

    print("\nCycling through all heights in MuJoCo viewer (2 s each).")
    print("Close the viewer window to quit.\n")

    import time
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        # Side-view camera: look at the robot from the +y side so the
        # forward-knee squat geometry is clearly visible.
        viewer.cam.lookat[:] = [0.0, 0.0, 0.5]
        viewer.cam.distance  = 2.5
        viewer.cam.azimuth   = 90.0   # look from +y side
        viewer.cam.elevation = -15.0  # slightly above

        for h, q_sol, ok in results:
            if not viewer.is_running():
                break
            mj_data.qpos[:] = pin_q_to_mj(q_sol, mj_model.nq)
            mj_data.qvel[:] = 0
            mujoco.mj_forward(mj_model, mj_data)
            viewer.sync()
            print(f"  height = {h:.2f} m  (converged={ok})")
            t0 = time.time()
            while viewer.is_running() and time.time() - t0 < 2.0:
                viewer.sync()

        # Hold final pose
        while viewer.is_running():
            viewer.sync()
