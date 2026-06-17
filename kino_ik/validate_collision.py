##
# Interactive self-collision validator for the G1 tuck.
#
# Move the legs (tuck slider) and the freed sagittal arms (shoulder pitch / elbow)
# and watch, live:
#   - the capsule "sausage-man" min clearance + which pair  (what the kino NLP
#     constrains — collision_model.PAIRS / radii), spheres drawn in the viewer
#     (red on the offending pair when below MARGIN), and
#   - the FCL MESH self-collision count (physical ground truth, adjacent/always-
#     touching pairs auto-disabled).
# Use it to confirm the flight tuck prior is collision-free and to calibrate the
# capsule radii so the capsule flag agrees with FCL.
#
# Usage:
#   conda run -n env_sbo python kino_ik/validate_collision.py           # interactive
#   conda run -n env_sbo python kino_ik/validate_collision.py --check   # headless assert
#
# Keys: LEFT/RIGHT tuck +-0.05 | UP/DOWN tuck +-0.01 | W/S shoulder_pitch | E/D
#       elbow | P jump to flight prior | R reset | T print config | ESC quit
##

import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pinocchio as pin

from kino_ik import collision_model as cm
from kino_ik.kino_nlp import (LEG_JOINTS, FREE_ARM_JOINTS,
                              FLIGHT_TUCK_LEGS, FLIGHT_TUCK_ARMS)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_XML  = os.path.join(_REPO_ROOT, "models", "g1", "g1_29dof_rev_1_0.xml")
_URDF = os.path.join(_REPO_ROOT, "models", "g1", "g1_29dof_rev_1_0.urdf")
_PKG  = [os.path.join(_REPO_ROOT, "models", "g1")]
_BASE_Z = 1.0   # arbitrary float-base height (self-collision is base-invariant)

# Leg tuck endpoints (from ik/viz_crouch_configs.py), LEG_JOINTS order.
STANDING = np.array([0.03, 0.0, 0.0, 0.05, -0.03, 0.0,
                     0.03, 0.0, 0.0, 0.05, -0.03, 0.0])
MAX_CROUCH = np.array([-2.5, 0.0, 0.0, 2.87, 0.5, 0.0,
                       -2.5, 0.0, 0.0, 2.87, 0.5, 0.0])
# Non-freed arm DOF held at defaults (shoulder roll/yaw), so the sausages are
# placed at a realistic arm pose.
_ARM_FIXED = {"left_shoulder_roll_joint": 0.3,  "left_shoulder_yaw_joint": 0.0,
              "right_shoulder_roll_joint": -0.3, "right_shoulder_yaw_joint": 0.0}

# ── pinocchio model + collision model ────────────────────────────────────────
_model = pin.buildModelFromUrdf(_URDF, pin.JointModelFreeFlyer())
_data  = _model.createData()
_caps  = cm.build_capsules(_model)
_leg_qi  = [_model.joints[_model.getJointId(j)].idx_q for j in LEG_JOINTS]
_armf_qi = [_model.joints[_model.getJointId(j)].idx_q for j in FREE_ARM_JOINTS]
_fix_qi  = {n: _model.joints[_model.getJointId(n)].idx_q for n in _ARM_FIXED}
MAX_CROUCH = np.clip(MAX_CROUCH,
                     [_model.lowerPositionLimit[i] for i in _leg_qi],
                     [_model.upperPositionLimit[i] for i in _leg_qi])


def _pin_q(ql, sp, el):
    """Build a pinocchio config (base at _BASE_Z, identity orientation)."""
    q = pin.neutral(_model)
    q[2] = _BASE_Z
    for i, v in zip(_leg_qi, ql):
        q[i] = v
    # freed arms: FREE_ARM_JOINTS = [L_sp, L_el, R_sp, R_el], symmetric sp/el
    for i, v in zip(_armf_qi, [sp, el, sp, el]):
        q[i] = v
    for n, qi in _fix_qi.items():
        q[qi] = _ARM_FIXED[n]
    return q


def _tuck_legs(t):
    return (1.0 - t) * STANDING + t * MAX_CROUCH


# FCL ground-truth checker with always-touching pairs disabled (neutral + stand).
_safe_qs = [pin.neutral(_model), _pin_q(STANDING, 0.0, 0.0)]
_gm, _gd, _disabled = cm.build_fcl_checker(_model, _URDF, _PKG, _safe_qs)
print(f"[validate] FCL pairs {len(_gm.collisionPairs)}, "
      f"auto-disabled (always-touching) {len(_disabled)}")


def _status(q):
    """Return (capsule_clearance, capsule_pair, fcl_collisions, spheres)."""
    sph = cm.world_spheres_numeric(_model, _data, q, _caps)
    dmin, pair, _, _ = cm.min_pair_distance(sph)
    fcl = cm.fcl_collisions(_model, _data, _gm, _gd, _disabled, q)
    return dmin, pair, fcl, sph


# ─────────────────────────────────────────────────────────────────────────────
# Headless check
# ─────────────────────────────────────────────────────────────────────────────

def run_check():
    print("\n[validate] === flight tuck prior ===")
    q = _pin_q(FLIGHT_TUCK_LEGS, FLIGHT_TUCK_ARMS[0], FLIGHT_TUCK_ARMS[1])
    dmin, pair, fcl, _ = _status(q)
    print(f"  capsule min clearance : {dmin*100:+.1f} cm  ({pair[0]}<->{pair[1]})  "
          f"margin={cm.MARGIN*100:.0f} cm  -> {'OK' if dmin > cm.MARGIN else 'TOO CLOSE'}")
    print(f"  FCL mesh collisions   : {len(fcl)}  "
          f"-> {'OK' if not fcl else 'COLLIDING ' + str(fcl)}")

    print("\n[validate] === tuck sweep (legs STANDING->MAX_CROUCH, arms at prior) ===")
    print("    t     capsule(cm)  closest pair        FCL")
    for t in np.linspace(0.0, 1.0, 11):
        q = _pin_q(_tuck_legs(t), FLIGHT_TUCK_ARMS[0], FLIGHT_TUCK_ARMS[1])
        dmin, pair, fcl, _ = _status(q)
        print(f"  {t:4.2f}   {dmin*100:+7.1f}     {pair[0]:7s}<->{pair[1]:8s}  "
              f"{len(fcl)}{' '+str(fcl[:2]) if fcl else ''}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Interactive viewer
# ─────────────────────────────────────────────────────────────────────────────

def run_interactive():
    import time
    import mujoco
    import mujoco.viewer

    mj = mujoco.MjModel.from_xml_path(_XML)
    mjd = mujoco.MjData(mj)

    def mj_qi(name):
        return mj.jnt_qposadr[mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_JOINT, name)]
    mj_leg = [mj_qi(j) for j in LEG_JOINTS]
    mj_armf = [mj_qi(j) for j in FREE_ARM_JOINTS]
    mj_fix = {n: mj_qi(n) for n in _ARM_FIXED}

    st = {"t": 0.90, "sp": float(FLIGHT_TUCK_ARMS[0]), "el": float(FLIGHT_TUCK_ARMS[1]),
          "dirty": True}

    def apply():
        ql = _tuck_legs(st["t"])
        mjd.qpos[0:3] = [0.0, 0.0, _BASE_Z]
        mjd.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        for i, v in zip(mj_leg, ql):
            mjd.qpos[i] = v
        for i, v in zip(mj_armf, [st["sp"], st["el"], st["sp"], st["el"]]):
            mjd.qpos[i] = v
        for n, i in mj_fix.items():
            mjd.qpos[i] = _ARM_FIXED[n]
        mujoco.mj_forward(mj, mjd)

    def report():
        ql = _tuck_legs(st["t"])
        q = _pin_q(ql, st["sp"], st["el"])
        dmin, pair, fcl, sph = _status(q)
        flag = "COLLISION" if dmin <= cm.MARGIN else "ok"
        print(f"t={st['t']:.2f} sp={st['sp']:+.3f} el={st['el']:+.3f} | "
              f"capsule {dmin*100:+5.1f}cm {pair[0]}<->{pair[1]} [{flag}] | "
              f"FCL {len(fcl)}" + (f" {fcl[:2]}" if fcl else ""))
        return sph, dmin, pair

    def key_cb(key):
        if   key == 263: st["t"] = max(0.0, st["t"] - 0.05)   # LEFT
        elif key == 262: st["t"] = min(1.0, st["t"] + 0.05)   # RIGHT
        elif key == 264: st["t"] = max(0.0, st["t"] - 0.01)   # DOWN
        elif key == 265: st["t"] = min(1.0, st["t"] + 0.01)   # UP
        elif key == 87:  st["sp"] += 0.05                      # W
        elif key == 83:  st["sp"] -= 0.05                      # S
        elif key == 69:  st["el"] += 0.05                      # E
        elif key == 68:  st["el"] -= 0.05                      # D
        elif key == 80:  st["t"] = 0.90; st["sp"] = float(FLIGHT_TUCK_ARMS[0]); \
                         st["el"] = float(FLIGHT_TUCK_ARMS[1])  # P -> prior
        elif key == 82:  st["t"] = 0.0; st["sp"] = 0.0; st["el"] = 0.0  # R reset
        elif key == 84:  report()                              # T print
        st["dirty"] = True

    def draw(scn, sph, offending):
        scn.ngeom = 0
        for name, (centers, r) in sph.items():
            hot = offending is not None and name in offending
            rgba = np.array([1, 0.2, 0.2, 0.5]) if hot else np.array([0.3, 0.6, 1.0, 0.35])
            for c in centers:
                if scn.ngeom >= scn.maxgeom:
                    return
                g = scn.geoms[scn.ngeom]
                mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE,
                                    np.array([r, 0, 0.]), np.asarray(c, float),
                                    np.eye(3).reshape(9), rgba)
                g.category = int(mujoco.mjtCatBit.mjCAT_DECOR)
                scn.ngeom += 1

    print("\n[validate] keys: LEFT/RIGHT tuck+-.05  UP/DOWN +-.01  W/S shoulder  "
          "E/D elbow  P prior  R reset  T print\n")
    with mujoco.viewer.launch_passive(mj, mjd, key_callback=key_cb) as v:
        v.cam.azimuth, v.cam.elevation, v.cam.distance = 160, -15, 2.5
        v.cam.lookat[:] = [0.0, 0.0, _BASE_Z]
        while v.is_running():
            if st["dirty"]:
                apply()
                sph, dmin, pair = report()
                if v.user_scn is not None:
                    draw(v.user_scn, sph, pair if dmin <= cm.MARGIN else None)
                v.sync()
                st["dirty"] = False
            time.sleep(0.02)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="G1 self-collision validator")
    ap.add_argument("--check", action="store_true",
                    help="Headless: report tuck-prior + tuck-sweep clearances and exit")
    args = ap.parse_args()
    if args.check:
        run_check()
    else:
        run_interactive()
