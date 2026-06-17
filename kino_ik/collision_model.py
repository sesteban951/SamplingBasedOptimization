##
# Shared self-collision model for the G1 — a coarse "sausage man": each major
# link is a capsule, rendered as a few spheres along its axis, so distances are
# smooth sphere-sphere norms (differentiable for the kino NLP) AND cheap to check
# numerically (validator).  Single source of truth used by:
#   - kino_ik/kino_nlp.py        (symbolic, cpin FK)  -> hard distance constraints
#   - kino_ik/validate_collision.py (numeric, pin FK) -> interactive validation
#
# Capsule axes are taken from the URDF joint placements (exact, auto-mirrored);
# radii are coarse limb radii to be calibrated against the FCL mesh ground truth
# in the validator.  FCL/coal distance is NOT differentiable through CasADi, so
# the optimizer cannot use it directly — hence this analytic sphere model.
##

import numpy as np

# Capsule radii by name (metres) — initial guesses; calibrate in the validator.
# Calibrated against FCL mesh ground truth (validate_collision.py): the trunk is
# narrower at the abdomen where the tucked thighs approach, so torso/thigh radii
# are slimmer than a naive bound to avoid false triggers at the intended tuck.
_RADIUS = {
    'torso': 0.08, 'pelvis': 0.10,
    'l_thigh': 0.05, 'r_thigh': 0.05,
    'l_shin':  0.045, 'r_shin': 0.045,
    'l_foot':  0.03, 'r_foot':  0.03,
    'l_uarm':  0.04, 'r_uarm':  0.04,
    'l_farm':  0.045, 'r_farm': 0.045,
}

N_SPHERES = 3          # spheres sampled along each capsule axis
MARGIN    = 0.01       # default safety margin [m] (extra clearance beyond radii)

# Foot capsule axis in ankle_roll_link frame (heel -> toe, from FOOT_CORNERS x-range).
_FOOT_P0 = np.array([-0.05, 0.0, -0.03])
_FOOT_P1 = np.array([ 0.12, 0.0, -0.03])
# Torso / pelvis axes (link frame, hand-set to wrap the trunk).
_TORSO_P0, _TORSO_P1   = np.array([0, 0, -0.05]), np.array([0, 0, 0.30])
_PELVIS_P0, _PELVIS_P1 = np.array([0, 0, -0.12]), np.array([0, 0, 0.00])
_HAND_EXT = 1.7        # extend forearm capsule past the wrist to cover the hand

# (name, link_frame, distal_joint or None).  Limb capsules span link origin ->
# the distal joint's placement (in the link frame); torso/pelvis/foot use the
# explicit axes above.
_SPECS = [
    ('torso',   'torso_link',            None),
    ('pelvis',  'pelvis',                None),
    ('l_thigh', 'left_hip_yaw_link',     'left_knee_joint'),
    ('r_thigh', 'right_hip_yaw_link',    'right_knee_joint'),
    ('l_shin',  'left_knee_link',        'left_ankle_pitch_joint'),
    ('r_shin',  'right_knee_link',       'right_ankle_pitch_joint'),
    ('l_foot',  'left_ankle_roll_link',  None),
    ('r_foot',  'right_ankle_roll_link', None),
    ('l_uarm',  'left_shoulder_yaw_link',  'left_elbow_joint'),
    ('r_uarm',  'right_shoulder_yaw_link', 'right_elbow_joint'),
    ('l_farm',  'left_elbow_link',       'left_wrist_roll_joint'),
    ('r_farm',  'right_elbow_link',      'right_wrist_roll_joint'),
]

# Curated non-adjacent pairs that can actually touch in tuck/crouch/landing
# (kinematically adjacent links — share a joint — are excluded; they always
# "touch").  Tune as needed.
PAIRS = [
    # legs <-> legs
    ('l_thigh', 'r_thigh'), ('l_shin', 'r_shin'),
    ('l_thigh', 'r_shin'),  ('r_thigh', 'l_shin'),
    ('l_foot', 'r_foot'),   ('l_shin', 'r_foot'), ('r_shin', 'l_foot'),
    # legs <-> trunk (deep tuck folds the legs onto the body)
    ('l_thigh', 'torso'), ('r_thigh', 'torso'),
    ('l_shin', 'torso'),  ('r_shin', 'torso'),
    ('l_foot', 'torso'),  ('r_foot', 'torso'),
    ('l_foot', 'pelvis'), ('r_foot', 'pelvis'),
    # foot <-> thigh (deep tuck folds the foot up toward the hip — real FCL mode)
    ('l_foot', 'l_thigh'), ('r_foot', 'r_thigh'),
    ('l_foot', 'r_thigh'), ('r_foot', 'l_thigh'),
    # arms <-> legs / trunk (freed shoulder-pitch + elbow swing)
    ('l_farm', 'l_thigh'), ('l_farm', 'r_thigh'),
    ('r_farm', 'r_thigh'), ('r_farm', 'l_thigh'),
    ('l_farm', 'l_shin'),  ('r_farm', 'r_shin'),
    ('l_farm', 'pelvis'),  ('r_farm', 'pelvis'),
    # arms <-> arms
    ('l_farm', 'r_farm'),
]


def build_capsules(model):
    """Return [{name, frame_id, p0, p1, r}] with axes from URDF joint placements."""
    caps = []
    for name, link, distal in _SPECS:
        fid = model.getFrameId(link)
        r = _RADIUS[name]
        if name in ('l_foot', 'r_foot'):
            p0, p1 = _FOOT_P0.copy(), _FOOT_P1.copy()
        elif name == 'torso':
            p0, p1 = _TORSO_P0.copy(), _TORSO_P1.copy()
        elif name == 'pelvis':
            p0, p1 = _PELVIS_P0.copy(), _PELVIS_P1.copy()
        else:
            p0 = np.zeros(3)
            p1 = np.array(model.jointPlacements[model.getJointId(distal)].translation)
            if name in ('l_farm', 'r_farm'):
                p1 = p1 * _HAND_EXT
        caps.append({'name': name, 'frame_id': int(fid), 'p0': p0, 'p1': p1, 'r': float(r)})
    return caps


def _axis_samples(p0, p1, n):
    """Local sphere centers along the capsule axis (n>=2 -> includes endpoints)."""
    if n <= 1:
        return [0.5 * (p0 + p1)]
    return [p0 + (i / (n - 1.0)) * (p1 - p0) for i in range(n)]


def capsule_spheres(caps, frame_pose, vecfn, n=N_SPHERES):
    """Generic world sphere centers.  Works for numpy or CasADi depending on the
    callables passed in.

    caps       : list from build_capsules
    frame_pose : frame_id -> (translation(3), rotation(3x3))   world placement
    vecfn      : converts a python/np 3-vector to the target type (np.asarray or ca.DM)
    returns    : {name: (list_of_centers, radius)}
    """
    out = {}
    for c in caps:
        t, R = frame_pose(c['frame_id'])
        centers = [t + R @ vecfn(lc) for lc in _axis_samples(c['p0'], c['p1'], n)]
        out[c['name']] = (centers, c['r'])
    return out


def world_spheres_numeric(model, data, q, caps, n=N_SPHERES):
    """Numeric world sphere centers via pinocchio FK."""
    import pinocchio as pin
    pin.framesForwardKinematics(model, data, q)

    def frame_pose(fid):
        M = data.oMf[fid]
        return np.array(M.translation), np.array(M.rotation)

    return capsule_spheres(caps, frame_pose, np.asarray, n)


def build_fcl_checker(model, urdf, pkg_dirs, safe_qs):
    """FCL mesh self-collision checker (ground truth) with always-touching pairs
    auto-disabled.  Any pair in collision at one of the `safe_qs` (e.g. neutral
    and standing) is structurally always overlapping (adjacent links / shared
    joint volume) and is disabled — no SRDF needed.

    Returns (geom_model, geom_data, disabled_set)."""
    import pinocchio as pin
    gm = pin.GeometryModel()
    pin.buildGeomFromUrdf(model, urdf, pin.COLLISION, gm, pkg_dirs)
    gm.addAllCollisionPairs()
    gd = gm.createData()
    data = model.createData()
    disabled = set()
    for q in safe_qs:
        pin.computeCollisions(model, data, gm, gd, np.asarray(q), False)
        for k in range(len(gm.collisionPairs)):
            if gd.collisionResults[k].isCollision():
                disabled.add(k)
    return gm, gd, disabled


def fcl_collisions(model, data, gm, gd, disabled, q):
    """Real (non-disabled) FCL mesh self-collisions at q -> list of (nameA,nameB)."""
    import pinocchio as pin
    pin.computeCollisions(model, data, gm, gd, np.asarray(q), False)
    out = []
    for k in range(len(gm.collisionPairs)):
        if k in disabled:
            continue
        if gd.collisionResults[k].isCollision():
            cp = gm.collisionPairs[k]
            out.append((gm.geometryObjects[cp.first].name,
                        gm.geometryObjects[cp.second].name))
    return out


def min_pair_distance(spheres, pairs=PAIRS):
    """Numeric: smallest signed clearance over all checked pairs.

    spheres : {name: (centers, radius)}  (numpy centers)
    returns : (min_clearance, (nameA, nameB), pa, pb)  where clearance =
              ||pa-pb|| - (rA+rB).  Negative => interpenetration.
    """
    best = (np.inf, None, None, None)
    for a, b in pairs:
        if a not in spheres or b not in spheres:
            continue
        (ca_, ra), (cb_, rb) = spheres[a], spheres[b]
        for pa in ca_:
            for pb in cb_:
                d = float(np.linalg.norm(np.asarray(pa) - np.asarray(pb))) - (ra + rb)
                if d < best[0]:
                    best = (d, (a, b), np.asarray(pa), np.asarray(pb))
    return best
