##
#
# Compute composite inertial properties for a Pinocchio robot model.
#
##

import numpy as np
import pinocchio as pin

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

URDF_FILE = "./models/g1/g1_29dof_rev_1_0.urdf"

Q_NOM = np.concatenate([
    [0, 0, 0.79],           # base position
    [0, 0, 0, 1],           # base orientation (quaternion)
    [0, 0, 0, 0, 0, 0,      # left leg  (hip, knee, ankle)
     0, 0, 0, 0, 0, 0,      # right leg (hip, knee, ankle)
     0, 0, 0,               # waist
     0.25,  0.25, 0, 1.0, 0, 0, 0,   # left arm
     0.25, -0.25, 0, 1.0, 0, 0, 0],  # right arm
])

PELVIS_JOINT_ID = 1  # free-flyer root joint; child link is the pelvis

np.set_printoptions(precision=6, suppress=True)

# ---------------------------------------------------------------------------
# Load model and run forward kinematics
# ---------------------------------------------------------------------------

model = pin.buildModelFromUrdf(URDF_FILE, pin.JointModelFreeFlyer())
data  = model.createData()

pin.forwardKinematics(model, data, Q_NOM)
pin.updateFramePlacements(model, data)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parallel_axis(m: float, r: np.ndarray) -> np.ndarray:
    """Inertia correction for shifting reference point by r (Steiner term)."""
    rx = pin.skew(r)
    return m * (rx.T @ rx)


def sym(A: np.ndarray) -> np.ndarray:
    """Symmetrize a matrix to eliminate numerical antisymmetry."""
    return 0.5 * (A + A.T)

# ---------------------------------------------------------------------------
# Total mass and system COM
# ---------------------------------------------------------------------------

total_mass  = sum(model.inertias[jid].mass for jid in range(1, model.njoints))

com_world = np.zeros(3)
for jid in range(1, model.njoints):
    I_joint = model.inertias[jid]
    oMj     = data.oMi[jid]
    com_world += I_joint.mass * (oMj.translation + oMj.rotation @ I_joint.lever)
com_world /= total_mass

print(f"Total mass              : {total_mass:.6f} kg")
print(f"COM (composite, world)  : {com_world}")
print(f"COM (pinocchio,  world) : {pin.centerOfMass(model, data, Q_NOM)}")

# ---------------------------------------------------------------------------
# Composite inertia about the system COM, in world frame
# ---------------------------------------------------------------------------

I_composite_world = np.zeros((3, 3))

for jid in range(1, model.njoints):
    I_joint = model.inertias[jid]
    oMj     = data.oMi[jid]
    R, p    = oMj.rotation, oMj.translation

    # Body inertia (about its own COM) rotated into world frame
    I_body_world = R @ I_joint.inertia @ R.T

    # Shift from body COM to system COM
    r = (p + R @ I_joint.lever) - com_world
    I_composite_world += I_body_world + parallel_axis(I_joint.mass, r)

I_composite_world = sym(I_composite_world)

print("\nComposite inertia about system COM (world frame):")
print(I_composite_world)

# ---------------------------------------------------------------------------
# Composite inertia in base / pelvis frames
# ---------------------------------------------------------------------------

R_base = data.oMi[PELVIS_JOINT_ID].rotation  # ^world R_base

I_composite_base = sym(R_base.T @ I_composite_world @ R_base)

print("\nComposite inertia about system COM (base / pelvis frame):")
print(I_composite_base)

# ---------------------------------------------------------------------------
# Principal moments of inertia
# ---------------------------------------------------------------------------

eigvals = np.sort(np.linalg.eigvalsh(I_composite_base))

print("\nPrincipal moments of inertia (about system COM, base frame):")
print(f"  I1: {eigvals[0]:.6f} kg·m²")
print(f"  I2: {eigvals[1]:.6f} kg·m²")
print(f"  I3: {eigvals[2]:.6f} kg·m²")

# ---------------------------------------------------------------------------
# Whole-body inertia about the pelvis origin, in pelvis frame
# ---------------------------------------------------------------------------

p_pelvis = data.oMi[PELVIS_JOINT_ID].translation

# Shift from system COM to pelvis origin
r_com_to_pelvis = com_world - p_pelvis
I_world_about_pelvis = sym(I_composite_world + parallel_axis(total_mass, r_com_to_pelvis))

I_pelvis_about_pelvis = sym(R_base.T @ I_world_about_pelvis @ R_base)

print("\nWhole-robot inertia about system COM, expressed in pelvis frame:")
print(I_composite_base)

print("\nWhole-robot inertia about pelvis origin, expressed in pelvis frame:")
print(I_pelvis_about_pelvis)