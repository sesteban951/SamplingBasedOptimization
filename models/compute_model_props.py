##
#
# Compute some model properties
#
##

import numpy as np
import pinocchio as pin

###########################################################
# PICK THE MODEL TO LOAD
###########################################################

# file path
urdf_file = "./models/g1/g1_29dof_rev_1_0.urdf"

# Load the model
model = pin.buildModelFromUrdf(urdf_file)
data = model.createData()

###########################################################
# MODEL INFO 
###########################################################

# Set print precision
np.set_printoptions(precision=4, suppress=True)

# nominal standing position 
q_nom_base = np.array([
    0, 0, 0.79,        #  base pos
    0, 0, 0, 1,        # base quat (x, y, z, w)
])  # base position (x, y, z)
q_nom_joints = np.array([    
    0, 0, 0, 0, 0, 0,             # left leg (hip, knee, ankle)
    0, 0, 0, 0, 0, 0,             # right leg (hip, knee, ankle)
    0, 0, 0,                      # waist
    0.25,  0.25, 0, 1.0, 0, 0, 0, # left arm
    0.25, -0.25, 0, 1.0, 0, 0, 0  # right arm
])

# Forward kinematics and update frame placements
pin.forwardKinematics(model, data, q_nom_joints)
pin.updateFramePlacements(model, data)


###########################################################
# COMPUTE COMPOSITE PROPERTIES
###########################################################

# compute total mass and COM
total_mass = 0.0
com_world = np.zeros(3)

for i in range(model.njoints):
    I_joint = model.inertias[i]
    mass = I_joint.mass
    
    # Position of this body's COM in world frame
    oMj = data.oMi[i]
    com_i_world = oMj.translation + oMj.rotation @ I_joint.lever
    
    total_mass += mass
    com_world += mass * com_i_world

com_world /= total_mass

print(f"Total mass: {total_mass:.4f} kg")
print(f"COM composite (world frame): {com_world}")
print(f"COM pinocchio (world frame): {pin.centerOfMass(model,data)}")

# compute composite inertia about COM in WORLD frame
I_composite_world = np.zeros((3, 3))

for i in range(model.njoints):
    I_joint = model.inertias[i]
    mass = I_joint.mass
    oMj = data.oMi[i]
    
    # Inertia in joint frame (about body's own COM)
    I_body = I_joint.inertia
    
    # Rotate to world frame
    R = oMj.rotation
    I_world = R @ I_body @ R.T
    
    # Position of body COM in world frame
    com_i_world = oMj.translation + R @ I_joint.lever
    
    # Parallel axis theorem: shift from body COM to system COM
    r = com_i_world - com_world
    r_cross = pin.skew(r)
    I_parallel = mass * (r_cross.T @ r_cross)
    
    # Add contribution
    I_composite_world += I_world + I_parallel

print(f"\nComposite inertia about COM (world frame):")
print(I_composite_world)

# 3. Express composite inertia in BODY (base) frame
# Get base frame orientation
base_joint_id = 1  # Usually joint 1 is the floating base
oM_base = data.oMi[base_joint_id]
R_base = oM_base.rotation  # world_R_base

# Transform: I_body = R_base^T * I_world * R_base
I_composite_body = R_base.T @ I_composite_world @ R_base

print(f"\nComposite inertia about COM (body/base frame):")
print(I_composite_body)

# 4. Principal moments of inertia
eigenvalues = np.linalg.eigvals(I_composite_body)
eigenvalues = np.sort(eigenvalues)

print(f"\nPrincipal moments of inertia:")
print(f"  I1: {eigenvalues[0]:.4f} kg⋅m²")
print(f"  I2: {eigenvalues[1]:.4f} kg⋅m²")
print(f"  I3: {eigenvalues[2]:.4f} kg⋅m²")