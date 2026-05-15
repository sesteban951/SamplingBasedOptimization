##
#
# G1 5-link planar model parameters.
# Extracted from g1_29dof_rev_1_0.urdf / g1_planar.xml at nominal upright stance.
#
# 5-link grouping (sagittal plane):
#   Torso  : pelvis + waist + torso + arms  (lumped)
#   L/R Thigh : hip_pitch + hip_roll + hip_yaw  (lumped)
#   L/R Shank : knee + ankle_pitch + ankle_roll (lumped)
#
##

# --- Inertial (from compute_model_props.py at nominal standing config) ---
M_TOTAL = 33.34      # kg    total robot mass
I_YY    = 3.300958   # kg*m^2  planar inertia about COM (sagittal / y-axis)
G       = 9.81       # m/s^2

# --- Nominal COM height at upright standing (pelvis at 0.790 m, COM 0.097 m below pelvis) ---
PZ_COM_NOM = 0.693   # m

# --- Hip joint offset from COM in body frame ---
# pelvis origin is 0.097 m above COM; hip_pitch is 0.1027 m below pelvis → 0.006 m below COM
# lateral offset from g1_planar.xml: left_hip_pitch_link y = 0.064452 + 0.052 ≈ 0.1185 m
HIP_OFFSET_X =  0.000   # m  forward/back (zero by sagittal symmetry)
HIP_OFFSET_Y =  0.1185  # m  lateral (± for left/right), same as srb.hip_offset
HIP_OFFSET_Z = -0.006   # m  below COM in body frame

# --- Leg link lengths (sagittal approximation) ---
# Thigh: hip_pitch → knee,  net 3D distance through hip_roll/yaw chain
L_THIGH = 0.335   # m
# Shank: knee_pitch → ankle_pitch
L_SHANK = 0.300   # m
# Foot: ankle_pitch → ground contact sphere (ankle_roll + sphere offset)
L_FOOT  = 0.048   # m

# Derived reach bounds (hip joint to ground contact)
# L_THIGH + L_SHANK + L_FOOT = 0.683 m but the hip chain has rotated frames that
# add a few mm; use 0.72 m to stay feasible at nominal stance height (0.693 m COM).
L_MAX = 0.72    # m  near-fully extended leg
L_MIN = 0.30    # m  practical lower bound (heavily bent knee)

# --- Joint limits (from g1_planar.xml actuated sagittal joints) ---
HIP_PITCH_MIN   = -2.5307   # rad
HIP_PITCH_MAX   =  2.8798   # rad
KNEE_MIN        = -0.0873   # rad  (nearly straight; G1 knee slightly hyperextends)
KNEE_MAX        =  2.8798   # rad
ANKLE_PITCH_MIN = -0.8727   # rad
ANKLE_PITCH_MAX =  0.5236   # rad

# --- Actuator torque limits ---
HIP_TORQUE_MAX   = 88.0    # N*m
KNEE_TORQUE_MAX  = 139.0   # N*m
ANKLE_TORQUE_MAX = 50.0    # N*m
