##
#
# 3D 5-link articulated walker: stance → flight → landing.
# Full joint-space trajectory optimization via Pinocchio + CasADi + IPOPT.
#
##

import numpy as np
import os
import casadi as ca
import pinocchio as pin

from srb_walker.five_link_dynamics import FiveLinkDynamics
from srb_walker.g1_5link_params import (
    HIP_PITCH_MIN, HIP_PITCH_MAX,
    KNEE_MIN, KNEE_MAX,
    HIP_TORQUE_MAX, KNEE_TORQUE_MAX,
)

##############################################################
# Setup
##############################################################

dyn = FiveLinkDynamics()
nq, nv, nj = dyn.nq, dyn.nv, dyn.nj   # 13, 12, 6

dt       = 0.02
T_stance = 0.5
T_flight = 0.4
T_land   = 0.5
T        = T_stance + T_flight + T_land

N_stance = int(T_stance / dt)   # 25
N_flight = int(T_flight / dt)   # 20
N_land   = int(T_land   / dt)   # 25
N        = N_stance + N_flight + N_land  # 70

stance_end = N_stance
flight_end = N_stance + N_flight

print(f"N={N}  stance=[0,{stance_end})  flight=[{stance_end},{flight_end})  land=[{flight_end},{N})")

##############################################################
# Nominal poses and foot positions
##############################################################

q0 = dyn.neutral_standing()      # neutral upright, feet at z=0
p_foot_L0, p_foot_R0 = dyn.foot_positions(q0)  # (≈ [0, ±0.1185, 0])
p_foot_L0 = p_foot_L0.copy()
p_foot_R0 = p_foot_R0.copy()

# Crouch pose initial guess (for stance push-off)
q_crouch = q0.copy()
q_crouch[8]  = 0.5   # l_hip_pitch
q_crouch[9]  = 1.0   # l_knee_pitch
q_crouch[11] = 0.5   # r_hip_pitch
q_crouch[12] = 1.0   # r_knee_pitch
q_crouch[2]  = 0.55  # lower base height when crouching

# Mid-flight leg swing (hips forward, knees up)
q_swing = q0.copy()
q_swing[8]  = -0.5   # l_hip_pitch (leg forward)
q_swing[9]  =  0.3   # l_knee_pitch
q_swing[11] = -0.5   # r_hip_pitch
q_swing[12] =  0.3   # r_knee_pitch
q_swing[2]  =  0.85  # apex height

##############################################################
# NLP
##############################################################

opti = ca.Opti()

Q  = opti.variable(nq, N + 1)   # config at each node
V  = opti.variable(nv, N + 1)   # velocity at each node
U  = opti.variable(nj, N)       # joint torques per interval
LL = opti.variable(3, N)        # left  contact force
LR = opti.variable(3, N)        # right contact force

##############################################################
# Boundary conditions
##############################################################

q0_ca = ca.DM(q0)
v0_ca = ca.DM.zeros(nv)

opti.subject_to(Q[:, 0] == q0_ca)
opti.subject_to(V[:, 0] == v0_ca)

# terminal: upright, stopped
opti.subject_to(Q[0:3, N] == q0_ca[0:3])     # same xy position
opti.subject_to(Q[3:7, N] == q0_ca[3:7])     # upright orientation
opti.subject_to(Q[7:,  N] == q0_ca[7:])      # same joint angles
opti.subject_to(V[:, N] == v0_ca)

##############################################################
# Dynamics constraints
##############################################################

for k in range(N):

    q_k = Q[:, k]
    v_k = V[:, k]
    u_k = U[:, k]
    lL_k = LL[:, k]
    lR_k = LR[:, k]

    # acceleration from ABA
    a_k = dyn.aba_fn(q_k, v_k, u_k, lL_k, lR_k)

    # Euler integration
    v_next = v_k + dt * a_k
    q_next = dyn.integrate_fn(q_k, v_k, dt)

    opti.subject_to(V[:, k + 1] == v_next)
    opti.subject_to(Q[:, k + 1] == q_next)

# quaternion norm at every node
for k in range(N + 1):
    opti.subject_to(ca.sumsqr(Q[3:7, k]) == 1.0)

##############################################################
# Contact / phase constraints
##############################################################

mu = 0.7
A_fc = ca.DM([[ 1,  0, -mu],
              [-1,  0, -mu],
              [ 0,  1, -mu],
              [ 0, -1, -mu],
              [ 0,  0,  -1]])
b_fc = ca.DM.zeros(5)

p_L0 = ca.DM(p_foot_L0)
p_R0 = ca.DM(p_foot_R0)

for k in range(N):

    q_k = Q[:, k]
    pL_k, pR_k = dyn.foot_pos_fn(q_k)

    if k < stance_end:
        # feet fixed at initial positions
        opti.subject_to(pL_k == p_L0)
        opti.subject_to(pR_k == p_R0)
        # friction cone + normal force
        opti.subject_to(A_fc @ LL[:, k] <= b_fc)
        opti.subject_to(A_fc @ LR[:, k] <= b_fc)

    elif k < flight_end:
        # no contact forces
        opti.subject_to(LL[:, k] == 0)
        opti.subject_to(LR[:, k] == 0)
        # feet stay above ground
        opti.subject_to(pL_k[2] >= 0)
        opti.subject_to(pR_k[2] >= 0)

    else:
        # land back at original foot positions
        opti.subject_to(pL_k == p_L0)
        opti.subject_to(pR_k == p_R0)
        opti.subject_to(A_fc @ LL[:, k] <= b_fc)
        opti.subject_to(A_fc @ LR[:, k] <= b_fc)

# base height (COM stays above min during flight)
pz_min = 0.55
for k in range(N + 1):
    opti.subject_to(Q[2, k] >= pz_min)

##############################################################
# Joint & torque limits
##############################################################

HIP_ROLL_MAX = 0.5
for k in range(N + 1):
    # hip roll ±
    opti.subject_to(opti.bounded(-HIP_ROLL_MAX, Q[7,  k], HIP_ROLL_MAX))
    opti.subject_to(opti.bounded(-HIP_ROLL_MAX, Q[10, k], HIP_ROLL_MAX))
    # hip pitch
    opti.subject_to(opti.bounded(HIP_PITCH_MIN, Q[8,  k], HIP_PITCH_MAX))
    opti.subject_to(opti.bounded(HIP_PITCH_MIN, Q[11, k], HIP_PITCH_MAX))
    # knee
    opti.subject_to(opti.bounded(KNEE_MIN, Q[9,  k], KNEE_MAX))
    opti.subject_to(opti.bounded(KNEE_MIN, Q[12, k], KNEE_MAX))

for k in range(N):
    # hip roll/pitch torque
    opti.subject_to(opti.bounded(-HIP_TORQUE_MAX,  U[0, k], HIP_TORQUE_MAX))
    opti.subject_to(opti.bounded(-HIP_TORQUE_MAX,  U[1, k], HIP_TORQUE_MAX))
    opti.subject_to(opti.bounded(-HIP_TORQUE_MAX,  U[3, k], HIP_TORQUE_MAX))
    opti.subject_to(opti.bounded(-HIP_TORQUE_MAX,  U[4, k], HIP_TORQUE_MAX))
    # knee torque
    opti.subject_to(opti.bounded(-KNEE_TORQUE_MAX, U[2, k], KNEE_TORQUE_MAX))
    opti.subject_to(opti.bounded(-KNEE_TORQUE_MAX, U[5, k], KNEE_TORQUE_MAX))

##############################################################
# Objective
##############################################################

w_tau   = 1e-3
w_lam   = 1e-5
w_state = 1.0
w_vel   = 0.5
w_term  = 20.0

J = 0
for k in range(N):
    # joint tracking cost (keep upright)
    q_err = Q[7:, k] - q0_ca[7:]    # joint angle deviation
    J += w_state * ca.sumsqr(q_err)
    J += w_vel   * ca.sumsqr(V[:, k])
    J += w_tau   * ca.sumsqr(U[:, k])

    if k < stance_end or k >= flight_end:
        J += w_lam * (ca.sumsqr(LL[:, k]) + ca.sumsqr(LR[:, k]))

# terminal
q_err_f = Q[7:, N] - q0_ca[7:]
J += w_term * (ca.sumsqr(Q[0:3, N] - q0_ca[0:3])
             + ca.sumsqr(q_err_f)
             + ca.sumsqr(V[:, N]))
opti.minimize(J)

##############################################################
# Initial guesses
##############################################################

for k in range(N + 1):
    if k < stance_end:
        alpha = k / stance_end
        q_g = (1 - alpha) * q0 + alpha * q_crouch
    elif k < flight_end:
        alpha = (k - stance_end) / N_flight
        # parabolic apex
        pz = 0.85 + 0.1 * np.sin(np.pi * alpha)
        q_g = q0.copy()
        q_g[2] = pz
        # swing: interpolate from crouch to swing to landing pose
        q_g[8]  = q_crouch[8]  + alpha * (q_swing[8]  - q_crouch[8])
        q_g[9]  = q_crouch[9]  + alpha * (q_swing[9]  - q_crouch[9])
        q_g[11] = q_crouch[11] + alpha * (q_swing[11] - q_crouch[11])
        q_g[12] = q_crouch[12] + alpha * (q_swing[12] - q_crouch[12])
    else:
        alpha = (k - flight_end) / N_land
        q_g = (1 - alpha) * q_crouch + alpha * q0

    opti.set_initial(Q[:, k], q_g)
    opti.set_initial(V[:, k], 0.0)

for k in range(N):
    opti.set_initial(U[:, k], 0.0)
    if k < stance_end or k >= flight_end:
        # each foot carries half the robot's weight
        f_z = 33.34 * 9.81 / 2.0
        opti.set_initial(LL[:, k], [0, 0, f_z])
        opti.set_initial(LR[:, k], [0, 0, f_z])
    else:
        opti.set_initial(LL[:, k], 0.0)
        opti.set_initial(LR[:, k], 0.0)

##############################################################
# Solve
##############################################################

opti.solver("ipopt", {}, {
    "max_iter": 500,
    "print_level": 5,
    "linear_solver": "mumps",
    "warm_start_init_point": "yes",
})

print("Solving...")
sol = opti.solve()

##############################################################
# Extract and save
##############################################################

Q_sol  = np.array(sol.value(Q)).T    # (N+1, nq)
V_sol  = np.array(sol.value(V)).T    # (N+1, nv)
U_sol  = np.array(sol.value(U)).T    # (N,   nj)
LL_sol = np.array(sol.value(LL)).T   # (N,   3)
LR_sol = np.array(sol.value(LR)).T   # (N,   3)

time_arr = np.linspace(0, T, N + 1)

# foot positions at each node
p_feet = np.zeros((N + 1, 6))
for k in range(N + 1):
    pL, pR = dyn.foot_positions(Q_sol[k, :])
    p_feet[k, 0:3] = pL
    p_feet[k, 3:6] = pR

# contact schedule
c_sched = np.zeros(N + 1)
c_sched[:stance_end]  = 1.0
c_sched[flight_end:]  = 1.0

print(f"\nTerminal base pos: {Q_sol[N, 0:3]}")
print(f"Terminal base quat: {Q_sol[N, 3:7]}")
print(f"Max torque used: {np.abs(U_sol).max():.1f} N·m")

save_dir = "./results/srb_walker/five_link_jump/"
os.makedirs(save_dir, exist_ok=True)
np.savetxt(save_dir + "time.csv",    time_arr, delimiter=",")
np.savetxt(save_dir + "q_opt.csv",   Q_sol,    delimiter=",")
np.savetxt(save_dir + "v_opt.csv",   V_sol,    delimiter=",")
np.savetxt(save_dir + "u_opt.csv",   U_sol,    delimiter=",")
np.savetxt(save_dir + "lam_L.csv",   LL_sol,   delimiter=",")
np.savetxt(save_dir + "lam_R.csv",   LR_sol,   delimiter=",")
np.savetxt(save_dir + "p_feet.csv",  p_feet,   delimiter=",")
np.savetxt(save_dir + "c_sched.csv", c_sched,  delimiter=",")
print(f"Saved to {save_dir}")
