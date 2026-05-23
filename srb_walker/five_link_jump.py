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

# Fixed node counts per phase — NLP structure stays constant.
# Time steps dt_s / dt_f / dt_l become decision variables so that
# T_phase = N_phase * dt_phase is optimized within the bounds below.
N_stance = 30
N_flight = 20
N_land   = 30
N        = N_stance + N_flight + N_land   # 70

stance_end = N_stance
flight_end = N_stance + N_flight

# Phase duration bounds [s] — edit these to taste
T_STANCE_MIN, T_STANCE_MAX = 0.2, 2.5
T_FLIGHT_MIN, T_FLIGHT_MAX = 0.5, 1.5
T_LAND_MIN,   T_LAND_MAX   = 0.2, 2.5

# Initial guesses for phase durations [s]
T_stance_guess = 0.5
T_flight_guess = 0.7
T_land_guess   = 0.5

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

# Mid-flight tuck pose (knees pulled up, body upright)
q_swing = q0.copy()
q_swing[8]  =  0.4   # l_hip_pitch (slight forward flex)
q_swing[9]  =  0.8   # l_knee_pitch (knees bent/tucked)
q_swing[11] =  0.4   # r_hip_pitch
q_swing[12] =  0.8   # r_knee_pitch
q_swing[2]  =  0.95  # apex height above standing (0.689m)

##############################################################
# NLP
##############################################################

opti = ca.Opti()

Q  = opti.variable(nq, N + 1)   # config at each node
V  = opti.variable(nv, N + 1)   # velocity at each node
U  = opti.variable(nj, N)       # joint torques per interval
LL = opti.variable(3, N)        # left  contact force
LR = opti.variable(3, N)        # right contact force

# Phase time steps (seconds per interval within each phase)
dt_s = opti.variable()   # stance
dt_f = opti.variable()   # flight
dt_l = opti.variable()   # landing

opti.subject_to(opti.bounded(T_STANCE_MIN / N_stance, dt_s, T_STANCE_MAX / N_stance))
opti.subject_to(opti.bounded(T_FLIGHT_MIN / N_flight, dt_f, T_FLIGHT_MAX / N_flight))
opti.subject_to(opti.bounded(T_LAND_MIN   / N_land,   dt_l, T_LAND_MAX   / N_land))

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

    q_k  = Q[:, k]
    v_k  = V[:, k]
    u_k  = U[:, k]
    lL_k = LL[:, k]
    lR_k = LR[:, k]

    # time step for this interval
    if k < stance_end:
        dt_k = dt_s
    elif k < flight_end:
        dt_k = dt_f
    else:
        dt_k = dt_l

    # acceleration from ABA
    a_k = dyn.aba_fn(q_k, v_k, u_k, lL_k, lR_k)

    # Euler integration
    v_next = v_k + dt_k * a_k
    q_next = dyn.integrate_fn(q_k, v_k, dt_k)

    opti.subject_to(V[:, k + 1] == v_next)
    opti.subject_to(Q[:, k + 1] == q_next)

# quaternion norm at every node
for k in range(N + 1):
    opti.subject_to(ca.sumsqr(Q[3:7, k]) == 1.0)

##############################################################
# Contact / phase constraints
##############################################################

mu = 0.5
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

# base height: general floor + must actually be airborne during flight
pz_min = 0.45
# L_MAX=0.72m → max COM height with feet on ground is ~0.73m.
# Only enforce flight floor on interior flight nodes (not at phase boundaries
# where foot contact forces transition and the kinematic limit kicks in).
pz_flight_min = 0.71
for k in range(N + 1):
    opti.subject_to(Q[2, k] >= pz_min)
for k in range(stance_end + 1, flight_end):
    opti.subject_to(Q[2, k] >= pz_flight_min)

# enforce left-right leg symmetry during flight (prevents yaw spin from asymmetric leg motion)
for k in range(stance_end, flight_end + 1):
    opti.subject_to(Q[8,  k] == Q[11, k])   # hip pitch L == R
    opti.subject_to(Q[9,  k] == Q[12, k])   # knee L == R
    opti.subject_to(Q[7,  k] == -Q[10, k])  # hip roll antisymmetric (mirror)

# limit body pitch: relaxed on ground, tight in the air (~10 deg during flight)
PITCH_MAX_GROUND = 0.35
PITCH_MAX_FLIGHT = 0.18
for k in range(N + 1):
    if stance_end <= k <= flight_end:
        opti.subject_to(opti.bounded(-PITCH_MAX_FLIGHT, Q[4, k], PITCH_MAX_FLIGHT))
    else:
        opti.subject_to(opti.bounded(-PITCH_MAX_GROUND, Q[4, k], PITCH_MAX_GROUND))

##############################################################
# Joint & torque limits
##############################################################

HIP_ROLL_MAX = 0.5
# tighter bounds during flight to prevent legs clipping through the body
HIP_PITCH_FLIGHT_MIN = -0.3   # don't let legs swing far backward
HIP_PITCH_FLIGHT_MAX =  1.8   # allow tuck but not extreme fold
KNEE_FLIGHT_MAX      =  2.2   # knees can bend for tuck but not fold into body
HIP_ROLL_FLIGHT_MAX  =  0.2   # keep legs close to sagittal plane in the air

for k in range(N + 1):
    if stance_end <= k <= flight_end:
        opti.subject_to(opti.bounded(-HIP_ROLL_FLIGHT_MAX, Q[7,  k], HIP_ROLL_FLIGHT_MAX))
        opti.subject_to(opti.bounded(-HIP_ROLL_FLIGHT_MAX, Q[10, k], HIP_ROLL_FLIGHT_MAX))
        opti.subject_to(opti.bounded(HIP_PITCH_FLIGHT_MIN, Q[8,  k], HIP_PITCH_FLIGHT_MAX))
        opti.subject_to(opti.bounded(HIP_PITCH_FLIGHT_MIN, Q[11, k], HIP_PITCH_FLIGHT_MAX))
        opti.subject_to(opti.bounded(KNEE_MIN, Q[9,  k], KNEE_FLIGHT_MAX))
        opti.subject_to(opti.bounded(KNEE_MIN, Q[12, k], KNEE_FLIGHT_MAX))
    else:
        opti.subject_to(opti.bounded(-HIP_ROLL_MAX, Q[7,  k], HIP_ROLL_MAX))
        opti.subject_to(opti.bounded(-HIP_ROLL_MAX, Q[10, k], HIP_ROLL_MAX))
        opti.subject_to(opti.bounded(HIP_PITCH_MIN, Q[8,  k], HIP_PITCH_MAX))
        opti.subject_to(opti.bounded(HIP_PITCH_MIN, Q[11, k], HIP_PITCH_MAX))
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

# Pinocchio quat: Q[3]=qx (roll), Q[4]=qy (pitch), Q[5]=qz (yaw)
w_no_twist        = 50.0    # roll + yaw throughout
w_no_twist_flight = 300.0   # extra pitch/roll/yaw penalty during flight

w_tau      = 1e-3
w_lam      = 5e-4    # higher than before to discourage force spikes
w_lam_rate = 1e-4    # penalise sudden changes in contact force
w_state    = 1.0
w_vel      = 0.5
w_term     = 20.0
w_height   = 2.0     # reward flight height to avoid split degenerate solution

J = 0
for k in range(N):
    J += w_tau      * ca.sumsqr(U[:, k])
    J += w_vel      * ca.sumsqr(V[:, k])
    # penalise roll, pitch, and yaw to keep body upright
    J += w_no_twist * (Q[3, k]**2 + Q[4, k]**2 + Q[5, k]**2)
    if stance_end <= k < flight_end:
        # penalise all rotations strongly during flight
        J += w_no_twist_flight * (Q[3, k]**2 + Q[4, k]**2 + Q[5, k]**2)

    if k < stance_end:
        J += w_lam * (ca.sumsqr(LL[:, k]) + ca.sumsqr(LR[:, k]))
        if k > 0:
            J += w_lam_rate * (ca.sumsqr(LL[:, k] - LL[:, k-1])
                             + ca.sumsqr(LR[:, k] - LR[:, k-1]))
    elif k >= flight_end:
        q_err = Q[7:, k] - q0_ca[7:]
        J += w_state * ca.sumsqr(q_err)
        J += w_lam * (ca.sumsqr(LL[:, k]) + ca.sumsqr(LR[:, k]))
        if k > flight_end:
            J += w_lam_rate * (ca.sumsqr(LL[:, k] - LL[:, k-1])
                             + ca.sumsqr(LR[:, k] - LR[:, k-1]))
    else:
        q_err = Q[7:, k] - q0_ca[7:]
        J += w_state * ca.sumsqr(q_err)
        J -= w_height * Q[2, k]   # maximize height during flight

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

opti.set_initial(dt_s, T_stance_guess / N_stance)
opti.set_initial(dt_f, T_flight_guess / N_flight)
opti.set_initial(dt_l, T_land_guess   / N_land)

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

dt_s_val = float(sol.value(dt_s))
dt_f_val = float(sol.value(dt_f))
dt_l_val = float(sol.value(dt_l))

T_s = N_stance * dt_s_val
T_f = N_flight * dt_f_val
T_l = N_land   * dt_l_val
print(f"\nOptimized phase durations:  stance={T_s:.3f}s  flight={T_f:.3f}s  land={T_l:.3f}s  total={T_s+T_f+T_l:.3f}s")

time_arr = np.concatenate([
    np.arange(N_stance + 1) * dt_s_val,
    T_s + np.arange(1, N_flight + 1) * dt_f_val,
    T_s + T_f + np.arange(1, N_land + 1) * dt_l_val,
])

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
