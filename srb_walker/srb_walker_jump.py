##
#
# 5-link 3D SRB traj opt: stance -> flight -> land.
#
##

import numpy as np
import os
import casadi as ca

from srb_walker.srb_5link import FiveLinkSRB
from srb_walker.g1_5link_params import PZ_COM_NOM


##############################################################
# Setup
##############################################################

srb = FiveLinkSRB()
f   = srb.f_disc
nq  = srb.nq   # 7
nv  = srb.nv   # 6
nx  = nq + nv  # 13

# timing
dt       = 0.02
T_stance = 0.6
T_flight = 0.4
T_land   = 0.6
T        = T_stance + T_flight + T_land

N_stance = int(T_stance / dt)
N_flight = int(T_flight / dt)
N_land   = int(T_land   / dt)
N        = int(T / dt)

stance_end = N_stance
flight_end = N_stance + N_flight

# ----------------------------------------------------------
# Decision Variables
# ----------------------------------------------------------

opti = ca.Opti()

X        = opti.variable(nx, N + 1)   # state [p, quat, v, w]
p_L_land = opti.variable(2)            # landing left  foot [x, y]
p_R_land = opti.variable(2)            # landing right foot [x, y]
F_L      = opti.variable(3, N)         # left  contact force  [Fx, Fy, Fz]
F_R      = opti.variable(3, N)         # right contact force  [Fx, Fy, Fz]
M_L      = opti.variable(3, N)         # left  ankle moment   [Mx, My, Mz]
M_R      = opti.variable(3, N)         # right ankle moment   [Mx, My, Mz]

# ----------------------------------------------------------
# Boundary Conditions
# ----------------------------------------------------------

pz_nom = PZ_COM_NOM   # 0.693 m
quat0  = np.array([1.0, 0.0, 0.0, 0.0])   # upright

x0 = np.array([0, 0, pz_nom,
               quat0[0], quat0[1], quat0[2], quat0[3],
               0, 0, 0,
               0, 0, 0])
x0_ca = ca.DM(x0)

# initial foot positions (directly below hips)
p0_L = ca.DM([0.0,  srb.hip_offset_y])
p0_R = ca.DM([0.0, -srb.hip_offset_y])

# goal: same position, upright, stopped
x_goal = x0.copy()
x_goal_ca = ca.DM(x_goal)

p_L_goal = ca.DM([0.0,  srb.hip_offset_y])
p_R_goal = ca.DM([0.0, -srb.hip_offset_y])

# ----------------------------------------------------------
# Dynamics Constraints
# ----------------------------------------------------------

opti.subject_to(X[:, 0] == x0_ca)

epsilon = 0.05
opti.subject_to(X[:, N] >= x_goal_ca - epsilon)
opti.subject_to(X[:, N] <= x_goal_ca + epsilon)

for k in range(N):

    p_com = X[0:3, k]
    quat  = X[3:7, k]

    # STANCE
    if k < stance_end:
        p_L = ca.vertcat(p0_L, 0.0)
        p_R = ca.vertcat(p0_R, 0.0)

        r_L = p_L - p_com
        r_R = p_R - p_com

        F_total = F_L[:, k] + F_R[:, k]
        M_total = (
            ca.cross(r_L, F_L[:, k])
          + ca.cross(r_R, F_R[:, k])
          + M_L[:, k] + M_R[:, k]
        )

        opti.subject_to(srb.leg_reach_sq_L(p_com, quat, p_L) <= srb.L_max**2)
        opti.subject_to(srb.leg_reach_sq_R(p_com, quat, p_R) <= srb.L_max**2)
        opti.subject_to(srb.leg_reach_sq_L(p_com, quat, p_L) >= srb.L_min**2)
        opti.subject_to(srb.leg_reach_sq_R(p_com, quat, p_R) >= srb.L_min**2)

    # FLIGHT
    elif k < flight_end:
        F_total = ca.DM.zeros(3)
        M_total = ca.DM.zeros(3)

    # LANDING
    else:
        p_L = ca.vertcat(p_L_land, 0.0)
        p_R = ca.vertcat(p_R_land, 0.0)

        r_L = p_L - p_com
        r_R = p_R - p_com

        F_total = F_L[:, k] + F_R[:, k]
        M_total = (
            ca.cross(r_L, F_L[:, k])
          + ca.cross(r_R, F_R[:, k])
          + M_L[:, k] + M_R[:, k]
        )

        opti.subject_to(srb.leg_reach_sq_L(p_com, quat, p_L) <= srb.L_max**2)
        opti.subject_to(srb.leg_reach_sq_R(p_com, quat, p_R) <= srb.L_max**2)
        opti.subject_to(srb.leg_reach_sq_L(p_com, quat, p_L) >= srb.L_min**2)
        opti.subject_to(srb.leg_reach_sq_R(p_com, quat, p_R) >= srb.L_min**2)

    u = ca.vertcat(F_total, M_total)
    opti.subject_to(X[:, k + 1] == f(X[:, k], u, dt))

# COM height floor
pz_min = 0.45
for k in range(N + 1):
    opti.subject_to(X[2, k] >= pz_min)

# ----------------------------------------------------------
# Contact Constraints
# ----------------------------------------------------------

mu             = 1.0
M_ankle_x_max  = 50.0
M_ankle_y_max  = 50.0
M_ankle_z_max  = 10.0
F_leg_max      = 500.0

A_friction, b_friction = srb.friction_cone_matrix(mu)

for k in range(N):

    if k < stance_end or k >= flight_end:
        opti.subject_to(A_friction @ F_L[:, k] <= b_friction)
        opti.subject_to(A_friction @ F_R[:, k] <= b_friction)
        opti.subject_to(ca.sumsqr(F_L[:, k]) <= F_leg_max**2)
        opti.subject_to(ca.sumsqr(F_R[:, k]) <= F_leg_max**2)
        opti.subject_to(opti.bounded(-M_ankle_x_max, M_L[0, k], M_ankle_x_max))
        opti.subject_to(opti.bounded(-M_ankle_y_max, M_L[1, k], M_ankle_y_max))
        opti.subject_to(opti.bounded(-M_ankle_z_max, M_L[2, k], M_ankle_z_max))
        opti.subject_to(opti.bounded(-M_ankle_x_max, M_R[0, k], M_ankle_x_max))
        opti.subject_to(opti.bounded(-M_ankle_y_max, M_R[1, k], M_ankle_y_max))
        opti.subject_to(opti.bounded(-M_ankle_z_max, M_R[2, k], M_ankle_z_max))

    else:   # flight
        opti.subject_to(F_L[:, k] == 0)
        opti.subject_to(F_R[:, k] == 0)
        opti.subject_to(M_L[:, k] == 0)
        opti.subject_to(M_R[:, k] == 0)

# landing foot placement
landing_tol = 0.05
opti.subject_to(ca.sumsqr(p_L_land - p_L_goal) <= landing_tol**2)
opti.subject_to(ca.sumsqr(p_R_land - p_R_goal) <= landing_tol**2)

# ----------------------------------------------------------
# Objective
# ----------------------------------------------------------

J = 0
for k in range(N):

    J += srb.state_cost(X[:, k], x_goal_ca)

    if k < stance_end or k >= flight_end:
        J += srb.contact_cost(F_L[:, k], F_R[:, k], M_L[:, k], M_R[:, k])

    if k < N - 1:
        J += srb.force_rate_cost(
            F_L[:, k], F_R[:, k], M_L[:, k], M_R[:, k],
            F_L[:, k+1], F_R[:, k+1], M_L[:, k+1], M_R[:, k+1],
            dt,
        )

J += srb.foot_placement_cost(p_L_land, p_R_land, p_L_goal, p_R_goal)
J += srb.terminal_cost(X[:, N], x_goal_ca)
opti.minimize(J)

# ----------------------------------------------------------
# Initial Guesses
# ----------------------------------------------------------

for k in range(N + 1):
    alpha = k / N
    p_guess = (1 - alpha) * x0[:3] + alpha * x_goal[:3]
    opti.set_initial(X[0:3, k], p_guess)
    opti.set_initial(X[3:7, k], [1.0, 0.0, 0.0, 0.0])
    opti.set_initial(X[7:13, k], 0.0)

opti.set_initial(p_L_land, p_L_goal)
opti.set_initial(p_R_land, p_R_goal)

for k in range(N):
    if k < stance_end or k >= flight_end:
        opti.set_initial(F_L[:, k], [0.0, 0.0, srb.m * srb.g / 2])
        opti.set_initial(F_R[:, k], [0.0, 0.0, srb.m * srb.g / 2])
    else:
        opti.set_initial(F_L[:, k], [0.0, 0.0, 0.0])
        opti.set_initial(F_R[:, k], [0.0, 0.0, 0.0])

opti.set_initial(M_L, 0.0)
opti.set_initial(M_R, 0.0)

# ----------------------------------------------------------
# Solve
# ----------------------------------------------------------

opti.solver("ipopt")
sol = opti.solve()

# ----------------------------------------------------------
# Extract Solution
# ----------------------------------------------------------

X_sol   = sol.value(X)
pL_land = sol.value(p_L_land)
pR_land = sol.value(p_R_land)
FL_sol  = sol.value(F_L)
FR_sol  = sol.value(F_R)
ML_sol  = sol.value(M_L)
MR_sol  = sol.value(M_R)

# ----------------------------------------------------------
# Reconstruct Wrench Trajectory
# ----------------------------------------------------------

F_traj = (FL_sol + FR_sol).T   # (N, 3)
M_traj = np.zeros((N, 3))

for k in range(N):
    if k < stance_end:
        p_com = X_sol[0:3, k]
        p_L   = np.array([float(p0_L[0]), float(p0_L[1]), 0.0])
        p_R   = np.array([float(p0_R[0]), float(p0_R[1]), 0.0])
    elif k < flight_end:
        continue
    else:
        p_com = X_sol[0:3, k]
        p_L   = np.array([pL_land[0], pL_land[1], 0.0])
        p_R   = np.array([pR_land[0], pR_land[1], 0.0])

    r_L = p_L - p_com
    r_R = p_R - p_com
    M_traj[k, :] = (
        np.cross(r_L, FL_sol[:, k])
      + np.cross(r_R, FR_sol[:, k])
      + ML_sol[:, k] + MR_sol[:, k]
    )

U = np.hstack((F_traj, M_traj))   # (N, 6)

# ----------------------------------------------------------
# Accelerations
# ----------------------------------------------------------

X_sol = X_sol.T              # (N+1, nx)
q_opt = X_sol[:, :nq]        # (N+1, 7): [p, quat]
v_opt = X_sol[:, nq:]        # (N+1, 6): [v, w]
a_opt = np.zeros_like(v_opt)

for k in range(N):
    xdot        = np.array(srb.f_cont(X_sol[k, :], U[k, :])).squeeze()
    a_opt[k, :] = xdot[nq:]
a_opt[N, :] = a_opt[N - 1, :]

# ----------------------------------------------------------
# Contact Schedule & Foot Positions
# ----------------------------------------------------------

contact_schedule = np.zeros(N + 1)
for k in range(N + 1):
    if k < stance_end or k >= flight_end:
        contact_schedule[k] = 1.0

# feet.csv: [pL_x, pL_y, pR_x, pR_y]
feet = np.full((N, 4), np.nan)
feet[:stance_end, 0:2] = np.array([float(p0_L[0]), float(p0_L[1])])
feet[:stance_end, 2:4] = np.array([float(p0_R[0]), float(p0_R[1])])
feet[flight_end:, 0:2] = np.array([pL_land[0], pL_land[1]])
feet[flight_end:, 2:4] = np.array([pR_land[0], pR_land[1]])

foot_positions = np.array([
    [float(p0_L[0]), float(p0_L[1]), 0.0, float(p0_R[0]), float(p0_R[1]), 0.0],
    [pL_land[0],     pL_land[1],     0.0, pR_land[0],     pR_land[1],     0.0],
])

# ----------------------------------------------------------
# Print
# ----------------------------------------------------------

print(f"\nLanding foot positions:")
print(f"  Left  foot: x={pL_land[0]:.3f},  y={pL_land[1]:.3f}")
print(f"  Right foot: x={pR_land[0]:.3f},  y={pR_land[1]:.3f}")

print(f"\nPhase summary:")
print(f"  Stance:  k=0        -> {stance_end-1:<3}  (t=0.00 -> {stance_end*dt:.2f}s)")
print(f"  Flight:  k={stance_end:<3}      -> {flight_end-1:<3}  (t={stance_end*dt:.2f} -> {flight_end*dt:.2f}s)")
print(f"  Landing: k={flight_end:<3}      -> {N-1:<3}  (t={flight_end*dt:.2f} -> {N*dt:.2f}s)")

print(f"\nTerminal state:")
print(f"  p_com = {X_sol[N, 0:3]}")
print(f"  quat  = {X_sol[N, 3:7]}")
print(f"  v_com = {X_sol[N, 7:10]}")

# ----------------------------------------------------------
# Save
# ----------------------------------------------------------

time_arr = np.linspace(0, T, N + 1)

save_dir = "./results/srb_walker/jump/"
os.makedirs(save_dir, exist_ok=True)

np.savetxt(save_dir + "time.csv",             time_arr,         delimiter=",")
np.savetxt(save_dir + "q_opt.csv",            q_opt,            delimiter=",")
np.savetxt(save_dir + "v_opt.csv",            v_opt,            delimiter=",")
np.savetxt(save_dir + "a_opt.csv",            a_opt,            delimiter=",")
np.savetxt(save_dir + "tau_opt.csv",          U,                delimiter=",")
np.savetxt(save_dir + "feet.csv",             feet,             delimiter=",")
np.savetxt(save_dir + "contact_schedule.csv", contact_schedule, delimiter=",")
np.savetxt(save_dir + "foot_positions.csv",   foot_positions,   delimiter=",")

print(f"\nSaved results to {save_dir}")
