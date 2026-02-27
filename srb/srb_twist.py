##
#
# Single Rigid Body Traj Opt with leg kinematics.
#
##

# standard imports
import numpy as np
import os

# casadi import
import casadi as ca

# custom imports
from utils.kinematics import kin
from srb.srb import SRBDynamics


class SRB_Jump(SRBDynamics):

    # initialize the class
    def __init__(self):

        super().__init__()

        # simple quadratic penalty on states
        self.Qx = ca.diag(ca.vertcat(
            1.0, 1.0, 1.0,  # px, py, pz
            1.0, 1.0, 1.0,  # qx, qy, qz
            1.0, 1.0, 1.0,  # vx, vy, vz
            1.0, 1.0, 1.0   # wx, wy, wz
        ))

        # foot placement cost weights
        self.Q_foot = 100.0

        # penalize forces and moments
        self.Q_force = 0.005
        self.Q_moment = 0.0005

        # terminal weights
        self.Qx_f = 100.0 * self.Qx


    ###############################################################
    # Cost Functions
    ###############################################################

    # State-only running cost (no input cost)
    def state_cost(self, x, x_goal):
        """Cost on state tracking only"""
        
        # compute errors
        pos_err = x[0:3] - x_goal[0:3]
        vel_err = x[7:10] - x_goal[7:10]
        omega_err = x[10:13] - x_goal[10:13]

        quat_err = kin.quat_diff_ca(x[3:7], x_goal[3:7])
        quat_err_log = kin.quat_log_ca(quat_err)

        # state error vector
        e_x = ca.vertcat(
            pos_err,
            quat_err_log,
            vel_err,
            omega_err
        )

        # state cost
        cost = 0.5 * e_x.T @ self.Qx @ e_x

        return cost
    
    # Contact-based input cost
    def contact_cost(self, F_L, F_R, M_L, M_R):
        """Cost on contact forces and moments"""
        
        cost = (
            0.5 * self.Q_force * (ca.sumsqr(F_L) + ca.sumsqr(F_R))
          + 0.5 * self.Q_moment * (ca.sumsqr(M_L) + ca.sumsqr(M_R))
        )
        
        return cost
    
    # Foot placement cost
    def foot_placement_cost(self, p_L, p_R, p_L_des, p_R_des):
        """Cost on foot placement tracking"""
        
        cost =(
            0.5 * self.Q_foot * ca.sumsqr(p_L - p_L_des)
          + 0.5 * self.Q_foot * ca.sumsqr(p_R - p_R_des)
        )
        
        return cost
    
    # Terminal cost (UNCHANGED)
    def terminal_cost(self, x, x_goal):
        
        # compute errors
        pos_err   = x[0:3]   - x_goal[0:3]
        vel_err   = x[7:10]  - x_goal[7:10]
        omega_err = x[10:13] - x_goal[10:13]

        quat_err = kin.quat_diff_ca(x[3:7], x_goal[3:7])
        quat_err_log = kin.quat_log_ca(quat_err)
        
        # state error vector
        e = ca.vertcat(
            pos_err,
            quat_err_log,
            vel_err,
            omega_err
        )

        # terminal cost
        cost_terminal = e.T @ self.Qx_f @ e

        return cost_terminal


##############################################################
# Trajectory Optimization
##############################################################

# create the dynamics object
srb = SRB_Jump()
f = srb.f_disc
nq = srb.nq
nv = srb.nv
nx = nq + nv
nu = srb.nu

# fix timings
dt = 0.02        # time step
T_stance = 0.7   # stance duration
T_flight = 0.6   # flight duration
T_land = 0.5     # landing duration
T = T_stance + T_flight + T_land  # total trajectory duration

# number of steps
N_stance = int(T_stance / dt)
N_flight = int(T_flight / dt)
N_land = int(T_land / dt)
N = int(T / dt) 

# phase boundaries
stance_end = N_stance
flight_end = N_stance + N_flight

# ----------------------------------------------------------
# Setup the optimization problem
# ----------------------------------------------------------

# make the optimizer
opti = ca.Opti()

# horizon variables
X = opti.variable(nx, N + 1)  # states over the horizon

# decision variables
p_L_land = opti.variable(2)  # single landing position
p_R_land = opti.variable(2)  # single landing position
F_L = opti.variable(3, N)   # forces along left leg
F_R = opti.variable(3, N)   # forces along right leg
M_L = opti.variable(3, N)   # ankle moments at left ankle
M_R = opti.variable(3, N)   # ankle moments at right ankle

# initial condition
pitch0_deg = 1.0
quat0 = np.array([
    np.cos(np.radians(pitch0_deg) / 2),  # qw
    0,                                   # qx
    np.sin(np.radians(pitch0_deg) / 2),  # qy
    0                                    # qz
])
x0 = np.array([0, 0, 0.69,  # p_com
               quat0[0], quat0[1], quat0[2], quat0[3],  # quaternion
               0, 0, 0,    # v_com
               0, 0, 0])   # w_body
x0_ca = ca.DM(x0)
p0_L = np.array([0,  srb.hip_offset]) # initial left foot position
p0_R = np.array([0, -srb.hip_offset]) # initial right foot position
p0_L = ca.DM(p0_L)
p0_R = ca.DM(p0_R)

# desired goal state - jump forward, land upright with 180 deg CCW yaw twist
px_goal = 0.5
yaw_goal = 2*np.pi
quat_goal = np.array([
    np.cos(yaw_goal / 2),  # qw
    0.0,                   # qx
    0.0,                   # qy
    np.sin(yaw_goal / 2),  # qz
])
x_goal = np.array([px_goal, 0, 0.69,           # p_com (forward, same height)
                   quat_goal[0], quat_goal[1], # quaternion (upright, yaw = +pi)
                   quat_goal[2], quat_goal[3],
                   0, 0, 0,                    # v_com (stopped)
                   0, 0, 0])                   # w_body
x_goal_ca = ca.DM(x_goal)

# Desired landing foot positions
p_L_goal = ca.DM([x_goal[0],  srb.hip_offset])
p_R_goal = ca.DM([x_goal[0], -srb.hip_offset])

# ----------------------------------------------------------
# Dynamics Constraints
# ----------------------------------------------------------

# set the initial condition 
opti.subject_to(X[:, 0] == x0_ca)

# set box constraint on terminal condition
epsilon = 0.005
x_terminal_lb = x_goal_ca - epsilon
x_terminal_ub = x_goal_ca + epsilon
opti.subject_to(X[:, N] >= x_terminal_lb)
opti.subject_to(X[:, N] <= x_terminal_ub)

# kinematic limits
L_max = 0.75   # [m] max leg length
L_min = 0.30   # [m] min leg length

# compute the dynamics constraints
for k in range(N):

    # STANCE
    if k < stance_end:
        # Extract COM position
        p_com = X[0:3, k]
        
        # stance is prescribed intial condition
        p_L = ca.vertcat(p0_L, 0)
        p_R = ca.vertcat(p0_R, 0)
        
        # moment arms from COM to feet
        r_L = p_L - p_com
        r_R = p_R - p_com

        # total wrench from contacts (in world frame)
        F_total = F_L[:, k] + F_R[:, k]
        M_total = (
              ca.cross(r_L, F_L[:, k]) 
            + ca.cross(r_R, F_R[:, k]) 
            + M_L[:, k] + M_R[:, k]
        )

        # constrain the leg length
        opti.subject_to(ca.sumsqr(r_L) <= L_max**2)
        opti.subject_to(ca.sumsqr(r_R) <= L_max**2)
        opti.subject_to(ca.sumsqr(r_L) >= L_min**2)
        opti.subject_to(ca.sumsqr(r_R) >= L_min**2)

    # FLIGHT
    elif (k>= stance_end) and (k < flight_end):
        F_total = ca.DM.zeros(3)
        M_total = ca.DM.zeros(3)

    # LANDING
    else:
        # Extract COM position
        p_com = X[0:3, k]
        
        # feet are decision variables but at fixed height
        p_L = ca.vertcat(p_L_land, 0)
        p_R = ca.vertcat(p_R_land, 0)

        # moment arms from COM to feet
        r_L = p_L - p_com
        r_R = p_R - p_com
    
        # total wrench from contacts (in world frame)
        F_total = F_L[:, k] + F_R[:, k]
        M_total = (
              ca.cross(r_L, F_L[:, k]) 
            + ca.cross(r_R, F_R[:, k])
            + M_L[:, k] + M_R[:, k]
        )

        # constrain the leg length
        opti.subject_to(ca.sumsqr(r_L) <= L_max**2)
        opti.subject_to(ca.sumsqr(r_R) <= L_max**2)
        opti.subject_to(ca.sumsqr(r_L) >= L_min**2)
        opti.subject_to(ca.sumsqr(r_R) >= L_min**2)

    # Combined wrench as control input
    u = ca.vertcat(F_total, M_total)
    
    # Dynamics
    x_next = f(X[:, k], u, dt)
    opti.subject_to(X[:, k + 1] == x_next)

# add z_com constraints
pz_min = 0.20
pz_max = 2.0
for k in range(N+1):
    opti.subject_to(X[2, k] >= pz_min)  # enforce constant height
    opti.subject_to(X[2, k] <= pz_max)  # enforce constant height

# ----------------------------------------------------------
# Contact Constraints
# ----------------------------------------------------------

# Contact parameters
mu = 1.0              # friction coefficient
M_ankle_x_max = 50.0  # [N*m]
M_ankle_y_max = 50.0  # [N*m]
M_ankle_z_max = 10.0  # [N*m]
F_leg_max = 500.0     # [N] max force per leg

# Get friction cone constraint matrices
A_friction, b_friction = srb.friction_cone_matrix(mu)

for k in range(N):

    # STANCE
    if k < stance_end:
        # friction cone constraints
        opti.subject_to(A_friction @ F_L[:, k] <= b_friction)
        opti.subject_to(A_friction @ F_R[:, k] <= b_friction)

        # force limits
        opti.subject_to(ca.sumsqr(F_L[:, k]) <= F_leg_max**2)
        opti.subject_to(ca.sumsqr(F_R[:, k]) <= F_leg_max**2)

        # moment limits
        opti.subject_to(opti.bounded(-M_ankle_x_max, M_L[0, k], M_ankle_x_max))
        opti.subject_to(opti.bounded(-M_ankle_y_max, M_L[1, k], M_ankle_y_max))
        opti.subject_to(opti.bounded(-M_ankle_z_max, M_L[2, k], M_ankle_z_max))

        opti.subject_to(opti.bounded(-M_ankle_x_max, M_R[0, k], M_ankle_x_max))
        opti.subject_to(opti.bounded(-M_ankle_y_max, M_R[1, k], M_ankle_y_max))
        opti.subject_to(opti.bounded(-M_ankle_z_max, M_R[2, k], M_ankle_z_max))

    # FLIGHT
    elif (k>= stance_end) and (k < flight_end):
        # no contact forces or moments
        opti.subject_to(F_L[:, k] == 0)
        opti.subject_to(F_R[:, k] == 0)
        opti.subject_to(M_L[:, k] == 0)
        opti.subject_to(M_R[:, k] == 0)

    # LANDING
    else:
        # friction cone constraints
        opti.subject_to(A_friction @ F_L[:, k] <= b_friction)
        opti.subject_to(A_friction @ F_R[:, k] <= b_friction)

        # force limits
        opti.subject_to(ca.sumsqr(F_L[:, k]) <= F_leg_max**2)
        opti.subject_to(ca.sumsqr(F_R[:, k]) <= F_leg_max**2)

        # moment limits
        opti.subject_to(opti.bounded(-M_ankle_x_max, M_L[0, k], M_ankle_x_max))
        opti.subject_to(opti.bounded(-M_ankle_y_max, M_L[1, k], M_ankle_y_max))
        opti.subject_to(opti.bounded(-M_ankle_z_max, M_L[2, k], M_ankle_z_max))
        
        opti.subject_to(opti.bounded(-M_ankle_x_max, M_R[0, k], M_ankle_x_max))
        opti.subject_to(opti.bounded(-M_ankle_y_max, M_R[1, k], M_ankle_y_max))
        opti.subject_to(opti.bounded(-M_ankle_z_max, M_R[2, k], M_ankle_z_max))

# landing foot placement constraint
landing_tol = 0.1  # foot landing tolerance
opti.subject_to(ca.sumsqr(p_L_land - p_L_goal) <= landing_tol**2)
opti.subject_to(ca.sumsqr(p_R_land - p_R_goal) <= landing_tol**2)

# ----------------------------------------------------------
# Objective Function
# ----------------------------------------------------------

# stance yaw-overrun penalty settings (encourage twist to happen in flight)
W_stance_yaw = 200.0
yaw_allow = 0.15  # [rad]

# total cost
J = 0
for k in range(N):

    # phase-aware state objective:
    # no state tracking in stance; twist in flight; track final state in landing
    if k < stance_end:
        # Penalize only excess yaw progress toward the desired twist direction.
        qk = X[3:7, k]  # [qw, qx, qy, qz]
        yaw_k = ca.atan2(
            2.0 * (qk[0] * qk[3] + qk[1] * qk[2]),
            1.0 - 2.0 * (qk[2] * qk[2] + qk[3] * qk[3])
        )
        yaw_progress = ca.sign(yaw_goal) * yaw_k
        yaw_excess = ca.fmax(0.0, yaw_progress - yaw_allow)
        J += 0.5 * W_stance_yaw * yaw_excess**2
        x_ref_k = None
    elif k < flight_end:
        alpha = (k - stance_end + 1) / N_flight
        yaw_k = alpha * yaw_goal
        quat_k = np.array([
            np.cos(yaw_k / 2),
            0.0,
            0.0,
            np.sin(yaw_k / 2),
        ])
        x_ref_k = x0.copy()
        x_ref_k[0] = alpha * px_goal
        x_ref_k[1] = 0.0
        x_ref_k[2] = x_goal[2]
        x_ref_k[3:7] = quat_k
        x_ref_k[7:13] = 0.0
    else:
        x_ref_k = x_goal

    # state cost only during flight + landing
    if x_ref_k is not None:
        J += srb.state_cost(X[:, k], ca.DM(x_ref_k))

    # contact cost
    if k < stance_end or k >= flight_end:
        J += srb.contact_cost(F_L[:, k], F_R[:, k], M_L[:, k], M_R[:, k])

# foot placement cost
J += srb.foot_placement_cost(p_L_land, p_R_land, p_L_goal, p_R_goal)

# Terminal cost (optional since we have a terminal constraint, but can help convergence)
J += srb.terminal_cost(X[:, N], x_goal_ca)

# set the objective
opti.minimize(J)

# ----------------------------------------------------------
# Initial Guesses
# ----------------------------------------------------------

# state trajectory
for k in range(N + 1):

    # interp coeff
    alpha = k / N

    # com position
    p_com_guess = (1 - alpha) * x0[:3] + alpha * x_goal[:3]
    opti.set_initial(X[0:3, k], p_com_guess)

    # quaternion initial guess aligned with phase-aware yaw objective
    if k < stance_end:
        yaw_k = 0.0
    elif k < flight_end:
        alpha_f = (k - stance_end + 1) / N_flight
        yaw_k = alpha_f * yaw_goal
    else:
        yaw_k = yaw_goal

    quat_guess = [np.cos(yaw_k / 2), 0.0, 0.0, np.sin(yaw_k / 2)]
    opti.set_initial(X[3:7, k], quat_guess)

# # landing foot positions
opti.set_initial(p_L_land, p_L_goal)
opti.set_initial(p_R_land, p_R_goal)

# wrenches: phase-aware
for k in range(N):
    if k < stance_end or k >= flight_end:
        # contact phases: split weight evenly
        opti.set_initial(F_L[:, k], [0, 0, srb.m * srb.g / 2])
        opti.set_initial(F_R[:, k], [0, 0, srb.m * srb.g / 2])
    else:
        # flight: zero
        opti.set_initial(F_L[:, k], [0, 0, 0])
        opti.set_initial(F_R[:, k], [0, 0, 0])

# moments: zero
opti.set_initial(M_L, 0)
opti.set_initial(M_R, 0)

# ----------------------------------------------------------
# Solve the optimization
# ----------------------------------------------------------

# solve the problem
opti.solver("ipopt")
sol = opti.solve()

# ----------------------------------------------------------
# Extract solutions
# ----------------------------------------------------------

X_sol    = sol.value(X)         # (nx, N+1)
pL_land  = sol.value(p_L_land)  # (2,)
pR_land  = sol.value(p_R_land)  # (2,)
FL_sol   = sol.value(F_L)       # (3, N)
FR_sol   = sol.value(F_R)       # (3, N)
ML_sol   = sol.value(M_L)       # (3, N)
MR_sol   = sol.value(M_R)       # (3, N)

# ----------------------------------------------------------
# Reconstruct wrench trajectory
# ----------------------------------------------------------

# Forces (in world frame)
F = (FL_sol + FR_sol).T  # (N, 3)

# Moments (in world frame)
M = np.zeros((N, 3))
for k in range(N):

    if k < stance_end:
        p_com = X_sol[0:3, k]
        p_L = np.array([float(p0_L[0]), float(p0_L[1]), 0.0])
        p_R = np.array([float(p0_R[0]), float(p0_R[1]), 0.0])

    elif k < flight_end:
        M[k, :] = 0.0
        continue

    else:
        p_com = X_sol[0:3, k]
        p_L = np.array([pL_land[0], pL_land[1], 0.0])  # was p0_L — bug
        p_R = np.array([pR_land[0], pR_land[1], 0.0])

    r_L = p_L - p_com
    r_R = p_R - p_com
    M[k, :] = (
          np.cross(r_L, FL_sol[:, k])
        + np.cross(r_R, FR_sol[:, k])
        + ML_sol[:, k] + MR_sol[:, k]
    )

# Total wrench trajectory in world frame
U = np.hstack((F, M))  # (N, 6)

# ----------------------------------------------------------
# Compute accelerations
# ----------------------------------------------------------

X_sol = X_sol.T          # (N+1, nx)
q_opt = X_sol[:, 0:nq]   # (N+1, nq)
v_opt = X_sol[:, nq:nx]  # (N+1, nv)
a_opt = np.zeros_like(v_opt)

for k in range(N):
    xdot = np.array(srb.f_cont(X_sol[k, :], U[k, :])).squeeze()  # (nx,)
    a_opt[k, :] = xdot[nq:nx]
# last step: use last control (hold)
a_opt[N, :] = a_opt[N-1, :]

# ----------------------------------------------------------
# Print results
# ----------------------------------------------------------
print(f"\nOptimal landing foot positions:")
print(f"  Left  foot: x={pL_land[0]:.3f}, y={pL_land[1]:.3f}")
print(f"  Right foot: x={pR_land[0]:.3f}, y={pR_land[1]:.3f}")

print(f"\nPhase summary:")
print(f"  Stance:  k=0  -> {stance_end-1}   (t=0.00 -> {stance_end*dt:.2f}s)")
print(f"  Flight:  k={stance_end} -> {flight_end-1}  (t={stance_end*dt:.2f} -> {flight_end*dt:.2f}s)")
print(f"  Landing: k={flight_end} -> {N-1}  (t={flight_end*dt:.2f} -> {N*dt:.2f}s)")

print(f"\nTerminal state:")
print(f"  p_com = {X_sol[N, 0:3]}")
print(f"  quat  = {X_sol[N, 3:7]}")
print(f"  v_com = {X_sol[N, 7:10]}")

# ----------------------------------------------------------
# Save
# ----------------------------------------------------------
time = np.linspace(0, T, N+1)

# feet trajectory for each control step k:
# columns = [pL_x, pL_y, pR_x, pR_y]
# stance: fixed at p0_L/p0_R, flight: undefined (NaN), landing: fixed at optimized landing feet
feet = np.full((N, 4), np.nan)
feet[:stance_end, 0:2] = np.array([float(p0_L[0]), float(p0_L[1])])
feet[:stance_end, 2:4] = np.array([float(p0_R[0]), float(p0_R[1])])
feet[flight_end:, 0:2] = np.array([float(pL_land[0]), float(pL_land[1])])
feet[flight_end:, 2:4] = np.array([float(pR_land[0]), float(pR_land[1])])

save_dir = "./results/srb/srb_twist/"
os.makedirs(save_dir, exist_ok=True)

np.savetxt(save_dir + "time.csv",    time,  delimiter=",")
np.savetxt(save_dir + "q_opt.csv",   q_opt, delimiter=",")
np.savetxt(save_dir + "v_opt.csv",   v_opt, delimiter=",")
np.savetxt(save_dir + "a_opt.csv",   a_opt, delimiter=",")
np.savetxt(save_dir + "tau_opt.csv", U,     delimiter=",")
np.savetxt(save_dir + "feet.csv",    feet,  delimiter=",")

print(f"\nSaved results to {save_dir}")
