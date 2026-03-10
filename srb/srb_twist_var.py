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


class SRB_Twist(SRBDynamics):

    # initialize the class
    def __init__(self):

        super().__init__()

        # simple quadratic penalty on states
        self.Qx = ca.diag(ca.vertcat(
            1.0, 1.0, 1.0,  # px, py, pz
            50.0, 50.0, 50.0,  # qx, qy, qz
            1.0, 1.0, 1.0,  # vx, vy, vz
            1.0, 1.0, 1.0   # wx, wy, wz
        ))

        # foot placement cost weights
        self.Q_foot = 100.0

        # penalize forces and moments
        self.Q_force = 1e-4
        self.Q_moment = 1e-4
        self.Q_force_dot  = 1e-4
        self.Q_moment_dot = 1e-4

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
    
    def force_rate_cost(self, F_L_k, F_R_k, M_L_k, M_R_k,
                              F_L_k1, F_R_k1, M_L_k1, M_R_k1,
                              dt):
        """Cost on force and moment rate of change"""
        dF_L = (F_L_k1 - F_L_k) / dt
        dF_R = (F_R_k1 - F_R_k) / dt
        dM_L = (M_L_k1 - M_L_k) / dt
        dM_R = (M_R_k1 - M_R_k) / dt
        return (
            0.5 * self.Q_force_dot  * (ca.sumsqr(dF_L) + ca.sumsqr(dF_R))
          + 0.5 * self.Q_moment_dot * (ca.sumsqr(dM_L) + ca.sumsqr(dM_R))
        )
    
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
srb = SRB_Twist()
f = srb.f_disc
nq = srb.nq
nv = srb.nv
nx = nq + nv
nu = srb.nu

# nominal timings used to define the fixed node allocation and initialize the solver
dt_nom = 0.02          # nominal time step
T_stance_nom = 0.5     # nominal stance duration
T_flight_nom = 0.5     # nominal flight duration
T_land_nom = 0.5       # nominal landing duration
T_nom = T_stance_nom + T_flight_nom + T_land_nom  # nominal total trajectory duration

# number of steps
N_stance = int(T_stance_nom / dt_nom)
N_flight = int(T_flight_nom / dt_nom)
N_land = int(T_land_nom / dt_nom)
N = int(T_nom / dt_nom)

# phase boundaries
stance_end = N_stance
flight_end = N_stance + N_flight

# stance yaw allowance for limiting pre-rotation toward desired twist direction
yaw_allow = 0.15  # [rad]

# ----------------------------------------------------------
# Setup the optimization problem
# ----------------------------------------------------------

# make the optimizer
opti = ca.Opti()

# horizon variables
X = opti.variable(nx, N + 1)  # states over the horizon

# phase duration decision variables
T_stance = opti.variable()
T_flight = opti.variable()
T_land = opti.variable()

# phase-specific time steps
dt_stance = T_stance / N_stance
dt_flight = T_flight / N_flight
dt_land = T_land / N_land

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
x0 = np.array([0, 0, 0.77,  # p_com
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
x_goal = np.array([px_goal, 0, 0.77,           # p_com (forward, same height)
                   quat_goal[0], quat_goal[1], # quaternion (upright, yaw = +pi)
                   quat_goal[2], quat_goal[3],
                   0, 0, 0,                    # v_com (stopped)
                   0, 0, 0])                   # w_body
x_goal_ca = ca.DM(x_goal)

# Build automated yaw keyframes for flight quaternion references.
yaw_start = kin.quat_to_yaw(quat0)
quat_slerp_keyframes = kin.build_yaw_slerp_keyframes(yaw_start, yaw_goal)

# Desired landing foot positions in body-frame XY (COM-centered)
p_L_goal = ca.DM([0.0,  srb.hip_offset])
p_R_goal = ca.DM([0.0, -srb.hip_offset])

# Touchdown-to-world mapping for landing feet.
# p_L_land and p_R_land are body-frame XY coordinates and remain constant decision vars.
# Their world-frame touchdown positions are locked for the entire landing phase.
q_touchdown = X[3:7, flight_end]
yaw_touchdown = kin.quat_to_yaw_ca(q_touchdown)
c_td = ca.cos(yaw_touchdown)
s_td = ca.sin(yaw_touchdown)
Rz_touchdown = ca.vertcat(
    ca.horzcat(c_td, -s_td),
    ca.horzcat(s_td,  c_td),
)
p_com_touchdown_xy = X[0:2, flight_end]
p_L_land_xy_W = p_com_touchdown_xy + Rz_touchdown @ p_L_land
p_R_land_xy_W = p_com_touchdown_xy + Rz_touchdown @ p_R_land
p_L_land_W = ca.vertcat(p_L_land_xy_W, 0)
p_R_land_W = ca.vertcat(p_R_land_xy_W, 0)

# ----------------------------------------------------------
# Dynamics Constraints
# ----------------------------------------------------------

# set the initial condition 
opti.subject_to(X[:, 0] == x0_ca)

# bound the phase durations
T_stance_min, T_stance_max = 0.4, 1.0
T_flight_min, T_flight_max = 0.2, 1.0
T_land_min, T_land_max = 0.2, 1.0
opti.subject_to(opti.bounded(T_stance_min, T_stance, T_stance_max))
opti.subject_to(opti.bounded(T_flight_min, T_flight, T_flight_max))
opti.subject_to(opti.bounded(T_land_min, T_land, T_land_max))

# set box constraint on terminal condition
epsilon = 0.01
x_terminal_lb = x_goal_ca - epsilon
x_terminal_ub = x_goal_ca + epsilon
opti.subject_to(X[:, N] >= x_terminal_lb)
opti.subject_to(X[:, N] <= x_terminal_ub)

# kinematic limits
L_max = 0.8   # [m] max leg length
L_min = 0.45   # [m] min leg length

# compute the dynamics constraints
for k in range(N):
    if k < stance_end:
        dt_k = dt_stance
    elif k < flight_end:
        dt_k = dt_flight
    else:
        dt_k = dt_land

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

        # hard bound on pre-rotation toward desired twist direction during stance
        qk = X[3:7, k]  # [qw, qx, qy, qz]
        yaw_k = ca.atan2(
            2.0 * (qk[0] * qk[3] + qk[1] * qk[2]),
            1.0 - 2.0 * (qk[2] * qk[2] + qk[3] * qk[3])
        )
        yaw_progress = ca.sign(yaw_goal) * yaw_k
        opti.subject_to(yaw_progress <= yaw_allow)

    # FLIGHT
    elif (k>= stance_end) and (k < flight_end):
        F_total = ca.DM.zeros(3)
        M_total = ca.DM.zeros(3)

    # LANDING
    else:
        # Extract COM position
        p_com = X[0:3, k]
        
        # landing feet are locked in world after touchdown
        p_L = p_L_land_W
        p_R = p_R_land_W

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
    x_next = f(X[:, k], u, dt_k)
    opti.subject_to(X[:, k + 1] == x_next)

# add z_com constraints
pz_min = 0.45
# pz_max = 2.0
for k in range(N+1):
    opti.subject_to(X[2, k] >= pz_min)  # enforce constant height
    # opti.subject_to(X[2, k] <= pz_max)  # enforce constant height

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

# total cost
J = 0
for k in range(N):
    if k < stance_end:
        dt_k = dt_stance
        phase_k = "stance"
    elif k < flight_end:
        dt_k = dt_flight
        phase_k = "flight"
    else:
        dt_k = dt_land
        phase_k = "landing"

    # phase-aware state objective:
    # no state tracking in stance; twist in flight; track final state in landing
    if k < stance_end:
        # # Penalize only excess yaw progress toward the desired twist direction.
        # qk = X[3:7, k]  # [qw, qx, qy, qz]
        # yaw_k = ca.atan2(
        #     2.0 * (qk[0] * qk[3] + qk[1] * qk[2]),
        #     1.0 - 2.0 * (qk[2] * qk[2] + qk[3] * qk[3])
        # )
        # yaw_progress = ca.sign(yaw_goal) * yaw_k
        # yaw_excess = ca.fmax(0.0, yaw_progress - yaw_allow)
        # J += 0.5 * W_stance_yaw * yaw_excess**2
        x_ref_k = None
    elif k < flight_end:
        alpha = (k - stance_end + 1) / N_flight
        quat_k = kin.sample_piecewise_slerp(alpha, quat_slerp_keyframes)
        # Encourage in-flight yaw rate consistent with completing yaw_goal over the
        # optimized flight duration.
        x_ref_k = ca.vertcat(
            alpha * px_goal,
            0.0,
            x_goal[2],
            quat_k[0],
            quat_k[1],
            quat_k[2],
            quat_k[3],
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            yaw_goal / T_flight
        )
    else:
        x_ref_k = x_goal_ca

    # state cost only during flight + landing
    if x_ref_k is not None:
        J += dt_k * srb.state_cost(X[:, k], x_ref_k)

    # contact cost
    if phase_k in ("stance", "landing"):
        J += dt_k * srb.contact_cost(F_L[:, k], F_R[:, k], M_L[:, k], M_R[:, k])

    # force rate cost
    if k < N - 1:
        if k + 1 < stance_end:
            phase_k1 = "stance"
        elif k + 1 < flight_end:
            phase_k1 = "flight"
        else:
            phase_k1 = "landing"

        # Skip rate penalties across hybrid phase boundaries.
        if phase_k1 == phase_k:
            J += dt_k * srb.force_rate_cost(F_L[:, k], F_R[:, k], M_L[:, k], M_R[:, k], 
                                            F_L[:, k+1], F_R[:, k+1], M_L[:, k+1], M_R[:, k+1], 
                                            dt_k)

    # if k < N - 1:
    #     J += srb.force_rate_cost(F_L[:, k], F_R[:, k], M_L[:, k], M_R[:, k], 
    #                              F_L[:, k+1], F_R[:, k+1], M_L[:, k+1], M_R[:, k+1], 
    #                              dt_k)

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
        quat_guess = kin.yaw_to_quat(yaw_k)
    elif k < flight_end:
        alpha_f = (k - stance_end + 1) / N_flight
        quat_guess = kin.sample_piecewise_slerp(alpha_f, quat_slerp_keyframes)
    else:
        yaw_k = yaw_goal
        quat_guess = kin.yaw_to_quat(yaw_k)
    opti.set_initial(X[3:7, k], quat_guess)

# # landing foot positions
opti.set_initial(p_L_land, p_L_goal)
opti.set_initial(p_R_land, p_R_goal)
opti.set_initial(T_stance, T_stance_nom)
opti.set_initial(T_flight, T_flight_nom)
opti.set_initial(T_land, T_land_nom)

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
opti.solver("ipopt", {"ipopt": {"max_iter": 5000}})
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
T_stance_sol = float(sol.value(T_stance))
T_flight_sol = float(sol.value(T_flight))
T_land_sol = float(sol.value(T_land))
T_sol = T_stance_sol + T_flight_sol + T_land_sol
dt_stance_sol = T_stance_sol / N_stance
dt_flight_sol = T_flight_sol / N_flight
dt_land_sol = T_land_sol / N_land

# ----------------------------------------------------------
# Reconstruct wrench trajectory
# ----------------------------------------------------------

# Forces (in world frame)
F = (FL_sol + FR_sol).T  # (N, 3)

# Convert solved body-frame landing feet to world frame at touchdown and lock.
q_touchdown_sol = X_sol[3:7, flight_end]
yaw_touchdown_sol = kin.quat_to_yaw(q_touchdown_sol)
c_td_sol = np.cos(yaw_touchdown_sol)
s_td_sol = np.sin(yaw_touchdown_sol)
Rz_touchdown_sol = np.array([
    [c_td_sol, -s_td_sol],
    [s_td_sol,  c_td_sol],
])
p_com_touchdown_xy_sol = X_sol[0:2, flight_end]
pL_land_world_xy = p_com_touchdown_xy_sol + Rz_touchdown_sol @ pL_land
pR_land_world_xy = p_com_touchdown_xy_sol + Rz_touchdown_sol @ pR_land

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
        p_L = np.array([pL_land_world_xy[0], pL_land_world_xy[1], 0.0])
        p_R = np.array([pR_land_world_xy[0], pR_land_world_xy[1], 0.0])

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
print(f"  Left  foot (body): x={pL_land[0]:.3f}, y={pL_land[1]:.3f}")
print(f"  Right foot (body): x={pR_land[0]:.3f}, y={pR_land[1]:.3f}")
print(f"  Left  foot (world): x={pL_land_world_xy[0]:.3f}, y={pL_land_world_xy[1]:.3f}")
print(f"  Right foot (world): x={pR_land_world_xy[0]:.3f}, y={pR_land_world_xy[1]:.3f}")

print(f"\nOptimal phase durations:")
print(f"  Stance:  {T_stance_sol:.3f}s (dt={dt_stance_sol:.4f}s, nodes={N_stance})")
print(f"  Flight:  {T_flight_sol:.3f}s (dt={dt_flight_sol:.4f}s, nodes={N_flight})")
print(f"  Landing: {T_land_sol:.3f}s (dt={dt_land_sol:.4f}s, nodes={N_land})")
print(f"  Total:   {T_sol:.3f}s")

print(f"\nPhase summary:")
print(f"  Stance:  k=0  -> {stance_end-1}   (t=0.00 -> {T_stance_sol:.2f}s)")
print(f"  Flight:  k={stance_end} -> {flight_end-1}  (t={T_stance_sol:.2f} -> {T_stance_sol + T_flight_sol:.2f}s)")
print(f"  Landing: k={flight_end} -> {N-1}  (t={T_stance_sol + T_flight_sol:.2f} -> {T_sol:.2f}s)")

print(f"\nTerminal state:")
print(f"  p_com = {X_sol[N, 0:3]}")
print(f"  quat  = {X_sol[N, 3:7]}")
print(f"  v_com = {X_sol[N, 7:10]}")

# ----------------------------------------------------------
# Save
# ----------------------------------------------------------
dt_schedule = np.concatenate((
    np.full(N_stance, dt_stance_sol),
    np.full(N_flight, dt_flight_sol),
    np.full(N_land, dt_land_sol),
))
time = np.zeros(N + 1)
time[1:] = np.cumsum(dt_schedule)

# feet trajectory for each control step k:
# columns = [pL_x, pL_y, pR_x, pR_y]
# stance: fixed at p0_L/p0_R, flight: undefined (NaN), landing: fixed at optimized landing feet
feet = np.full((N, 4), np.nan)
feet[:stance_end, 0:2] = np.array([float(p0_L[0]), float(p0_L[1])])
feet[:stance_end, 2:4] = np.array([float(p0_R[0]), float(p0_R[1])])
feet[flight_end:, 0:2] = np.array([float(pL_land_world_xy[0]), float(pL_land_world_xy[1])])
feet[flight_end:, 2:4] = np.array([float(pR_land_world_xy[0]), float(pR_land_world_xy[1])])

save_dir = "./results/srb/srb_twist/"
os.makedirs(save_dir, exist_ok=True)

np.savetxt(save_dir + "time.csv",    time,  delimiter=",")
np.savetxt(save_dir + "q_opt.csv",   q_opt, delimiter=",")
np.savetxt(save_dir + "v_opt.csv",   v_opt, delimiter=",")
np.savetxt(save_dir + "a_opt.csv",   a_opt, delimiter=",")
np.savetxt(save_dir + "tau_opt.csv", U,     delimiter=",")
np.savetxt(save_dir + "feet.csv",    feet,  delimiter=",")

print(f"\nSaved results to {save_dir}")
