##
#
# Generalized Aerial Maneuver Optimizer for Single Rigid Body.
# Supports arbitrary roll/pitch/yaw in-air maneuvers via config.
#
# Usage: python -m srb.srb_aerial srb.config.twist_360
#
##

# standard imports
import sys
import importlib
import numpy as np
import os

# casadi import
import casadi as ca

# custom imports
from utils.kinematics import kin
from utils.interpolation import interp
from srb.srb import SRBDynamics


class SRB_Aerial(SRBDynamics):

    def __init__(self, costs):

        super().__init__()

        # simple quadratic penalty on states
        self.Qx = ca.diag(ca.vertcat(*costs.Qx_diag))

        # foot placement cost weights
        self.Q_foot = costs.Q_foot
        self.Q_foot_world = costs.Q_foot_world

        # penalize forces and moments
        self.Q_force = costs.Q_force
        self.Q_moment = costs.Q_moment
        self.Q_force_dot = costs.Q_force_dot
        self.Q_moment_dot = costs.Q_moment_dot

        # terminal weights
        self.Qx_f = costs.Qx_terminal_scale * self.Qx

    ###############################################################
    # Cost Functions
    ###############################################################

    def state_cost(self, x, x_goal):
        pos_err = x[0:3] - x_goal[0:3]
        vel_err = x[7:10] - x_goal[7:10]
        omega_err = x[10:13] - x_goal[10:13]

        quat_err = kin.quat_diff_ca(x[3:7], x_goal[3:7])
        quat_err_log = kin.quat_log_ca(quat_err)

        e_x = ca.vertcat(pos_err, quat_err_log, vel_err, omega_err)
        cost = 0.5 * e_x.T @ self.Qx @ e_x
        return cost

    def contact_cost(self, F_L, F_R, M_L, M_R):
        cost = (
            0.5 * self.Q_force * (ca.sumsqr(F_L) + ca.sumsqr(F_R))
          + 0.5 * self.Q_moment * (ca.sumsqr(M_L) + ca.sumsqr(M_R))
        )
        return cost

    def force_rate_cost(self, F_L_k, F_R_k, M_L_k, M_R_k,
                              F_L_k1, F_R_k1, M_L_k1, M_R_k1, dt):
        dF_L = (F_L_k1 - F_L_k) / dt
        dF_R = (F_R_k1 - F_R_k) / dt
        dM_L = (M_L_k1 - M_L_k) / dt
        dM_R = (M_R_k1 - M_R_k) / dt
        return (
            0.5 * self.Q_force_dot * (ca.sumsqr(dF_L) + ca.sumsqr(dF_R))
          + 0.5 * self.Q_moment_dot * (ca.sumsqr(dM_L) + ca.sumsqr(dM_R))
        )

    def foot_placement_cost(self, p_L, p_R, p_L_des, p_R_des):
        cost = (
            0.5 * self.Q_foot * ca.sumsqr(p_L - p_L_des)
          + 0.5 * self.Q_foot * ca.sumsqr(p_R - p_R_des)
        )
        return cost

    def world_foot_placement_cost(self, p_L_W, p_R_W, p_L_des_W, p_R_des_W):
        cost = (
            0.5 * self.Q_foot_world * ca.sumsqr(p_L_W - p_L_des_W)
          + 0.5 * self.Q_foot_world * ca.sumsqr(p_R_W - p_R_des_W)
        )
        return cost

    def terminal_cost(self, x, x_goal):
        pos_err = x[0:3] - x_goal[0:3]
        vel_err = x[7:10] - x_goal[7:10]
        omega_err = x[10:13] - x_goal[10:13]

        quat_err = kin.quat_diff_ca(x[3:7], x_goal[3:7])
        quat_err_log = kin.quat_log_ca(quat_err)

        e = ca.vertcat(pos_err, quat_err_log, vel_err, omega_err)
        cost_terminal = e.T @ self.Qx_f @ e
        return cost_terminal


##############################################################
# Load Config
##############################################################

if len(sys.argv) < 2:
    print("Usage: python -m srb.srb_aerial <config_module>")
    print("  e.g. python -m srb.srb_aerial srb.config.twist_360")
    sys.exit(1)

config_module = importlib.import_module(sys.argv[1])
cfg = config_module.config

##############################################################
# Build initial and goal states from config
##############################################################

# Initial state
roll0, pitch0, yaw0 = np.radians(cfg.initial.rpy_deg)
quat0 = kin.euler_ZYX_to_quat(roll0, pitch0, yaw0)
x0 = np.array([
    *cfg.initial.p_com,
    quat0[0], quat0[1], quat0[2], quat0[3],
    *cfg.initial.v_com,
    *cfg.initial.w_body,
])

# Goal state
roll_g, pitch_g, yaw_g = np.radians(cfg.goal.rpy_deg)
quat_goal = kin.euler_ZYX_to_quat(roll_g, pitch_g, yaw_g)
x_goal = np.array([
    *cfg.goal.p_com,
    quat_goal[0], quat_goal[1], quat_goal[2], quat_goal[3],
    *cfg.goal.v_com,
    *cfg.goal.w_body,
])

x0_ca = ca.DM(x0)
x_goal_ca = ca.DM(x_goal)

##############################################################
# Trajectory Optimization
##############################################################

# create the dynamics object
srb = SRB_Aerial(cfg.costs)
f = srb.f_disc
nq = srb.nq
nv = srb.nv
nx = nq + nv
nu = srb.nu

# nominal timings
dt_nom = cfg.timing.dt_nom
T_stance_nom = cfg.timing.T_stance_nom
T_flight_nom = cfg.timing.T_flight_nom
T_land_nom = cfg.timing.T_land_nom
T_nom = T_stance_nom + T_flight_nom + T_land_nom

# number of steps
N_stance = int(T_stance_nom / dt_nom)
N_flight = int(T_flight_nom / dt_nom)
N_land = int(T_land_nom / dt_nom)
N = N_stance + N_flight + N_land

# phase boundaries
stance_end = N_stance
flight_end = N_stance + N_flight

##############################################################
# SLERP keyframes from maneuver config (relative rotation)
##############################################################

roll_man, pitch_man, yaw_man = np.radians(cfg.maneuver.rpy_deg)
quat_slerp_keyframes = interp.build_general_slerp_keyframes(
    quat0, roll_man, pitch_man, yaw_man
)

##############################################################
# Precompute flight-phase angular velocity reference (body frame)
##############################################################

alpha_samples = [j / N_flight for j in range(N_flight + 1)]
quat_samples = [interp.sample_piecewise_slerp(a, quat_slerp_keyframes) for a in alpha_samples]

# Per-segment rotation in body frame (rad, not divided by time)
omega_ref_body = []
for j in range(len(quat_samples) - 1):
    q_diff = kin.quat_diff(quat_samples[j], quat_samples[j + 1])
    log_diff_world = kin.quat_log(q_diff)
    log_diff_body = kin.world_to_body(log_diff_world, quat_samples[j])
    omega_ref_body.append(log_diff_body)

##############################################################
# Desired foot positions
##############################################################

p0_L = np.array([0, srb.hip_offset])
p0_R = np.array([0, -srb.hip_offset])
p0_L_ca = ca.DM(p0_L)
p0_R_ca = ca.DM(p0_R)

p_L_goal = ca.DM([0.0, srb.hip_offset])
p_R_goal = ca.DM([0.0, -srb.hip_offset])

# World-frame foot goals from goal COM state
yaw_goal_val = kin.quat_to_yaw(quat_goal)
Rz_goal = kin.yaw_to_rot_matrix(yaw_goal_val)
p_com_goal_xy = np.array(cfg.goal.p_com[:2])
p_L_goal_W = ca.DM(p_com_goal_xy + (Rz_goal @ np.array([0.0, srb.hip_offset, 0.0]))[:2])
p_R_goal_W = ca.DM(p_com_goal_xy + (Rz_goal @ np.array([0.0, -srb.hip_offset, 0.0]))[:2])

##############################################################
# Setup the optimization problem
##############################################################

opti = ca.Opti()

# horizon variables
X = opti.variable(nx, N + 1)

# phase duration decision variables
T_stance = opti.variable()
T_flight = opti.variable()
T_land = opti.variable()

# phase-specific time steps
dt_stance = T_stance / N_stance
dt_flight = T_flight / N_flight
dt_land = T_land / N_land

# decision variables
p_L_land = opti.variable(2)
p_R_land = opti.variable(2)
F_L = opti.variable(3, N)
F_R = opti.variable(3, N)
M_L = opti.variable(3, N)
M_R = opti.variable(3, N)

##############################################################
# Touchdown-to-world mapping for landing feet (yaw-only)
##############################################################

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

##############################################################
# Constraints
##############################################################

# initial condition
opti.subject_to(X[:, 0] == x0_ca)

# phase duration bounds
T_stance_min, T_stance_max = cfg.timing.T_stance_bounds
T_flight_min, T_flight_max = cfg.timing.T_flight_bounds
T_land_min, T_land_max = cfg.timing.T_land_bounds
opti.subject_to(opti.bounded(T_stance_min, T_stance, T_stance_max))
opti.subject_to(opti.bounded(T_flight_min, T_flight, T_flight_max))
opti.subject_to(opti.bounded(T_land_min, T_land, T_land_max))

# terminal constraint (split: position/velocity box + orientation angular distance)
epsilon = cfg.constraints.terminal_epsilon
opti.subject_to(X[0:3, N] >= x_goal_ca[0:3] - epsilon)
opti.subject_to(X[0:3, N] <= x_goal_ca[0:3] + epsilon)
opti.subject_to(X[7:13, N] >= x_goal_ca[7:13] - epsilon)
opti.subject_to(X[7:13, N] <= x_goal_ca[7:13] + epsilon)

# orientation: angular-distance constraint via quat_log(quat_diff)
q_err_terminal = kin.quat_diff_ca(X[3:7, N], ca.DM(quat_goal))
log_err_terminal = kin.quat_log_ca(q_err_terminal)
opti.subject_to(ca.sumsqr(log_err_terminal) <= epsilon**2)

# kinematic limits
L_max = cfg.constraints.L_max
L_min = cfg.constraints.L_min

# contact parameters
mu = cfg.constraints.mu
M_ankle_x_max = cfg.constraints.M_ankle_x_max
M_ankle_y_max = cfg.constraints.M_ankle_y_max
M_ankle_z_max = cfg.constraints.M_ankle_z_max
F_leg_max = cfg.constraints.F_leg_max

A_friction, b_friction = srb.friction_cone_matrix(mu)

# dynamics and phase constraints
for k in range(N):
    if k < stance_end:
        dt_k = dt_stance
    elif k < flight_end:
        dt_k = dt_flight
    else:
        dt_k = dt_land

    # STANCE
    if k < stance_end:
        p_com = X[0:3, k]
        p_L = ca.vertcat(p0_L_ca, 0)
        p_R = ca.vertcat(p0_R_ca, 0)
        r_L = p_L - p_com
        r_R = p_R - p_com

        F_total = F_L[:, k] + F_R[:, k]
        M_total = (
              ca.cross(r_L, F_L[:, k])
            + ca.cross(r_R, F_R[:, k])
            + M_L[:, k] + M_R[:, k]
        )

        opti.subject_to(ca.sumsqr(r_L) <= L_max**2)
        opti.subject_to(ca.sumsqr(r_R) <= L_max**2)
        opti.subject_to(ca.sumsqr(r_L) >= L_min**2)
        opti.subject_to(ca.sumsqr(r_R) >= L_min**2)

        # stance pre-rotation constraint (isotropic angular-distance cap relative to quat0)
        q_rel = kin.quat_diff_ca(ca.DM(quat0), X[3:7, k])
        log_rel = kin.quat_log_ca(q_rel)
        opti.subject_to(ca.sumsqr(log_rel) <= cfg.constraints.stance_rotation_allow**2)

    # FLIGHT
    elif (k >= stance_end) and (k < flight_end):
        F_total = ca.DM.zeros(3)
        M_total = ca.DM.zeros(3)

    # LANDING
    else:
        p_com = X[0:3, k]
        p_L = p_L_land_W
        p_R = p_R_land_W
        r_L = p_L - p_com
        r_R = p_R - p_com

        F_total = F_L[:, k] + F_R[:, k]
        M_total = (
              ca.cross(r_L, F_L[:, k])
            + ca.cross(r_R, F_R[:, k])
            + M_L[:, k] + M_R[:, k]
        )

        opti.subject_to(ca.sumsqr(r_L) <= L_max**2)
        opti.subject_to(ca.sumsqr(r_R) <= L_max**2)
        opti.subject_to(ca.sumsqr(r_L) >= L_min**2)
        opti.subject_to(ca.sumsqr(r_R) >= L_min**2)

    # combined wrench
    u = ca.vertcat(F_total, M_total)

    # dynamics
    x_next = f(X[:, k], u, dt_k)
    opti.subject_to(X[:, k + 1] == x_next)

# z_com constraint
pz_min = cfg.constraints.pz_min
for k in range(N + 1):
    opti.subject_to(X[2, k] >= pz_min)

# contact constraints
for k in range(N):
    # STANCE
    if k < stance_end:
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

    # FLIGHT
    elif (k >= stance_end) and (k < flight_end):
        opti.subject_to(F_L[:, k] == 0)
        opti.subject_to(F_R[:, k] == 0)
        opti.subject_to(M_L[:, k] == 0)
        opti.subject_to(M_R[:, k] == 0)

    # LANDING
    else:
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

# landing foot placement constraint
landing_tol = cfg.constraints.landing_tol
opti.subject_to(ca.sumsqr(p_L_land - p_L_goal) <= landing_tol**2)
opti.subject_to(ca.sumsqr(p_R_land - p_R_goal) <= landing_tol**2)

# touchdown uprightness constraint (enforce small roll/pitch at touchdown)
q_td = X[3:7, flight_end]
roll_td = ca.atan2(2.0 * (q_td[0] * q_td[1] + q_td[2] * q_td[3]),
                   1.0 - 2.0 * (q_td[1]**2 + q_td[2]**2))
sinp = 2.0 * (q_td[0] * q_td[2] - q_td[3] * q_td[1])
pitch_td = ca.asin(ca.fmin(ca.fmax(sinp, -1.0), 1.0))
opti.subject_to(opti.bounded(-cfg.constraints.touchdown_rp_max, roll_td, cfg.constraints.touchdown_rp_max))
opti.subject_to(opti.bounded(-cfg.constraints.touchdown_rp_max, pitch_td, cfg.constraints.touchdown_rp_max))

##############################################################
# Objective Function
##############################################################

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

    # phase-aware state objective
    if k < stance_end:
        x_ref_k = None
    elif k < flight_end:
        alpha = (k - stance_end + 1) / N_flight
        quat_k = interp.sample_piecewise_slerp(alpha, quat_slerp_keyframes)

        # body-frame angular velocity reference (scaled by N_flight / T_flight)
        j = k - stance_end
        omega_ref_k = ca.DM(omega_ref_body[j]) * N_flight / T_flight

        x_ref_k = ca.vertcat(
            alpha * x_goal[0] + (1 - alpha) * x0[0],
            alpha * x_goal[1] + (1 - alpha) * x0[1],
            x_goal[2],
            quat_k[0], quat_k[1], quat_k[2], quat_k[3],
            0.0, 0.0, 0.0,
            omega_ref_k[0], omega_ref_k[1], omega_ref_k[2],
        )
    else:
        x_ref_k = x_goal_ca

    # state cost only during flight + landing
    if x_ref_k is not None:
        J += dt_k * srb.state_cost(X[:, k], x_ref_k)

    # contact cost
    if phase_k in ("stance", "landing"):
        J += dt_k * srb.contact_cost(F_L[:, k], F_R[:, k], M_L[:, k], M_R[:, k])

    # force rate cost (skip across phase boundaries)
    if k < N - 1:
        if k + 1 < stance_end:
            phase_k1 = "stance"
        elif k + 1 < flight_end:
            phase_k1 = "flight"
        else:
            phase_k1 = "landing"

        if phase_k1 == phase_k:
            J += dt_k * srb.force_rate_cost(
                F_L[:, k], F_R[:, k], M_L[:, k], M_R[:, k],
                F_L[:, k+1], F_R[:, k+1], M_L[:, k+1], M_R[:, k+1],
                dt_k)

# foot placement cost (body frame)
J += srb.foot_placement_cost(p_L_land, p_R_land, p_L_goal, p_R_goal)

# foot placement cost (world frame)
J += srb.world_foot_placement_cost(p_L_land_xy_W, p_R_land_xy_W, p_L_goal_W, p_R_goal_W)

# terminal cost
J += srb.terminal_cost(X[:, N], x_goal_ca)

opti.minimize(J)

##############################################################
# Initial Guesses
##############################################################

# state trajectory
for k in range(N + 1):
    alpha = k / N

    # com position
    p_com_guess = (1 - alpha) * x0[:3] + alpha * x_goal[:3]
    opti.set_initial(X[0:3, k], p_com_guess)

    # quaternion initial guess aligned with phase
    if k < stance_end:
        quat_guess = quat0.copy()
    elif k < flight_end:
        alpha_f = (k - stance_end + 1) / N_flight
        quat_guess = interp.sample_piecewise_slerp(alpha_f, quat_slerp_keyframes)
    else:
        quat_guess = quat_goal.copy()
    opti.set_initial(X[3:7, k], quat_guess)

# landing foot positions
opti.set_initial(p_L_land, p_L_goal)
opti.set_initial(p_R_land, p_R_goal)
opti.set_initial(T_stance, T_stance_nom)
opti.set_initial(T_flight, T_flight_nom)
opti.set_initial(T_land, T_land_nom)

# wrenches: phase-aware
for k in range(N):
    if k < stance_end or k >= flight_end:
        opti.set_initial(F_L[:, k], [0, 0, srb.m * srb.g / 2])
        opti.set_initial(F_R[:, k], [0, 0, srb.m * srb.g / 2])
    else:
        opti.set_initial(F_L[:, k], [0, 0, 0])
        opti.set_initial(F_R[:, k], [0, 0, 0])

opti.set_initial(M_L, 0)
opti.set_initial(M_R, 0)

##############################################################
# Solve
##############################################################

opti.solver("ipopt", {"ipopt": {"max_iter": cfg.solver.max_iter}})
sol = opti.solve()

##############################################################
# Extract solutions
##############################################################

X_sol = sol.value(X)
pL_land = sol.value(p_L_land)
pR_land = sol.value(p_R_land)
FL_sol = sol.value(F_L)
FR_sol = sol.value(F_R)
ML_sol = sol.value(M_L)
MR_sol = sol.value(M_R)
T_stance_sol = float(sol.value(T_stance))
T_flight_sol = float(sol.value(T_flight))
T_land_sol = float(sol.value(T_land))
T_sol = T_stance_sol + T_flight_sol + T_land_sol
dt_stance_sol = T_stance_sol / N_stance
dt_flight_sol = T_flight_sol / N_flight
dt_land_sol = T_land_sol / N_land

##############################################################
# Reconstruct wrench trajectory
##############################################################

F = (FL_sol + FR_sol).T  # (N, 3)

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

M = np.zeros((N, 3))
for k in range(N):
    if k < stance_end:
        p_com = X_sol[0:3, k]
        p_L = np.array([float(p0_L_ca[0]), float(p0_L_ca[1]), 0.0])
        p_R = np.array([float(p0_R_ca[0]), float(p0_R_ca[1]), 0.0])
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

U = np.hstack((F, M))  # (N, 6)

##############################################################
# Compute accelerations
##############################################################

X_sol = X_sol.T          # (N+1, nx)
q_opt = X_sol[:, 0:nq]
v_opt = X_sol[:, nq:nx]
a_opt = np.zeros_like(v_opt)

for k in range(N):
    xdot = np.array(srb.f_cont(X_sol[k, :], U[k, :])).squeeze()
    a_opt[k, :] = xdot[nq:nx]
a_opt[N, :] = a_opt[N-1, :]

##############################################################
# Print results
##############################################################

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

##############################################################
# Save
##############################################################

dt_schedule = np.concatenate((
    np.full(N_stance, dt_stance_sol),
    np.full(N_flight, dt_flight_sol),
    np.full(N_land, dt_land_sol),
))
time = np.zeros(N + 1)
time[1:] = np.cumsum(dt_schedule)

feet = np.full((N, 4), np.nan)
feet[:stance_end, 0:2] = np.array([float(p0_L_ca[0]), float(p0_L_ca[1])])
feet[:stance_end, 2:4] = np.array([float(p0_R_ca[0]), float(p0_R_ca[1])])
feet[flight_end:, 0:2] = np.array([float(pL_land_world_xy[0]), float(pL_land_world_xy[1])])
feet[flight_end:, 2:4] = np.array([float(pR_land_world_xy[0]), float(pR_land_world_xy[1])])

save_dir = cfg.save_dir
os.makedirs(save_dir, exist_ok=True)

np.savetxt(save_dir + "time.csv",    time,  delimiter=",")
np.savetxt(save_dir + "q_opt.csv",   q_opt, delimiter=",")
np.savetxt(save_dir + "v_opt.csv",   v_opt, delimiter=",")
np.savetxt(save_dir + "a_opt.csv",   a_opt, delimiter=",")
np.savetxt(save_dir + "tau_opt.csv", U,     delimiter=",")
np.savetxt(save_dir + "feet.csv",    feet,  delimiter=",")

print(f"\nSaved results to {save_dir}")
