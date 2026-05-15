##
#
# Standing Squat Optimizer for Single Rigid Body.
# Single contact domain -- feet remain fixed on the ground.
# COM z-reference follows a triangle profile: start -> squat depth -> goal.
#
# Usage: python -m srb.srb_squat
#
##

# standard imports
import numpy as np
import os

# casadi import
import casadi as ca

# custom imports
from utils.kinematics import kin
from utils.interpolation import interp
from srb.srb import SRBDynamics


class SRB_Squat(SRBDynamics):

    def __init__(self, Qx_diag, Q_force, Q_moment, Q_force_dot, Q_moment_dot,
                 Qx_terminal_scale):

        super().__init__()

        self.Qx = ca.diag(ca.vertcat(*Qx_diag))
        self.Q_force = Q_force
        self.Q_moment = Q_moment
        self.Q_force_dot = Q_force_dot
        self.Q_moment_dot = Q_moment_dot
        self.Qx_f = Qx_terminal_scale * self.Qx

    ###############################################################
    # Cost Functions
    ###############################################################

    def state_cost(self, x, x_ref):
        pos_err = x[0:3] - x_ref[0:3]
        vel_err = x[7:10] - x_ref[7:10]
        omega_err = x[10:13] - x_ref[10:13]

        quat_err = kin.quat_diff_ca(x[3:7], x_ref[3:7])
        quat_err_log = kin.quat_log_ca(quat_err)

        e_x = ca.vertcat(pos_err, quat_err_log, vel_err, omega_err)
        return 0.5 * e_x.T @ self.Qx @ e_x

    def contact_cost(self, F_L, F_R, M_L, M_R):
        return (
            0.5 * self.Q_force * (ca.sumsqr(F_L) + ca.sumsqr(F_R))
          + 0.5 * self.Q_moment * (ca.sumsqr(M_L) + ca.sumsqr(M_R))
        )

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

    def terminal_cost(self, x, x_goal):
        pos_err = x[0:3] - x_goal[0:3]
        vel_err = x[7:10] - x_goal[7:10]
        omega_err = x[10:13] - x_goal[10:13]

        quat_err = kin.quat_diff_ca(x[3:7], x_goal[3:7])
        quat_err_log = kin.quat_log_ca(quat_err)

        e = ca.vertcat(pos_err, quat_err_log, vel_err, omega_err)
        return e.T @ self.Qx_f @ e


##############################################################
# Parameters (self-contained, no external config)
##############################################################

# initial state
p_com_0 = [0.0, 0.0, 0.693]
rpy_deg_0 = [0.0, 0.0, 0.0]

# goal state (return to standing)
p_com_goal = [0.0, 0.0, 0.693]
rpy_deg_goal = [0.0, 0.0, 0.0]

# squat depth: minimum COM z the trajectory should reach
squat_depth_z = 0.15

# timing
dt_nom = 0.02
T_nom = 1.5
T_bounds = [0.8, 3.0]

# cost weights
#           px   py   pz   qlog_x qlog_y qlog_z  vx   vy   vz   wx   wy   wz
Qx_diag = [20,  20, 500,    50,    50,    50,    10,  10,  10,  10,  10,  10]
Qx_terminal_scale = 100.0
Q_force = 1e-5
Q_moment = 1e-5
Q_force_dot = 1e-6
Q_moment_dot = 1e-6

# constraints
L_min = 0.15
L_max = 0.80
mu = 1.0
M_ankle_x_max = 50.0
M_ankle_y_max = 50.0
M_ankle_z_max = 10.0
F_leg_max = 500.0
pz_min = 0.30
terminal_epsilon = 0.1

# solver
max_iter = 3000

# output
save_dir = "./results/srb/srb_squat/"


##############################################################
# Build initial and goal states
##############################################################

roll0, pitch0, yaw0 = np.radians(rpy_deg_0)
quat0 = kin.euler_ZYX_to_quat(roll0, pitch0, yaw0)
x0 = np.array([
    *p_com_0,
    quat0[0], quat0[1], quat0[2], quat0[3],
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
])

roll_g, pitch_g, yaw_g = np.radians(rpy_deg_goal)
quat_goal = kin.euler_ZYX_to_quat(roll_g, pitch_g, yaw_g)
x_goal = np.array([
    *p_com_goal,
    quat_goal[0], quat_goal[1], quat_goal[2], quat_goal[3],
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
])

x0_ca = ca.DM(x0)
x_goal_ca = ca.DM(x_goal)


##############################################################
# Trajectory Optimization
##############################################################

srb = SRB_Squat(Qx_diag, Q_force, Q_moment, Q_force_dot, Q_moment_dot,
                Qx_terminal_scale)
f = srb.f_disc
nq = srb.nq
nv = srb.nv
nx = nq + nv
nu = srb.nu

N = int(T_nom / dt_nom)

# foot positions (fixed throughout)
p0_L = np.array([0.0, srb.hip_offset])
p0_R = np.array([0.0, -srb.hip_offset])
p0_L_ca = ca.DM(p0_L)
p0_R_ca = ca.DM(p0_R)

A_friction, b_friction = srb.friction_cone_matrix(mu)


##############################################################
# Z-reference: triangle profile (down to squat depth, back up)
##############################################################

z_start = p_com_0[2]
z_goal = p_com_goal[2]

def z_ref(alpha):
    if alpha <= 0.5:
        t = alpha / 0.5
        return (1 - t) * z_start + t * squat_depth_z
    else:
        t = (alpha - 0.5) / 0.5
        return (1 - t) * squat_depth_z + t * z_goal


##############################################################
# Setup the optimization problem
##############################################################

opti = ca.Opti()

X = opti.variable(nx, N + 1)
T_stance = opti.variable()
dt_k = T_stance / N

F_L = opti.variable(3, N)
F_R = opti.variable(3, N)
M_L = opti.variable(3, N)
M_R = opti.variable(3, N)


##############################################################
# Constraints
##############################################################

# initial condition
opti.subject_to(X[:, 0] == x0_ca)

# phase duration bounds
opti.subject_to(opti.bounded(T_bounds[0], T_stance, T_bounds[1]))

# terminal constraint (box + quaternion distance)
opti.subject_to(X[0:3, N] >= x_goal_ca[0:3] - terminal_epsilon)
opti.subject_to(X[0:3, N] <= x_goal_ca[0:3] + terminal_epsilon)
opti.subject_to(X[7:13, N] >= x_goal_ca[7:13] - terminal_epsilon)
opti.subject_to(X[7:13, N] <= x_goal_ca[7:13] + terminal_epsilon)

q_err_terminal = kin.quat_diff_ca(X[3:7, N], ca.DM(quat_goal))
log_err_terminal = kin.quat_log_ca(q_err_terminal)
opti.subject_to(ca.sumsqr(log_err_terminal) <= terminal_epsilon**2)

# dynamics and contact constraints
for k in range(N):
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

    u = ca.vertcat(F_total, M_total)
    x_next = f(X[:, k], u, dt_k)
    opti.subject_to(X[:, k + 1] == x_next)

    # leg length bounds
    opti.subject_to(ca.sumsqr(r_L) <= L_max**2)
    opti.subject_to(ca.sumsqr(r_R) <= L_max**2)
    opti.subject_to(ca.sumsqr(r_L) >= L_min**2)
    opti.subject_to(ca.sumsqr(r_R) >= L_min**2)

    # friction cone
    opti.subject_to(A_friction @ F_L[:, k] <= b_friction)
    opti.subject_to(A_friction @ F_R[:, k] <= b_friction)

    # force limits
    opti.subject_to(ca.sumsqr(F_L[:, k]) <= F_leg_max**2)
    opti.subject_to(ca.sumsqr(F_R[:, k]) <= F_leg_max**2)

    # ankle moment limits
    opti.subject_to(opti.bounded(-M_ankle_x_max, M_L[0, k], M_ankle_x_max))
    opti.subject_to(opti.bounded(-M_ankle_y_max, M_L[1, k], M_ankle_y_max))
    opti.subject_to(opti.bounded(-M_ankle_z_max, M_L[2, k], M_ankle_z_max))
    opti.subject_to(opti.bounded(-M_ankle_x_max, M_R[0, k], M_ankle_x_max))
    opti.subject_to(opti.bounded(-M_ankle_y_max, M_R[1, k], M_ankle_y_max))
    opti.subject_to(opti.bounded(-M_ankle_z_max, M_R[2, k], M_ankle_z_max))

# COM height floor
for k in range(N + 1):
    opti.subject_to(X[2, k] >= pz_min)


##############################################################
# Objective Function
##############################################################

J = 0
for k in range(N):
    alpha = k / N

    # z reference: triangle profile
    z_k = z_ref(alpha)

    # xy reference: linear interpolation from start to goal
    xy_k = interp.lerp(np.array(p_com_0[0:2]), np.array(p_com_goal[0:2]), alpha)

    # orientation reference: slerp from initial to goal
    quat_k = interp.slerp(quat0, quat_goal, alpha)

    x_ref_k = ca.vertcat(
        xy_k[0], xy_k[1], z_k,
        quat_k[0], quat_k[1], quat_k[2], quat_k[3],
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
    )

    # state cost
    J += dt_k * srb.state_cost(X[:, k], x_ref_k)

    # contact cost
    J += dt_k * srb.contact_cost(F_L[:, k], F_R[:, k], M_L[:, k], M_R[:, k])

    # force rate cost
    if k < N - 1:
        J += dt_k * srb.force_rate_cost(
            F_L[:, k], F_R[:, k], M_L[:, k], M_R[:, k],
            F_L[:, k+1], F_R[:, k+1], M_L[:, k+1], M_R[:, k+1],
            dt_k)

# terminal cost
J += srb.terminal_cost(X[:, N], x_goal_ca)

opti.minimize(J)


##############################################################
# Initial Guesses
##############################################################

for k in range(N + 1):
    alpha = k / N

    # position: xy lerp, z follows squat reference
    p_guess = interp.lerp(np.array(p_com_0), np.array(p_com_goal), alpha)
    p_guess[2] = z_ref(alpha)
    opti.set_initial(X[0:3, k], p_guess)

    # quaternion: slerp from initial to goal
    quat_guess = interp.slerp(quat0, quat_goal, alpha)
    opti.set_initial(X[3:7, k], quat_guess)

opti.set_initial(T_stance, T_nom)

for k in range(N):
    opti.set_initial(F_L[:, k], [0, 0, srb.m * srb.g / 2])
    opti.set_initial(F_R[:, k], [0, 0, srb.m * srb.g / 2])

opti.set_initial(M_L, 0)
opti.set_initial(M_R, 0)


##############################################################
# Solve
##############################################################

opti.solver("ipopt", {"ipopt": {"max_iter": max_iter}})
sol = opti.solve()


##############################################################
# Extract solutions
##############################################################

X_sol = sol.value(X)
FL_sol = sol.value(F_L)
FR_sol = sol.value(F_R)
ML_sol = sol.value(M_L)
MR_sol = sol.value(M_R)
T_sol = float(sol.value(T_stance))
dt_sol = T_sol / N


##############################################################
# Reconstruct wrench trajectory
##############################################################

F = (FL_sol + FR_sol).T  # (N, 3)

M_wrench = np.zeros((N, 3))
for k in range(N):
    p_com = X_sol[0:3, k]
    p_L = np.array([float(p0_L_ca[0]), float(p0_L_ca[1]), 0.0])
    p_R = np.array([float(p0_R_ca[0]), float(p0_R_ca[1]), 0.0])
    r_L = p_L - p_com
    r_R = p_R - p_com
    M_wrench[k, :] = (
          np.cross(r_L, FL_sol[:, k])
        + np.cross(r_R, FR_sol[:, k])
        + ML_sol[:, k] + MR_sol[:, k]
    )

U = np.hstack((F, M_wrench))  # (N, 6)


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

print(f"\nOptimal stance duration: {T_sol:.3f}s (dt={dt_sol:.4f}s, nodes={N})")
print(f"\nCOM z range: {q_opt[:, 2].min():.3f} -> {q_opt[:, 2].max():.3f} m")
print(f"Squat depth target: {squat_depth_z:.3f} m")
print(f"Achieved min z: {q_opt[:, 2].min():.3f} m")

print(f"\nTerminal state:")
print(f"  p_com = {X_sol[N, 0:3]}")
print(f"  quat  = {X_sol[N, 3:7]}")
print(f"  v_com = {X_sol[N, 7:10]}")


##############################################################
# Save
##############################################################

time_vec = np.linspace(0, T_sol, N + 1)

feet = np.zeros((N, 4))
feet[:, 0:2] = np.array([float(p0_L_ca[0]), float(p0_L_ca[1])])
feet[:, 2:4] = np.array([float(p0_R_ca[0]), float(p0_R_ca[1])])

os.makedirs(save_dir, exist_ok=True)

np.savetxt(save_dir + "time.csv",    time_vec, delimiter=",")
np.savetxt(save_dir + "q_opt.csv",   q_opt,    delimiter=",")
np.savetxt(save_dir + "v_opt.csv",   v_opt,    delimiter=",")
np.savetxt(save_dir + "a_opt.csv",   a_opt,    delimiter=",")
np.savetxt(save_dir + "tau_opt.csv", U,        delimiter=",")
np.savetxt(save_dir + "feet.csv",    feet,     delimiter=",")

print(f"\nSaved results to {save_dir}")
