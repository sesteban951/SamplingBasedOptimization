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
from srb.srb import SRBDynamics2D


class SRB_Jump2D(SRBDynamics2D):

    def __init__(self):

        super().__init__()

        # simple quadratic penalalty on states
        self.Qx = ca.diag(ca.vertcat(
            1.0, 1.0,    # px, pz
            15.0,         # theta
            1.0, 1.0,    # vx, vz
            3.0          # w
        ))

        # foot placement cost weights
        self.Q_foot    = 1.0

        # penalize forces and moments
        self.Q_force   = 1e-4
        self.Q_moment  = 1e-4
        self.Q_force_dot    = 1e-4
        self.Q_moment_dot   = 1e-4

        # terminal weights
        self.Qx_f = 10.0 * self.Qx


    ###############################################################
    # Cost Functions
    ###############################################################

    def state_cost(self, x, x_goal):
        """Cost on state tracking only"""
        e = x - x_goal
        return 0.5 * e.T @ self.Qx @ e

    def contact_cost(self, F_L, F_R, M_L, M_R):
        """Cost on contact forces and moments"""
        cost = (
            0.5 * self.Q_force  * (ca.sumsqr(F_L) + ca.sumsqr(F_R))
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

    def foot_placement_cost(self, p_L, p_R, p_L_des, p_R_des):
        """Cost on foot placement tracking"""
        cost = (
            0.5 * self.Q_foot * ca.sumsqr(p_L - p_L_des)
          + 0.5 * self.Q_foot * ca.sumsqr(p_R - p_R_des)
        )
        return cost

    def terminal_cost(self, x, x_goal):
        """Terminal state cost."""
        e = x - x_goal
        return e.T @ self.Qx_f @ e


##############################################################
# Helpers
##############################################################

def cross2d(r, F):
    """
    2D scalar cross product (y-component of r × F).
    r = [rx, rz], F = [Fx, Fz]
    Returns: rx * Fz - rz * Fx
    """
    return r[0] * F[1] - r[1] * F[0]


##############################################################
# Trajectory Optimization
##############################################################

# create the dynamics object
srb = SRB_Jump2D()
f   = srb.f_disc
nx  = srb.nq + srb.nv
nq  = srb.nq         
nv  = srb.nv         

# fix timings
dt       = 0.02
T_stance = 1.0
T_flight = 0.5
T_land   = 0.8
T        = T_stance + T_flight + T_land

# number of steps
N_stance = int(T_stance / dt)
N_flight = int(T_flight / dt)
N_land   = int(T_land   / dt)
N        = int(T / dt)

# phase boundaries
stance_end = N_stance
flight_end = N_stance + N_flight

# ----------------------------------------------------------
# Decision Variables
# ----------------------------------------------------------

# make the optimizer
opti = ca.Opti()

# horizon variables
X         = opti.variable(nx, N + 1)  # state trajectory

# decision variables
p_L_land  = opti.variable(1)          # landing foot x position (left)
p_R_land  = opti.variable(1)          # landing foot x position (right)
F_L       = opti.variable(2, N)       # left  contact force  [Fx, Fz]
F_R       = opti.variable(2, N)       # right contact force  [Fx, Fz]
M_L       = opti.variable(1, N)       # left  ankle moment   [My]
M_R       = opti.variable(1, N)       # right ankle moment   [My]

# ----------------------------------------------------------
# Initial and Goal States
# ----------------------------------------------------------

# Initial state: standing upright
pz_nom = 0.77
x0 = np.array([
    0.0, pz_nom,  # px, pz
    0.0,          # theta
    0.0, 0.0,     # vx, vz
    0.0           # w
])
x0_ca = ca.DM(x0)

# Initial foot x positions (directly below COM)
p0_L = ca.DM([x0[0]])  # left  foot x
p0_R = ca.DM([x0[0]])  # right foot x

# Goal state: jumped forward, same height, upright, stopped
px_goal = 0.0
x_goal = np.array([
    px_goal, pz_nom,  # px, pz,
    0.0,              # theta
    0.0, 0.0,         # vx, vz, 
    0.0               # w
])
x_goal_ca = ca.DM(x_goal)

# Desired landing foot positions
p_L_goal = ca.DM([px_goal])
p_R_goal = ca.DM([px_goal])

# ----------------------------------------------------------
# Dynamics Constraints
# ----------------------------------------------------------

# set the initial state constraint
opti.subject_to(X[:, 0] == x0_ca)

# Box constraint on terminal state
epsilon = 0.005
opti.subject_to(X[:, N] >= x_goal_ca - epsilon)
opti.subject_to(X[:, N] <= x_goal_ca + epsilon)

# kinematic limits
L_max = 0.80   # [m] max leg length
L_min = 0.45   # [m] min leg length

# dynamics constraints
for k in range(N):

    # STANCE
    if k < stance_end:
        # Extract COM position
        p_com = X[0:2, k]  # [px, pz]

        # foot positions as 2D vectors [x, z]
        p_L = ca.vertcat(p0_L, 0.0)
        p_R = ca.vertcat(p0_R, 0.0)

        # moment arms from foot to COM
        r_L = p_L - p_com
        r_R = p_R - p_com

        # total force and moment (world frame)
        F_total = F_L[:, k] + F_R[:, k]
        M_total = (
            cross2d(r_L, F_L[:, k])
          + cross2d(r_R, F_R[:, k])
          + M_L[:, k] + M_R[:, k]
        )

        # constrain the leg length
        opti.subject_to(ca.sumsqr(r_L) <= L_max**2)
        opti.subject_to(ca.sumsqr(r_R) <= L_max**2)
        opti.subject_to(ca.sumsqr(r_L) >= L_min**2)
        opti.subject_to(ca.sumsqr(r_R) >= L_min**2)

    # FLIGHT
    elif (k>= stance_end) and (k < flight_end):
        F_total = ca.DM.zeros(2)
        M_total = ca.DM.zeros(1)

    # LANDING
    else:
        # Extract COM position
        p_com = X[0:2, k]

        # foot positions: x is a decision variable, z is ground
        p_L = ca.vertcat(p_L_land, 0.0)
        p_R = ca.vertcat(p_R_land, 0.0)

        # moment arms from COM to foot
        r_L = p_L - p_com
        r_R = p_R - p_com

        # total wrench from contacts (in world frame)
        F_total = F_L[:, k] + F_R[:, k]
        M_total = (
            cross2d(r_L, F_L[:, k])
          + cross2d(r_R, F_R[:, k])
          + M_L[:, k] + M_R[:, k]
        )

        # constrain the leg length
        opti.subject_to(ca.sumsqr(r_L) <= L_max**2)
        opti.subject_to(ca.sumsqr(r_R) <= L_max**2)
        opti.subject_to(ca.sumsqr(r_L) >= L_min**2)
        opti.subject_to(ca.sumsqr(r_R) >= L_min**2)

    # Combined wrench as input to SRB
    u = ca.vertcat(F_total, M_total)

    # Dynamics rollout
    x_next = f(X[:, k], u, dt)
    opti.subject_to(X[:, k + 1] == x_next)


# COM height bounds
pz_min = 0.45
# pz_max = 1.0
for k in range(N + 1):
    opti.subject_to(X[1, k] >= pz_min)
    # opti.subject_to(X[1, k] <= pz_max)

# ----------------------------------------------------------
# Contact Constraints
# ----------------------------------------------------------

# Contact parameters
mu           = 1.0
M_ankle_max  = 50.0   # [N*m]
F_leg_max    = 500.0  # [N]

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
        opti.subject_to(opti.bounded(-M_ankle_max, M_L[:, k], M_ankle_max))
        opti.subject_to(opti.bounded(-M_ankle_max, M_R[:, k], M_ankle_max))

    # FLIGHT
    elif (k>= stance_end) and (k < flight_end):
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
        opti.subject_to(opti.bounded(-M_ankle_max, M_L[:, k], M_ankle_max))
        opti.subject_to(opti.bounded(-M_ankle_max, M_R[:, k], M_ankle_max))

# Landing foot placement
landing_tol = 0.01
opti.subject_to(ca.sumsqr(p_L_land - p_L_goal) <= landing_tol**2)
opti.subject_to(ca.sumsqr(p_R_land - p_R_goal) <= landing_tol**2)


# ----------------------------------------------------------
# Objective Function
# ----------------------------------------------------------

# total cost
J = 0
for k in range(N):

    # state cost
    J += srb.state_cost(X[:, k], x_goal_ca)
    
    # contact cost
    if k < stance_end or k >= flight_end:
        J += srb.contact_cost(F_L[:, k], F_R[:, k], M_L[:, k], M_R[:, k])

    # force rate cost
    if k < N - 1:
        J += srb.force_rate_cost(F_L[:, k], F_R[:, k], M_L[:, k], M_R[:, k], 
                                 F_L[:, k+1], F_R[:, k+1], M_L[:, k+1], M_R[:, k+1], 
                                 dt)

# foot placement cost
J += srb.foot_placement_cost(p_L_land, p_R_land, p_L_goal, p_R_goal)

# terminal cost
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
    p_com_guess = (1 - alpha) * x0[:2] + alpha * x_goal[:2]

    opti.set_initial(X[0:2, k], p_com_guess) # px, pz
    opti.set_initial(X[2,   k], 0.0)         # theta
    opti.set_initial(X[3:5, k], 0.0)         # vx, vz
    opti.set_initial(X[5,   k], 0.0)         # w

# landing foot positions
opti.set_initial(p_L_land, p_L_goal)
opti.set_initial(p_R_land, p_R_goal)

# wrenches: phase aware
for k in range(N):
    if k < stance_end or k >= flight_end:
        # split weight evenly between two feet
        opti.set_initial(F_L[:, k], [0.0, srb.m * srb.g / 2])
        opti.set_initial(F_R[:, k], [0.0, srb.m * srb.g / 2])
    else:
        # flight: zero
        opti.set_initial(F_L[:, k], [0.0, 0.0])
        opti.set_initial(F_R[:, k], [0.0, 0.0])

# moments: zero
opti.set_initial(M_L, 0.0)
opti.set_initial(M_R, 0.0)

# ----------------------------------------------------------
# Solve the optimization
# ----------------------------------------------------------

opti.solver("ipopt")
sol = opti.solve()

# ----------------------------------------------------------
# Extract Solutions
# ----------------------------------------------------------

X_sol    = sol.value(X)         # (nx, N+1)
pL_land  = sol.value(p_L_land)  # scalar
pR_land  = sol.value(p_R_land)  # scalar
FL_sol   = sol.value(F_L)       # (2, N)
FR_sol   = sol.value(F_R)       # (2, N)
ML_sol   = sol.value(M_L)       # (1, N)
MR_sol   = sol.value(M_R)       # (1, N)

# ----------------------------------------------------------
# Reconstruct Wrench Trajectory
# ----------------------------------------------------------

# forces in world frame
F_traj = (FL_sol + FR_sol).T  # (N, 2)

# Moment (in world frame)
M_traj = np.zeros((N, 1))
for k in range(N):

    if k < stance_end:
        p_com  = X_sol[0:2, k]
        p_L  = np.array([float(p0_L[0]), 0.0])
        p_R  = np.array([float(p0_R[0]), 0.0])

    elif k < flight_end:
        M_traj[k] = 0.0
        continue

    else:
        p_com  = X_sol[0:2, k]
        p_L  = np.array([float(pL_land), 0.0])
        p_R  = np.array([float(pR_land), 0.0])

    r_L = p_L - p_com
    r_R = p_R - p_com
    M_traj[k] = (
        cross2d(r_L, FL_sol[:, k]) 
      + cross2d(r_R, FR_sol[:, k])
      + ML_sol[k] + MR_sol[k]
    )

# Total wrench trajectory in world frame
U = np.hstack((F_traj, M_traj))  # (N, 3): [Fx, Fz, My]

# ----------------------------------------------------------
# Compute Accelerations
# ----------------------------------------------------------

X_sol = X_sol.T              # (N+1, nx)
q_opt = X_sol[:, :nq]        # (N+1, 3)
v_opt = X_sol[:, nq:]        # (N+1, 3)
a_opt = np.zeros_like(v_opt)

for k in range(N):
    xdot     = np.array(srb.f_cont(X_sol[k, :], U[k, :])).squeeze()
    a_opt[k, :] = xdot[nq:]
a_opt[N, :] = a_opt[N - 1, :]

# ----------------------------------------------------------
# Create contact schedule signal
# ----------------------------------------------------------

contact_schedule = np.zeros((N+1, ))  # (N+1)
for k in range(N + 1):
    if k < stance_end:
        contact_schedule[k] = 1.0  # stance
    elif k < flight_end:
        contact_schedule[k] = 0.0  # flight
    else:
        contact_schedule[k] = 1.0  # landing

foot_positions = np.array([
    [float(p0_L[0]), 0.0, float(p0_R[0]), 0.0],
    [float(pL_land), 0.0, float(pR_land), 0.0],
])

# ----------------------------------------------------------
# Print Results
# ----------------------------------------------------------

print(f"\nOptimal landing foot positions:")
print(f"  Left  foot: x = {float(pL_land):.3f} m")
print(f"  Right foot: x = {float(pR_land):.3f} m")

print(f"\nPhase summary:")
print(f"  Stance:  k=0        -> {stance_end-1:<3}  (t=0.00 -> {stance_end*dt:.2f}s)")
print(f"  Flight:  k={stance_end:<3}      -> {flight_end-1:<3}  (t={stance_end*dt:.2f} -> {flight_end*dt:.2f}s)")
print(f"  Landing: k={flight_end:<3}      -> {N-1:<3}  (t={flight_end*dt:.2f} -> {N*dt:.2f}s)")

print(f"\nTerminal state:")
print(f"  p_com = [{X_sol[N, 0]:.3f}, {X_sol[N, 1]:.3f}]")
print(f"  theta = {np.degrees(X_sol[N, 2]):.2f} deg")
print(f"  v_com = [{X_sol[N, 3]:.3f}, {X_sol[N, 4]:.3f}]")
print(f"  w     = {X_sol[N, 5]:.4f} rad/s")

# ----------------------------------------------------------
# Save
# ----------------------------------------------------------

time = np.linspace(0, T, N + 1)

# feet trajectory for each control step k:
# columns = [pL_x, pL_y, pR_x, pR_y]
# stance: fixed at p0_L/p0_R, flight: undefined (NaN), landing: fixed at optimized landing feet
feet = np.full((N, 4), np.nan)
feet[:stance_end, 0:2] = np.array([float(p0_L[0]), 0.0])
feet[:stance_end, 2:4] = np.array([float(p0_R[0]), 0.0])
feet[flight_end:, 0:2] = np.array([float(pL_land), 0.0])
feet[flight_end:, 2:4] = np.array([float(pR_land), 0.0])

save_dir = "./results/srb/srb_jump_2d/"
os.makedirs(save_dir, exist_ok=True)

np.savetxt(save_dir + "time.csv",    time,    delimiter=",")
np.savetxt(save_dir + "q_opt.csv",   q_opt,   delimiter=",")
np.savetxt(save_dir + "v_opt.csv",   v_opt,   delimiter=",")
np.savetxt(save_dir + "a_opt.csv",   a_opt,   delimiter=",")
np.savetxt(save_dir + "tau_opt.csv", U,       delimiter=",")
np.savetxt(save_dir + "contact_schedule.csv", contact_schedule, delimiter=",")
np.savetxt(save_dir + "foot_positions.csv", foot_positions, delimiter=",")
np.savetxt(save_dir + "feet.csv",    feet,  delimiter=",")

print(f"\nSaved results to {save_dir}")
