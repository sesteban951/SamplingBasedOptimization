##
#
# Single Rigid Body Traj Opt with continuous wrench.
#
##

# standard imports
import numpy as np
import os

# casadi import
import casadi as ca

# custom imports
from utils.kinematics import kin


##############################################################
# Single Rigid Body for Nonlinear Programming
##############################################################

class SRBDynamics:

    # initialize the class
    def __init__(self):
        
        # state dimension
        self.nq = 7  # [p_com, quat]
        self.nv = 6  # [v_com, w_body]

        # input dimension
        self.nu = 6   # [F, M]

        # system parameters (pulled from pinocchio + 29dof urdf)
        # nominal ocnfiguration is arms down, standing straight
        self.pz_com = 0.69  # center of mass height in world frame
        self.m = 33.34      # mass [kg]
        self.g = 9.81       # gravity [m/s^2]
        self.I = ca.vertcat(
            ca.horzcat(3.7475,  0.0001,  0.087),
            ca.horzcat(0.0001,  3.301 , -0.0009),
            ca.horzcat(0.087 , -0.0009,  0.5165),
        ) # body frame inertia matrix [kg*m^2]

        # simple quadratic penalalty on states
        self.Qx = ca.diag(ca.vertcat(
            1.0, 1.0, 1.0,  # px, py, pz
            1.0, 1.0, 1.0,  # qx, qy, qz
            1.0, 1.0, 1.0,     # vx, vy, vz
            1.0, 1.0, 1.0      # wx, wy, wz
        ))

        # foot placement cost weights
        self.Q_foot = 100.0

        # penalize forces and moments
        self.Q_force = 0.005
        self.Q_moment = 0.0005

        # terminal weights
        self.Qx_f = 500.0 * self.Qx

    ###############################################################
    # Dynamics
    ###############################################################

    # SRB model continuous dynamics
    # https://arxiv.org/pdf/2207.04163
    def f_cont(self, x, u):
        
        # extract the state
        p_com =  x[0:3]    # position in world frame
        quat =   x[3:7]    # orientation quaternion q_BW, body in world, [w,x,y,z]
        v_com =  x[7:10]   # linear velocity in world frame
        w_body = x[10:13]  # body frame angular velocity

        # extract the inputs (world frame wrench)
        F = u[0:3]  # force applied to the body in world frame
        M = u[3:6]  # moment applied to the body in world frame

        # rotation of body expressed in world frame
        R_BW = kin.quat_to_rot_matrix_ca(quat)

        # net force in the world frame
        F_net_W = F + ca.DM([0, 0, -self.m * self.g])

        # net moment about COM
        M_net_W = M
        M_net_B = R_BW.T @ M_net_W  # express moment in body frame

        # translational dynamics
        p_com_dot = v_com
        v_com_dot = (1.0 / self.m) * F_net_W

        # quaternion rate
        w_body_quat = ca.vertcat(0, w_body)  # augment angular velocity to quaternion form [0, wx, wy, wz]
        quat_dot = 0.5 * kin.quat_mult_ca(quat, w_body_quat)
        
        # angular dynamics
        w_body_dot = ca.solve(self.I, M_net_B - ca.cross(w_body, self.I @ w_body))

        # build the dynamics vector
        x_dot = ca.vertcat(
            p_com_dot,
            quat_dot,
            v_com_dot,
            w_body_dot
        )

        return x_dot
    
    # SRB model discrete dynamics using Euler integration
    def f_disc(self, x, u, dt):

        # Euler integration
        k1 = self.f_cont(x, u)
        x_next = x + dt * k1

        # RK2
        # k1 = self.f_cont(x, u)
        # k2 = self.f_cont(x + 0.5 * dt * k1, u)
        # x_next = x + dt * k2

        # RK4
        # k1 = self.f_cont(x, u)
        # k2 = self.f_cont(x + 0.5 * dt * k1, u)
        # k3 = self.f_cont(x + 0.5 * dt * k2, u)
        # k4 = self.f_cont(x + dt * k3, u)
        # x_next = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        
        # project back to unit quaternion manifold
        quat_next = x_next[3:7]
        quat_next = quat_next / (ca.norm_2(quat_next) + 1e-12)
        
        # rebuild state to avoid slice assignment
        x_next = ca.vertcat(
            x_next[0:3],     # p
            quat_next,       # quat
            x_next[7:10],    # v
            x_next[10:13]    # w_body
        )
        
        return x_next
    
    # friction cone matrix for single force
    def friction_cone_matrix(self, mu):

        # build the friction cone matrix
        A = ca.vertcat(
            ca.horzcat( 1,  0, -mu),
            ca.horzcat(-1,  0, -mu),
            ca.horzcat( 0,  1, -mu),
            ca.horzcat( 0, -1, -mu),
            ca.horzcat( 0,  0,  -1)
        )
        b = ca.DM.zeros(5, 1)

        return A, b
    
    ###############################################################
    # Cost Functions
    ###############################################################

    # # generate a reference trajectory
    # def make_reference(self, T, N):

    #     # intial state

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

# state input indices
IDX_P = slice(0, 3)
IDX_Q = slice(3, 7)
IDX_V = slice(7, 10)
IDX_W = slice(10, 13)

# force input indices
IDX_F = slice(0, 3)
IDX_M = slice(3, 6)

# moment input indices
IDX_FX = 0
IDX_FY = 1
IDX_FZ = 2
IDX_MX = 3
IDX_MY = 4
IDX_MZ = 5

##############################################################
# Trajectory Optimization
##############################################################

# create the dynamics object
srb = SRBDynamics()
f = srb.f_disc
nq = srb.nq
nv = srb.nv
nx = nq + nv
nu = srb.nu

# fix timings
dt = 0.04        # time step
T_stance = 0.5   # stance duration
T_flight = 0.4   # flight duration
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

# contact decision variables
p_L = opti.variable(2, N)  
p_R = opti.variable(2, N)
F_L = opti.variable(3, N)
F_R = opti.variable(3, N)
M_L = opti.variable(3, N)
M_R = opti.variable(3, N)

# initial condition
x0 = np.array([0, 0, 0.69,  # p_com
               1, 0, 0, 0, # quaternion
               0, 0, 0,    # v_com
               0, 0, 0])   # w_body
x0_ca = ca.DM(x0)
p0_L = np.array([0, 0.1185])  # initial left foot position
p0_R = np.array([0, -0.1185]) # initial right foot position
p0_L = ca.DM(p0_L)
p0_R = ca.DM(p0_R)

# desired goal state - jump forward, stay upright
x_goal = np.array([1.0, 0, 0.69,  # p_com (forward, same height)
                   1.0, 0, 0, 0,     # quaternion (upright)
                   0, 0, 0,        # v_com (stopped)
                   0, 0, 0])       # w_body
x_goal_ca = ca.DM(x_goal)

# set the initial condition 
opti.subject_to(X[:, 0] == x0_ca)

# set terminal constraint
# opti.subject_to(X[:, N] == x_goal_ca)

# compute the dynamics constraints
for k in range(N):
    # Extract COM position
    p_com = X[0:3, k]
    
    # 3D foot positions (on flat ground z=0)
    p_L_3d = ca.vertcat(p_L[:, k], 0)
    p_R_3d = ca.vertcat(p_R[:, k], 0)
    
    # Moment arms from COM to feet
    r_L = p_L_3d - p_com
    r_R = p_R_3d - p_com
    
    # Total wrench from contacts (in world frame)
    F_total = F_L[:, k] + F_R[:, k]
    M_total = ca.cross(r_L, F_L[:, k]) + ca.cross(r_R, F_R[:, k]) + M_L[:, k] + M_R[:, k]
    
    # Combined wrench as control input
    u = ca.vertcat(F_total, M_total)
    
    # Dynamics
    x_next = f(X[:, k], u, dt)
    opti.subject_to(X[:, k + 1] == x_next)

# add z_com constraints
pz_min = 0.5
pz_max = 0.8
for k in range(N+1):
    opti.subject_to(X[2, k] > pz_min)  # enforce constant height
    opti.subject_to(X[2, k] < pz_max)  # enforce constant height

# ----------------------------------------------------------
# Contact Constraints
# ----------------------------------------------------------

# Contact parameters
mu = 1.0                      # friction coefficient
hip_offset = 0.1185           # y-distance from base to each foot
M_ankle_x_max = 50.0          # [N*m]
M_ankle_y_max = 50.0          # [N*m]
M_ankle_z_max = 10.0          # [N*m]
F_max = 500.0                 # [N] max force per foot

# Get friction cone constraint matrices
A_friction, b_friction = srb.friction_cone_matrix(mu)

# STANCE PHASE (k = 0 to stance_end - 1)
for k in range(stance_end):
    # Feet fixed on ground at initial location
    opti.subject_to(p_L[:, k] == p0_L)
    opti.subject_to(p_R[:, k] == p0_R)
    
    # Friction cone constraints
    opti.subject_to(A_friction @ F_L[:, k] <= b_friction)
    opti.subject_to(A_friction @ F_R[:, k] <= b_friction)
    
    # Force magnitude limits
    opti.subject_to(ca.norm_2(F_L[:, k]) <= F_max)
    opti.subject_to(ca.norm_2(F_R[:, k]) <= F_max)
    
    # Ankle moment limits (roll, pitch, and yaw)
    opti.subject_to(opti.bounded(-M_ankle_x_max, M_L[0, k], M_ankle_x_max))
    opti.subject_to(opti.bounded(-M_ankle_y_max, M_L[1, k], M_ankle_y_max))
    opti.subject_to(opti.bounded(-M_ankle_z_max, M_L[2, k], M_ankle_z_max))
    
    opti.subject_to(opti.bounded(-M_ankle_x_max, M_R[0, k], M_ankle_x_max))
    opti.subject_to(opti.bounded(-M_ankle_y_max, M_R[1, k], M_ankle_y_max))
    opti.subject_to(opti.bounded(-M_ankle_z_max, M_R[2, k], M_ankle_z_max))

# FLIGHT PHASE (k = stance_end to flight_end - 1)
for k in range(stance_end, flight_end):
    # No contact forces or moments
    opti.subject_to(F_L[:, k] == 0)
    opti.subject_to(F_R[:, k] == 0)
    opti.subject_to(M_L[:, k] == 0)
    opti.subject_to(M_R[:, k] == 0)

# LANDING PHASE (k = flight_end to N - 1)
for k in range(flight_end, N):
    # Friction cone constraints
    opti.subject_to(A_friction @ F_L[:, k] <= b_friction)
    opti.subject_to(A_friction @ F_R[:, k] <= b_friction)
    
    # Force magnitude limits
    opti.subject_to(ca.norm_2(F_L[:, k]) <= F_max)
    opti.subject_to(ca.norm_2(F_R[:, k]) <= F_max)
    
    # Ankle moment limits (roll, pitch, and yaw)
    opti.subject_to(opti.bounded(-M_ankle_x_max, M_L[0, k], M_ankle_x_max))
    opti.subject_to(opti.bounded(-M_ankle_y_max, M_L[1, k], M_ankle_y_max))
    opti.subject_to(opti.bounded(-M_ankle_z_max, M_L[2, k], M_ankle_z_max))
    
    opti.subject_to(opti.bounded(-M_ankle_x_max, M_R[0, k], M_ankle_x_max))
    opti.subject_to(opti.bounded(-M_ankle_y_max, M_R[1, k], M_ankle_y_max))
    opti.subject_to(opti.bounded(-M_ankle_z_max, M_R[2, k], M_ankle_z_max))

# ----------------------------------------------------------
# Objective Function
# ----------------------------------------------------------

# Desired landing foot positions
p_L_des = ca.DM([x_goal[0], hip_offset])
p_R_des = ca.DM([x_goal[0], -hip_offset])

J = 0
for k in range(N):
    J += srb.state_cost(X[:, k], x_goal_ca)
    J += srb.contact_cost(F_L[:, k], F_R[:, k], M_L[:, k], M_R[:, k])
    
    # Add foot placement cost only during landing phase
    if k >= flight_end:
        J += srb.foot_placement_cost(p_L[:, k], p_R[:, k], p_L_des, p_R_des)

# Terminal cost
J += srb.terminal_cost(X[:, N], x_goal_ca)

# set the objective
opti.minimize(J)

# ----------------------------------------------------------
# Initial Guesses
# ----------------------------------------------------------

# State trajectory
for k in range(N+1):
    opti.set_initial(X[3:7, k], [1, 0, 0, 0])  # keep quaternion upright

# Foot positions - interpolate from start to goal
for k in range(N):
    if k < stance_end:
        # Stance: at initial position
        opti.set_initial(p_L[:, k], [0, hip_offset])
        opti.set_initial(p_R[:, k], [0, -hip_offset])
    elif k < flight_end:
        # Flight: interpolate
        alpha = (k - stance_end) / N_flight
        x_foot = (1 - alpha) * 0 + alpha * x_goal[0]
        opti.set_initial(p_L[:, k], [x_foot, hip_offset])
        opti.set_initial(p_R[:, k], [x_foot, -hip_offset])
    else:
        # Landing: at goal position
        opti.set_initial(p_L[:, k], [x_goal[0], hip_offset])
        opti.set_initial(p_R[:, k], [x_goal[0], -hip_offset])

# Forces: phase-aware initial guess
for k in range(N):
    if k < stance_end or k >= flight_end:
        # Contact phases: support weight
        opti.set_initial(F_L[:, k], [0, 0, srb.m * srb.g / 2])
        opti.set_initial(F_R[:, k], [0, 0, srb.m * srb.g / 2])
    else:
        # Flight phase: zero force
        opti.set_initial(F_L[:, k], [0, 0, 0])
        opti.set_initial(F_R[:, k], [0, 0, 0])

# Moments: zero initial guess
opti.set_initial(M_L, 0)
opti.set_initial(M_R, 0)

# ----------------------------------------------------------
# Solve the optimization
# ----------------------------------------------------------

# solve the problem
opti.solver("ipopt")
sol = opti.solve()

# Extract solutions
X_sol = sol.value(X)
pL_sol = sol.value(p_L)
pR_sol = sol.value(p_R)
FL_sol = sol.value(F_L)
FR_sol = sol.value(F_R)
ML_sol = sol.value(M_L)
MR_sol = sol.value(M_R)

# ----------------------------------------------------------
# Save
# ----------------------------------------------------------

# create the time array
time = np.linspace(0, T, N+1)

# save the solution as csv
save_dir = "./results/srb_jump/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
np.savetxt(save_dir + "times.csv", time, delimiter=",")
np.savetxt(save_dir + "states.csv", X_sol.T, delimiter=",")

print(f"\nSaved results to {save_dir}")
