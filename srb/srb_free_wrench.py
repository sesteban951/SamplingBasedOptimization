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
            10.0, 10.0, 10.0,  # px, py, pz
            10.0, 10.0, 10.0,  # qx, qy, qz
            1.0, 1.0, 1.0,     # vx, vy, vz
            1.0, 1.0, 1.0      # wx, wy, wz
        ))

        # penalize forces and moments
        self.Qu = ca.diag(ca.vertcat(
            0.1, 0.1, 0.1, # fx, fy, fz
            0.1, 0.1, 0.1  # mx, my, mz
        ))

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
        # R_BW = self._quat_to_rotmat(quat)
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
    
    ###############################################################
    # Cost Functions
    ###############################################################

    # running cost
    def running_cost(self, x, u, x_goal):

        # compute errors
        pos_err = x[0:3] - x_goal[0:3]
        vel_err = x[7:10] - x_goal[7:10]
        omega_err = x[10:13] - x_goal[10:13]

        quat_err = kin.quat_diff_ca(x[3:7], x_goal[3:7])  # like err = x_goal - x
        quat_err_log = kin.quat_log_ca(quat_err)

        # state error vector
        e_x = ca.vertcat(
            pos_err,
            quat_err_log,
            vel_err,
            omega_err
        )

        # state cost
        cost_state = 0.5 * e_x.T @ self.Qx @ e_x

        # compute errors
        F = u[0:3]
        M = u[3:6]
        
        # input error vector
        e_u = ca.vertcat(F, M)

        # input cost
        cost_input = 0.5 * e_u.T @ self.Qu @ e_u

        # total cost
        cost_tot = cost_input + cost_state

        return cost_tot
    
    # terminal cost
    def terminal_cost(self, x, x_goal):

        # compute errors (same as running_cost)
        pos_err   = x[0:3]   - x_goal[0:3]
        vel_err   = x[7:10]  - x_goal[7:10]
        omega_err = x[10:13] - x_goal[10:13]

        quat_err = kin.quat_diff_ca(x[3:7], x_goal[3:7])  # like err = x_goal - x
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

# optimization settings
dt = 0.04        # time step
T = 2.0          # total time
N = int(T / dt)  # number of intervals

# ----------------------------------------------------------
# Setup the optimization problem
# ----------------------------------------------------------

# make the optimizer
opti = ca.Opti()

# horizon variables
X = opti.variable(nx, N + 1)  # states over the horizon
U = opti.variable(nu, N)      # inputs over the horizon

# initial condition
x0 = np.array([0, 0, 0.8,  # p_com
               1, 0, 0, 0, # quaternion
               0, 0, 0,    # v_com
               0, 0, 0])   # w_body
x0_ca = ca.DM(x0)

# desired goal state
pitch_goal = np.deg2rad(-270.0) 
x_goal = np.array([0.0, 0, 0.8, # p_com
                   np.cos(pitch_goal/2), 0, np.sin(pitch_goal/2), 0, # quaternion
                   0, 0, 0,     # v_com
                   0, 0, 0])    # w_body
x_goal_ca = ca.DM(x_goal)

# set the initial condition 
opti.subject_to(X[:, 0] == x0_ca)

# system dynamics constraints at each time step
for k in range(N):
    x_next = f(X[:, k], U[:, k], dt)
    opti.subject_to(X[:, k + 1] == x_next)

# state constraints
z_min = 0.2
for k in range(N+1):
    opti.subject_to(X[2, k] >= z_min)  # z com min height

# force limits
F_max = 750.0 # [N]
for k in range(N):
    opti.subject_to(opti.bounded(-F_max, U[IDX_F, k], F_max))

# moment limits
M_max = 500.0 # [N*m]
for k in range(N):
    opti.subject_to(opti.bounded(-M_max, U[IDX_M, k], M_max))

# objective function 
J = 0
for k in range(N):
    J += srb.running_cost(X[:, k], U[:, k], x_goal)

# set the terminal constraint or cost
# J += srb.terminal_cost(X[:, N], x_goal)
opti.subject_to(X[:, N] == x_goal_ca)

# set the objective
opti.minimize(J)

# initial guesses
opti.set_initial(X, np.tile(x0.reshape(-1, 1), (1, N+1)))
opti.set_initial(U, 0)

# better force guess: support weight evenly
opti.set_initial(U[IDX_FZ, :], srb.m * srb.g) 

# ----------------------------------------------------------
# Solve the optimization
# ----------------------------------------------------------

# solver settings
opti.solver(
    "ipopt",
)
sol = opti.solve()
X_sol = sol.value(X)
U_sol = sol.value(U)

# ----------------------------------------------------------
# Save
# ----------------------------------------------------------

# create the time array
time = np.linspace(0, T, N+1)

# save the solution as csv
X_sol_T = X_sol.T
U_sol_T = U_sol.T
save_dir = "./results/srb_free_wrench/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
time_file =  save_dir + "times.csv"
state_file = save_dir + "states.csv"
input_file = save_dir + "inputs.csv"
np.savetxt(time_file, time, delimiter=",")
np.savetxt(state_file, X_sol_T, delimiter=",")
np.savetxt(input_file, U_sol_T, delimiter=",")

print(f"Saved results to {save_dir}")
