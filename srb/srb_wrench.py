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
from utils import interpolation.interp

##############################################################
# Single Rigid Body Dynamics
##############################################################

class SRBDynamics:

    # initialize the class
    def __init__(self):
        
        # state dimension
        self.nq = 7  # [p_com, quat]
        self.nv = 6  # [v_com, w_body]

        # input dimension
        self.nu = 6   # [F, M]

        # system parameters
        self.m = 33.34      # mass [kg]
        self.g = 9.81       # gravity [m/s^2]
        self.I = ca.vertcat(
            ca.horzcat( 3.0413, -0.2082, -0.5213),
            ca.horzcat(-0.2082,  3.3526,  0.097 ),
            ca.horzcat(-0.5213,  0.097 ,  1.0543),
        )

        # simple quadratic 
        w_pos = 10.0
        w_ori = 8.0
        w_vel = 2.0
        w_omega = 2.0
        self.Qx = ca.diag(ca.vertcat(
            w_pos, w_pos, w_pos,        # p_com
            w_ori, w_ori, w_ori,        # quat vector part
            w_vel, w_vel, w_vel,        # v_com
            w_omega, w_omega, w_omega   # w_body
        ))

        # penalize forces and positions
        w_force = 0.01
        w_moment = 0.01
        self.Qu = ca.diag(ca.vertcat(
            w_force, w_force, w_force,    # F
            w_moment, w_moment, w_moment  # M
        ))

        # terminal weights (usually larger than running)
        self.Qx_f = 500.0 * self.Qx

    ###############################################################
    # Dynamics
    ###############################################################

    # SRB model continuous dynamics
    # https://arxiv.org/pdf/2207.04163
    def f_cont(self, x, u):
        
        # extract the state
        p_com =  x[0:3]    # position in world frame
        quat =   x[3:7]    # orientation quaternion q_BW (body -> world), [w,x,y,z]
        v_com =  x[7:10]   # linear velocity in world frame
        w_body = x[10:13]  # body frame angular velocity

        # extract the inputs (world frame)
        F_left =  u[0:3]  # left foot force    in world frame
        F_right = u[3:6]  # right foot force   in world frame
        M_left =  u[6:9]  # left foot moment   in world frame
        M_right = u[9:12] # right foot moment  in world frame

        # rotation of body expressed in world frame
        R_BW = self._quat_to_rotmat(quat)

        # net force in the world frame
        F_net_W = F_left + F_right + ca.DM([0, 0, -self.m * self.g])

        # choose a a random point on the ground for each foot
        p_left = ca.DM([0.0,   0.1, 0.0])   # left foot position in world frame
        p_right = ca.DM([0.0, -0.1, 0.0])  # right foot position in world frame

        # net moment about COM
        M_net_W = (ca.cross(p_left - p_com, F_left) 
                 + ca.cross(p_right - p_com, F_right)
                 + M_left + M_right)
        M_net_B = R_BW.T @ M_net_W  # express moment in body frame

        # translational dynamics
        p_com_dot = v_com
        v_com_dot = (1.0 / self.m) * F_net_W

        # quaternion rate
        w_body_quat = ca.vertcat(0, w_body)  # augment angular velocity to quaternion form [0, wx, wy, wz]
        quat_dot = 0.5 * self._quat_hamilton(quat, w_body_quat)
        
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
        
        # get the continuous dynamics vector
        x_dot = self.f_cont(x, u)

        # Euler integration
        x_next = x + dt * x_dot

        # normalize the quaternion
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
    # Helper Functions
    ###############################################################

    # quaternion to rotation matrix
    def _quat_to_rotmat(self, q):

        # extract quaternion components
        qw = q[0]
        qx = q[1]
        qy = q[2]
        qz = q[3]
        
        # compute rotation matrix
        R_00 = 1 - 2*(qy*qy + qz*qz)
        R_01 = 2*(qx*qy - qz*qw)
        R_02 = 2*(qx*qz + qy*qw)
        R_10 = 2*(qx*qy + qz*qw)
        R_11 = 1 - 2*(qx*qx + qz*qz)
        R_12 = 2*(qy*qz - qx*qw)
        R_20 = 2*(qx*qz - qy*qw)
        R_21 = 2*(qy*qz + qx*qw)
        R_22 = 1 - 2*(qx*qx + qy*qy)
        R = ca.vertcat(
            ca.horzcat(R_00, R_01, R_02),
            ca.horzcat(R_10, R_11, R_12),
            ca.horzcat(R_20, R_21, R_22)
        )

        return R
    
    # rotation matrix to quaternion
    def _rotmat_to_euler_ZYX(self, R):

        R_20 = R[2, 0]
        R_20 = np.clip(R_20, -1.0, 1.0)  # numerical safety

        # compute pitch
        p = np.arcsin(-R_20)

        # check gimbal lock
        eps = 1e-8
        if abs(np.cos(p)) > eps:
            r = np.arctan2(R[2, 1], R[2, 2])
            y  = np.arctan2(R[1, 0], R[0, 0])
        else:
            # gimbal lock: yaw-roll coupling
            # set roll = 0 and compute yaw from other terms
            r = 0.0
            y  = np.arctan2(-R[0, 1], R[1, 1])

        return y, p, r

    
    # quaternion Hamilton product
    def _quat_hamilton(self, a, b):
        """
        Hamilton product c = a ⊗ b for quaternions in [w, x, y, z] convention.
        Returns a 4x1 SX vector.
        """

        # unpck the quaternions
        aw, ax, ay, az = a[0], a[1], a[2], a[3]
        bw, bx, by, bz = b[0], b[1], b[2], b[3]

        # compute the Hamilton product
        c = ca.vertcat(
            aw*bw - ax*bx - ay*by - az*bz,
            aw*bx + ax*bw + ay*bz - az*by,
            aw*by - ax*bz + ay*bw + az*bx,
            aw*bz + ax*by - ay*bx + az*bw
        )

        return c
    
    # quaternion conjugate
    def _quat_conj(self, q):
        return ca.vertcat(q[0], -q[1], -q[2], -q[3])

    # quaternion error
    def _quat_error(self, a, b):
        
        # compute the conjugate of a
        a_conj = self._quat_conj(a)

        # compute the error quaternion
        q_err = self._quat_hamilton(b, a_conj)

        return q_err
    
    # friction cone matrix for single force
    def _friction_cone_matrix(self, mu):

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

    # running cost
    def running_cost(self, x, u, x_goal):

        # compute errors
        pos_err = x[0:3] - x_goal[0:3]
        vel_err = x[7:10] - x_goal[7:10]
        omega_err = x[10:13] - x_goal[10:13]

        quat_err = self._quat_error(x[3:7], x_goal[3:7])
        quat_err = ca.if_else(quat_err[0] < 0, -quat_err, quat_err)
        quat_err_img = quat_err[1:4]

        # state error vector
        e_x = ca.vertcat(
            pos_err,
            quat_err_img,
            vel_err,
            omega_err
        )

        # state cost
        cost_state = 0.5 * e_x.T @ self.Qx @ e_x

        # compute errors
        F_left = u[0:3]
        F_right = u[3:6]
        M_left_W = u[6:9]
        M_right_W = u[9:12]

        # input error vector
        e_u = ca.vertcat(
            F_left,
            F_right,
            M_left_W,
            M_right_W
        )

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

        quat_err = self._quat_error(x[3:7], x_goal[3:7])
        quat_err = ca.if_else(quat_err[0] < 0, -quat_err, quat_err)
        quat_err_img = quat_err[1:4]
        
        # state error vector
        e = ca.vertcat(
            pos_err,
            quat_err_img,
            vel_err,
            omega_err
        )

        # terminal cost
        cost_terminal = e.T @ self.Qx_f @ e

        return cost_terminal

##############################################################

# state input indices
IDX_P      = slice(0, 3)
IDX_Q      = slice(3, 7)
IDX_V      = slice(7, 10)
IDX_W      = slice(10, 13)

# force input indices
IDX_FL_X   = 0
IDX_FL_Y   = 1
IDX_FL_Z   = 2
IDX_FR_X   = 3
IDX_FR_Y   = 4
IDX_FR_Z   = 5
IDX_FL     = slice(0, 3)
IDX_FR     = slice(3, 6)

# moment input indices
IDX_ML_X   = 6
IDX_ML_Y   = 7
IDX_ML_Z   = 8
IDX_MR_X   = 9
IDX_MR_Y   = 10
IDX_MR_Z   = 11
IDX_ML     = slice(6, 9)
IDX_MR     = slice(9, 12)

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
T = 4.0          # total time
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
x0 = np.array([0, 0, 1,    # p_com
               1, 0, 0, 0, # quaternion
               0, 0, 0,    # v_com
               0, 0, 0])   # w_body

# desired goal state
pitch_goal = np.deg2rad(-270) 
x_goal = np.array([0.0, 0, 0.5, # p_com
                   np.cos(pitch_goal/2), 0, np.sin(pitch_goal/2), 0, # quaternion
                   0, 0, 0,     # v_com
                   0, 0, 0])    # w_body

# set the initial condition 
opti.subject_to(X[:, 0] == x0)

# system dynamics constraints at each time step
for k in range(N):
    x_next = f(X[:, k], U[:, k], dt)
    opti.subject_to(X[:, k + 1] == x_next)

# state constraints
z_min = 0.2
for k in range(N+1):
    opti.subject_to(X[2, k] >= z_min)  # z com min height

# force limits
mu = 1.0
A, b = srb._friction_cone_matrix(mu)
for k in range(N):
    opti.subject_to(A @ U[IDX_FL, k] <= b)
    opti.subject_to(A @ U[IDX_FR, k] <= b)

# moment limits
m_max = 500.0  
for k in range(N):
    opti.subject_to(opti.bounded(-m_max, U[IDX_ML, k], m_max))
    opti.subject_to(opti.bounded(-m_max, U[IDX_MR, k], m_max))

# objective function 
J = 0
for k in range(N):
    J += srb.running_cost(X[:, k], U[:, k], x_goal)

# either terminal cost or final state constraint, not both
# J += srb.terminal_cost(X[:, N], x_goal)
opti.subject_to(X[:, N] == x_goal)

# set the objective
opti.minimize(J)

# initial guesses
opti.set_initial(X, np.tile(x0.reshape(-1, 1), (1, N+1)))
opti.set_initial(U, 0)

# better force guess: support weight evenly
opti.set_initial(U[2, :], 0.5 * srb.m * srb.g)  # fLz
opti.set_initial(U[5, :], 0.5 * srb.m * srb.g)  # fRz

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
save_dir = "./SRB/results/squat/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
time_file =  "./SRB/results/squat/time.csv"
state_file = "./SRB/results/squat/states.csv"
input_file = "./SRB/results/squat/inputs.csv"
np.savetxt(time_file, time, delimiter=",")
np.savetxt(state_file, X_sol_T, delimiter=",")
np.savetxt(input_file, U_sol_T, delimiter=",")
print(f"Saved time to {time_file}")
print(f"Saved states to {state_file}")
print(f"Saved inputs to {input_file}")
