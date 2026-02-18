##
#
# Single Rigid Body base class.
#
##

# for base class
from __future__ import annotations
from abc import ABC

# casadi import
import casadi as ca

# custom imports
from utils.kinematics import kin


##############################################################
# Single Rigid Body (SRB 3D) for Nonlinear Programming
##############################################################

class SRBDynamics(ABC):

    # state indices
    IDX_P = slice(0, 3)
    IDX_Q = slice(3, 7)
    IDX_V = slice(7, 10)
    IDX_W = slice(10, 13)
    IDX_PX = 0
    IDX_PY = 1
    IDX_PZ = 2
    IDX_QW = 3
    IDX_QX = 4
    IDX_QY = 5
    IDX_QZ = 6
    IDX_VX = 7
    IDX_VY = 8
    IDX_VZ = 9
    IDX_WX = 10
    IDX_WY = 11
    IDX_WZ = 12

    # force indices
    IDX_F = slice(0, 3)
    IDX_M = slice(3, 6)
    IDX_FX = 0
    IDX_FY = 1
    IDX_FZ = 2
    IDX_MX = 3
    IDX_MY = 4
    IDX_MZ = 5

    # initialize the class
    def __init__(self):
        
        # state dimension
        self.nq = 7  # [p_com, quat]
        self.nv = 6  # [v_com, w_body]

        # input dimension
        self.nu = 6   # [F, M]

        # system parameters (pulled from pinocchio + 29dof urdf)
        # nominal configuration is arms down, standing straight
        self.pz_com = 0.693  # center of mass height in world frame
        self.m = 33.34      # mass [kg]
        self.g = 9.81       # gravity [m/s^2]
        self.I = ca.vertcat(
            ca.horzcat(3.747533,  0.000051,  0.086972),
            ca.horzcat(0.000051,  3.300958, -0.000894),
            ca.horzcat(0.086972, -0.000894,  0.516523),
        ) # inertia matrix about COM [kg*m^2]
        # self.I = ca.vertcat(
        #     ca.horzcat(4.061963,  0.000039,  0.101482),
        #     ca.horzcat(0.000039,  3.616058, -0.000628),
        #     ca.horzcat(0.101482, -0.000628,  0.517193),
        # ) # inertia matrix about base frame [kg*m^2]

        # nominal G1 offset from base to foot
        self.hip_offset = 0.1185


    ###############################################################
    # Dynamics
    ###############################################################

    def f_cont(self, x, u):
        """
        Continuous time dynamics for the single rigid body (SRB) model.
        https://arxiv.org/pdf/2207.04163
        
        Args:
            x: state vector [p_com, quat, v_com, w_body]
                p_com : position of COM in world frame
                quat  : orientation of body as quaternion q_BW (body in world), [w,x,y,z]
                v_com : linear velocity of COM in world frame
                w_body: angular velocity of body in body frame
            u: input vector [F, M]
                F : net external force applied to body in world frame
                M : net external moment applied to body in world frame
        Returns:
            x_dot: time derivative of state vector
        """
        
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
        quat_next = quat_next / ca.norm_2(quat_next)
        
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


##############################################################
# Single Rigid Body (SRB 2D) for Nonlinear Programming
##############################################################

class SRBDynamics2D(ABC):

    # state indices
    IDX_P  = slice(0, 2)
    IDX_PX = 0
    IDX_PZ = 1
    IDX_Q  = 2

    # force indices
    IDX_VX = 3
    IDX_VZ = 4
    IDX_WY = 5

    # input indices
    IDX_F = slice(0, 2)
    IDX_FX = 0
    IDX_FZ = 1
    IDX_MY = 2

    def __init__(self):

        # state/input dimensions
        self.nq = 3   # [x, z, theta]
        self.nv = 3   # [vx, vz, w]
        self.nu = 3   # [Fx, Fz, My]

        # system parameters (same robot, planar sagittal slice)
        self.m  = 33.34   # mass [kg]
        self.g  = 9.81    # gravity [m/s^2]
        self.I = 3.300958  # planar inertia about COM y-axis [kg*m^2]
        # self.I = 3.616058  # planar inertia about base frame y-axis [kg*m^2]


    ###############################################################
    # Dynamics
    ###############################################################

    def f_cont(self, x, u):
        """
        Continuous planar SRB dynamics.

        Args:
            x: state vector [x, z, theta, vx, vz, w]
                x, z : position of COM in world frame
                theta : orientation of body in world frame
                vx, vz : linear velocity of COM in world frame
                w : angular velocity of body in world frame
            u: input vector [Fx, Fz, My]
                Fx : horizontal force applied to the body in world frame
                Fz : vertical force applied to the body in world frame
                My : moment applied to the body about the y-axis in world frame
        Returns:            
            x_dot: time derivative of state vector
        """

        # unpack state
        vx = x[self.IDX_VX]
        vz = x[self.IDX_VZ]
        w  = x[self.IDX_WY]

        # unpack inputs
        Fx = u[self.IDX_FX]
        Fz = u[self.IDX_FZ]
        My = u[self.IDX_MY]

        # translational dynamics (world frame)
        px_dot = vx
        pz_dot = vz
        vx_dot = Fx / self.m
        vz_dot = Fz / self.m - self.g

        # rotational dynamics
        theta_dot = w
        w_dot     = My / self.I

        x_dot = ca.vertcat(
            px_dot,
            pz_dot,
            theta_dot,
            vx_dot,
            vz_dot,
            w_dot,
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
        # x_next = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        return x_next

    def friction_cone_matrix(self, mu):
        """
        2D friction cone for a single contact force [Fx, Fz].
        Constraints: A @ f <= b
           Fx <=  mu * Fz
          -Fx <=  mu * Fz
          -Fz <= 0  (no pull)
        """
        A = ca.vertcat(
            ca.horzcat( 1, -mu),
            ca.horzcat(-1, -mu),
            ca.horzcat( 0,  -1),
        )
        b = ca.DM.zeros(3, 1)

        return A, b
