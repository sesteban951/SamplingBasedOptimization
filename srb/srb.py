##
#
# Single Rigid Body base class.
#
##

# for base class
from __future__ import annotations
from abc import ABC, abstractmethod

# casadi import
import casadi as ca

# custom imports
from utils.kinematics import kin


##############################################################
# Single Rigid Body for Nonlinear Programming
##############################################################

class SRBDynamics(ABC):

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
        ) # body frame inertia matrix [kg*m^2] (from pinocchio)
        # self.I = ca.vertcat(
        #     ca.horzcat(4.06196200e+00,  3.91260469e-05,  1.01482291e-01),
        #     ca.horzcat(3.91260469e-05,  3.61605836e+00, -6.28007265e-04),
        #     ca.horzcat(1.01482291e-01, -6.28007265e-04,  5.17192159e-01),
        # ) # body frame inertia matrix [kg*m^2] (from mujoco)

        # nominal G1 offset from base to foot
        self.hip_offset = 0.1185


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
