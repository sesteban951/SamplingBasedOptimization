##
#
# 5-link 3D SRB for the G1 robot.
#
##

from __future__ import annotations
import casadi as ca
from srb.srb import SRBDynamics
from utils.kinematics import kin
from srb_walker.g1_5link_params import (
    HIP_OFFSET_X, HIP_OFFSET_Y, HIP_OFFSET_Z,
    L_THIGH, L_SHANK, L_FOOT, L_MAX, L_MIN,
)


class FiveLinkSRB(SRBDynamics):
    """
    SRBDynamics (3D) parameterized for the G1 as a 5-link walker.

    Physical parameters (m, I, g) are inherited from SRBDynamics unchanged.
    Adds 3D hip-position and leg-reachability helpers so the traj-opt can
    enforce accurate kinematic constraints from each hip joint.
    """

    def __init__(self):
        super().__init__()

        # lateral hip offset (±y in body frame)
        self.hip_offset_x = HIP_OFFSET_X
        self.hip_offset_y = HIP_OFFSET_Y
        self.hip_offset_z = HIP_OFFSET_Z

        self.l_thigh = L_THIGH
        self.l_shank = L_SHANK
        self.l_foot  = L_FOOT
        self.L_max   = L_MAX
        self.L_min   = L_MIN

        # cost weights
        self.Qx = ca.diag(ca.vertcat(
            1.0, 1.0, 1.0,       # px, py, pz
            15.0, 15.0, 15.0,    # quat log (x, y, z)
            1.0, 1.0, 1.0,       # vx, vy, vz
            3.0, 3.0, 3.0        # wx, wy, wz
        ))
        self.Q_foot       = 1.0
        self.Q_force      = 1e-4
        self.Q_moment     = 1e-4
        self.Q_force_dot  = 1e-4
        self.Q_moment_dot = 1e-4
        self.Qx_f = 10.0 * self.Qx


    ###############################################################
    # Leg Kinematics
    ###############################################################

    def _hip_offset_body(self, side):
        """Hip offset vector in body frame.  side: +1 = left, -1 = right."""
        return ca.DM([
            self.hip_offset_x,
            side * self.hip_offset_y,
            self.hip_offset_z,
        ])

    def hip_pos_L(self, p_com, quat):
        """Left hip joint position in world frame."""
        R = kin.quat_to_rot_matrix_ca(quat)
        return p_com + R @ self._hip_offset_body(+1)

    def hip_pos_R(self, p_com, quat):
        """Right hip joint position in world frame."""
        R = kin.quat_to_rot_matrix_ca(quat)
        return p_com + R @ self._hip_offset_body(-1)

    def leg_reach_sq_L(self, p_com, quat, p_foot):
        """Squared distance from left hip to foot contact point."""
        return ca.sumsqr(p_foot - self.hip_pos_L(p_com, quat))

    def leg_reach_sq_R(self, p_com, quat, p_foot):
        """Squared distance from right hip to foot contact point."""
        return ca.sumsqr(p_foot - self.hip_pos_R(p_com, quat))


    ###############################################################
    # Cost Functions
    ###############################################################

    def state_cost(self, x, x_goal):
        pos_err   = x[0:3]   - x_goal[0:3]
        vel_err   = x[7:10]  - x_goal[7:10]
        omega_err = x[10:13] - x_goal[10:13]
        quat_err     = kin.quat_diff_ca(x[3:7], x_goal[3:7])
        quat_err_log = kin.quat_log_ca(quat_err)
        e = ca.vertcat(pos_err, quat_err_log, vel_err, omega_err)
        return 0.5 * e.T @ self.Qx @ e

    def contact_cost(self, F_L, F_R, M_L, M_R):
        return (
            0.5 * self.Q_force  * (ca.sumsqr(F_L) + ca.sumsqr(F_R))
          + 0.5 * self.Q_moment * (ca.sumsqr(M_L) + ca.sumsqr(M_R))
        )

    def force_rate_cost(self, F_L_k, F_R_k, M_L_k, M_R_k,
                              F_L_k1, F_R_k1, M_L_k1, M_R_k1, dt):
        dF_L = (F_L_k1 - F_L_k) / dt
        dF_R = (F_R_k1 - F_R_k) / dt
        dM_L = (M_L_k1 - M_L_k) / dt
        dM_R = (M_R_k1 - M_R_k) / dt
        return (
            0.5 * self.Q_force_dot  * (ca.sumsqr(dF_L) + ca.sumsqr(dF_R))
          + 0.5 * self.Q_moment_dot * (ca.sumsqr(dM_L) + ca.sumsqr(dM_R))
        )

    def foot_placement_cost(self, p_L, p_R, p_L_des, p_R_des):
        return (
            0.5 * self.Q_foot * ca.sumsqr(p_L - p_L_des)
          + 0.5 * self.Q_foot * ca.sumsqr(p_R - p_R_des)
        )

    def terminal_cost(self, x, x_goal):
        pos_err   = x[0:3]   - x_goal[0:3]
        vel_err   = x[7:10]  - x_goal[7:10]
        omega_err = x[10:13] - x_goal[10:13]
        quat_err     = kin.quat_diff_ca(x[3:7], x_goal[3:7])
        quat_err_log = kin.quat_log_ca(quat_err)
        e = ca.vertcat(pos_err, quat_err_log, vel_err, omega_err)
        return e.T @ self.Qx_f @ e
