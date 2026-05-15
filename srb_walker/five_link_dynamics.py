##
#
# Articulated 3D 5-link walker dynamics via Pinocchio + CasADi.
#
# State:  q  (nq=13)  [base_pos(3), base_quat_xyzw(4), joints(6)]
#         v  (nv=12)  [base_lin_vel(3), base_ang_vel(3), joint_vel(6)]
# Control: tau (6)    joint torques [l_roll, l_pitch, l_knee, r_roll, r_pitch, r_knee]
# Contact: lam_L, lam_R (3 each)  world-frame point forces at foot frames
#
##

import os
import numpy as np
import casadi as ca
import pinocchio as pin
import pinocchio.casadi as cpin

_URDF = os.path.join(os.path.dirname(__file__),
                     "../models/g1/g1_5link_3d.urdf")


class FiveLinkDynamics:

    def __init__(self, urdf_path=_URDF):
        self.model = pin.buildModelFromUrdf(
            urdf_path, pin.JointModelFreeFlyer()
        )
        self.data  = self.model.createData()

        self.nq  = self.model.nq   # 13
        self.nv  = self.model.nv   # 12
        self.nj  = 6               # actuated joints
        self.g   = 9.81

        self.l_foot_id = self.model.getFrameId("l_foot")
        self.r_foot_id = self.model.getFrameId("r_foot")

        # nominal: base at z=0.689 so feet sit exactly at z=0, legs straight
        # Pinocchio quaternion convention: (x, y, z, w)
        self.q0 = pin.neutral(self.model)
        self.q0[2] = 0.689

        self._build_casadi_fns()

    # ------------------------------------------------------------------
    # Build symbolic CasADi functions (called once at construction)
    # ------------------------------------------------------------------

    def _build_casadi_fns(self):
        cmodel = cpin.Model(self.model)
        cdata  = cmodel.createData()

        q      = ca.SX.sym("q",   self.nq)
        v      = ca.SX.sym("v",   self.nv)
        tau    = ca.SX.sym("tau", self.nj)
        lam_L  = ca.SX.sym("lL",  3)
        lam_R  = ca.SX.sym("lR",  3)

        # ---- foot positions (world frame) ----------------------------
        cpin.forwardKinematics(cmodel, cdata, q)
        cpin.updateFramePlacements(cmodel, cdata)

        p_L = cdata.oMf[self.l_foot_id].translation
        p_R = cdata.oMf[self.r_foot_id].translation

        self.foot_pos_fn = ca.Function(
            "foot_pos", [q], [p_L, p_R],
            ["q"], ["p_L", "p_R"]
        )

        # ---- contact Jacobians (linear part, world-aligned) ----------
        cpin.computeJointJacobians(cmodel, cdata, q)

        J6_L = cpin.getFrameJacobian(
            cmodel, cdata, self.l_foot_id,
            pin.LOCAL_WORLD_ALIGNED
        )
        J6_R = cpin.getFrameJacobian(
            cmodel, cdata, self.r_foot_id,
            pin.LOCAL_WORLD_ALIGNED
        )
        J_L = J6_L[:3, :]   # (3, nv)
        J_R = J6_R[:3, :]

        self.jac_fn = ca.Function(
            "contact_jac", [q], [J_L, J_R],
            ["q"], ["J_L", "J_R"]
        )

        # ---- forward dynamics (ABA) ----------------------------------
        # floating base is unactuated; contact forces enter via J^T * lam
        tau_gen = ca.vertcat(ca.SX.zeros(6), tau) \
                + J_L.T @ lam_L \
                + J_R.T @ lam_R

        a = cpin.aba(cmodel, cdata, q, v, tau_gen)

        self.aba_fn = ca.Function(
            "aba", [q, v, tau, lam_L, lam_R], [a],
            ["q", "v", "tau", "lam_L", "lam_R"], ["a"]
        )

        # ---- Lie-group integration: q_next = q ⊕ (v * dt) -----------
        dt_sym = ca.SX.sym("dt")
        q_next = cpin.integrate(cmodel, q, v * dt_sym)

        self.integrate_fn = ca.Function(
            "integrate", [q, v, dt_sym], [q_next],
            ["q", "v", "dt"], ["q_next"]
        )

    # ------------------------------------------------------------------
    # Convenience: foot positions for a given numpy q
    # ------------------------------------------------------------------

    def foot_positions(self, q_np):
        pin.forwardKinematics(self.model, self.data, q_np)
        pin.updateFramePlacements(self.model, self.data)
        p_L = self.data.oMf[self.l_foot_id].translation.copy()
        p_R = self.data.oMf[self.r_foot_id].translation.copy()
        return p_L, p_R

    # ------------------------------------------------------------------
    # Nominal upright config with feet flat on the ground
    # ------------------------------------------------------------------

    def neutral_standing(self, base_xy=None):
        """Return q0 with base positioned so feet are at z=0."""
        q = self.q0.copy()
        if base_xy is not None:
            q[0:2] = base_xy
        return q
