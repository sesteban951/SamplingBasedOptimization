##
#
# Newton-Raphson IK for the Unitree G1 robot.
#
# Given a desired pelvis pose and desired foot poses (world frame),
# solves for leg joint angles using:
#   - SE(3) error via pin.log6  (maps rotation error onto so(3) Lie algebra)
#   - pin.integrate             (manifold-correct quaternion update on SO(3))
#   - damped least-squares      (Levenberg-Marquardt)
#
##

import os
import numpy as np
import pinocchio as pin

_REPO_ROOT    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DEFAULT_URDF = os.path.join(_REPO_ROOT, "models", "g1", "g1_29dof_rev_1_0.urdf")


class G1IK:

    L_FOOT = "left_ankle_roll_link"
    R_FOOT = "right_ankle_roll_link"

    LEG_JOINTS = [
        "left_hip_pitch_joint",   "left_hip_roll_joint",   "left_hip_yaw_joint",
        "left_knee_joint",        "left_ankle_pitch_joint", "left_ankle_roll_joint",
        "right_hip_pitch_joint",  "right_hip_roll_joint",  "right_hip_yaw_joint",
        "right_knee_joint",       "right_ankle_pitch_joint","right_ankle_roll_joint",
    ]

    # Height of ankle_roll_link above floor at all-zero joints / pelvis z=0.79
    ANKLE_HEIGHT = 0.0332
    HIP_WIDTH    = 0.11851

    def __init__(self, urdf_path: str = _DEFAULT_URDF):
        self.model = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
        self.data  = self.model.createData()

        self.l_foot_id = self.model.getFrameId(self.L_FOOT)
        self.r_foot_id = self.model.getFrameId(self.R_FOOT)

        # Velocity-vector indices and configuration indices for the 12 leg DOFs
        self._leg_vidx = []
        self._leg_qidx = []
        for name in self.LEG_JOINTS:
            jid = self.model.getJointId(name)
            self._leg_vidx.append(self.model.joints[jid].idx_v)
            self._leg_qidx.append(self.model.joints[jid].idx_q)

        # Per-DOF joint limits for the 12 leg joints (from URDF)
        lo = self.model.lowerPositionLimit
        hi = self.model.upperPositionLimit
        self._leg_lo = np.array([lo[qi] for qi in self._leg_qidx])
        self._leg_hi = np.array([hi[qi] for qi in self._leg_qidx])


    # ------------------------------------------------------------------
    # Main solver
    # ------------------------------------------------------------------

    def solve(
        self,
        q0:       np.ndarray,
        oMl_des:  pin.SE3,
        oMr_des:  pin.SE3,
        max_iter: int   = 300,
        tol:      float = 1e-6,
        alpha:    float = 0.5,
        damp:     float = 1e-4,
    ):
        """
        Newton-Raphson IK on SE(3) for both feet simultaneously.

        The base pose encoded in q0 is held fixed throughout — only the 12
        leg joint angles are updated.

        Error:      e = log6(T_curr^{-1} * T_des)   ← Lie-algebra twist (local frame)
        Jacobian:   J = getFrameJacobian(..., LOCAL)  ← consistent with local error
        Update:     q = integrate(model, q, alpha * dv)  ← correct on SO(3) manifold

        Args:
            q0:       Initial configuration (nq=36).
            oMl_des:  Desired left-foot SE3 in world frame.
            oMr_des:  Desired right-foot SE3 in world frame.
            max_iter: Maximum Newton-Raphson iterations.
            tol:      Convergence threshold on stacked error norm.
            alpha:    Line-search / step-damping factor (reduce if diverging).
            damp:     Levenberg-Marquardt regularization on JJ^T.

        Returns:
            q:       Solved configuration (nq,).
            success: True if ‖e‖ < tol at convergence.
            errs:    List of error norms, one per iteration.
        """
        q   = q0.copy()
        nv  = self.model.nv
        errs = []

        # Binary mask: 1 for leg DOFs, 0 for base + non-leg joints
        leg_mask = np.zeros(nv)
        leg_mask[self._leg_vidx] = 1.0

        for _ in range(max_iter):
            # computeJointJacobians runs FK + populates all joint Jacobians in data
            pin.computeJointJacobians(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)

            oMl = self.data.oMf[self.l_foot_id]
            oMr = self.data.oMf[self.r_foot_id]

            # SE(3) error in local foot frame: log(T_curr^{-1} * T_des)
            # pin.log6 maps the group element back to se(3), giving a 6-vector
            # [v; ω] that captures position + orientation error on the manifold.
            e_l = pin.log6(oMl.actInv(oMl_des)).vector  # (6,) [v; ω]
            e_r = pin.log6(oMr.actInv(oMr_des)).vector  # (6,)
            e   = np.concatenate([e_l, e_r])             # (12,)

            err_norm = np.linalg.norm(e)
            errs.append(err_norm)
            if err_norm < tol:
                self._warn_limits(q)
                return q, True, errs

            # LOCAL Jacobians — matches the local-frame error from log6
            J_l = pin.getFrameJacobian(self.model, self.data, self.l_foot_id, pin.LOCAL)
            J_r = pin.getFrameJacobian(self.model, self.data, self.r_foot_id, pin.LOCAL)
            J   = np.vstack([J_l, J_r])  # (12, nv)

            # Zero base + non-leg columns so they are never updated
            J_leg = J * leg_mask[np.newaxis, :]

            # Damped least-squares: dv = J^T (J J^T + λI)^{-1} e
            JJT = J_leg @ J_leg.T + damp * np.eye(12)
            dv  = J_leg.T @ np.linalg.solve(JJT, e)  # (nv,)
            dv *= leg_mask

            # Manifold-correct integration: exponential map on SO(3) for quaternion
            q = pin.integrate(self.model, q, alpha * dv)

            # Clamp leg joints to URDF limits to stay in the correct kinematic branch
            for qi, lo, hi in zip(self._leg_qidx, self._leg_lo, self._leg_hi):
                q[qi] = np.clip(q[qi], lo, hi)

        self._warn_limits(q)
        return q, False, errs

    def _warn_limits(self, q: np.ndarray):
        for name, qi, lo, hi in zip(self.LEG_JOINTS, self._leg_qidx, self._leg_lo, self._leg_hi):
            val = q[qi]
            if abs(val - lo) < 1e-3:
                print(f"[G1IK] WARNING: {name} at lower limit ({val:.4f} rad)")
            elif abs(val - hi) < 1e-3:
                print(f"[G1IK] WARNING: {name} at upper limit ({val:.4f} rad)")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def floor_targets(self, q: np.ndarray):
        """
        Default foot targets: feet flat on floor (z = ANKLE_HEIGHT),
        laterally offset by ±HIP_WIDTH along the pelvis y-axis projected onto
        the floor plane.  Accounts for yaw (and small pitch/roll) of the base.

        Returns (oMl_des, oMr_des) as pin.SE3.
        """
        pin.forwardKinematics(self.model, self.data, q)
        p_base = self.data.oMi[1].translation.copy()
        R_base = self.data.oMi[1].rotation.copy()

        # Pelvis lateral (y) axis projected to the world floor plane and normalised
        y_base  = R_base[:, 1]                 # pelvis y-axis in world frame
        y_floor = np.array([y_base[0], y_base[1], 0.0])
        norm    = np.linalg.norm(y_floor)
        if norm > 1e-6:
            y_floor /= norm
        else:
            y_floor = np.array([0.0, 1.0, 0.0])  # fallback: pure pitch/roll, no yaw

        p_l = p_base + self.HIP_WIDTH * y_floor
        p_r = p_base - self.HIP_WIDTH * y_floor
        p_l[2] = self.ANKLE_HEIGHT
        p_r[2] = self.ANKLE_HEIGHT

        R_flat = np.eye(3)
        return pin.SE3(R_flat, p_l), pin.SE3(R_flat, p_r)

    def standing_config(self, com_height: float = 0.79) -> np.ndarray:
        """
        Build a squat-biased initial guess with pelvis at com_height.
        Knee starts positive (proper squat branch) scaled by how far the pelvis
        is below the nominal 0.79 m standing height, so Newton-Raphson converges
        to the physically correct solution rather than the hyperextended branch.
        """
        NOMINAL_H = 0.79
        q = pin.neutral(self.model)
        q[2] = com_height

        # Approximate squat angles: scale with pelvis drop.
        # Minimum knee bias keeps NR on the correct branch even when pelvis is
        # above nominal (e.g. with the CoM→pelvis offset in the pipeline).
        drop = max(0.0, NOMINAL_H - com_height)
        hip_p  =  drop * 1.5
        knee   =  max(0.05, drop * 3.0)   # never zero — lower limit is -0.087 rad
        ankle_p = -drop * 1.5

        for name, angle in [
            ("left_hip_pitch_joint",   hip_p),
            ("right_hip_pitch_joint",  hip_p),
            ("left_knee_joint",        knee),
            ("right_knee_joint",       knee),
            ("left_ankle_pitch_joint", ankle_p),
            ("right_ankle_pitch_joint",ankle_p),
        ]:
            jid = self.model.getJointId(name)
            qi  = self.model.joints[jid].idx_q
            q[qi] = angle

        return q
