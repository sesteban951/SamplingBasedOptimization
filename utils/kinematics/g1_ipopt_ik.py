##
#
# IPOPT-based IK for the Unitree G1 robot.
#
# Formulation:
#   min  w_foot    * (||p_l - p_l_des||² + ||R_l - R_l_des||²_F + right foot)
#      + w_inertia * ||I_G(q) - R @ I_srb @ R^T||²_F   (commented out until enabled)
#   s.t. q_leg_lo <= q_legs <= q_leg_hi     (box)
#
# Base pose (pelvis position + orientation) is a fixed parameter each call,
# matching the NR solver strategy.  The CoM equality constraint is intentionally
# omitted: the SRB trajectory was optimised with CoM ≈ pelvis, so enforcing
# com(q) = p_srb_com forces the pelvis ~0.085 m higher than the SRB intended,
# making the legs kinematically unable to reach the floor at near-standing heights.
# The correct fix is to bake the CoM–pelvis offset into the SRB trajectory itself.
#
# Decision variables: leg joints (12)
# Two cdata objects are kept for when the inertia cost is re-enabled.
#
# The NLP is compiled once in __init__ and re-solved with different parameters,
# amortising CasADi/IPOPT compilation cost across the full trajectory.
#
# Interface is compatible with G1IK.solve():
#   q_sol, ok, errs = ik.solve(q0, oMl_des, oMr_des)
#
##

import os
import numpy as np
import casadi as ca
import pinocchio as pin
import pinocchio.casadi as cpin

_REPO_ROOT    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DEFAULT_URDF = os.path.join(_REPO_ROOT, "models", "g1", "g1_29dof_rev_1_0.urdf")

# SRB inertia tensor from models/srb/srb.xml  (fullinertia: Ixx Iyy Izz Ixy Ixz Iyz)
_SRB_INERTIA_DEFAULT = np.array([
    [3.7475,  0.0001,  0.087 ],
    [0.0001,  3.301,  -0.0009],
    [0.087,  -0.0009,  0.5165],
])


class G1IPOPTIK:

    L_FOOT = "left_ankle_roll_link"
    R_FOOT = "right_ankle_roll_link"

    LEG_JOINTS = [
        "left_hip_pitch_joint",    "left_hip_roll_joint",    "left_hip_yaw_joint",
        "left_knee_joint",         "left_ankle_pitch_joint", "left_ankle_roll_joint",
        "right_hip_pitch_joint",   "right_hip_roll_joint",   "right_hip_yaw_joint",
        "right_knee_joint",        "right_ankle_pitch_joint","right_ankle_roll_joint",
    ]

    ANKLE_HEIGHT = 0.0332
    HIP_WIDTH    = 0.11851

    def __init__(self, urdf_path: str = _DEFAULT_URDF,
                 w_foot: float     = 1.0,
                 w_inertia: float  = 0.0,
                 I_srb: np.ndarray = None,
                 ipopt_opts: dict  = None):
        """
        Args:
            I_srb: 3×3 SRB inertia tensor [kg·m²].  Defaults to the G1 SRB model values.
                   Used as I_des in the inertia cost ||I_G(q) - I_srb||²_F.
        """
        self.model = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
        self.data  = self.model.createData()

        self.cmodel    = cpin.Model(self.model)
        self.cdata_fk  = self.cmodel.createData()   # for framesForwardKinematics (feet)
        self.cdata_dyn = self.cmodel.createData()   # for ccrba (CoM + centroidal inertia)

        self.l_foot_id = self.model.getFrameId(self.L_FOOT)
        self.r_foot_id = self.model.getFrameId(self.R_FOOT)

        self._leg_qidx = []
        for name in self.LEG_JOINTS:
            jid = self.model.getJointId(name)
            self._leg_qidx.append(self.model.joints[jid].idx_q)

        lo = self.model.lowerPositionLimit
        hi = self.model.upperPositionLimit
        self._leg_lo = np.array([lo[qi] for qi in self._leg_qidx])
        self._leg_hi = np.array([hi[qi] for qi in self._leg_qidx])

        self.w_foot    = w_foot
        self.w_inertia = w_inertia

        I_srb = _SRB_INERTIA_DEFAULT if I_srb is None else np.asarray(I_srb)
        self._I_srb_flat = I_srb.flatten('F')   # column-major for ca.reshape(,3,3)

        self._solver = self._build_solver(ipopt_opts)

    # ------------------------------------------------------------------
    # NLP construction — called once, solver reused across trajectory
    # ------------------------------------------------------------------

    def _build_solver(self, ipopt_opts: dict = None) -> ca.Function:
        nq    = self.model.nq
        nv    = self.model.nv
        n_leg = len(self._leg_qidx)

        # ── Decision variables ───────────────────────────────────────────
        # pelvis xyz (3) + leg joints (12) = 15
        p_pelvis = ca.SX.sym("p_pelvis", 3)
        q_legs   = ca.SX.sym("q_legs",   n_leg)
        x        = ca.vertcat(p_pelvis, q_legs)

        # ── Parameters ───────────────────────────────────────────────────
        # quat_pelvis(4) + p_com_des(3) +
        # p_l_des(3) + R_l_flat(9) + p_r_des(3) + R_r_flat(9) + I_des_flat(9)
        # + q_prev_legs(n_leg) + w_reg(1) + w_inertia(1) + w_foot(1)
        # R_*_flat and I_des_flat use column-major (Fortran) order for ca.reshape(,3,3).
        quat_pelvis  = ca.SX.sym("quat_pelvis",  4)
        p_com_des    = ca.SX.sym("p_com_des",    3)
        p_l_des      = ca.SX.sym("p_l_des",      3)
        R_l_flat     = ca.SX.sym("R_l_flat",     9)
        p_r_des      = ca.SX.sym("p_r_des",      3)
        R_r_flat     = ca.SX.sym("R_r_flat",     9)
        I_des_flat   = ca.SX.sym("I_des_flat",   9)
        q_prev_legs  = ca.SX.sym("q_prev_legs",  n_leg)
        w_reg_sym    = ca.SX.sym("w_reg",         1)
        w_inertia_sym = ca.SX.sym("w_inertia",    1)
        w_foot_sym   = ca.SX.sym("w_foot",        1)
        params = ca.vertcat(quat_pelvis, p_com_des,
                            p_l_des, R_l_flat, p_r_des, R_r_flat, I_des_flat,
                            q_prev_legs, w_reg_sym, w_inertia_sym, w_foot_sym)

        # ── Assemble full q ───────────────────────────────────────────────
        # pelvis xyz from decision vars, orientation from parameter,
        # leg joints from decision vars, upper body stays zero.
        q_full = ca.SX.zeros(nq)
        for j in range(3):
            q_full[j]     = p_pelvis[j]
        for j in range(4):
            q_full[3 + j] = quat_pelvis[j]
        for i, qi in enumerate(self._leg_qidx):
            q_full[qi]    = q_legs[i]

        # ── Foot FK (cdata_fk) ────────────────────────────────────────────
        cpin.framesForwardKinematics(self.cmodel, self.cdata_fk, q_full)

        p_l = self.cdata_fk.oMf[self.l_foot_id].translation
        R_l = self.cdata_fk.oMf[self.l_foot_id].rotation
        p_r = self.cdata_fk.oMf[self.r_foot_id].translation
        R_r = self.cdata_fk.oMf[self.r_foot_id].rotation

        # ── CoM + centroidal inertia (cdata_dyn) ──────────────────────────
        # ccrba computes centroidal momentum matrix, CoM, and centroidal inertia I_G.
        # Velocity is zero here — I_G depends only on q, not v.
        v_zeros = ca.SX.zeros(nv)
        cpin.ccrba(self.cmodel, self.cdata_dyn, q_full, v_zeros)

        com = self.cdata_dyn.com[0]          # (3,)  CoM position
        I_G = self.cdata_dyn.Ig.inertia      # (3,3) centroidal rotational inertia (world frame)

        # ── Rotate I_srb from body frame → world frame ────────────────────
        # I_srb is defined in the SRB body frame.  I_G(q) from ccrba is in the
        # world frame.  Rotate: I_des = R_body @ I_srb @ R_body^T so the
        # comparison is frame-consistent at every pose along the trajectory.
        # R_body is derived from quat_pelvis [qx, qy, qz, qw] (pinocchio convention).
        qx = quat_pelvis[0]; qy = quat_pelvis[1]; qz = quat_pelvis[2]; qw = quat_pelvis[3]
        R_body = ca.vertcat(
            ca.horzcat(1-2*(qy**2+qz**2),   2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)),
            ca.horzcat(  2*(qx*qy+qz*qw), 1-2*(qx**2+qz**2),   2*(qy*qz-qx*qw)),
            ca.horzcat(  2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw), 1-2*(qx**2+qy**2)),
        )  # (3,3)
        I_srb_3x3 = ca.reshape(I_des_flat, 3, 3)
        I_des     = R_body @ I_srb_3x3 @ R_body.T   # rotate body-frame I_des to world frame

        # ── Cost ─────────────────────────────────────────────────────────
        R_l_des = ca.reshape(R_l_flat, 3, 3)
        R_r_des = ca.reshape(R_r_flat, 3, 3)

        e_lp = p_l - p_l_des
        e_rp = p_r - p_r_des
        e_lR = ca.reshape(R_l - R_l_des, 9, 1)
        e_rR = ca.reshape(R_r - R_r_des, 9, 1)
        cost_foot = (ca.dot(e_lp, e_lp) + ca.dot(e_rp, e_rp) +
                     ca.dot(e_lR, e_lR) + ca.dot(e_rR, e_rR))

        e_I          = ca.reshape(I_G - I_des, 9, 1)
        cost_inertia = ca.dot(e_I, e_I)

        e_reg    = q_legs - q_prev_legs
        cost_reg = ca.dot(e_reg, e_reg)
        cost = w_foot_sym * cost_foot + w_inertia_sym * cost_inertia + w_reg_sym * cost_reg

        # ── CoM equality constraint ───────────────────────────────────────
        g = com - p_com_des   # (3,) enforced == 0

        nlp = {"x": x, "f": cost, "g": g, "p": params}

        opts = {
            "ipopt.max_iter":                   300,
            "ipopt.tol":                        1e-6,
            "ipopt.acceptable_tol":             1e-4,
            "ipopt.acceptable_constr_viol_tol": 1e-4,
            "ipopt.print_level":                0,
            "print_time":                       0,
        }
        if ipopt_opts:
            opts.update(ipopt_opts)

        return ca.nlpsol("g1_ipopt_ik", "ipopt", nlp, opts)

    # ------------------------------------------------------------------
    # Main solver interface (drop-in compatible with G1IK.solve)
    # ------------------------------------------------------------------

    def solve(self, q0: np.ndarray, oMl_des: pin.SE3, oMr_des: pin.SE3,
              q_prev: np.ndarray = None, w_reg: float = 0.0,
              I_des: np.ndarray = None, w_inertia: float = None,
              w_foot: float = None):
        """
        Solve IK via IPOPT.

        Args:
            q0:        Initial configuration (nq=36).
                       q0[0:3] is used as the target CoM position (p_com_des) — this
                       matches _build_q0 which copies the SRB CoM into q0[0:3].
                       q0[3:7] is the fixed pelvis orientation (from SRB quaternion).
                       q0[7:19] seeds the leg-joint warm-start.
            oMl_des:   Desired left-foot pose (pin.SE3).
            oMr_des:   Desired right-foot pose (pin.SE3).
            q_prev:    Previous frame's full configuration (nq=36), used for joint
                       continuity regularisation.  None → uses q0.
            w_reg:     Weight on ||q_legs - q_prev_legs||² regularisation.
                       0.0 (default) disables it.
            I_des:     3×3 desired centroidal inertia in body frame [kg·m²].
                       None → uses the default SRB inertia (self._I_srb_flat).
            w_inertia: Weight on ||I_G(q) - I_des||²_F cost.
                       None → uses self.w_inertia.  Set > 0 to activate.
                       For contact phases use a small value (e.g. 1e-3).
                       For flight (w_foot=0) use a larger value (e.g. 0.5).
            w_foot:    Weight on foot placement cost.
                       None → uses self.w_foot (1.0).  Set 0.0 for flight-only
                       inertia tracking where foot targets are irrelevant.

        Returns:
            q_sol:   Full configuration (nq=36).
            success: True if IPOPT returned Solve_Succeeded or Solved_To_Acceptable_Level.
            errs:    [pos_err_m] — summed L2 foot position error at the solution.
        """
        n_leg        = len(self._leg_qidx)
        q_legs_0     = np.array([q0[qi] for qi in self._leg_qidx])
        p_com_des    = q0[0:3].copy()   # SRB CoM stored in q0[0:3] by _build_q0
        q_ref        = q_prev if q_prev is not None else q0
        q_prev_legs  = np.array([q_ref[qi] for qi in self._leg_qidx])

        I_des_flat    = I_des.flatten('F') if I_des is not None else self._I_srb_flat
        w_inertia_val = w_inertia if w_inertia is not None else self.w_inertia
        w_foot_val    = w_foot    if w_foot    is not None else self.w_foot

        p_val = np.concatenate([
            q0[3:7],                           # quat_pelvis (fixed orientation)
            p_com_des,                         # CoM equality target
            oMl_des.translation,               # p_l_des
            oMl_des.rotation.flatten('F'),     # R_l_flat (col-major)
            oMr_des.translation,               # p_r_des
            oMr_des.rotation.flatten('F'),     # R_r_flat
            I_des_flat,                        # desired inertia (body frame, col-major)
            q_prev_legs,                       # joint regularisation reference
            [w_reg],                           # regularisation weight
            [w_inertia_val],                   # inertia cost weight
            [w_foot_val],                      # foot placement cost weight
        ])

        x0  = np.concatenate([p_com_des, q_legs_0])
        # Pelvis z bounds derived from the initial guess so the solver works at any
        # height (e.g. robot standing on an elevated box).  A fixed margin of 0.3 m
        # above and below the initial pelvis z keeps the search well-bounded.
        _pz_margin = 0.3
        lbx = np.concatenate([[-np.inf, -np.inf, max(0.2, p_com_des[2] - _pz_margin)], self._leg_lo])
        ubx = np.concatenate([[ np.inf,  np.inf,               p_com_des[2] + _pz_margin], self._leg_hi])

        sol = self._solver(
            x0=x0, lbx=lbx, ubx=ubx,
            lbg=np.zeros(3), ubg=np.zeros(3),
            p=p_val,
        )

        x_sol      = np.array(sol["x"]).flatten()
        q_legs_sol = x_sol[3:]

        q_sol      = q0.copy()
        q_sol[0:3] = x_sol[0:3]     # pelvis xyz solved by IPOPT
        for i, qi in enumerate(self._leg_qidx):
            q_sol[qi] = q_legs_sol[i]

        status  = self._solver.stats()["return_status"]
        success = status in ("Solve_Succeeded", "Solved_To_Acceptable_Level")

        pin.framesForwardKinematics(self.model, self.data, q_sol)
        p_l_curr = self.data.oMf[self.l_foot_id].translation
        p_r_curr = self.data.oMf[self.r_foot_id].translation
        pos_err  = (np.linalg.norm(p_l_curr - oMl_des.translation) +
                    np.linalg.norm(p_r_curr - oMr_des.translation))

        self._warn_limits(q_sol)
        return q_sol, success, [pos_err]

    # ------------------------------------------------------------------
    # Helpers shared with G1IK
    # ------------------------------------------------------------------

    def standing_config(self, com_height: float = 0.79) -> np.ndarray:
        """Squat-biased initial guess — identical logic to G1IK.standing_config."""
        NOMINAL_H = 0.79
        q    = pin.neutral(self.model)
        q[2] = com_height
        drop    = max(0.0, NOMINAL_H - com_height)
        hip_p   =  drop * 1.5
        knee    =  max(0.05, drop * 3.0)
        ankle_p = -drop * 1.5
        for name, angle in [
            ("left_hip_pitch_joint",    hip_p),
            ("right_hip_pitch_joint",   hip_p),
            ("left_knee_joint",         knee),
            ("right_knee_joint",        knee),
            ("left_ankle_pitch_joint",  ankle_p),
            ("right_ankle_pitch_joint", ankle_p),
        ]:
            jid = self.model.getJointId(name)
            q[self.model.joints[jid].idx_q] = angle
        return q

    def _warn_limits(self, q: np.ndarray):
        lo = self.model.lowerPositionLimit
        hi = self.model.upperPositionLimit
        for name, qi in zip(self.LEG_JOINTS, self._leg_qidx):
            val = q[qi]
            if abs(val - lo[qi]) < 1e-3:
                print(f"[G1IPOPTIK] WARNING: {name} at lower limit ({val:.4f} rad)")
            elif abs(val - hi[qi]) < 1e-3:
                print(f"[G1IPOPTIK] WARNING: {name} at upper limit ({val:.4f} rad)")
