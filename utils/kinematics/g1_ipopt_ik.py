##
#
# IPOPT-based IK for the Unitree G1 robot.
#
# Decision variables: pelvis xyz (3) + leg joints (12) = 15
# Hard constraint:    com(q) = p_com_des  (3 equalities)
# Cost:
#   w_foot    * (||p_l - p_l_des||² + ||R_l - R_l_des||²_F + right foot)
#   w_inertia * ||I_G(q) - I_des||²_F
#   w_reg     * ||q_legs - q_prev_legs||²
#   w_sym     * bilateral symmetry terms
#   w_foot_vel* ||p_foot(t) - p_foot(t-1)||²  (foot-velocity, anti-slide)
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
        """
        self.model = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
        self.data  = self.model.createData()

        self.cmodel    = cpin.Model(self.model)
        self.cdata_fk  = self.cmodel.createData()   # for framesForwardKinematics (feet)
        self.cdata_dyn = self.cmodel.createData()   # for ccrba (CoM + centroidal inertia)

        self.l_foot_id = self.model.getFrameId(self.L_FOOT)
        self.r_foot_id = self.model.getFrameId(self.R_FOOT)
        self.l_knee_id = self.model.getFrameId("left_knee_link")
        self.r_knee_id = self.model.getFrameId("right_knee_link")

        self._leg_qidx = []
        self._leg_vidx = []
        for name in self.LEG_JOINTS:
            jid = self.model.getJointId(name)
            self._leg_qidx.append(self.model.joints[jid].idx_q)
            self._leg_vidx.append(self.model.joints[jid].idx_v)

        lo = self.model.lowerPositionLimit
        hi = self.model.upperPositionLimit
        self._leg_lo = np.array([lo[qi] for qi in self._leg_qidx])
        self._leg_hi = np.array([hi[qi] for qi in self._leg_qidx])
        self._leg_vel_lim = np.array([self.model.velocityLimit[vi] for vi in self._leg_vidx])

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

        # ── Parameters (total 63) ─────────────────────────────────────────
        # quat_pelvis(4) + p_com_des(3) +
        # p_l_des(3) + R_l_flat(9) + p_r_des(3) + R_r_flat(9) + I_des_flat(9) +
        # q_prev_legs(12) + w_reg(1) + w_inertia(1) + w_foot(1) + w_sym(1) +
        # p_l_prev(3) + p_r_prev(3) + w_foot_vel(1) + floor_z(1)  →  total 63
        # g layout (17 rows): com_eq(3) + lfoot_floor(1) + rfoot_floor(1)
        #                   + lfoot_pos(3) + rfoot_pos(3)
        #                   + lfoot_rot(3) + rfoot_rot(3)
        # foot_hard=True:  pins foot position and orientation to target (hard equality).
        # foot_hard=False: lbg/ubg open → foot pose constraints inactive.
        quat_pelvis   = ca.SX.sym("quat_pelvis",  4)
        p_com_des     = ca.SX.sym("p_com_des",    3)
        p_l_des       = ca.SX.sym("p_l_des",      3)
        R_l_flat      = ca.SX.sym("R_l_flat",     9)
        p_r_des       = ca.SX.sym("p_r_des",      3)
        R_r_flat      = ca.SX.sym("R_r_flat",     9)
        I_des_flat    = ca.SX.sym("I_des_flat",   9)
        q_prev_legs   = ca.SX.sym("q_prev_legs",  n_leg)
        w_reg_sym     = ca.SX.sym("w_reg",         1)
        w_inertia_sym = ca.SX.sym("w_inertia",     1)
        w_foot_sym    = ca.SX.sym("w_foot",        1)
        w_sym_sym     = ca.SX.sym("w_sym",         1)
        p_l_prev_sym  = ca.SX.sym("p_l_prev",     3)
        p_r_prev_sym  = ca.SX.sym("p_r_prev",     3)
        w_foot_vel_sym = ca.SX.sym("w_foot_vel",  1)
        floor_z_sym   = ca.SX.sym("floor_z",       1)
        params = ca.vertcat(quat_pelvis, p_com_des,
                            p_l_des, R_l_flat, p_r_des, R_r_flat, I_des_flat,
                            q_prev_legs, w_reg_sym, w_inertia_sym, w_foot_sym,
                            w_sym_sym,
                            p_l_prev_sym, p_r_prev_sym, w_foot_vel_sym,
                            floor_z_sym)

        # ── Assemble full q ───────────────────────────────────────────────
        q_full = ca.SX.zeros(nq)
        for j in range(3):
            q_full[j]     = p_pelvis[j]
        for j in range(4):
            q_full[3 + j] = quat_pelvis[j]
        for i, qi in enumerate(self._leg_qidx):
            q_full[qi]    = q_legs[i]

        # ── Foot FK ───────────────────────────────────────────────────────
        cpin.framesForwardKinematics(self.cmodel, self.cdata_fk, q_full)
        p_l = self.cdata_fk.oMf[self.l_foot_id].translation
        R_l = self.cdata_fk.oMf[self.l_foot_id].rotation
        p_r = self.cdata_fk.oMf[self.r_foot_id].translation
        R_r = self.cdata_fk.oMf[self.r_foot_id].rotation

        # ── CoM + centroidal inertia ──────────────────────────────────────
        v_zeros = ca.SX.zeros(nv)
        cpin.ccrba(self.cmodel, self.cdata_dyn, q_full, v_zeros)
        com = self.cdata_dyn.com[0]
        I_G = self.cdata_dyn.Ig.inertia

        # ── Rotate I_srb body frame → world frame ─────────────────────────
        qx = quat_pelvis[0]; qy = quat_pelvis[1]; qz = quat_pelvis[2]; qw = quat_pelvis[3]
        R_body = ca.vertcat(
            ca.horzcat(1-2*(qy**2+qz**2),   2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)),
            ca.horzcat(  2*(qx*qy+qz*qw), 1-2*(qx**2+qz**2),   2*(qy*qz-qx*qw)),
            ca.horzcat(  2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw), 1-2*(qx**2+qy**2)),
        )
        I_srb_3x3 = ca.reshape(I_des_flat, 3, 3)
        I_des     = R_body @ I_srb_3x3 @ R_body.T

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

        # Bilateral symmetry: pitch/knee/ankle_pitch equal; roll/ankle_roll mirrored.
        # LEG_JOINTS order: [L_hp, L_hr, L_hy, L_k, L_ap, L_ar,  R_hp, R_hr, R_hy, R_k, R_ap, R_ar]
        cost_sym = (
            (q_legs[0] - q_legs[6])**2 +   # hip_pitch  L == R
            (q_legs[1] + q_legs[7])**2 +   # hip_roll   L == -R
            q_legs[2]**2 + q_legs[8]**2 +  # hip_yaw    L ≈ 0, R ≈ 0
            (q_legs[3] - q_legs[9])**2 +   # knee       L == R
            (q_legs[4] - q_legs[10])**2 +  # ankle_pitch L == R
            (q_legs[5] + q_legs[11])**2    # ankle_roll  L == -R
        )

        # Foot-velocity cost: penalise foot position change relative to previous frame.
        # Prevents lateral foot sliding during contact phases.
        e_lv = p_l - p_l_prev_sym
        e_rv = p_r - p_r_prev_sym
        cost_foot_vel = ca.dot(e_lv, e_lv) + ca.dot(e_rv, e_rv)

        cost = (w_foot_sym * cost_foot + w_inertia_sym * cost_inertia
                + w_reg_sym * cost_reg + w_sym_sym * cost_sym
                + w_foot_vel_sym * cost_foot_vel)

        # ── Constraints ───────────────────────────────────────────────────
        # [0:3]   CoM equality: com(q) = p_com_des
        # [3]     left foot floor:  p_l_z >= floor_z  (lbg=0, ubg=inf)
        # [4]     right foot floor: p_r_z >= floor_z  (lbg=0, ubg=inf)
        # [5:8]   left foot position equality  (activated by lbg/ubg in solve())
        # [8:11]  right foot position equality (activated by lbg/ubg in solve())
        # [11:14] left foot orientation — 3-vector rotation error via skew(R_des^T @ R)
        # [14:17] right foot orientation
        # floor_z = ANKLE_HEIGHT during contact; -1000 during flight (inactive).
        g_lfoot_floor = p_l[2] - floor_z_sym
        g_rfoot_floor = p_r[2] - floor_z_sym
        g_lfoot_pos   = p_l - p_l_des
        g_rfoot_pos   = p_r - p_r_des

        # Rotation error: skew-symmetric part of R_des^T @ R_actual.
        # Zero iff R_actual == R_des; independent of sign / branch cuts.
        def _rot_err(R_act, R_des_mat):
            E    = R_des_mat.T @ R_act
            skew = (E - E.T) / 2
            return ca.vertcat(skew[2, 1], skew[0, 2], skew[1, 0])

        R_l_des_mat = ca.reshape(R_l_flat, 3, 3)
        R_r_des_mat = ca.reshape(R_r_flat, 3, 3)
        g_lfoot_rot = _rot_err(R_l, R_l_des_mat)
        g_rfoot_rot = _rot_err(R_r, R_r_des_mat)

        g = ca.vertcat(com - p_com_des, g_lfoot_floor, g_rfoot_floor,
                       g_lfoot_pos, g_rfoot_pos,
                       g_lfoot_rot, g_rfoot_rot)

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
              w_foot: float = None, w_sym: float = 0.0,
              p_l_prev: np.ndarray = None, p_r_prev: np.ndarray = None,
              w_foot_vel: float = 0.0,
              floor_z: float = -1000.0,
              foot_hard: bool = False,
              q_dot_max=None, dt: float = 0.02):
        """
        Solve IK via IPOPT.

        Args:
            q0:        Initial configuration (nq=36).
                       q0[0:3] is used as p_com_des (SRB CoM height from _build_q0).
                       q0[3:7] is the fixed pelvis orientation.
            oMl_des:   Desired left-foot pose (pin.SE3).
            oMr_des:   Desired right-foot pose (pin.SE3).
            q_prev:    Previous frame config for regularisation / velocity limits.
            w_reg:     Weight on ||q_legs - q_prev_legs||² regularisation.
            I_des:     3×3 desired centroidal inertia (body frame). None → SRB default.
            w_inertia: Weight on inertia cost. None → self.w_inertia.
            w_foot:    Weight on foot placement cost. None → self.w_foot.
            w_sym:     Weight on bilateral symmetry cost.
            foot_hard: If True, foot xyz position and orientation are enforced as hard
                       equality constraints instead of being part of the soft cost.
                       Orientation uses a 3-vector skew-symmetric rotation error
                       (skew(R_des^T @ R)), giving 3 independent constraints per foot.
                       Use during contact phases (stance, landing) to guarantee foot
                       pose regardless of competing costs or CoM mismatch.
            q_dot_max: Per-joint velocity limit [rad/s] as box constraint on leg joints.
                       None → uses URDF velocity limits.  0.0 → disabled.
                       Only active when q_prev is provided.
            dt:        Timestep [s] for the velocity box constraint window.

        Returns:
            q_sol:   Full configuration (nq=36).
            success: True if IPOPT returned Solve_Succeeded or Solved_To_Acceptable_Level.
            errs:    [pos_err_m] — summed L2 foot position error at the solution.
        """
        n_leg        = len(self._leg_qidx)
        q_legs_0     = np.array([q0[qi] for qi in self._leg_qidx])
        p_com_des    = q0[0:3].copy()
        q_ref        = q_prev if q_prev is not None else q0
        q_prev_legs  = np.array([q_ref[qi] for qi in self._leg_qidx])

        I_des_flat    = I_des.flatten('F') if I_des is not None else self._I_srb_flat
        w_inertia_val = w_inertia if w_inertia is not None else self.w_inertia
        w_foot_val    = w_foot    if w_foot    is not None else self.w_foot
        # Foot-velocity: default to current foot targets so cost is zero on first frame
        p_l_prev_val  = oMl_des.translation if p_l_prev is None else np.asarray(p_l_prev)
        p_r_prev_val  = oMr_des.translation if p_r_prev is None else np.asarray(p_r_prev)

        p_val = np.concatenate([
            q0[3:7],                           # quat_pelvis (fixed orientation)
            p_com_des,                         # CoM equality target
            oMl_des.translation,               # p_l_des
            oMl_des.rotation.flatten('F'),     # R_l_flat (col-major)
            oMr_des.translation,               # p_r_des
            oMr_des.rotation.flatten('F'),     # R_r_flat
            I_des_flat,                        # desired inertia (body frame, col-major)
            q_prev_legs,                       # regularisation reference
            [w_reg],
            [w_inertia_val],
            [w_foot_val],
            [w_sym],
            p_l_prev_val,                      # previous left-foot position
            p_r_prev_val,                      # previous right-foot position
            [w_foot_vel],                      # foot-velocity cost weight
            [floor_z],                         # foot floor lower bound
        ])

        x0 = np.concatenate([p_com_des, q_legs_0])
        _pz_margin = 0.3

        # Joint velocity box constraint: tighten leg bounds to |q - q_prev| ≤ q_dot_max * dt.
        # Only active when a previous frame is provided.
        if q_prev is not None and q_dot_max != 0.0:
            _qdmax = (self._leg_vel_lim if q_dot_max is None
                      else np.broadcast_to(np.asarray(q_dot_max, dtype=float),
                                           self._leg_vel_lim.shape).copy())
            _window = _qdmax * dt
            leg_lo_eff = np.maximum(self._leg_lo, q_prev_legs - _window)
            leg_hi_eff = np.minimum(self._leg_hi, q_prev_legs + _window)
        else:
            leg_lo_eff = self._leg_lo
            leg_hi_eff = self._leg_hi

        lbx = np.concatenate([[-np.inf, -np.inf, max(0.2, p_com_des[2] - _pz_margin)], leg_lo_eff])
        ubx = np.concatenate([[ np.inf,  np.inf,               p_com_des[2] + _pz_margin], leg_hi_eff])

        # g layout (17 rows):
        #   [0:3]   com_eq      — always equality
        #   [3]     lfoot_floor — always >= 0
        #   [4]     rfoot_floor — always >= 0
        #   [5:8]   lfoot_pos   — equality when foot_hard, else inactive
        #   [8:11]  rfoot_pos   — equality when foot_hard, else inactive
        #   [11:14] lfoot_rot   — equality when foot_hard, else inactive
        #   [14:17] rfoot_rot   — equality when foot_hard, else inactive
        _BIG  = np.inf
        _zero = np.zeros(6)   # pos + rot per foot (3+3)
        _open = np.full(6, _BIG)
        if foot_hard:
            lbg = np.concatenate([[0., 0., 0., 0., 0.], _zero, _zero])
            ubg = np.concatenate([[0., 0., 0., _BIG, _BIG], _zero, _zero])
        else:
            lbg = np.concatenate([[0., 0., 0., 0., 0.], -_open, -_open])
            ubg = np.concatenate([[0., 0., 0., _BIG, _BIG],  _open,  _open])

        sol = self._solver(
            x0=x0, lbx=lbx, ubx=ubx,
            lbg=lbg, ubg=ubg,
            p=p_val,
        )

        x_sol      = np.array(sol["x"]).flatten()
        q_legs_sol = x_sol[3:]

        q_sol      = q0.copy()
        q_sol[0:3] = x_sol[0:3]
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
