##
#
# IPOPT-based IK for the Unitree G1 robot.
#
# Formulation:
#   min  w_foot    * (||p_l - p_l_des||² + ||R_l - R_l_des||²_F + right foot)
#      + w_inertia * ||q_legs - q_legs_ref||²   ← identity placeholder;
#                                                   replace with I_ik(q)-I_des(q)
#   s.t. q_leg_lo <= q_legs <= q_leg_hi           (box constraints via lbx/ubx)
#
# Base pose is a parameter (held fixed during optimisation), matching the
# Newton-Raphson strategy in g1_ik.py.  The NLP is compiled once in __init__
# and re-solved with different parameters each call, amortising CasADi/IPOPT
# compilation overhead across the full trajectory.
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
                 w_foot: float = 1.0,
                 w_inertia: float = 1e-4,
                 ipopt_opts: dict = None):
        self.model = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
        self.data  = self.model.createData()

        self.cmodel = cpin.Model(self.model)
        self.cdata  = self.cmodel.createData()

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

        self._solver = self._build_solver(ipopt_opts)

    # ------------------------------------------------------------------
    # NLP construction — called once, solver reused across trajectory
    # ------------------------------------------------------------------

    def _build_solver(self, ipopt_opts: dict = None) -> ca.Function:
        nq    = self.model.nq
        n_leg = len(self._leg_qidx)

        # Decision variables: 12 leg joint angles
        q_legs = ca.SX.sym("q_legs", n_leg)

        # Parameters packed as one vector:
        #   [q_base(7), p_l_des(3), R_l_flat(9), p_r_des(3), R_r_flat(9), q_legs_ref(12)]
        # R_*_flat uses column-major (Fortran) order to match ca.reshape(,3,3).
        q_base     = ca.SX.sym("q_base",      7)
        p_l_des    = ca.SX.sym("p_l_des",     3)
        R_l_flat   = ca.SX.sym("R_l_flat",    9)
        p_r_des    = ca.SX.sym("p_r_des",     3)
        R_r_flat   = ca.SX.sym("R_r_flat",    9)
        q_legs_ref = ca.SX.sym("q_legs_ref", n_leg)
        params = ca.vertcat(q_base, p_l_des, R_l_flat, p_r_des, R_r_flat, q_legs_ref)

        # Assemble full q: base from parameter, leg joints from decision vars,
        # upper-body DOFs zero (they don't appear in the ankle FK chain).
        q_full = ca.SX.zeros(nq)
        for j in range(7):
            q_full[j] = q_base[j]
        for i, qi in enumerate(self._leg_qidx):
            q_full[qi] = q_legs[i]

        # Symbolic FK: framesForwardKinematics runs FK + updates all frame placements
        cpin.framesForwardKinematics(self.cmodel, self.cdata, q_full)

        p_l = self.cdata.oMf[self.l_foot_id].translation
        R_l = self.cdata.oMf[self.l_foot_id].rotation
        p_r = self.cdata.oMf[self.r_foot_id].translation
        R_r = self.cdata.oMf[self.r_foot_id].rotation

        # Desired rotation matrices (column-major reshape matches flatten('F'))
        R_l_des = ca.reshape(R_l_flat, 3, 3)
        R_r_des = ca.reshape(R_r_flat, 3, 3)

        # Foot cost: position L2² + rotation Frobenius²
        e_lp = p_l - p_l_des
        e_rp = p_r - p_r_des
        e_lR = ca.reshape(R_l - R_l_des, 9, 1)
        e_rR = ca.reshape(R_r - R_r_des, 9, 1)
        cost_foot = (ca.dot(e_lp, e_lp) + ca.dot(e_rp, e_rp) +
                     ca.dot(e_lR, e_lR) + ca.dot(e_rR, e_rR))

        # Inertia cost placeholder: identity regularisation toward q_legs_ref.
        # Swap in cpin.crba-based inertia residual here when ready.
        e_reg = q_legs - q_legs_ref
        cost_inertia = ca.dot(e_reg, e_reg)

        cost = self.w_foot * cost_foot + self.w_inertia * cost_inertia

        nlp = {"x": q_legs, "f": cost, "p": params}

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

    def solve(self, q0: np.ndarray, oMl_des: pin.SE3, oMr_des: pin.SE3):
        """
        Solve IK via IPOPT.

        Args:
            q0:      Initial configuration (nq=36).  q0[0:7] is held fixed as the
                     base pose; q0[7:19] seeds the leg-joint warm-start.
            oMl_des: Desired left-foot pose (pin.SE3).
            oMr_des: Desired right-foot pose (pin.SE3).

        Returns:
            q_sol:   Full configuration (nq=36) with solved leg joints.
            success: True if IPOPT returned Solve_Succeeded or Solved_To_Acceptable_Level.
            errs:    [pos_err_m] — summed L2 foot position error at the solution,
                     matches the errs[-1] interface used by run_ik_trajectory.
        """
        q_legs_0 = np.array([q0[qi] for qi in self._leg_qidx])

        # Column-major flatten so ca.reshape(R_flat, 3, 3) reconstructs correctly
        p_val = np.concatenate([
            q0[0:7],
            oMl_des.translation,
            oMl_des.rotation.flatten('F'),
            oMr_des.translation,
            oMr_des.rotation.flatten('F'),
            q_legs_0,               # regularisation reference = warm-start
        ])

        sol = self._solver(
            x0=q_legs_0,
            lbx=self._leg_lo,
            ubx=self._leg_hi,
            p=p_val,
        )

        q_legs_sol = np.array(sol["x"]).flatten()

        q_sol = q0.copy()
        for i, qi in enumerate(self._leg_qidx):
            q_sol[qi] = q_legs_sol[i]

        status  = self._solver.stats()["return_status"]
        success = status in ("Solve_Succeeded", "Solved_To_Acceptable_Level")

        # Evaluate final foot position error via standard pinocchio (cheap post-check)
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
