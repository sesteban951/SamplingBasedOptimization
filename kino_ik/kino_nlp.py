##
#
# Kino-dynamics centroidal NLP (Option B: discrete centroidal-momentum balance).
#
# Solves for the full-body joint trajectory (q, v) whose centroidal momentum is
# driven by the SRB's external wrench, while the feet are planted (contact) and
# the base pose / momentum track the SRB reference.
#
# State X = {q, v} over nodes k = 0..N.  The floating base b = {p, qu, v_b, w}
# is the base subset of (q, v) — tracked softly, not pinned.  Arms/waist are
# pinned to defaults so the only inertia-shaping DOF are the 12 legs (the SRB's
# variable inertia was sampled from leg tuck only — see ik/sample_inertia_workspace.py).
#
# Decision variables per node k = 0..N:
#   Q[0:3,  k]  — base (CoM≈pelvis) XYZ            [free]
#   Q[3:7,  k]  — base quaternion [qx,qy,qz,qw]    [free, soft-tracked to SRB]
#   Q[leg,  k]  — 12 leg joint angles               [free]
#   Q[arm,  k]  — 17 arm/waist joints               [fixed = defaults]
#   V[0:6,  k]  — base spatial velocity             [free, soft-tracked]
#   V[leg,  k]  — 12 leg joint velocities            [free]
#   V[arm,  k]  — 17 arm velocities                  [fixed = 0]
#
# Hard constraints:
#   com(q_k) = p_com_srb[k]                               CoM position (was soft)
#   q_{k+1} = integrate(q_k, dt_k * v_k)                  manifold config integration
#   p_foot(q_k)  = [foot_xy_srb, floor_z]   (contact)     foot position
#   log3(R_foot) = 0                        (contact)     foot flat
#   Q[arm,k] = defaults,  V[arm,k] = 0                    arm/waist pinned
#   leg q / v box limits ; ||quat|| = 1
#
# Soft costs:
#   w_mom    * ||hg(q_k,v_k) - H_srb[k]||^2        centroidal momentum tracking
#                                                  (linear part = m*v_com tracking)
#   w_dym    * ||(hg_{k+1}-hg_k) - dt*W_ext[k]||^2 centroidal momentum balance (was hard)
#   w_quat   * ||Q[3:7,k] - quat_srb[k]||^2        base orientation
#   w_wbase  * ||V[3:6,k] - w_body_srb[k]||^2      base angular velocity
#   w_sym    * bilateral_symmetry(q_legs_k)
#   w_qreg   * ||q_legs[k] - q_legs_warm[k]||^2
#   w_vsmooth* ||V[k+1] - V[k]||^2
#
# CCRBA appears inside the NLP (momentum balance + cost), so the L-BFGS Hessian
# approximation is used: CasADi AD-differentiates ccrba exactly once (the
# constraint Jacobian), never a second time for an exact Hessian.
#
##

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import casadi as ca
import pinocchio as pin
import pinocchio.casadi as cpin

_REPO_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_URDF = os.path.join(_REPO_ROOT, "models", "g1", "g1_29dof_rev_1_0.urdf")

LEG_JOINTS = [
    "left_hip_pitch_joint",    "left_hip_roll_joint",    "left_hip_yaw_joint",
    "left_knee_joint",         "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint",   "right_hip_roll_joint",   "right_hip_yaw_joint",
    "right_knee_joint",        "right_ankle_pitch_joint","right_ankle_roll_joint",
]

L_FOOT = "left_ankle_roll_link"
R_FOOT = "right_ankle_roll_link"


def _skew(u):
    return np.array([[0.0, -u[2], u[1]], [u[2], 0.0, -u[0]], [-u[1], u[0], 0.0]])


class _HgJac(ca.Callback):
    """Analytic Jacobian of centroidal momentum hg(q,v) = Ag(q)v.

    d hg/d v     = Ag                                              (exact)
    d hg/d q_tan = [ dHdq[0:3] ; dHdq[3:6] + [l]xJcom - [c]x dHdq[0:3] ]
                   (pinocchio dHdq is about the world origin; the transport
                    terms re-reference it to the moving CoM — validated vs FD)
    d hg/d q_raw = d hg/d q_tan @ pinv(J_int)   (quaternion bridge, J_int FD)
    """
    def __init__(self, name, model, opts={}):
        ca.Callback.__init__(self)
        self.model = model
        self.data  = model.createData()
        self.data2 = model.createData()
        self.data3 = model.createData()
        self.nq = model.nq
        self.nv = model.nv
        self.construct(name, opts)

    def get_n_in(self):  return 3      # q, v, hg_nominal (unused)
    def get_n_out(self): return 2      # d hg/dq (6 x nq), d hg/dv (6 x nv)

    def get_sparsity_in(self, i):
        return ca.Sparsity.dense([self.nq, self.nv, 6][i], 1)

    def get_sparsity_out(self, i):
        return ca.Sparsity.dense(6, self.nq if i == 0 else self.nv)

    def eval(self, arg):
        m = self.model
        q = np.array(arg[0]).flatten().copy()
        v = np.array(arg[1]).flatten()
        q[3:7] /= np.linalg.norm(q[3:7])
        pin.computeCentroidalDynamicsDerivatives(m, self.data, q, v, np.zeros(self.nv))
        dHdq = np.array(self.data.dHdq)
        Ag   = np.array(self.data.Ag)
        Jcom = pin.jacobianCenterOfMass(m, self.data2, q)
        pin.ccrba(m, self.data2, q, v)
        l = np.array(self.data2.hg.linear)
        c = np.array(pin.centerOfMass(m, self.data3, q))
        dqt = dHdq.copy()
        dqt[3:6, :] += _skew(l) @ Jcom - _skew(c) @ dHdq[0:3, :]
        # Analytic raw-quaternion bridge B = pinv(J_int), block-diagonal:
        #   J_int = blkdiag( R(3x3, base lin) , Qb(4x3, quat) , I(joints) )
        #   pinv  = blkdiag( R^T            , pinv(Qb)       , I        )
        # Qb = d(quat ⊗ Exp(w))/dw = 0.5[ w I3 + [v]x ; -v^T ]  (xyzw coeffs)
        # validated vs central-difference FD to ~1e-9 over a diverse battery.
        R  = pin.Quaternion(q[3:7]).matrix()
        qv = q[3:6]; qw = q[6]
        Qb = 0.5 * np.vstack([qw*np.eye(3) + _skew(qv), -qv.reshape(1, 3)])
        B  = np.zeros((self.nv, self.nq))
        B[0:3, 0:3]              = R.T
        B[3:6, 3:7]              = np.linalg.pinv(Qb)
        B[6:self.nv, 7:self.nq]  = np.eye(self.nq - 7)
        Jq = dqt @ B
        return [ca.DM(Jq), ca.DM(Ag)]


class HgCallback(ca.Callback):
    """Centroidal momentum hg(q,v) = Ag(q)v via pinocchio (numeric, no AD/compile).

    Provides an analytic Jacobian (see _HgJac) so CasADi never differentiates
    ccrba symbolically — avoids both the slow VM AD and the JIT compile blowup.
    """
    def __init__(self, name, model, opts={}):
        ca.Callback.__init__(self)
        self.model = model
        self.data  = model.createData()
        self.nq = model.nq
        self.nv = model.nv
        self._jac_refs = []          # keep Jac callbacks alive (CasADi holds raw refs)
        self.construct(name, opts)

    def get_n_in(self):  return 2
    def get_n_out(self): return 1

    def get_sparsity_in(self, i):
        return ca.Sparsity.dense(self.nq if i == 0 else self.nv, 1)

    def get_sparsity_out(self, i):
        return ca.Sparsity.dense(6, 1)

    def eval(self, arg):
        q = np.array(arg[0]).flatten().copy()
        v = np.array(arg[1]).flatten()
        q[3:7] /= np.linalg.norm(q[3:7])
        pin.ccrba(self.model, self.data, q, v)
        return [np.concatenate([np.array(self.data.hg.linear),
                                np.array(self.data.hg.angular)])]

    def has_jacobian(self): return True

    def get_jacobian(self, name, inames, onames, opts):
        j = _HgJac(name, self.model, opts)
        self._jac_refs.append(j)
        return j


_DEFAULT_WEIGHTS = {
    'w_mom':       1.0,      # centroidal momentum tracking (||hg - H_srb||^2)
    'w_dym':     100.0,      # centroidal momentum balance penalty (||dhg - dt*W_ext||^2)
    'w_com':    5000.0,      # UNUSED — CoM is now a hard constraint (kept for CLI compat)
    'w_quat':    500.0,      # base orientation tracking
    'w_wbase':    50.0,      # base angular velocity tracking
    'w_sym':       2.0,      # bilateral leg symmetry regularizer
    'w_qreg':      1.0,      # leg-angle regularizer toward warm start
    'w_vsmooth':   0.1,      # velocity smoothness regularizer
    'w_armreg':  5000.0,     # STRONG regularizer on freed shoulder DOF (escape hatch)
}

# Arm DOF freed for slack/momentum authority (sagittal shoulder pitch only).
FREE_ARM_JOINTS = ["left_shoulder_pitch_joint", "right_shoulder_pitch_joint"]


class KinoNLP:

    def __init__(self,
                 dt_vec,         # (N,)      timestep per interval
                 stance_end,     # int       first flight node index
                 flight_end,     # int       first landing node index
                 quat_srb_pin,   # (N+1, 4)  pinocchio convention [qx,qy,qz,qw]
                 w_body_srb,     # (N+1, 3)  body-frame angular velocity from SRB
                 H_srb,          # (N+1, 6)  centroidal momentum [linear; angular] world frame
                 W_ext,          # (N,   6)  external wrench [F-mg ; M] world frame, drives hg
                 p_com_srb,      # (N+1, 3)  SRB CoM position reference
                 p_foot_srb,     # (N+1, 4)  [pLx,pLy,pRx,pRy], NaN during flight
                 floor_z,        # float     ankle-frame floor height
                 q_arm_default,  # (nq,)     full config with arm/waist defaults
                 Q_warm,         # (N+1, nq) warm-start configuration
                 V_warm,         # (N+1, nv) warm-start velocity
                 weights=None,
                 hessian_mode="limited-memory",  # "limited-memory" (L-BFGS) or "exact"
                 urdf_path=_DEFAULT_URDF):

        self.hessian_mode = hessian_mode

        self.model = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
        self.data  = self.model.createData()
        nq = self.model.nq   # 36
        nv = self.model.nv   # 35

        cmodel = cpin.Model(self.model)
        self.cmodel = cmodel

        # Joint index maps
        self._leg_qidx = [self.model.joints[self.model.getJointId(j)].idx_q for j in LEG_JOINTS]
        self._leg_vidx = [self.model.joints[self.model.getJointId(j)].idx_v for j in LEG_JOINTS]
        # Freed arm DOF (sagittal shoulder pitch): movable but strongly regularized.
        self._free_qidx = [self.model.joints[self.model.getJointId(j)].idx_q for j in FREE_ARM_JOINTS]
        self._free_vidx = [self.model.joints[self.model.getJointId(j)].idx_v for j in FREE_ARM_JOINTS]
        # Remaining arm/waist DOF stay pinned to defaults.
        self._arm_qidx = sorted(set(range(7, nq)) - set(self._leg_qidx) - set(self._free_qidx))
        self._arm_vidx = sorted(set(range(6, nv)) - set(self._leg_vidx) - set(self._free_vidx))

        self._leg_qlo  = np.array([self.model.lowerPositionLimit[qi] for qi in self._leg_qidx])
        self._leg_qhi  = np.array([self.model.upperPositionLimit[qi] for qi in self._leg_qidx])
        self._leg_vmax = np.array([self.model.velocityLimit[vi]       for vi in self._leg_vidx])
        self._free_qlo  = np.array([self.model.lowerPositionLimit[qi] for qi in self._free_qidx])
        self._free_qhi  = np.array([self.model.upperPositionLimit[qi] for qi in self._free_qidx])
        self._free_vmax = np.array([self.model.velocityLimit[vi]       for vi in self._free_vidx])

        l_foot_id = self.model.getFrameId(L_FOOT)
        r_foot_id = self.model.getFrameId(R_FOOT)

        N = len(dt_vec)
        self.N           = N
        self.nq          = nq
        self.nv          = nv
        self.dt_vec      = np.asarray(dt_vec, dtype=float)
        self.stance_end  = stance_end
        self.flight_end  = flight_end
        self.quat_srb_pin = np.asarray(quat_srb_pin, dtype=float)
        self.w_body_srb   = np.asarray(w_body_srb,   dtype=float)
        self.H_srb        = np.asarray(H_srb,         dtype=float)
        self.W_ext        = np.asarray(W_ext,         dtype=float)
        self.p_com_srb    = np.asarray(p_com_srb,     dtype=float)
        self.p_foot_srb   = np.asarray(p_foot_srb,    dtype=float)
        self.floor_z      = float(floor_z)
        self.q_arm_default = np.asarray(q_arm_default, dtype=float)
        self.Q_warm       = np.asarray(Q_warm, dtype=float)
        self.V_warm       = np.asarray(V_warm, dtype=float)

        w = {**_DEFAULT_WEIGHTS, **(weights or {})}
        self.w_mom      = w['w_mom']
        self.w_dym      = w['w_dym']
        self.w_com      = w['w_com']   # unused (CoM hard); kept for CLI compatibility
        self.w_quat     = w['w_quat']
        self.w_wbase    = w['w_wbase']
        self.w_sym      = w['w_sym']
        self.w_qreg     = w['w_qreg']
        self.w_vsmooth  = w['w_vsmooth']
        self.w_armreg   = w['w_armreg']

        # ── Build compiled CasADi functions (FK, CoM, ccrba, integrate) ──────
        # framesForwardKinematics sets oMi (joints) + oMf (frames).
        # We compute CoM manually from oMi + inertia levers — O(n).
        print("[KinoNLP] Building CasADi functions (FK, CoM, ccrba, integrate)...")
        q_sx = ca.SX.sym('q', nq)
        v_sx = ca.SX.sym('v', nv)

        cdata_fk = cmodel.createData()
        cpin.framesForwardKinematics(cmodel, cdata_fk, q_sx)

        # CoM from oMi + inertia levers (one FK pass, no CRBA).
        # Use the numeric model (not cmodel) for masses/levers — they are constants.
        _total_mass = sum(self.model.inertias[i].mass for i in range(1, self.model.njoints))
        _com_sum = ca.SX.zeros(3)
        for i in range(1, self.model.njoints):
            m_i = self.model.inertias[i].mass
            if m_i < 1e-10:
                continue
            lever = ca.DM(np.array(self.model.inertias[i].lever))
            R_i   = cdata_fk.oMi[i].rotation
            t_i   = cdata_fk.oMi[i].translation
            _com_sum += m_i * (t_i + R_i @ lever)
        _com_sx = _com_sum / _total_mass

        # Single function: q → (com, p_l, p_r, R_l, R_r)
        self.f_fkcom = ca.Function('f_fkcom', [q_sx], [
            _com_sx,
            cdata_fk.oMf[l_foot_id].translation,
            cdata_fk.oMf[r_foot_id].translation,
            cdata_fk.oMf[l_foot_id].rotation,
            cdata_fk.oMf[r_foot_id].rotation,
        ])

        # Centroidal momentum hg(q,v) = Ag(q) v via a numeric pinocchio Callback
        # with an analytic Jacobian (HgCallback) — no AD through ccrba, no compile.
        self.f_hg  = HgCallback('hg_cb', self.model)
        self.f_h   = self.f_hg                       # alias for pipeline/verification
        self.f_com = ca.Function('f_com', [q_sx], [_com_sx])

        # Manifold integration q+ = integrate(q, dt*v)
        dt_sx = ca.SX.sym('dt')
        self.f_integrate = ca.Function('f_integrate', [q_sx, v_sx, dt_sx],
                                       [cpin.integrate(cmodel, q_sx, dt_sx * v_sx)])

        print("[KinoNLP] Functions built. Assembling Opti problem...")
        self._build_opti()
        print("[KinoNLP] Opti ready.")

    # -------------------------------------------------------------------------

    def _build_opti(self):
        N  = self.N
        nq = self.nq
        nv = self.nv

        opti = ca.Opti()
        Q = opti.variable(nq, N + 1)   # (nq, N+1)
        V = opti.variable(nv, N + 1)   # (nv, N+1)

        J = 0.0

        # ── Fixed DOF equality constraints (arms/waist only — base is free) ──
        # Arm velocities pinned to 0 at every node; arm angles pinned only at
        # node 0 — integration (v_arm=0) then holds them at default for all k.
        for qi in self._arm_qidx:
            opti.subject_to(Q[qi, 0] == float(self.q_arm_default[qi]))
        for k in range(N + 1):
            for vi in self._arm_vidx:
                opti.subject_to(V[vi, k] == 0.0)

        # Quaternion unit-norm only at node 0 — integrate() yields unit quats for k>=1
        opti.subject_to(ca.dot(Q[3:7, 0], Q[3:7, 0]) == 1.0)

        # ── Per-node constraints and costs ────────────────────────────────────
        for k in range(N + 1):
            Q_k = Q[:, k]
            V_k = V[:, k]
            is_contact = (k < self.stance_end) or (k >= self.flight_end)

            com_k, p_l, p_r, R_l, R_r = self.f_fkcom(Q_k)
            hg_k = self.f_hg(Q_k, V_k)

            # Joint position limits — leg DOFs
            for i, qi in enumerate(self._leg_qidx):
                opti.subject_to(opti.bounded(
                    float(self._leg_qlo[i]), Q[qi, k], float(self._leg_qhi[i])))

            # Joint velocity limits — leg DOFs
            for i, vi in enumerate(self._leg_vidx):
                opti.subject_to(opti.bounded(
                    -float(self._leg_vmax[i]), V[vi, k], float(self._leg_vmax[i])))

            # Freed shoulder DOF: box limits (movable) + strong regularizer below
            for i, qi in enumerate(self._free_qidx):
                opti.subject_to(opti.bounded(
                    float(self._free_qlo[i]), Q[qi, k], float(self._free_qhi[i])))
            for i, vi in enumerate(self._free_vidx):
                opti.subject_to(opti.bounded(
                    -float(self._free_vmax[i]), V[vi, k], float(self._free_vmax[i])))

            # ── Foot constraints — contact nodes only (hard FK targets) ───────
            if is_contact and not np.isnan(self.p_foot_srb[k, 0]):
                # Position: xy at SRB foot target, z on the floor
                opti.subject_to(p_l[0] == self.p_foot_srb[k, 0])
                opti.subject_to(p_l[1] == self.p_foot_srb[k, 1])
                opti.subject_to(p_l[2] == self.floor_z)
                opti.subject_to(p_r[0] == self.p_foot_srb[k, 2])
                opti.subject_to(p_r[1] == self.p_foot_srb[k, 3])
                opti.subject_to(p_r[2] == self.floor_z)
                # Orientation: feet flat (R=I).  Antisymmetric (vee) residual
                # vee(R-R^T)=0 is smooth at identity (unlike log3's 0/0) and is
                # 3 independent constraints that drive R->I near the warm start.
                def _vee_err(R):
                    return ca.vertcat(R[2, 1] - R[1, 2],
                                      R[0, 2] - R[2, 0],
                                      R[1, 0] - R[0, 1])
                opti.subject_to(_vee_err(R_l) == 0)
                opti.subject_to(_vee_err(R_r) == 0)

            # ── Costs ─────────────────────────────────────────────────────────

            # Centroidal momentum tracking (linear part = m*v_com tracking)
            e_h = hg_k - self.H_srb[k]
            J += self.w_mom * ca.dot(e_h, e_h)

            # CoM position — HARD equality constraint (was soft w_com cost)
            opti.subject_to(com_k == self.p_com_srb[k])

            # Base orientation tracking
            e_quat = Q[3:7, k] - self.quat_srb_pin[k]
            J += self.w_quat * ca.dot(e_quat, e_quat)

            # Base angular velocity tracking
            e_w = V[3:6, k] - self.w_body_srb[k]
            J += self.w_wbase * ca.dot(e_w, e_w)

            # Leg joint regularization toward warm start
            q_legs_warm_k = np.array([self.Q_warm[k, qi] for qi in self._leg_qidx])
            e_qreg = ca.vertcat(*[Q[qi, k] for qi in self._leg_qidx]) - q_legs_warm_k
            J += self.w_qreg * ca.dot(e_qreg, e_qreg)

            # STRONG regularization on freed shoulder DOF — keep arms at default
            # (q->default, v->0) unless the legs genuinely can't comply.  Large
            # resulting arm motion is then a diagnostic, not a free crutch.
            if self._free_qidx:
                q_free_def = np.array([self.q_arm_default[qi] for qi in self._free_qidx])
                e_armq = ca.vertcat(*[Q[qi, k] for qi in self._free_qidx]) - q_free_def
                e_armv = ca.vertcat(*[V[vi, k] for vi in self._free_vidx])
                J += self.w_armreg * (ca.dot(e_armq, e_armq) + ca.dot(e_armv, e_armv))

            # Bilateral leg symmetry
            qi = self._leg_qidx
            J += self.w_sym * (
                (Q[qi[0], k] - Q[qi[6], k])**2  +   # hip_pitch  L == R
                (Q[qi[1], k] + Q[qi[7], k])**2  +   # hip_roll   L == -R
                Q[qi[2], k]**2 + Q[qi[8], k]**2 +   # hip_yaw    L,R ≈ 0
                (Q[qi[3], k] - Q[qi[9], k])**2  +   # knee       L == R
                (Q[qi[4], k] - Q[qi[10], k])**2 +   # ankle_pitch L == R
                (Q[qi[5], k] + Q[qi[11], k])**2     # ankle_roll  L == -R
            )

        # ── Centroidal momentum dynamics (SOFT cost) — the "dym" penalty ───────
        # Self-consistency: the trajectory's OWN momentum should evolve per the
        # wrench, hg(k+1) - hg(k) = dt*W_ext.  Now a soft penalty (was a hard
        # equality) so it cannot conflict with the hard CoM/foot constraints —
        # the dynamic consistency is traded off rather than enforced exactly.
        # This is NOT the same as tracking H_srb (the separate w_mom cost).
        for k in range(N):
            dt_k  = float(self.dt_vec[k])
            hg_k  = self.f_hg(Q[:, k],     V[:, k])
            hg_k1 = self.f_hg(Q[:, k + 1], V[:, k + 1])
            e_dym = (hg_k1 - hg_k) - dt_k * self.W_ext[k]
            J += self.w_dym * ca.dot(e_dym, e_dym)

        # ── Pin INITIAL linear momentum to the SRB (anchors start-at-rest) ─────
        # H_srb(0)=0 here (SRB starts at rest).  Now that dym is soft, this hard
        # pin anchors the node-0 CoM velocity directly so the trajectory starts
        # at the correct linear momentum; it is consistent with (not redundant
        # against) the hard CoM positions, which constrain average velocities.
        # LINEAR only: H_srb_ang=I_srb*w is rigid, but true A(q)v includes
        # leg-swing momentum during the tuck, so pinning angular would over-fit.
        hg0 = self.f_hg(Q[:, 0], V[:, 0])
        opti.subject_to(hg0[0:3] == self.H_srb[0, 0:3])

        # ── Manifold config integration (hard, TRAPEZOIDAL O(dt^2)) ────────────
        #   q_{k+1} = integrate(q_k, dt_k * 0.5*(v_k + v_{k+1}))
        # Using the average endpoint velocity (vs left-endpoint Euler) makes the
        # discrete CoM displacement match the instantaneous CoM velocity to
        # O(dt^2) instead of O(dt) — closes the discretization gap that caused
        # the residual CoM drift even with c_lin pinned to 0.
        for k in range(N):
            dt_k  = float(self.dt_vec[k])
            v_avg = 0.5 * (V[:, k] + V[:, k + 1])
            opti.subject_to(Q[:, k + 1] == self.f_integrate(Q[:, k], v_avg, dt_k))

        # ── Velocity smoothness cost (all intervals) ──────────────────────────
        for k in range(N - 1):
            e_v = V[:, k + 1] - V[:, k]
            J += self.w_vsmooth * ca.dot(e_v, e_v)

        # ── Warm start ────────────────────────────────────────────────────────
        opti.set_initial(Q, self.Q_warm.T)   # Q_warm is (N+1, nq) → need (nq, N+1)
        opti.set_initial(V, self.V_warm.T)

        opti.minimize(J)

        opti.solver("ipopt", {
            # hg is a numeric Callback with an analytic Jacobian → cannot expand
            # to SX, and no JIT needed (no symbolic ccrba graph to compile).
            "expand":                               False,
            "ipopt.max_iter":                       10000,
            "ipopt.tol":                            1e-4,
            "ipopt.acceptable_tol":                 1e-3,
            "ipopt.acceptable_constr_viol_tol":     1e-3,
            "ipopt.acceptable_dual_inf_tol":        1e1,
            "ipopt.acceptable_iter":                20,
            "ipopt.hessian_approximation":          self.hessian_mode,
            "ipopt.limited_memory_max_history":     30,
            "ipopt.print_level":                    5,
            "print_time":                           1,
        })

        self._opti = opti
        self._Q    = Q
        self._V    = V

    # -------------------------------------------------------------------------

    def solve(self):
        """
        Returns:
            Q_sol  (N+1, nq)  full configuration trajectory
            V_sol  (N+1, nv)  full velocity trajectory
            success bool
        """
        try:
            sol    = self._opti.solve()
            Q_sol  = np.array(sol.value(self._Q)).T
            V_sol  = np.array(sol.value(self._V)).T
            return Q_sol, V_sol, True
        except Exception as exc:
            print(f"[KinoNLP] IPOPT did not converge: {exc}")
            Q_sol = np.array(self._opti.debug.value(self._Q)).T
            V_sol = np.array(self._opti.debug.value(self._V)).T
            return Q_sol, V_sol, False

    # -------------------------------------------------------------------------
    # Verification helpers
    # -------------------------------------------------------------------------

    def momentum_residuals(self, Q_traj, V_traj):
        """Return per-node ||hg(q,v) - H_srb|| (numpy)."""
        residuals = np.zeros(self.N + 1)
        for k in range(self.N + 1):
            h = np.array(self.f_hg(Q_traj[k], V_traj[k])).flatten()
            residuals[k] = np.linalg.norm(h - self.H_srb[k])
        return residuals

    def momentum_dynamics_residuals(self, Q_traj, V_traj):
        """Return per-interval ||(hg_{k+1}-hg_k) - dt*W_ext|| (numpy).

        Confirms the hard centroidal momentum balance is satisfied."""
        residuals = np.zeros(self.N)
        for k in range(self.N):
            hg_k  = np.array(self.f_hg(Q_traj[k],     V_traj[k])).flatten()
            hg_k1 = np.array(self.f_hg(Q_traj[k + 1], V_traj[k + 1])).flatten()
            residuals[k] = np.linalg.norm((hg_k1 - hg_k) - self.dt_vec[k] * self.W_ext[k])
        return residuals

    def com_residuals(self, Q_traj, V_traj=None):
        """Return per-node ||com(q) - p_com_srb|| (numpy)."""
        residuals = np.zeros(self.N + 1)
        for k in range(self.N + 1):
            com = np.array(self.f_com(Q_traj[k])).flatten()
            residuals[k] = np.linalg.norm(com - self.p_com_srb[k])
        return residuals

    def foot_xy_residuals(self, Q_traj):
        """Return per-node max foot XY error for contact frames (NaN for flight)."""
        residuals = np.full(self.N + 1, np.nan)
        for k in range(self.N + 1):
            is_contact = (k < self.stance_end) or (k >= self.flight_end)
            if not is_contact or np.isnan(self.p_foot_srb[k, 0]):
                continue
            _, p_l, p_r, _, _ = [np.array(x).flatten() for x in self.f_fkcom(Q_traj[k])]
            e_l = np.linalg.norm(p_l[:2] - self.p_foot_srb[k, 0:2])
            e_r = np.linalg.norm(p_r[:2] - self.p_foot_srb[k, 2:4])
            residuals[k] = max(e_l, e_r)
        return residuals
