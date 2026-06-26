##
#
# Kino-dynamics centroidal NLP — restructured (lifted) formulation.
#
# Decision variables per node k = 0..N (unless noted "per interval", k = 0..N-1):
#   Q[:,k]   (nq)  pinocchio config [p_pelvis(3), quat xyzw(4), qj(29)]
#   V[:,k]   (nv)  pinocchio velocity [base spatial(6), qj_dot(29)]
#   C[:,k]   (3)   CoM position           c
#   Cd[:,k]  (3)   CoM velocity           c_dot
#   Hh[:,k]  (6)   centroidal momentum    h          = [linear; angular] (world)
#   Hd[:,k]  (6)   centroidal mom. rate   h_dot       (per interval)
#   Lam[:,k] (24)  contact-polygon forces lambda = [fL1..fL4, fR1..fR4] (world, per int.)
#                  4 corner point-forces per foot (no lumped ankle torque)
#
#   So  X = [c, c_dot, h, h_dot, quat, w, qj, qj_dot, lambda]   (quat=Q[3:7], w=V[3:6],
#   qj=Q[7:], qj_dot=V[6:] live inside Q,V; c,c_dot,h,h_dot,lambda are lifted).
#
# Objective (track the SRB reference s = [c, c_dot, quat, w, lambda]):
#   min  sum_k  w_com |c-c_srb|^2 + w_cdot |c_dot-c_dot_srb|^2
#              + w_quat |quat-quat_srb|^2 + w_wbase |w-w_srb|^2
#      + sum_k  w_lam |lambda-lambda_srb|^2
#      + w_ke * (1/2 qj_dot^T M_jj qj_dot)            joint KE (min-effort, all nodes)
#      + w_config |q_free - q_tuck|^2  (FLIGHT only)  posture prior (anti-flail):
#                  legs->0.9 tuck, sagittal arms->tuck arm pose
#      + small regularizers (leg q toward warm start, velocity smoothness)
#
# Defining / coupling constraints (hard):
#   c_k        = com(Q_k)                                          (FK)
#   m * c_dot_k = h_k[0:3]                                         (linear momentum)
#   h_k        = A_G(Q_k) V_k                                      (centroidal momentum)
#   h_dot_k    = A_G(Q_k) a_k + Adot_G(Q_k,V_k) V_k,  a_k=(V_{k+1}-V_k)/dt
#   h_dot_k    = [ sum_i f_i - (0,0,mg) ;  sum_i (p_i - c_k) x f_i ]   (Newton-Euler,
#                contact polygon)  p_i = FK world position of foot corner i
#   Q_{k+1}    = integrate(Q_k, dt_k * 0.5(V_k+V_{k+1}))           (trapezoidal manifold)
#   non-sagittal arm/waist pinned to defaults (q at node 0, v=0); sagittal arm
#     (shoulder pitch + elbow) freed like the legs; ||quat_0|| = 1
#   contact nodes: foot xy = SRB target, foot z = floor, feet flat (vee residual)
#   contact intervals: per point  f_z >= 0  and  |f_x|+|f_y| <= mu f_z   (unilateral
#                      + inner friction pyramid; normal = world +z, flat foot)
#   flight intervals: lambda_k = 0
#   leg joint position / velocity box limits (URDF limits)
#   self-collision: per capsule sphere-pair  ||p_a-p_b||^2 >= (r_a+r_b+margin)^2
#                   (all nodes; smooth, from kino_ik/collision_model.py)
#
# Centroidal dynamics via the Wensing & Orin (2016) factorization: A_G and the
# bias Adot_G q_dot are obtained from inverse-dynamics terms (mass matrix H and
# Coriolis C q_dot) through a kinematic transform iX_G^T, so neither ccrba nor
# (the expensive) dccrba is ever formed.  Realized here with O(n) RNEA:
#   T(q)   = [[R1, 0],[-skew(c - p_base) R1, R1]]          (R1 = base rotation)
#   g(q)   = rnea(q, 0, 0)                                  (gravity term)
#   h      = T (rnea(q,0,v) - g)[0:6]                       (= A_G v)
#   h_dot  = T (rnea(q,v,a) - g)[0:6]                       (= A_G a + Adot_G v)
# Verified vs pin.ccrba / computeCentroidalMomentumTimeVariation to ~1e-13
# (see kino_ik/verify_cmm_wensing_orin.py).
#
##

import os, sys, time, hashlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import casadi as ca
import pinocchio as pin
import pinocchio.casadi as cpin

from kino_ik import collision_model as cm

# Bump this whenever the NLP *structure* (variables / constraints / cost terms /
# their symbolic form) changes, so stale serialized solvers are not reloaded.
# (Weights and reference DATA are not part of the structure — weights are runtime
# parameters; reference data is hashed into the cache key.)
_CACHE_VERSION = "v11"  # v11: prepended standing lead-in nodes (n_lead) — mirror of the
                        #      terminal settle window at the front (no SRB cost, stand pull,
                        #      initial rest V_0=0).  Changes the symbolic graph.
                        # v10: foot orientation reformulated.  (1) flat + UPRIGHT always
                        #      (foot z-axis = world +z, R[2,2]=+1) instead of vee(R)=0
                        #      ("R symmetric"), which also admitted 180-deg flips that stab the
                        #      foot through the floor.  (2) heading locked per ground phase for
                        #      BOTH modes: free_foot_yaw=False pins the plant-frame heading to
                        #      the SRB-reference body yaw (generalises to any twist, no hardcoded
                        #      forward); free_foot_yaw=True leaves it free within +/-slack.
                        # v9: free_full_dof flag (free all joints except wrists);
                        #     free_waist_yaw + free_full_dof added to cache key
                        # v8: free_foot_yaw reformulated — control-frame (body-yaw) heading
                        #     with slack at the plant frame, locked across each ground phase
                        # v7: free_foot_yaw — flat-sole/free-then-locked landing foot heading
                        # v6: appended terminal settling nodes (no SRB cost, V_N=0)

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

# Sagittal-plane arm joints freed for momentum authority (the backflip is in the
# sagittal plane): shoulder pitch + elbow, both arms.  All other arm/waist DOF
# stay pinned to defaults.  Freed joints get box limits, the KE term, and the
# warm-start regularizer like the legs.
FREE_ARM_JOINTS = [
    "left_shoulder_pitch_joint",  "left_elbow_joint",
    "right_shoulder_pitch_joint", "right_elbow_joint",
]

# Wrist joints — kept pinned even under free_full_dof (negligible inertia/momentum
# benefit, only enlarges the NLP).  Drop from here to also free the wrists.
WRIST_JOINTS = [
    "left_wrist_roll_joint",  "left_wrist_pitch_joint",  "left_wrist_yaw_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

_G = 9.81   # gravity (m/s^2)


def _resolve_hsllib():
    """Locate the CoinHSL shared library for IPOPT's runtime HSL loader.

    Order: $IPOPT_HSLLIB (explicit), then the active env's lib dirs for
    libhsl.so / libcoinhsl.so.  Returns an absolute path or None.
    """
    env = os.environ.get("IPOPT_HSLLIB")
    if env and os.path.exists(env):
        return env
    for d in (os.path.join(sys.prefix, "lib"),
              os.path.join(sys.prefix, "lib", "x86_64-linux-gnu")):
        for name in ("libhsl.so", "libcoinhsl.so"):
            p = os.path.join(d, name)
            if os.path.exists(p):
                return p
    return None

# Foot contact polygon: 4 corner points per foot, in the ankle_roll_link frame
# (taken from the G1 MuJoCo collision spheres — heel pair + toe pair).  Contact
# forces are applied at these points; the moment about the CoM is r x F summed
# over all points (no lumped ankle torque).
FOOT_CORNERS = np.array([
    [-0.05,  0.025, -0.03],   # heel, inner/left
    [-0.05, -0.025, -0.03],   # heel, outer/right
    [ 0.12,  0.030, -0.03],   # toe,  inner/left
    [ 0.12, -0.030, -0.03],   # toe,  outer/right
])
N_PT = FOOT_CORNERS.shape[0]   # contact points per foot (4)

# Flight posture prior: the 0.9-tuck leg configuration (0.1*STANDING + 0.9*MAX_CROUCH),
# i.e. _FLIGHT_TUCK_LEGS from ik/pipeline_srb_ik.py / viz_crouch_configs.py.
# Order matches LEG_JOINTS: [hip_p, hip_r, hip_y, knee, ankle_p, ankle_r] x (L, R).
FLIGHT_TUCK_LEGS = np.array([
    -2.247, 0.0, 0.0,  2.588,  0.447, 0.0,   # left
    -2.247, 0.0, 0.0,  2.588,  0.447, 0.0,   # right
])

# Flight posture prior for the freed sagittal arms, aligned with FREE_ARM_JOINTS
# order [L_shoulder_pitch, L_elbow, R_shoulder_pitch, R_elbow].  Base (right) is
FLIGHT_TUCK_ARMS = np.array([-0.802, -0.599, -0.802, -0.599])

# Flight posture prior for waist_yaw when it is freed for twist authority
# (free_waist_yaw=True).  Neutral (0): pull the relative upper/lower-body twist back
# home in flight, letting it deviate only as the momentum dynamics require.
WAIST_YAW_FLIGHT_PRIOR = 0.0

# Standing leg config (near-straight), order matches LEG_JOINTS — mirrors
# STANDING in ik/viz_crouch_configs.py.  Used as the terminal "stand up" target
# that the strong settle-window regularizer (w_qreg_term) pulls the legs toward.
STANDING_LEGS = np.array([
    0.03, 0.0, 0.0, 0.05, -0.03, 0.0,   # left
    0.03, 0.0, 0.0, 0.05, -0.03, 0.0,   # right
])


def _skew_ca(u):
    return ca.vertcat(ca.horzcat(   0, -u[2],  u[1]),
                      ca.horzcat( u[2],    0, -u[0]),
                      ca.horzcat(-u[1],  u[0],    0))


def _skew_np(u):
    return np.array([[0.0, -u[2], u[1]], [u[2], 0.0, -u[0]], [-u[1], u[0], 0.0]])


def distribute_wrench_to_points(wrench_foot, corners=FOOT_CORNERS):
    """Distribute a per-foot net wrench [F(3); M_ankle(3)] onto the corner points.

    Min-norm point forces f_i (one per corner) reproducing the foot's net force
    and its moment about the ankle:   sum f_i = F,   sum r_i x f_i = M_ankle.
    Uses the pseudoinverse of the grasp matrix (corners assumed ~flat, world≈foot
    frame — exact enough for a reference; the NLP itself uses true FK).

    wrench_foot : (N, 12)  [F_L, M_L, F_R, M_R]   world frame
    returns     : (N, 4*3*2 = 24)  [fL1..fL4, fR1..fR4]
    """
    npt = corners.shape[0]
    G = np.zeros((6, 3 * npt))
    for i, r in enumerate(corners):
        G[0:3, 3 * i:3 * i + 3] = np.eye(3)
        G[3:6, 3 * i:3 * i + 3] = _skew_np(r)
    Gp = np.linalg.pinv(G)
    Nn = wrench_foot.shape[0]
    out = np.zeros((Nn, 6 * npt))
    for k in range(Nn):
        out[k, 0:3 * npt]        = Gp @ wrench_foot[k, 0:6]    # left foot
        out[k, 3 * npt:6 * npt]  = Gp @ wrench_foot[k, 6:12]   # right foot
    return out

#COST COEFF
_DEFAULT_WEIGHTS = {
    'w_com':    1000.0,   # CoM position tracking (was a hard constraint; now soft)
    'w_cdot':    100.0,   # CoM velocity tracking
    'w_quat':    500.0,   # base orientation tracking
    'w_wbase':    50.0,   # base angular velocity tracking
    'w_lam':       1e-3,  # per-foot wrench tracking (forces are O(100) N)
    'w_qreg':      1e1,   # leg-angle regularizer toward warm start (conditioning)
    'w_vsmooth':   1e-1,   # velocity smoothness regularizer (conditioning)
    'w_ke':        3e0,   # joint kinetic energy  1/2 qj_dot^T M_jj qj_dot (all nodes)
    'w_config':    35,   # flight posture prior toward the 0.9-tuck config
    'w_qreg_term': 1e3,   # STRONG terminal regularizer: on the appended settling
                          # nodes only, pulls the free DOF toward the standing
                          # config (legs->STANDING, arms->default) so the robot
                          # rises from the landing crouch into a static stand
}


class KinoNLP:

    def __init__(self,
                 dt_vec,         # (N,)      timestep per interval
                 stance_end,     # int       first flight node index
                 flight_end,     # int       first landing node index
                 quat_srb_pin,   # (N+1, 4)  pinocchio convention [qx,qy,qz,qw]
                 w_body_srb,     # (N+1, 3)  body-frame angular velocity from SRB
                 c_srb,          # (N+1, 3)  SRB CoM position reference
                 cd_srb,         # (N+1, 3)  SRB CoM velocity reference (world)
                 lam_srb,        # (N,  24)  per-foot-point force ref [fL1..fL4, fR1..fR4]
                 p_foot_srb,     # (N+1, 4)  [pLx,pLy,pRx,pRy], NaN during flight
                 floor_z,        # float     ankle-frame floor height
                 q_arm_default,  # (nq,)     full config with arm/waist defaults
                 Q_warm,         # (N+1, nq) warm-start configuration
                 V_warm,         # (N+1, nv) warm-start velocity
                 mu=1.0,         # friction coefficient (linearized pyramid)
                 free_foot_yaw=False,  # contact feet: flat sole; heading aligned to the body
                                       # yaw (leveled control frame) within +/-foot_yaw_slack,
                                       # chosen at the plant frame and locked (world-fixed) for
                                       # the rest of that ground phase — no skid.  vs rigid
                                       # flat+forward.  Needed when the body yaws (e.g. twist).
                 foot_yaw_slack=0.26,  # [rad] allowed foot-heading deviation from the body
                                       # heading at the plant frame (~15 deg). free_foot_yaw only.
                 foot_world_yaw=None,  # list[float] (radians) | None.  If set, pin the planted
                                       # foot heading to these ABSOLUTE WORLD yaw angles — one per
                                       # ground phase in plant order, last value repeats — instead
                                       # of the SRB-reference body yaw.  COMMANDS a twist regardless
                                       # of the SRB plan, e.g. [0, pi] = stance forward, land at
                                       # 180 deg.  Takes priority over free_foot_yaw.
                 free_waist_yaw=False, # free waist_yaw (in addition to legs + sagittal
                                       # arms) for yaw momentum authority (e.g. twist).
                 free_full_dof=False,  # free EVERY actuated joint (legs + full waist +
                                       # both full arms incl. wrists).  Overrides the
                                       # leg+sagittal-arm[+waist_yaw] default set.  Extra
                                       # DOF get a default-pose flight prior; sagittal
                                       # arms keep their tuck prior.
                 n_terminal=0,   # # of settling nodes appended after the SRB motion:
                                 #   no SRB tracking cost, but full dynamics/contact
                                 #   constraints + a terminal rest constraint V_N=0
                 n_lead=0,       # # of standing lead-in nodes prepended before the SRB
                                 #   motion: mirror of n_terminal at the front — no SRB
                                 #   tracking cost, strong w_qreg_term pull to the standing
                                 #   config, plus an initial rest constraint V_0=0
                 weights=None,
                 hessian_mode="exact",  # "exact" (default, ~5x faster) or "limited-memory"
                 expand=True,    # expand NLP to a single SX graph (much faster eval)
                 max_iter=10000,
                 linear_solver="mumps",   # IPOPT linear solver (mumps/ma57/ma97/...)
                 mu_strategy="monotone",  # IPOPT barrier strategy (monotone/adaptive)
                 self_collision=True,     # add capsule sphere-pair distance constraints
                 collision_margin=None,   # extra clearance [m]; default collision_model.MARGIN
                 cache_dir=None,          # dir for serialized solvers (skips rebuild)
                 use_cache=True,          # load/save the built solver to cache_dir
                 rebuild=False,           # force rebuild even if a cache file exists
                 cache_keep=2,            # keep this many most-recent caches; prune rest
                 urdf_path=_DEFAULT_URDF):

        self.hessian_mode  = hessian_mode
        self.expand        = bool(expand)
        self.max_iter      = int(max_iter)
        self.linear_solver = str(linear_solver)
        self.mu_strategy   = str(mu_strategy)

        self.model = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
        self.data  = self.model.createData()
        nq = self.model.nq   # 36
        nv = self.model.nv   # 35

        cmodel = cpin.Model(self.model)
        self.cmodel = cmodel

        # Joint index maps
        self._leg_qidx = [self.model.joints[self.model.getJointId(j)].idx_q for j in LEG_JOINTS]
        self._leg_vidx = [self.model.joints[self.model.getJointId(j)].idx_v for j in LEG_JOINTS]
        # Freed sagittal arm DOF (shoulder pitch + elbow): movable like the legs.
        # waist_yaw is appended here when free_waist_yaw is set (twist authority), so
        # it flows into every index-driven block below (free set, box limits, priors)
        # and out of the pinned arm/waist set.  Built per-instance — the module
        # constants stay immutable.
        self.free_waist_yaw = bool(free_waist_yaw)
        self.free_full_dof  = bool(free_full_dof)
        arm_free   = list(FREE_ARM_JOINTS)
        arm_flight = list(FLIGHT_TUCK_ARMS)
        if self.free_full_dof:
            # Free every actuated joint except legs, sagittal arms (already free), and
            # wrists.  Prior = default pose so the extra DOF regularize home but can
            # move.  Takes precedence over free_waist_yaw (which it already includes).
            _already = set(LEG_JOINTS) | set(arm_free) | set(WRIST_JOINTS)
            for _jid in range(1, self.model.njoints):
                _jname = self.model.names[_jid]
                if _jname == "root_joint" or _jname in _already:
                    continue
                arm_free.append(_jname)
                arm_flight.append(float(q_arm_default[self.model.joints[_jid].idx_q]))
        elif self.free_waist_yaw:
            arm_free.append("waist_yaw_joint")
            arm_flight.append(WAIST_YAW_FLIGHT_PRIOR)
        self._arm_free_qidx = [self.model.joints[self.model.getJointId(j)].idx_q for j in arm_free]
        self._arm_free_vidx = [self.model.joints[self.model.getJointId(j)].idx_v for j in arm_free]
        # All free actuated DOF (legs + sagittal arms [+ waist_yaw]): get box limits,
        # the KE term, and the warm-start regularizer.
        self._free_qidx = self._leg_qidx + self._arm_free_qidx
        self._free_vidx = self._leg_vidx + self._arm_free_vidx
        # Everything else in arm/waist stays pinned to defaults (q at node 0, v=0).
        self._arm_qidx = sorted(set(range(7, nq)) - set(self._free_qidx))
        self._arm_vidx = sorted(set(range(6, nv)) - set(self._free_vidx))

        # Position / velocity limits (box bounds) from the URDF for the free DOF.
        self._free_qlo  = np.array([self.model.lowerPositionLimit[qi] for qi in self._free_qidx])
        self._free_qhi  = np.array([self.model.upperPositionLimit[qi] for qi in self._free_qidx])
        self._free_vmax = np.array([self.model.velocityLimit[vi]       for vi in self._free_vidx])

        # Flight posture prior over the free DOF (legs + sagittal arms [+ waist_yaw]),
        # aligned with self._free_qidx = leg_qidx + arm_free_qidx.
        self._flight_prior = np.concatenate([FLIGHT_TUCK_LEGS, np.array(arm_flight)])

        l_foot_id = self.model.getFrameId(L_FOOT)
        r_foot_id = self.model.getFrameId(R_FOOT)

        N = len(dt_vec)
        self.N            = N
        self.nq           = nq
        self.nv           = nv
        self.dt_vec       = np.asarray(dt_vec, dtype=float)
        self.stance_end   = stance_end
        self.flight_end   = flight_end
        self.quat_srb_pin = np.asarray(quat_srb_pin, dtype=float)
        self.w_body_srb   = np.asarray(w_body_srb,   dtype=float)
        self.c_srb        = np.asarray(c_srb,        dtype=float)
        self.cd_srb       = np.asarray(cd_srb,       dtype=float)
        self.lam_srb      = np.asarray(lam_srb,      dtype=float)
        self.p_foot_srb   = np.asarray(p_foot_srb,   dtype=float)
        self.floor_z      = float(floor_z)
        self.q_arm_default = np.asarray(q_arm_default, dtype=float)
        self.Q_warm       = np.asarray(Q_warm, dtype=float)

        # Terminal "stand up" target over the free DOF (legs + sagittal arms),
        # aligned with self._free_qidx = leg_qidx + arm_free_qidx: legs go to the
        # near-straight STANDING config, the freed arms return to their defaults.
        self._stand_free = np.concatenate([
            STANDING_LEGS, self.q_arm_default[self._arm_free_qidx]])
        self.V_warm       = np.asarray(V_warm, dtype=float)

        self.mass = float(sum(self.model.inertias[i].mass
                              for i in range(1, self.model.njoints)))
        self.g    = _G
        self.mu   = float(mu)
        self.free_foot_yaw = bool(free_foot_yaw)
        self.foot_yaw_slack = float(foot_yaw_slack)
        self.foot_world_yaw = (None if foot_world_yaw is None
                               else [float(a) for a in np.atleast_1d(foot_world_yaw)])
        # Settling nodes appended after the SRB motion: the last `n_terminal`
        # nodes (k > N - n_terminal) carry no SRB tracking cost — only the
        # dynamics/contact constraints and the terminal rest constraint V_N=0.
        self.n_terminal = int(n_terminal)
        # Standing lead-in nodes prepended before the SRB motion: the first `n_lead`
        # nodes (k < n_lead) carry no SRB tracking cost — only the dynamics/contact
        # constraints, the strong w_qreg_term pull to the standing config, and the
        # initial rest constraint V_0=0.  Mirror of n_terminal at the front.
        self.n_lead = int(n_lead)
        self.self_collision   = bool(self_collision)
        self.collision_margin = cm.MARGIN if collision_margin is None else float(collision_margin)
        self._caps = cm.build_capsules(self.model)   # "sausage man" capsule model

        w = {**_DEFAULT_WEIGHTS, **(weights or {})}
        self.w_com     = w['w_com']
        self.w_cdot    = w['w_cdot']
        self.w_quat    = w['w_quat']
        self.w_wbase   = w['w_wbase']
        self.w_lam     = w['w_lam']
        self.w_qreg    = w['w_qreg']
        self.w_vsmooth = w['w_vsmooth']
        self.w_ke      = w['w_ke']
        self.w_config  = w['w_config']
        self.w_qreg_term = w['w_qreg_term']

        # Weights are runtime PARAMETERS of the cached solver (so tuning them does
        # not invalidate the cache).  This fixed order maps self.w_* <-> the
        # parameter vector passed to the solver function.
        self._weight_keys = ['w_com', 'w_cdot', 'w_quat', 'w_wbase', 'w_lam',
                             'w_qreg', 'w_vsmooth', 'w_ke', 'w_config',
                             'w_qreg_term']
        self.use_cache  = bool(use_cache)
        self.rebuild    = bool(rebuild)
        self.cache_keep = int(cache_keep)
        self.cache_dir  = cache_dir or os.path.join(_REPO_ROOT, ".kino_cache")

        # ── Build CasADi functions (FK/CoM, integrate, centroidal h & h_dot) ──
        print("[KinoNLP] Building CasADi functions (FK/CoM, integrate, centroidal)...")
        q_sx = ca.SX.sym('q', nq)
        v_sx = ca.SX.sym('v', nv)
        a_sx = ca.SX.sym('a', nv)
        zv   = ca.SX.zeros(nv)

        # FK data: base rotation, CoM, foot frames.
        cdata_fk = cmodel.createData()
        cpin.framesForwardKinematics(cmodel, cdata_fk, q_sx)
        cpin.centerOfMass(cmodel, cdata_fk, q_sx)
        R1   = cdata_fk.oMi[1].rotation          # base (free-flyer) orientation
        com  = cdata_fk.com[0]                    # CoM in world

        # Wensing-Orin force/momentum transform base -> centroidal (world-aligned):
        #   T = [[ R1 , 0 ], [ -skew(c - p_base) R1 , R1 ]]   (pinocchio [lin;ang] order)
        dvec = com - q_sx[0:3]
        T = ca.SX.zeros(6, 6)
        T[0:3, 0:3] = R1
        T[3:6, 0:3] = -_skew_ca(dvec) @ R1
        T[3:6, 3:6] = R1

        self.f_fkcom = ca.Function('f_fkcom', [q_sx], [
            com,
            cdata_fk.oMf[l_foot_id].translation,
            cdata_fk.oMf[r_foot_id].translation,
            cdata_fk.oMf[l_foot_id].rotation,
            cdata_fk.oMf[r_foot_id].rotation,
        ])
        self.f_com = ca.Function('f_com', [q_sx], [com])
        self.f_T   = ca.Function('f_T',   [q_sx], [T])

        # Inverse-dynamics terms via RNEA (O(n)); centroidal h & h_dot from them.
        cdata_id = cmodel.createData()
        g_term = cpin.rnea(cmodel, cdata_id, q_sx, zv, zv)          # gravity generalized force
        Mv     = cpin.rnea(cmodel, cdata_id, q_sx, zv, v_sx) - g_term   # H v
        Mav    = cpin.rnea(cmodel, cdata_id, q_sx, v_sx, a_sx) - g_term # H a + C v
        hg     = T @ Mv[0:6]                                         # = A_G v
        hgdot  = T @ Mav[0:6]                                        # = A_G a + Adot_G v

        self.f_hg    = ca.Function('f_hg',    [q_sx, v_sx],        [hg])
        self.f_hgdot = ca.Function('f_hgdot', [q_sx, v_sx, a_sx],  [hgdot])
        self.f_h     = self.f_hg                       # alias for pipeline/verification

        # Joint kinetic energy  KE_j = 1/2 qj_dot^T M_jj(q) qj_dot.  Zero the base
        # velocity so the quadratic form picks out the joint-joint mass block:
        #   M(q) [0; qj_dot] = rnea(q, 0, [0; qj_dot]) - g(q);  KE_j = 1/2 vz . (M vz)
        # Mass-weighted, PSD, frame-invariant; reuses RNEA (no crba).
        vz   = ca.vertcat(ca.SX.zeros(6), v_sx[6:])
        Mvz  = cpin.rnea(cmodel, cdata_id, q_sx, zv, vz) - g_term
        ke_j = 0.5 * ca.dot(vz, Mvz)
        self.f_kej = ca.Function('f_kej', [q_sx, v_sx], [ke_j])

        # Self-collision: per checked capsule-sphere pair, the smooth margin
        #   ||p_a(q) - p_b(q)||^2 - (r_a + r_b + margin)^2  >= 0.
        # Sphere centers from cpin FK (differentiable); see kino_ik/collision_model.py.
        self._n_coll = 0
        if self.self_collision:
            cdata_col = cmodel.createData()
            cpin.framesForwardKinematics(cmodel, cdata_col, q_sx)

            def _fpose(fid):
                M = cdata_col.oMf[fid]
                return M.translation, M.rotation

            sph = cm.capsule_spheres(self._caps, _fpose, ca.DM, cm.N_SPHERES)
            gvec = []
            for a, b in cm.PAIRS:
                (ca_, ra), (cb_, rb) = sph[a], sph[b]
                thr = (ra + rb + self.collision_margin) ** 2
                for pa in ca_:
                    for pb in cb_:
                        dd = pa - pb
                        gvec.append(ca.dot(dd, dd) - thr)
            self._n_coll = len(gvec)
            self.f_collpairs = ca.Function('f_collpairs', [q_sx], [ca.vertcat(*gvec)])
            print(f"[KinoNLP] self-collision: {len(cm.PAIRS)} pairs x "
                  f"{cm.N_SPHERES}^2 spheres = {self._n_coll} constraints/node")

        # Manifold integration q+ = integrate(q, dt*v)
        dt_sx = ca.SX.sym('dt')
        cdata_int = cmodel.createData()
        self.f_integrate = ca.Function('f_integrate', [q_sx, v_sx, dt_sx],
                                       [cpin.integrate(cmodel, q_sx, dt_sx * v_sx)])

        print("[KinoNLP] Functions built. Preparing solver...")
        self._build_or_load()
        print("[KinoNLP] Solver ready.")

    # -------------------------------------------------------------------------

    def _weight_vec(self):
        return np.array([getattr(self, k) for k in self._weight_keys], dtype=float)

    def _cache_key(self):
        """Hash of everything baked into the solver (NOT the weights, which are
        runtime parameters).  A change in any of these → rebuild."""
        h = hashlib.sha256()
        h.update(_CACHE_VERSION.encode())
        for s in (self.N, self.stance_end, self.flight_end, self.floor_z, self.mu,
                  self.free_foot_yaw, self.foot_yaw_slack,
                  self.free_waist_yaw, self.free_full_dof, self.n_terminal, self.n_lead,
                  self.hessian_mode, self.expand, self.linear_solver,
                  self.mu_strategy, self.max_iter, self.nq, self.nv,
                  self.self_collision, self.collision_margin,
                  cm.N_SPHERES, tuple(cm.PAIRS),
                  (None if self.foot_world_yaw is None
                   else tuple(round(a, 9) for a in self.foot_world_yaw))):
            h.update(repr(s).encode())
        if self.self_collision:
            for c in self._caps:
                h.update(repr((c['name'], c['frame_id'], c['r'])).encode())
                h.update(np.ascontiguousarray(c['p0'], dtype=float).tobytes())
                h.update(np.ascontiguousarray(c['p1'], dtype=float).tobytes())
        # _flight_prior covers both the leg and arm posture targets, and the
        # free-DOF index set is fixed by FREE_ARM_JOINTS (captured by nv/structure).
        for arr in (self.dt_vec, self.quat_srb_pin, self.w_body_srb, self.c_srb,
                    self.cd_srb, self.lam_srb, np.nan_to_num(self.p_foot_srb, nan=-9e9),
                    self.q_arm_default, self.Q_warm, self.V_warm,
                    FOOT_CORNERS, self._flight_prior, self._stand_free,
                    np.array(self._free_qidx, dtype=float)):
            h.update(np.ascontiguousarray(arr, dtype=float).tobytes())
        return h.hexdigest()[:16]

    def _build_or_load(self):
        if not self.use_cache:
            print("[KinoNLP] use_cache=False — building solver (not cached)...")
            self._build_opti()
            return
        os.makedirs(self.cache_dir, exist_ok=True)
        self._cache_path = os.path.join(
            self.cache_dir, f"kino_solve_{self._cache_key()}.casadi")
        if os.path.exists(self._cache_path) and not self.rebuild:
            t0 = time.time()
            self._f_solve = ca.Function.load(self._cache_path)
            os.utime(self._cache_path, None)   # mark recently used (LRU pruning)
            print(f"[KinoNLP] Loaded cached solver ({os.path.basename(self._cache_path)}) "
                  f"in {time.time()-t0:.1f}s — skipped build.")
        else:
            why = "forced rebuild" if self.rebuild else "no cache hit"
            print(f"[KinoNLP] {why} — building solver (one-time)...")
            self._build_opti()
        self._prune_cache()

    def _prune_cache(self):
        """Keep the `cache_keep` most-recently-used kino_solve_*.casadi files,
        delete the rest.  These files are large (exact-Hessian solvers can be
        multiple GB), so unbounded accumulation is a real disk concern."""
        if not self.use_cache or self.cache_keep <= 0:
            return
        files = [os.path.join(self.cache_dir, f) for f in os.listdir(self.cache_dir)
                 if f.startswith("kino_solve_") and f.endswith(".casadi")]
        files.sort(key=os.path.getmtime, reverse=True)     # newest first
        for p in files[self.cache_keep:]:
            try:
                sz = os.path.getsize(p) / 1e9
                os.remove(p)
                print(f"[KinoNLP] pruned old cache {os.path.basename(p)} ({sz:.1f} GB)")
            except OSError:
                pass

    # -------------------------------------------------------------------------

    def _contact_node(self, k):
        return (k < self.stance_end) or (k >= self.flight_end)

    def _build_opti(self):
        N  = self.N
        nq = self.nq
        nv = self.nv
        m  = self.mass

        opti = ca.Opti()
        Q   = opti.variable(nq, N + 1)
        V   = opti.variable(nv, N + 1)
        C   = opti.variable(3,  N + 1)     # CoM position c
        Cd  = opti.variable(3,  N + 1)     # CoM velocity c_dot
        Hh  = opti.variable(6,  N + 1)     # centroidal momentum h
        Hd  = opti.variable(6,  N)         # h_dot  (per interval)
        # Contact-polygon forces: 4 corner point-forces per foot, 3D each.
        #   Lam[0:12]  = [fL1, fL2, fL3, fL4]   (left foot, per corner)
        #   Lam[12:24] = [fR1, fR2, fR3, fR4]   (right foot, per corner)
        Lam = opti.variable(6 * N_PT, N)   # 24 x N  (per interval)

        # Cost weights as runtime parameters (so tuning them reuses the cache).
        w_par = opti.parameter(len(self._weight_keys))
        wp = {k: w_par[i] for i, k in enumerate(self._weight_keys)}

        J = 0.0
        g_vec = ca.DM([0.0, 0.0, m * self.g])

        # ── Arm/waist pinned; quaternion unit-norm at node 0 ──────────────────
        for qi in self._arm_qidx:
            opti.subject_to(Q[qi, 0] == float(self.q_arm_default[qi]))
        for k in range(N + 1):
            for vi in self._arm_vidx:
                opti.subject_to(V[vi, k] == 0.0)
        opti.subject_to(ca.dot(Q[3:7, 0], Q[3:7, 0]) == 1.0)

        # Settling nodes (the last n_terminal nodes, k > N - n_terminal) are
        # appended after the SRB motion ends: they keep the full dynamics /
        # contact / collision / limit machinery but drop the SRB tracking costs.
        def _is_settling(k):
            return self.n_terminal > 0 and k > (N - self.n_terminal)

        # Lead-in nodes (the first n_lead nodes, k < n_lead) are prepended before the
        # SRB motion begins: like the settling window, they keep the full dynamics /
        # contact / collision / limit machinery but drop the SRB tracking costs and are
        # pulled toward the standing config instead (plus an initial rest V_0=0).
        def _is_lead(k):
            return self.n_lead > 0 and k < self.n_lead

        # Plant-frame foot orientation for the current ground phase (free_foot_yaw): set
        # at the first planted frame of each phase (stance start / touchdown) and reused
        # to lock the heading of the rest of that phase so the planted foot cannot skid.
        R_l_td = R_r_td = None
        plant_idx = 0   # counts ground phases in plant order (for foot_world_yaw indexing)

        # ── Per-node defining constraints + tracking costs ────────────────────
        for k in range(N + 1):
            Q_k, V_k = Q[:, k], V[:, k]
            com_k, p_l, p_r, R_l, R_r = self.f_fkcom(Q_k)

            # Lifted defining constraints
            opti.subject_to(C[:, k] == com_k)                  # c = com(q)
            opti.subject_to(Hh[:, k] == self.f_hg(Q_k, V_k))   # h = A_G(q) v
            opti.subject_to(m * Cd[:, k] == Hh[0:3, k])        # m c_dot = h_linear

            # Self-collision avoidance (capsule sphere pairs), all nodes
            if self.self_collision:
                opti.subject_to(self.f_collpairs(Q_k) >= 0)

            # Free-joint (legs + sagittal arms) position / velocity box limits
            for i, qi in enumerate(self._free_qidx):
                opti.subject_to(opti.bounded(
                    float(self._free_qlo[i]), Q[qi, k], float(self._free_qhi[i])))
            for i, vi in enumerate(self._free_vidx):
                opti.subject_to(opti.bounded(
                    -float(self._free_vmax[i]), V[vi, k], float(self._free_vmax[i])))

            # Foot planting — contact nodes only (hard FK targets)
            if self._contact_node(k) and not np.isnan(self.p_foot_srb[k, 0]):
                opti.subject_to(p_l[0] == self.p_foot_srb[k, 0])
                opti.subject_to(p_l[1] == self.p_foot_srb[k, 1])
                opti.subject_to(p_l[2] == self.floor_z)
                opti.subject_to(p_r[0] == self.p_foot_srb[k, 2])
                opti.subject_to(p_r[1] == self.p_foot_srb[k, 3])
                opti.subject_to(p_r[2] == self.floor_z)

                # Feet flat AND upright (sole down) at EVERY contact node: foot z-axis ==
                # world +z.  R[0,2]=R[1,2]=0 makes the foot z-axis vertical; R[2,2]=+1 picks
                # the sole-DOWN branch — without it the foot z-axis may point down (R[2,2]=-1),
                # an inverted "flat" foot whose corners punch through the floor while
                # |R02|,|R12| stay ~0.  (Replaces the old vee(R)==0 "R symmetric" test, which
                # also admitted the whole 180-deg tilt/flip family that stabbed through.)
                for Rf in (R_l, R_r):
                    opti.subject_to(Rf[0, 2] == 0)
                    opti.subject_to(Rf[1, 2] == 0)
                    opti.subject_to(Rf[2, 2] == 1)

                # Heading (yaw): locked per ground phase at the plant frame so a planted foot
                # cannot pivot/skid.  The TWO modes differ ONLY in how the plant-frame heading
                # is chosen:
                #   free_foot_yaw=False (default) → heading FIXED to the SRB-reference body yaw
                #       at the plant frame (a constant).  Reads the maneuver's own stance/
                #       landing heading, so it generalises to any twist (180/360/...) with no
                #       hardcoded "forward".  Use when you want a determined foot heading.
                #   free_foot_yaw=True            → heading FREE within +/-foot_yaw_slack of the
                #       SOLVED body heading; let the feet choose a heading near the body.
                prev_planted = (k > 0 and self._contact_node(k - 1)
                                and not np.isnan(self.p_foot_srb[k - 1, 0]))
                if not prev_planted:
                    if self.foot_world_yaw is not None:
                        # COMMANDED heading: pin the foot forward-axis to an ABSOLUTE WORLD
                        # yaw for this ground phase (indexed in plant order; last value
                        # repeats).  Lets you enforce a twist independently of the SRB plan,
                        # e.g. foot_world_yaw=[0, pi] → stance forward, land at 180 deg.
                        psi = self.foot_world_yaw[min(plant_idx,
                                                      len(self.foot_world_yaw) - 1)]
                        hx, hy = float(np.cos(psi)), float(np.sin(psi))
                        opti.subject_to(R_l[0, 0] == hx);  opti.subject_to(R_l[1, 0] == hy)
                        opti.subject_to(R_r[0, 0] == hx);  opti.subject_to(R_r[1, 0] == hy)
                    elif self.free_foot_yaw:
                        # Body forward-axis horizontal comps from the SOLVED base quat
                        # Q[3:7] = [qx,qy,qz,qw]; allow the foot heading within +/-slack.
                        qx, qy, qz, qw = Q[3, k], Q[4, k], Q[5, k], Q[6, k]
                        b_x = 1.0 - 2.0 * (qy**2 + qz**2)
                        b_y = 2.0 * (qx * qy + qz * qw)
                        nb  = ca.sqrt(b_x**2 + b_y**2)
                        cs  = float(np.cos(self.foot_yaw_slack))
                        # foot_forward · body_forward_hat >= cos(slack)  ⟺  |Δyaw| <= slack.
                        # (foot_forward = (R[0,0],R[1,0]) is unit since the sole is flat.)
                        opti.subject_to(R_l[0, 0] * b_x + R_l[1, 0] * b_y >= cs * nb)
                        opti.subject_to(R_r[0, 0] * b_x + R_r[1, 0] * b_y >= cs * nb)
                    else:
                        # FIXED heading: foot forward-axis == SRB-reference body heading at
                        # this plant frame (constant), projected to horizontal.  quat_srb_pin
                        # is pinocchio [qx,qy,qz,qw].
                        qx, qy, qz, qw = self.quat_srb_pin[k]
                        b_x = 1.0 - 2.0 * (qy**2 + qz**2)
                        b_y = 2.0 * (qx * qy + qz * qw)
                        nb  = float(np.hypot(b_x, b_y))
                        hx, hy = (1.0, 0.0) if nb < 1e-6 else (b_x / nb, b_y / nb)
                        opti.subject_to(R_l[0, 0] == hx);  opti.subject_to(R_l[1, 0] == hy)
                        opti.subject_to(R_r[0, 0] == hx);  opti.subject_to(R_r[1, 0] == hy)
                    plant_idx += 1                   # advance ground-phase counter
                    R_l_td, R_r_td = R_l, R_r        # this phase's locked heading
                else:
                    # Already planted: lock heading to the plant frame's (world-fixed) so the
                    # foot cannot pivot/skid.  With the flat+upright rows above, pinning the
                    # forward-axis horizontal comps fixes the full R.
                    opti.subject_to(R_l[0, 0] == R_l_td[0, 0])
                    opti.subject_to(R_l[1, 0] == R_l_td[1, 0])
                    opti.subject_to(R_r[0, 0] == R_r_td[0, 0])
                    opti.subject_to(R_r[1, 0] == R_r_td[1, 0])

            # Tracking costs (s = [c, c_dot, quat, w, lambda]) — only where an
            # SRB reference exists; prepended lead-in and appended settling nodes
            # track nothing.
            if not _is_settling(k) and not _is_lead(k):
                e_c    = C[:, k]      - self.c_srb[k]
                e_cd   = Cd[:, k]     - self.cd_srb[k]
                e_quat = Q[3:7, k]    - self.quat_srb_pin[k]
                e_w    = V[3:6, k]    - self.w_body_srb[k]
                J += wp['w_com']  * ca.dot(e_c,  e_c)
                J += wp['w_cdot'] * ca.dot(e_cd, e_cd)
                J += wp['w_quat'] * ca.dot(e_quat, e_quat)
                J += wp['w_wbase']* ca.dot(e_w,  e_w)

            # Free-joint regularization.  On the appended settling window this
            # becomes a STRONG pull toward the standing config (w_qreg_term) so the
            # robot rises from the landing crouch into a static stand; elsewhere it
            # is the mild conditioning regularizer toward the warm-start config
            # (for the freed arms the warm start is the default pose, so it gently
            # keeps the arms home unless the dynamics genuinely need them).
            q_free_k = ca.vertcat(*[Q[qi, k] for qi in self._free_qidx])
            if _is_settling(k) or _is_lead(k):
                e_qreg = q_free_k - self._stand_free
                J += wp['w_qreg_term'] * ca.dot(e_qreg, e_qreg)
            else:
                q_free_warm_k = np.array([self.Q_warm[k, qi] for qi in self._free_qidx])
                e_qreg = q_free_k - q_free_warm_k
                J += wp['w_qreg'] * ca.dot(e_qreg, e_qreg)

            # Joint kinetic-energy regularizer (all nodes): minimum-effort
            # redundancy resolution — self-scoping, bites mainly in the swing
            # nullspace, damps flailing without a maneuver-specific reference.
            J += wp['w_ke'] * self.f_kej(Q_k, V_k)

            # Flight posture prior (config cost): pull the free DOF (legs at the
            # 0.9-tuck, arms at the sagittal tuck pose) toward the prior during
            # flight only, where they are underdetermined.
            if not self._contact_node(k):
                e_cfg = ca.vertcat(*[Q[qi, k] for qi in self._free_qidx]) - self._flight_prior
                J += wp['w_config'] * ca.dot(e_cfg, e_cfg)

        # ── Per-interval: integration, centroidal dynamics, wrench, lambda ────
        for k in range(N):
            dt_k = float(self.dt_vec[k])

            # Manifold config integration (trapezoidal, O(dt^2))
            v_avg = 0.5 * (V[:, k] + V[:, k + 1])
            opti.subject_to(Q[:, k + 1] == self.f_integrate(Q[:, k], v_avg, dt_k))

            # h_dot definition: h_dot = A_G(q_k) a_k + Adot_G(q_k,v_k) v_k
            a_k = (V[:, k + 1] - V[:, k]) / dt_k
            opti.subject_to(Hd[:, k] == self.f_hgdot(Q[:, k], V[:, k], a_k))

            # Newton-Euler from the contact polygon: each corner point applies a
            # 3D force at its FK world position; the moment about the CoM is the
            # sum of r_i x f_i (no lumped ankle torque — that emerges from CoP).
            is_flight = (self.stance_end <= k < self.flight_end)
            if is_flight:
                opti.subject_to(Lam[:, k] == 0)        # no contact in flight
            _, p_l, p_r, R_l, R_r = self.f_fkcom(Q[:, k])

            # Friction: feet are flat on the floor, so the contact normal is world
            # +z.  Per point: unilateral f_z >= 0 (no pulling) and an INNER
            # (diamond) friction pyramid |f_x| + |f_y| <= mu f_z.  The inner
            # pyramid guarantees the true cone |f_t| <= mu f_z is respected (the
            # box pyramid would allow up to sqrt(2)*mu), and being linear it has
            # no cone-apex gradient degeneracy at the f_z=0 points.  Contact only.
            mu = self.mu
            def _friction(f):
                opti.subject_to(f[2] >= 0.0)
                opti.subject_to( f[0] + f[1] <= mu * f[2])
                opti.subject_to( f[0] - f[1] <= mu * f[2])
                opti.subject_to(-f[0] + f[1] <= mu * f[2])
                opti.subject_to(-f[0] - f[1] <= mu * f[2])

            lin = -g_vec
            ang = ca.DM.zeros(3)
            for i in range(N_PT):
                corner = ca.DM(FOOT_CORNERS[i])
                f_Li = Lam[3 * i:3 * i + 3, k]
                f_Ri = Lam[3 * N_PT + 3 * i:3 * N_PT + 3 * i + 3, k]
                p_Li = p_l + R_l @ corner
                p_Ri = p_r + R_r @ corner
                lin = lin + f_Li + f_Ri
                ang = ang + ca.cross(p_Li - C[:, k], f_Li) \
                          + ca.cross(p_Ri - C[:, k], f_Ri)
                if not is_flight:
                    _friction(f_Li)
                    _friction(f_Ri)
            opti.subject_to(Hd[:, k] == ca.vertcat(lin, ang))

            # Wrench tracking (skip on lead-in / settling intervals — no SRB reference) +
            # velocity smoothness (kept everywhere; helps damp the settle / lead-in).
            # Settling intervals are k >= N - n_terminal; lead-in intervals are k < n_lead.
            _settling_iv = self.n_terminal > 0 and k >= (N - self.n_terminal)
            _lead_iv     = self.n_lead > 0 and k < self.n_lead
            if not (_settling_iv or _lead_iv):
                e_lam = Lam[:, k] - self.lam_srb[k]
                J += wp['w_lam'] * ca.dot(e_lam, e_lam)
            e_v = V[:, k + 1] - V[:, k]
            J += wp['w_vsmooth'] * ca.dot(e_v, e_v)

        # ── Terminal rest constraint ──────────────────────────────────────────
        # After the appended settling window the robot must come to a full stop.
        # Feet are already pinned by the contact constraints, so V_N = 0 lands the
        # trajectory in a static stand.
        if self.n_terminal > 0:
            opti.subject_to(V[:, N] == 0)

        # ── Initial rest constraint ───────────────────────────────────────────
        # Before the prepended standing lead-in the robot starts from a full stop.
        # Feet are already pinned by the contact constraints, so V_0 = 0 starts the
        # trajectory in a static stand (mirror of the terminal V_N = 0).
        if self.n_lead > 0:
            opti.subject_to(V[:, 0] == 0)

        # ── Warm start ────────────────────────────────────────────────────────
        opti.set_initial(Q, self.Q_warm.T)
        opti.set_initial(V, self.V_warm.T)
        # Lifted variables: seed from the warm-start configuration/velocity.
        C_warm  = np.zeros((3, N + 1))
        Cd_warm = np.zeros((3, N + 1))
        H_warm  = np.zeros((6, N + 1))
        for k in range(N + 1):
            C_warm[:, k] = np.array(self.f_com(self.Q_warm[k])).flatten()
            H_warm[:, k] = np.array(self.f_hg(self.Q_warm[k], self.V_warm[k])).flatten()
            Cd_warm[:, k] = H_warm[0:3, k] / m
        opti.set_initial(C,  C_warm)
        opti.set_initial(Cd, Cd_warm)
        opti.set_initial(Hh, H_warm)
        opti.set_initial(Lam, self.lam_srb.T)
        # Seed h_dot from a finite difference of the warm-start momentum (was 0).
        Hd_warm = np.zeros((6, N))
        for k in range(N):
            Hd_warm[:, k] = (H_warm[:, k + 1] - H_warm[:, k]) / float(self.dt_vec[k])
        opti.set_initial(Hd, Hd_warm)

        opti.minimize(J)

        ipopt_opts = {
            "expand":                               self.expand,
            "error_on_fail":                        False,  # return iterate, don't raise
            "ipopt.linear_solver":                  self.linear_solver,
            "ipopt.max_iter":                       self.max_iter,
            "ipopt.tol":                            1e-4,
            "ipopt.acceptable_tol":                 1e-3,
            "ipopt.acceptable_constr_viol_tol":     1e-3,
            "ipopt.acceptable_dual_inf_tol":        1e1,
            "ipopt.acceptable_iter":                20,
            "ipopt.hessian_approximation":          self.hessian_mode,
            "ipopt.limited_memory_max_history":     30,
            "ipopt.mu_strategy":                    self.mu_strategy,
            "ipopt.print_level":                    5,
            "print_time":                           1,
        }
        # HSL solvers (ma27/ma57/ma77/ma86/ma97) are loaded by IPOPT at runtime
        # from a shared library.  Point IPOPT at the CoinHSL build explicitly so
        # it resolves regardless of the loader search path; harmless for mumps.
        if self.linear_solver.startswith("ma"):
            hsllib = _resolve_hsllib()
            if hsllib:
                ipopt_opts["ipopt.hsllib"] = hsllib
                print(f"[KinoNLP] HSL linear solver '{self.linear_solver}' "
                      f"via {hsllib}")
            else:
                print(f"[KinoNLP] WARNING: linear_solver='{self.linear_solver}' "
                      "but no libhsl.so found (set IPOPT_HSLLIB) — IPOPT will "
                      "fall back / error.")
        opti.solver("ipopt", ipopt_opts)

        # Reusable solver: weights in, solved trajectory out.  This is what gets
        # serialized — the expensive symbolic build (incl. the exact Hessian) is
        # done once here and skipped on later loads.
        t0 = time.time()
        self._f_solve = opti.to_function(
            'kino_solve', [w_par], [Q, V, C, Cd, Hh, Hd, Lam],
            ['w'], ['Q', 'V', 'C', 'Cd', 'Hh', 'Hd', 'Lam'])
        print(f"[KinoNLP] Built solver function in {time.time()-t0:.1f}s")
        if self.use_cache:
            self._f_solve.save(self._cache_path)
            print(f"[KinoNLP] Saved solver cache → {os.path.basename(self._cache_path)}")

    # -------------------------------------------------------------------------

    def solve(self):
        """
        Returns:
            Q_sol  (N+1, nq), V_sol (N+1, nv), success bool

        Runs the (possibly cached) solver function with the current weights.
        Stores the solved lifted variables on self for post-solve comparison:
        Lam_sol (N,24), C_sol (N+1,3), Cd_sol, Hh_sol (N+1,6), Hd_sol (N,6).
        """
        outs = self._f_solve(ca.DM(self._weight_vec()))
        Q_sol        = np.array(outs[0]).T
        V_sol        = np.array(outs[1]).T
        self.C_sol   = np.array(outs[2]).T
        self.Cd_sol  = np.array(outs[3]).T
        self.Hh_sol  = np.array(outs[4]).T
        self.Hd_sol  = np.array(outs[5]).T
        self.Lam_sol = np.array(outs[6]).T
        success = True
        try:
            success = bool(self._f_solve.stats().get("success", True))
        except Exception:
            pass
        return Q_sol, V_sol, success

    # -------------------------------------------------------------------------
    # Verification helpers
    # -------------------------------------------------------------------------

    def com_residuals(self, Q_traj, V_traj=None):
        """Per-node ||com(q) - c_srb||."""
        res = np.zeros(self.N + 1)
        for k in range(self.N + 1):
            com = np.array(self.f_com(Q_traj[k])).flatten()
            res[k] = np.linalg.norm(com - self.c_srb[k])
        return res

    def cdot_residuals(self, Q_traj, V_traj):
        """Per-node ||h_linear/m - cd_srb||."""
        res = np.zeros(self.N + 1)
        for k in range(self.N + 1):
            h = np.array(self.f_hg(Q_traj[k], V_traj[k])).flatten()
            res[k] = np.linalg.norm(h[0:3] / self.mass - self.cd_srb[k])
        return res

    def momentum_dynamics_residuals(self, Q_traj, V_traj, lam=None):
        """Per-interval ||h_dot(q,v,a) - wrench(lambda)||: analytic h_dot from the
        solved (q,v) vs the contact-polygon wrench from the point forces.  Uses the
        solved lambda if given, else the SRB reference point forces.  Diagnostic."""
        if lam is None:
            lam = self.lam_srb
        res = np.zeros(self.N)
        g_vec = np.array([0.0, 0.0, self.mass * self.g])
        for k in range(self.N):
            dt_k = self.dt_vec[k]
            a_k  = (V_traj[k + 1] - V_traj[k]) / dt_k
            hd   = np.array(self.f_hgdot(Q_traj[k], V_traj[k], a_k)).flatten()
            _, p_l, p_r, R_l, R_r = [np.array(x) for x in self.f_fkcom(Q_traj[k])]
            p_l, p_r = p_l.flatten(), p_r.flatten()
            c    = np.array(self.f_com(Q_traj[k])).flatten()
            lin = -g_vec.copy()
            ang = np.zeros(3)
            for i, corner in enumerate(FOOT_CORNERS):
                f_Li = lam[k, 3 * i:3 * i + 3]
                f_Ri = lam[k, 3 * N_PT + 3 * i:3 * N_PT + 3 * i + 3]
                p_Li = p_l + R_l @ corner
                p_Ri = p_r + R_r @ corner
                lin = lin + f_Li + f_Ri
                ang = ang + np.cross(p_Li - c, f_Li) + np.cross(p_Ri - c, f_Ri)
            res[k] = np.linalg.norm(hd - np.concatenate([lin, ang]))
        return res

    def foot_xy_residuals(self, Q_traj):
        """Per-node max foot XY error for contact frames (NaN for flight)."""
        res = np.full(self.N + 1, np.nan)
        for k in range(self.N + 1):
            if not self._contact_node(k) or np.isnan(self.p_foot_srb[k, 0]):
                continue
            _, p_l, p_r, _, _ = [np.array(x).flatten() for x in self.f_fkcom(Q_traj[k])]
            e_l = np.linalg.norm(p_l[:2] - self.p_foot_srb[k, 0:2])
            e_r = np.linalg.norm(p_r[:2] - self.p_foot_srb[k, 2:4])
            res[k] = max(e_l, e_r)
        return res

    # -------------------------------------------------------------------------
    # Post-solve summary: per-term objective cost + grouped constraint violation
    # -------------------------------------------------------------------------

    def _is_settling_node(self, k):
        return self.n_terminal > 0 and k > (self.N - self.n_terminal)

    def _is_settling_interval(self, k):
        return self.n_terminal > 0 and k >= (self.N - self.n_terminal)

    def _is_lead_node(self, k):
        return self.n_lead > 0 and k < self.n_lead

    def _is_lead_interval(self, k):
        return self.n_lead > 0 and k < self.n_lead

    def cost_breakdown(self, Q, V, Lam=None, C=None, Cd=None):
        """Recompute every weighted objective term from a solved (Q, V, Lam),
        matching _build_opti term-for-term.  C/Cd default to the FK-consistent
        values (com(q), h_linear/m), which the defining constraints enforce.
        Returns an ordered dict of weighted term -> cost, plus 'total'.
        """
        N, m = self.N, self.mass
        Lam = (self.Lam_sol if (Lam is None and hasattr(self, "Lam_sol")) else Lam)
        if Lam is None:
            Lam = self.lam_srb
        if C is None:
            C = np.array([np.array(self.f_com(Q[k])).flatten() for k in range(N + 1)])
        if Cd is None:
            Cd = np.array([np.array(self.f_hg(Q[k], V[k])).flatten()[0:3] / m
                           for k in range(N + 1)])

        c = {k: 0.0 for k in ('com', 'cdot', 'quat', 'wbase',
                              'qreg', 'qreg_term', 'ke', 'config', 'lam', 'vsmooth')}
        for k in range(N + 1):
            if not self._is_settling_node(k) and not self._is_lead_node(k):
                e = C[k] - self.c_srb[k];            c['com']   += self.w_com   * e @ e
                e = Cd[k] - self.cd_srb[k];          c['cdot']  += self.w_cdot  * e @ e
                e = Q[k, 3:7] - self.quat_srb_pin[k];c['quat']  += self.w_quat  * e @ e
                e = V[k, 3:6] - self.w_body_srb[k];  c['wbase'] += self.w_wbase * e @ e
            q_free = Q[k, self._free_qidx]
            if self._is_settling_node(k) or self._is_lead_node(k):
                e = q_free - self._stand_free;       c['qreg_term'] += self.w_qreg_term * e @ e
            else:
                e = q_free - self.Q_warm[k, self._free_qidx]
                c['qreg'] += self.w_qreg * e @ e
            c['ke'] += self.w_ke * float(self.f_kej(Q[k], V[k]))
            if not self._contact_node(k):
                e = q_free - self._flight_prior;     c['config'] += self.w_config * e @ e
        for k in range(N):
            if not self._is_settling_interval(k) and not self._is_lead_interval(k):
                e = Lam[k] - self.lam_srb[k];        c['lam'] += self.w_lam * e @ e
            e = V[k + 1] - V[k];                     c['vsmooth'] += self.w_vsmooth * e @ e
        c['total'] = float(sum(c.values()))
        return c

    def constraint_violations(self, Q, V, Lam=None, C=None, Cd=None, Hh=None, Hd=None):
        """Grouped constraint-violation magnitudes for a solved trajectory.

        Each group maps to a numpy array of per-node (or per-interval) violation
        magnitudes (0 = satisfied).  Uses the solved lifted variables (self.*_sol)
        when present so the defining-constraint residuals reflect what IPOPT
        actually returned; otherwise it FK-recomputes them (then those groups are
        ~0 by construction).  Returns an ordered dict {group_name: np.ndarray}.
        """
        N, m = self.N, self.mass
        Lam = (self.Lam_sol if (Lam is None and hasattr(self, "Lam_sol")) else Lam)
        if Lam is None:
            Lam = self.lam_srb
        if C is None:  C  = getattr(self, "C_sol", None)
        if Cd is None: Cd = getattr(self, "Cd_sol", None)
        if Hh is None: Hh = getattr(self, "Hh_sol", None)
        if Hd is None: Hd = getattr(self, "Hd_sol", None)
        if C is None:
            C = np.array([np.array(self.f_com(Q[k])).flatten() for k in range(N + 1)])
        if Hh is None:
            Hh = np.array([np.array(self.f_hg(Q[k], V[k])).flatten() for k in range(N + 1)])
        if Cd is None:
            Cd = Hh[:, 0:3] / m
        if Hd is None:
            Hd = np.zeros((N, 6))
            for k in range(N):
                a = (V[k + 1] - V[k]) / self.dt_vec[k]
                Hd[k] = np.array(self.f_hgdot(Q[k], V[k], a)).flatten()

        g_vec = np.array([0.0, 0.0, m * self.g])
        gr = {}

        # Lifted defining constraints
        gr['c = com(q)']      = np.array([np.linalg.norm(C[k] - np.array(self.f_com(Q[k])).flatten())
                                          for k in range(N + 1)])
        gr['h = A_G v']       = np.array([np.linalg.norm(Hh[k] - np.array(self.f_hg(Q[k], V[k])).flatten())
                                          for k in range(N + 1)])
        gr['m*cdot = h_lin']  = np.array([np.linalg.norm(m * Cd[k] - Hh[k, 0:3]) for k in range(N + 1)])

        # h_dot definition (vs analytic A_G a + Adot_G v) and Newton-Euler (vs wrench)
        hdot_def = np.zeros(N); newton = np.zeros(N)
        for k in range(N):
            a = (V[k + 1] - V[k]) / self.dt_vec[k]
            hdot_def[k] = np.linalg.norm(Hd[k] - np.array(self.f_hgdot(Q[k], V[k], a)).flatten())
            _, p_l, p_r, R_l, R_r = [np.array(x) for x in self.f_fkcom(Q[k])]
            p_l, p_r = p_l.flatten(), p_r.flatten()
            lin = -g_vec.copy(); ang = np.zeros(3)
            for i, corner in enumerate(FOOT_CORNERS):
                f_L = Lam[k, 3 * i:3 * i + 3]
                f_R = Lam[k, 3 * N_PT + 3 * i:3 * N_PT + 3 * i + 3]
                p_Li = p_l + R_l @ corner; p_Ri = p_r + R_r @ corner
                lin = lin + f_L + f_R
                ang = ang + np.cross(p_Li - C[k], f_L) + np.cross(p_Ri - C[k], f_R)
            newton[k] = np.linalg.norm(Hd[k] - np.concatenate([lin, ang]))
        gr['hdot = A_G a+bias'] = hdot_def
        gr['Newton-Euler']      = newton

        # Manifold integration defect
        integ = np.zeros(N)
        for k in range(N):
            v_avg = 0.5 * (V[k] + V[k + 1])
            q_next = np.array(self.f_integrate(Q[k], v_avg, float(self.dt_vec[k]))).flatten()
            integ[k] = np.linalg.norm(pin.difference(self.model, q_next, Q[k + 1]))
        gr['integration'] = integ

        # Foot planting (position + flatness) on contact nodes
        fpos, fflat = [], []
        for k in range(N + 1):
            if not self._contact_node(k) or np.isnan(self.p_foot_srb[k, 0]):
                continue
            _, p_l, p_r, R_l, R_r = [np.array(x) for x in self.f_fkcom(Q[k])]
            p_l, p_r = p_l.flatten(), p_r.flatten()
            fpos.append(max(abs(p_l[0] - self.p_foot_srb[k, 0]),
                            abs(p_l[1] - self.p_foot_srb[k, 1]),
                            abs(p_l[2] - self.floor_z),
                            abs(p_r[0] - self.p_foot_srb[k, 2]),
                            abs(p_r[1] - self.p_foot_srb[k, 3]),
                            abs(p_r[2] - self.floor_z)))
            # flat + upright at every contact node (BOTH modes): foot z-axis == world +z,
            # i.e. |R02|, |R12|, and 1 - R22 all ~0.  (Heading/skid is enforced separately
            # and is mode/phase dependent, so it is not folded into this flatness metric.)
            fflat.append(max(abs(R_l[0, 2]), abs(R_l[1, 2]), abs(1.0 - R_l[2, 2]),
                             abs(R_r[0, 2]), abs(R_r[1, 2]), abs(1.0 - R_r[2, 2])))
        gr['foot position'] = np.array(fpos)
        gr['foot flatness'] = np.array(fflat)

        # Contact-force friction / unilateral and flight lambda = 0
        fric, flight = [], []
        for k in range(N):
            if self.stance_end <= k < self.flight_end:
                flight.append(np.linalg.norm(Lam[k]))
                continue
            v = 0.0
            for i in range(2 * N_PT):
                f = Lam[k, 3 * i:3 * i + 3]
                v = max(v, -f[2],
                        f[0] + f[1] - self.mu * f[2], f[0] - f[1] - self.mu * f[2],
                        -f[0] + f[1] - self.mu * f[2], -f[0] - f[1] - self.mu * f[2])
            fric.append(max(0.0, v))
        gr['friction/unilateral'] = np.array(fric)
        gr['flight lambda = 0']   = np.array(flight)

        # Free-DOF box limits (position + velocity)
        qf = Q[:, self._free_qidx]
        qlim = np.maximum(np.maximum(0.0, self._free_qlo[None, :] - qf),
                          np.maximum(0.0, qf - self._free_qhi[None, :])).max(axis=1)
        vf = V[:, self._free_vidx]
        vlim = np.maximum(0.0, np.abs(vf) - self._free_vmax[None, :]).max(axis=1)
        gr['joint pos limits'] = qlim
        gr['joint vel limits'] = vlim

        # Self-collision (squared-distance margin units, m^2)
        if self.self_collision:
            coll = np.array([max(0.0, -np.array(self.f_collpairs(Q[k])).flatten().min())
                             for k in range(N + 1)])
            gr['self-collision'] = coll

        # Terminal rest
        if self.n_terminal > 0:
            gr['terminal V_N=0'] = np.array([np.linalg.norm(V[self.N])])

        # Initial rest (prepended standing lead-in)
        if self.n_lead > 0:
            gr['initial V_0=0'] = np.array([np.linalg.norm(V[0])])

        return gr

    def solver_stats(self):
        """IPOPT-reported final objective / primal infeasibility / status from the
        last solve (empty dict if unavailable, e.g. after --skip-solve)."""
        f = getattr(self, "_f_solve", None)
        if f is None:
            return {}
        try:
            s = f.stats()
        except Exception:
            return {}
        out = {'success': s.get('success'), 'return_status': s.get('return_status')}
        it = s.get('iterations') or {}
        for key in ('obj', 'inf_pr', 'inf_du'):
            seq = it.get(key)
            if seq:
                out[key] = seq[-1]
        out['iter_count'] = (len(it['obj']) if it.get('obj') else s.get('iter_count'))
        return out

    def format_summary(self, Q, V, Lam=None, stats=None, header=None,
                       C=None, Cd=None, Hh=None, Hd=None):
        """Human-readable summary table for a solved trajectory: per-term
        weighted objective cost + total, and grouped constraint violations.
        `header` is a list of extra lines printed at the top; `stats` an optional
        solver_stats() dict.  Returns (text, costs, viol).
        """
        costs = self.cost_breakdown(Q, V, Lam, C=C, Cd=Cd)
        viol  = self.constraint_violations(Q, V, Lam, C=C, Cd=Cd, Hh=Hh, Hd=Hd)
        total = costs['total']
        L = []
        L.append("=" * 60)
        L.append("           KINO OPTIMIZATION SUMMARY")
        L.append("=" * 60)
        for h in (header or []):
            L.append(h)
        if stats:
            sline = (f"solver success = {stats.get('success')}"
                     f"   status = {stats.get('return_status')}"
                     f"   iters = {stats.get('iter_count')}")
            L.append(sline)
            if 'obj' in stats:
                L.append(f"IPOPT final objective      : {stats['obj']:.6g}")
            if 'inf_pr' in stats:
                L.append(f"IPOPT primal infeasibility : {stats['inf_pr']:.3e}"
                         "   (max constraint violation, scaled)")
        L.append("")
        L.append("-- Total cost per trajectory (weighted objective terms) --")
        L.append(f"  {'term':<14}{'cost':>14}{'% of total':>12}")
        for k, v in costs.items():
            if k == 'total':
                continue
            pct = (100.0 * v / total) if total else 0.0
            L.append(f"  {k:<14}{v:>14.4g}{pct:>11.1f}%")
        L.append(f"  {'TOTAL':<14}{total:>14.4g}{100.0:>11.1f}%")
        L.append("")
        L.append("-- Constraint violations (recomputed from solution) --")
        L.append(f"  {'group':<22}{'mean':>13}{'max':>13}")
        overall_max = 0.0
        for g, arr in viol.items():
            if arr is None or len(arr) == 0:
                L.append(f"  {g:<22}{'(n/a)':>13}{'(n/a)':>13}")
                continue
            mx = float(np.max(arr))
            overall_max = max(overall_max, mx)
            L.append(f"  {g:<22}{float(np.mean(arr)):>13.3e}{mx:>13.3e}")
        L.append(f"  {'OVERALL MAX':<22}{'':>13}{overall_max:>13.3e}")
        L.append("=" * 60)
        return "\n".join(L), costs, viol
