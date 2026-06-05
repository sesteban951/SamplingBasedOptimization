##
#
# Generalized Aerial Maneuver Optimizer for Single Rigid Body.
# Supports arbitrary roll/pitch/yaw in-air maneuvers via config.
#
# Usage: python -m srb.srb_aerial srb.config.twist_360
#
##

# standard imports
import sys
import importlib
import numpy as np
import os

# casadi import
import casadi as ca

# custom imports
from utils.kinematics import kin
from utils.interpolation import interp
from srb.srb import SRBDynamics

_WORKSPACE_COEFFS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "ik", "results", "workspace_poly_coeffs.csv",
)
_WORKSPACE_COEFFS_2D = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "ik", "results", "workspace_poly_coeffs_2d.csv",
)
_WORKSPACE_COEFFS_2D_UPPER = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "ik", "results", "workspace_poly_coeffs_2d_upper.csv",
)


class SRB_Aerial(SRBDynamics):

    def __init__(self, costs):

        super().__init__()

        # simple quadratic penalty on states
        self.Qx = ca.diag(ca.vertcat(*costs.Qx_diag))

        # foot placement cost weights
        self.Q_foot = costs.Q_foot
        self.Q_foot_world = costs.Q_foot_world

        # penalize forces and moments
        self.Q_force = costs.Q_force
        self.Q_moment = costs.Q_moment
        self.Q_force_dot = costs.Q_force_dot
        self.Q_moment_dot = costs.Q_moment_dot

        # terminal weights
        self.Qx_f = costs.Qx_terminal_scale * self.Qx

    ###############################################################
    # Cost Functions
    ###############################################################

    def state_cost(self, x, x_goal):
        pos_err = x[0:3] - x_goal[0:3]
        vel_err = x[7:10] - x_goal[7:10]
        omega_err = x[10:13] - x_goal[10:13]

        quat_err = kin.quat_diff_ca(x[3:7], x_goal[3:7])
        quat_err_log = kin.quat_log_ca(quat_err)

        e_x = ca.vertcat(pos_err, quat_err_log, vel_err, omega_err)
        cost = 0.5 * e_x.T @ self.Qx @ e_x
        return cost

    def contact_cost(self, F_L, F_R, M_L, M_R):
        cost = (
            0.5 * self.Q_force * (ca.sumsqr(F_L) + ca.sumsqr(F_R))
          + 0.5 * self.Q_moment * (ca.sumsqr(M_L) + ca.sumsqr(M_R))
        )
        return cost

    def force_rate_cost(self, F_L_k, F_R_k, M_L_k, M_R_k,
                              F_L_k1, F_R_k1, M_L_k1, M_R_k1, dt):
        dF_L = (F_L_k1 - F_L_k) / dt
        dF_R = (F_R_k1 - F_R_k) / dt
        dM_L = (M_L_k1 - M_L_k) / dt
        dM_R = (M_R_k1 - M_R_k) / dt
        return (
            0.5 * self.Q_force_dot * (ca.sumsqr(dF_L) + ca.sumsqr(dF_R))
          + 0.5 * self.Q_moment_dot * (ca.sumsqr(dM_L) + ca.sumsqr(dM_R))
        )

    def foot_placement_cost(self, p_L, p_R, p_L_des, p_R_des):
        cost = (
            0.5 * self.Q_foot * ca.sumsqr(p_L - p_L_des)
          + 0.5 * self.Q_foot * ca.sumsqr(p_R - p_R_des)
        )
        return cost

    def world_foot_placement_cost(self, p_L_W, p_R_W, p_L_des_W, p_R_des_W):
        cost = (
            0.5 * self.Q_foot_world * ca.sumsqr(p_L_W - p_L_des_W)
          + 0.5 * self.Q_foot_world * ca.sumsqr(p_R_W - p_R_des_W)
        )
        return cost

    def terminal_cost(self, x, x_goal):
        pos_err = x[0:3] - x_goal[0:3]
        vel_err = x[7:10] - x_goal[7:10]
        omega_err = x[10:13] - x_goal[10:13]

        quat_err = kin.quat_diff_ca(x[3:7], x_goal[3:7])
        quat_err_log = kin.quat_log_ca(quat_err)

        e = ca.vertcat(pos_err, quat_err_log, vel_err, omega_err)
        cost_terminal = e.T @ self.Qx_f @ e
        return cost_terminal


##############################################################
# Load Config
##############################################################

if len(sys.argv) < 2:
    print("Usage: python -m srb.srb_aerial <config_module>")
    print("  e.g. python -m srb.srb_aerial srb.config.twist_360")
    sys.exit(1)

config_module = importlib.import_module(sys.argv[1])
cfg = config_module.config
print(f"[dbg] config loaded: {sys.argv[1]}", flush=True)

# World-frame ground heights for each contact phase.
# Config values for p_com z and pz_min are relative to these surfaces;
# we add the offsets here so the rest of the file works in world frame.
_stance_gz  = cfg.constraints.stance_ground_z
_landing_gz = cfg.constraints.landing_ground_z

_var_inertia = cfg.solver.variable_inertia

_TUCK_SYM_NPZ = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "ik", "results", "inertia_tuck_sym.npz",
)

if _var_inertia:
    if not os.path.exists(_TUCK_SYM_NPZ):
        raise FileNotFoundError(
            f"variable_inertia=True but symmetric tuck curve not found: {_TUCK_SYM_NPZ}\n"
            "Run: conda run -n env_sbo python ik/sample_inertia_workspace.py"
        )
    _sym = np.load(_TUCK_SYM_NPZ)
    _tuck_grid = _sym['tuck']                       # (N_TUCK,)

    # Degree-3 polynomial fits from the symmetric sweep (secondary DOFs = 0).
    # Symmetric sweep gives a clean monotone curve that matches the actual
    # backflip leg configuration — no lateral spread noise from random DOF sampling.
    # Delta formulation: I_full(t) = srb.I + (I_poly(t) - I_poly(0))
    # ensures I_full(0) == srb.I exactly → no discontinuity at phase boundaries.
    _poly_Ixx = np.polyfit(_tuck_grid, _sym['Ixx_sym'], 3)
    _poly_Iyy = np.polyfit(_tuck_grid, _sym['Iyy_sym'], 3)
    _poly_Izz = np.polyfit(_tuck_grid, _sym['Izz_sym'], 3)
    _Ixx0 = float(np.polyval(_poly_Ixx, 0.0))
    _Iyy0 = float(np.polyval(_poly_Iyy, 0.0))
    _Izz0 = float(np.polyval(_poly_Izz, 0.0))

    _poly_fL = [np.polyfit(_tuck_grid, _sym['foot_pos_L_sym'][:, i], 3) for i in range(3)]
    _poly_fR = [np.polyfit(_tuck_grid, _sym['foot_pos_R_sym'][:, i], 3) for i in range(3)]

    print(f"[dbg] tuck sym curve: Iyy ∈ [{_sym['Iyy_sym'].min():.3f}, {_sym['Iyy_sym'].max():.3f}] kg·m²")
    print(f"[dbg] delta Iyy at full tuck: {float(np.polyval(_poly_Iyy, 1.0)) - _Iyy0:+.3f} kg·m²")

##############################################################
# Build initial and goal states from config
##############################################################

# Initial state
roll0, pitch0, yaw0 = np.radians(cfg.initial.rpy_deg)
quat0 = kin.euler_ZYX_to_quat(roll0, pitch0, yaw0)
x0 = np.array([
    *cfg.initial.p_com,
    quat0[0], quat0[1], quat0[2], quat0[3],
    *cfg.initial.v_com,
    *cfg.initial.w_body,
])

# Goal state
roll_g, pitch_g, yaw_g = np.radians(cfg.goal.rpy_deg)
quat_goal = kin.euler_ZYX_to_quat(roll_g, pitch_g, yaw_g)
x_goal = np.array([
    *cfg.goal.p_com,
    quat_goal[0], quat_goal[1], quat_goal[2], quat_goal[3],
    *cfg.goal.v_com,
    *cfg.goal.w_body,
])

# Lift config-relative heights into world frame.
x0[2]     += _stance_gz   # p_com z was above stance surface
x_goal[2] += _landing_gz  # goal  z was above landing surface

x0_ca = ca.DM(x0)
x_goal_ca = ca.DM(x_goal)

##############################################################
# Trajectory Optimization
##############################################################

print("[dbg] creating SRB_Aerial...", flush=True)
srb = SRB_Aerial(cfg.costs)
f = srb.f_disc
nq = srb.nq
nv = srb.nv
nx = nq + nv
nu = srb.nu

# nominal timings
dt_nom = cfg.timing.dt_nom
T_stance_nom = cfg.timing.T_stance_nom
T_flight_nom = cfg.timing.T_flight_nom
T_land_nom = cfg.timing.T_land_nom
T_nom = T_stance_nom + T_flight_nom + T_land_nom

# number of steps
N_stance = int(T_stance_nom / dt_nom)
N_flight = int(T_flight_nom / dt_nom)
N_land = int(T_land_nom / dt_nom)
N = N_stance + N_flight + N_land

# phase boundaries
stance_end = N_stance
flight_end = N_stance + N_flight

##############################################################
# SLERP keyframes from maneuver config (relative rotation)
##############################################################

roll_man, pitch_man, yaw_man = np.radians(cfg.maneuver.rpy_deg)
quat_slerp_keyframes = interp.build_general_slerp_keyframes(
    quat0, roll_man, pitch_man, yaw_man
)

##############################################################
# Precompute flight-phase angular velocity reference (body frame)
##############################################################

alpha_samples = [j / N_flight for j in range(N_flight + 1)]
quat_samples = [interp.sample_piecewise_slerp(a, quat_slerp_keyframes) for a in alpha_samples]

# Per-segment rotation in body frame (rad, not divided by time)
omega_ref_body = []
for j in range(len(quat_samples) - 1):
    q_diff = kin.quat_diff(quat_samples[j], quat_samples[j + 1])
    log_diff_world = kin.quat_log(q_diff)
    log_diff_body = kin.world_to_body(log_diff_world, quat_samples[j])
    omega_ref_body.append(log_diff_body)

##############################################################
# Desired foot positions
##############################################################

p0_L = np.array([0, srb.hip_offset])
p0_R = np.array([0, -srb.hip_offset])
p0_L_ca = ca.DM(p0_L)
p0_R_ca = ca.DM(p0_R)

p_L_goal = ca.DM([0.0, srb.hip_offset])
p_R_goal = ca.DM([0.0, -srb.hip_offset])

# World-frame foot goals from goal COM state
yaw_goal_val = kin.quat_to_yaw(quat_goal)
Rz_goal = kin.yaw_to_rot_matrix(yaw_goal_val)
p_com_goal_xy = np.array(cfg.goal.p_com[:2])
p_L_goal_W = ca.DM(p_com_goal_xy + (Rz_goal @ np.array([0.0, srb.hip_offset, 0.0]))[:2])
p_R_goal_W = ca.DM(p_com_goal_xy + (Rz_goal @ np.array([0.0, -srb.hip_offset, 0.0]))[:2])

##############################################################
# Setup the optimization problem
##############################################################

def _to_I_mat(i6):
    """6-vector [Ixx,Iyy,Izz,Ixy,Ixz,Iyz] → symmetric 3×3 (CasADi or numpy)."""
    return ca.vertcat(
        ca.horzcat(i6[0], i6[3], i6[4]),
        ca.horzcat(i6[3], i6[1], i6[5]),
        ca.horzcat(i6[4], i6[5], i6[2]),
    )

def _polyval_ca(coeffs, t_sym):
    """Horner's method polynomial evaluation — works with CasADi symbolics."""
    result = ca.DM(float(coeffs[0]))
    for c in coeffs[1:]:
        result = result * t_sym + float(c)
    return result

def _I_mat_from_tuck(t_sym):
    """Full robot inertia = srb.I + delta_I(tuck).  delta=0 at tuck=0 by construction.
    All three diagonals vary: Ixx/Iyy decrease with tuck (sagittal leg tuck),
    Izz increases slightly (mass moves forward in sagittal plane).
    Twist-axis (Izz) control via arm configuration is a separate future extension."""
    I_np = np.array(srb.I)
    Ixx = float(I_np[0, 0]) + _polyval_ca(_poly_Ixx, t_sym) - _Ixx0
    Iyy = float(I_np[1, 1]) + _polyval_ca(_poly_Iyy, t_sym) - _Iyy0
    Izz = float(I_np[2, 2]) + _polyval_ca(_poly_Izz, t_sym) - _Izz0
    return ca.vertcat(
        ca.horzcat(Ixx, 0.0, 0.0),
        ca.horzcat(0.0, Iyy, 0.0),
        ca.horzcat(0.0, 0.0, Izz),
    )

print("[dbg] building opti...", flush=True)
opti = ca.Opti()

# horizon variables
X = opti.variable(nx, N + 1)

# phase duration decision variables
T_stance = opti.variable()
T_flight = opti.variable()
T_land = opti.variable()

# phase-specific time steps
dt_stance = T_stance / N_stance
dt_flight = T_flight / N_flight
dt_land = T_land / N_land

# decision variables
p_L_land = opti.variable(2)
p_R_land = opti.variable(2)
F_L = opti.variable(3, N)
F_R = opti.variable(3, N)
M_L = opti.variable(3, N)
M_R = opti.variable(3, N)

# Tuck parameter t_k ∈ [0,1] at each flight node.
# t=0: legs extended (liftoff/touchdown); t=1: fully tucked.
# Inertia I(t) and foot positions foot(t) are derived from tuck via polynomial fits,
# ensuring physical consistency between the aerial configuration and body inertia.
if _var_inertia:
    tuck_flight = opti.variable(N_flight + 1)   # one scalar per flight node
    opti.subject_to(opti.bounded(0.0, tuck_flight, 1.0))
    opti.subject_to(tuck_flight[0] == 0.0)              # legs extended at liftoff
    opti.subject_to(tuck_flight[N_flight] == 0.0)       # legs extended at touchdown
    if cfg.solver.max_tuck_dot is not None:
        _dt_max = cfg.solver.max_tuck_dot * dt_nom
        for k in range(N_flight):
            opti.subject_to(opti.bounded(-_dt_max, tuck_flight[k + 1] - tuck_flight[k], _dt_max))
    # Angular momentum L = I·ω in body frame (auxiliary variable — avoids ca.solve).
    # Dynamics: L_{k+1} = L_k + dt*M_net_B  (linear, enforced in dynamics loop).
    # Coupling: I_mat(tuck_k) @ w_k = L_k   (bilinear, enforced after loop).
    L_var = opti.variable(3, N + 1)

    def _tuck_at(k):
        """CasADi tuck expression at trajectory node k (0 outside flight)."""
        if stance_end <= k <= flight_end:
            return tuck_flight[k - stance_end]
        return ca.DM(0.0)

    def _I_mat_at(k):
        return _I_mat_from_tuck(_tuck_at(k))

##############################################################
# Touchdown-to-world mapping for landing feet (yaw-only)
##############################################################

q_touchdown = X[3:7, flight_end]
yaw_touchdown = kin.quat_to_yaw_ca(q_touchdown)
c_td = ca.cos(yaw_touchdown)
s_td = ca.sin(yaw_touchdown)
Rz_touchdown = ca.vertcat(
    ca.horzcat(c_td, -s_td),
    ca.horzcat(s_td,  c_td),
)
p_com_touchdown_xy = X[0:2, flight_end]
p_L_land_xy_W = p_com_touchdown_xy + Rz_touchdown @ p_L_land
p_R_land_xy_W = p_com_touchdown_xy + Rz_touchdown @ p_R_land
p_L_land_W = ca.vertcat(p_L_land_xy_W, _landing_gz)
p_R_land_W = ca.vertcat(p_R_land_xy_W, _landing_gz)

##############################################################
# Constraints
##############################################################

# initial condition
opti.subject_to(X[:, 0] == x0_ca)

# phase duration bounds
T_stance_min, T_stance_max = cfg.timing.T_stance_bounds
T_flight_min, T_flight_max = cfg.timing.T_flight_bounds
T_land_min, T_land_max = cfg.timing.T_land_bounds
opti.subject_to(opti.bounded(T_stance_min, T_stance, T_stance_max))
opti.subject_to(opti.bounded(T_flight_min, T_flight, T_flight_max))
opti.subject_to(opti.bounded(T_land_min, T_land, T_land_max))

# terminal constraint (split: position/velocity box + orientation angular distance)
epsilon = cfg.constraints.terminal_epsilon
opti.subject_to(X[0:3, N] >= x_goal_ca[0:3] - epsilon)
opti.subject_to(X[0:3, N] <= x_goal_ca[0:3] + epsilon)
opti.subject_to(X[7:13, N] >= x_goal_ca[7:13] - epsilon)
opti.subject_to(X[7:13, N] <= x_goal_ca[7:13] + epsilon)

# orientation: angular-distance constraint via quat_log(quat_diff)
q_err_terminal = kin.quat_diff_ca(X[3:7, N], ca.DM(quat_goal))
log_err_terminal = kin.quat_log_ca(q_err_terminal)
opti.subject_to(ca.sumsqr(log_err_terminal) <= epsilon**2)

# kinematic limits
L_max = cfg.constraints.L_max
L_min = cfg.constraints.L_min
pz_min = cfg.constraints.pz_min

# contact parameters
mu = cfg.constraints.mu
M_ankle_x_max = cfg.constraints.M_ankle_x_max
M_ankle_y_max = cfg.constraints.M_ankle_y_max
M_ankle_z_max = cfg.constraints.M_ankle_z_max
F_leg_max = cfg.constraints.F_leg_max

A_friction, b_friction = srb.friction_cone_matrix(mu)

# workspace pz boundary functions (defined here so available inside the loop)
if cfg.constraints.workspace_pz_2d:
    if not os.path.exists(_WORKSPACE_COEFFS_2D):
        raise FileNotFoundError(
            f"workspace_pz_2d=True but coeffs not found: {_WORKSPACE_COEFFS_2D}\n"
            "Run: conda run -n env_sbo python ik/squat_workspace.py"
        )
    _raw2d = np.loadtxt(_WORKSPACE_COEFFS_2D, delimiter=",", comments="#")
    _mono2d = [(int(r[0]), int(r[1])) for r in _raw2d]
    _c2d    = [float(r[2]) for r in _raw2d]

    def _pz_boundary_2d(px_sym, pitch_sym):
        val = 0.0
        for (i, j), c in zip(_mono2d, _c2d):
            val = val + c * px_sym**i * pitch_sym**j
        return val

if cfg.constraints.workspace_pz:
    if not os.path.exists(_WORKSPACE_COEFFS):
        raise FileNotFoundError(
            f"workspace_pz=True but coeffs not found: {_WORKSPACE_COEFFS}\n"
            "Run: conda run -n env_sbo python ik/squat_workspace.py"
        )
    _poly = np.loadtxt(_WORKSPACE_COEFFS, delimiter=",")

    def _pz_boundary(px_sym):
        val = float(_poly[0])
        for c in _poly[1:]:
            val = val * px_sym + float(c)
        return val

if cfg.constraints.workspace_pz_2d_upper:
    if not os.path.exists(_WORKSPACE_COEFFS_2D_UPPER):
        raise FileNotFoundError(
            f"workspace_pz_2d_upper=True but coeffs not found: {_WORKSPACE_COEFFS_2D_UPPER}\n"
            "Run: conda run -n env_sbo python ik/squat_workspace_upper.py"
        )
    _raw2d_upper = np.loadtxt(_WORKSPACE_COEFFS_2D_UPPER, delimiter=",", comments="#")
    _mono2d_upper = [(int(r[0]), int(r[1])) for r in _raw2d_upper]
    _c2d_upper    = [float(r[2]) for r in _raw2d_upper]

    def _pz_ceiling_2d(px_sym, pitch_sym):
        val = 0.0
        for (i, j), c in zip(_mono2d_upper, _c2d_upper):
            val = val + c * px_sym**i * pitch_sym**j
        return val

print("[dbg] dynamics loop...", flush=True)
for k in range(N):
    if k < stance_end:
        dt_k = dt_stance
    elif k < flight_end:
        dt_k = dt_flight
    else:
        dt_k = dt_land

    # STANCE
    if k < stance_end:
        p_com = X[0:3, k]
        p_L = ca.vertcat(p0_L_ca, cfg.constraints.stance_ground_z)
        p_R = ca.vertcat(p0_R_ca, cfg.constraints.stance_ground_z)
        r_L = p_L - p_com
        r_R = p_R - p_com

        F_total = F_L[:, k] + F_R[:, k]
        M_total = (
              ca.cross(r_L, F_L[:, k])
            + ca.cross(r_R, F_R[:, k])
            + M_L[:, k] + M_R[:, k]
        )

        opti.subject_to(ca.sumsqr(r_L) <= L_max**2)
        opti.subject_to(ca.sumsqr(r_R) <= L_max**2)

        opti.subject_to(ca.sumsqr(r_L) >= L_min**2)
        opti.subject_to(ca.sumsqr(r_R) >= L_min**2)

        # stance pre-rotation constraint (isotropic angular-distance cap relative to quat0)
        q_rel = kin.quat_diff_ca(ca.DM(quat0), X[3:7, k])
        log_rel = kin.quat_log_ca(q_rel)
        opti.subject_to(ca.sumsqr(log_rel) <= cfg.constraints.stance_rotation_allow**2)

        # yaw-specific constraint to prevent twisting during stance
        yaw_k = kin.quat_to_yaw_ca(X[3:7, k])
        opti.subject_to(opti.bounded(-cfg.constraints.stance_yaw_max, yaw_k, cfg.constraints.stance_yaw_max))

        if cfg.constraints.px_stance_max is not None:
            opti.subject_to(X[0, k] <= cfg.constraints.px_stance_max)

    # FLIGHT
    elif (k >= stance_end) and (k < flight_end):
        F_total = ca.DM.zeros(3)
        M_total = ca.DM.zeros(3)

    # LANDING
    else:
        p_com = X[0:3, k]
        p_L = p_L_land_W
        p_R = p_R_land_W
        r_L = p_L - p_com
        r_R = p_R - p_com

        F_total = F_L[:, k] + F_R[:, k]
        M_total = (
              ca.cross(r_L, F_L[:, k])
            + ca.cross(r_R, F_R[:, k])
            + M_L[:, k] + M_R[:, k]
        )

        opti.subject_to(ca.sumsqr(r_L) <= L_max**2)
        opti.subject_to(ca.sumsqr(r_R) <= L_max**2)
        opti.subject_to(ca.sumsqr(r_L) >= L_min**2)
        opti.subject_to(ca.sumsqr(r_R) >= L_min**2)

    # dynamics
    if _var_inertia:
        # d(I·ω)/dt = M_ext using angular momentum L = I·ω as auxiliary variable.
        # Avoids ca.solve (which creates a dense symbolic Hessian).
        # L dynamics are LINEAR; coupling I@w=L is BILINEAR — much better for IPOPT.
        q_k  = X[3:7, k]
        v_k  = X[7:10, k]
        w_k  = X[10:13, k]
        R_BW = kin.quat_to_rot_matrix_ca(q_k)   # body-to-world rotation

        F_net_W = F_total + ca.vertcat(0.0, 0.0, -srb.m * srb.g)
        M_net_B = R_BW.T @ M_total

        # L dynamics: L_{k+1} = L_k + dt * M_net_B  (angular momentum conservation)
        opti.subject_to(L_var[:, k + 1] == L_var[:, k] + dt_k * M_net_B)

        # Position / quaternion / velocity dynamics (unchanged from fixed-I formulation)
        w_quat = ca.vertcat(0, w_k)
        q_dot  = 0.5 * kin.quat_mult_ca(q_k, w_quat)
        p_next = X[0:3, k] + dt_k * v_k
        q_next = q_k + dt_k * q_dot
        q_next = q_next / ca.norm_2(q_next)
        v_next = v_k + dt_k * (F_net_W / srb.m)

        # w_{k+1} is not explicitly stepped; it is pinned by the coupling L=I@w below.
        opti.subject_to(X[0:10, k + 1] == ca.vertcat(p_next, q_next, v_next))
    else:
        u      = ca.vertcat(F_total, M_total)
        x_next = f(X[:, k], u, dt_k)
        opti.subject_to(X[:, k + 1] == x_next)

# Leg extension at liftoff (last stance node) and touchdown (first landing node)
if cfg.constraints.L_extension_min > 0:
    L_ext2 = cfg.constraints.L_extension_min ** 2

    k = stance_end - 1
    p_com_k = X[0:3, k]
    r_L_lo = ca.vertcat(p0_L_ca[0], p0_L_ca[1], cfg.constraints.stance_ground_z) - p_com_k
    r_R_lo = ca.vertcat(p0_R_ca[0], p0_R_ca[1], cfg.constraints.stance_ground_z) - p_com_k
    opti.subject_to(ca.sumsqr(r_L_lo) >= L_ext2)
    opti.subject_to(ca.sumsqr(r_R_lo) >= L_ext2)

    k = flight_end
    p_com_k = X[0:3, k]
    r_L_td = p_L_land_W - p_com_k
    r_R_td = p_R_land_W - p_com_k
    opti.subject_to(ca.sumsqr(r_L_td) >= L_ext2)
    opti.subject_to(ca.sumsqr(r_R_td) >= L_ext2)

# L = I·ω coupling: bilinear constraint tying angular momentum, inertia, and
# angular velocity at every node.  Inertia is derived from the tuck polynomial.
if _var_inertia:
    for k in range(N + 1):
        opti.subject_to(_I_mat_at(k) @ X[10:13, k] == L_var[:, k])

print("[dbg] pz constraints...", flush=True)
_c_pz = cfg.constraints.pitch_pz_coupling

def _pitch_from_quat(k):
    # Use atan2 instead of asin to avoid infinite gradient at sinp = ±1.
    # asin'(x) = 1/sqrt(1-x²) → ∞ at x=±1; fmin/fmax clamp produces 0*∞ → NaN
    # in the Jacobian when IPOPT perturbs denormalized quaternions.
    # atan2(y, x) has finite gradient everywhere except (0,0), which can't occur.
    # X[3:7] = [qw, qx, qy, qz] (pinocchio scalar-first convention).
    q_k  = X[3:7, k]
    qw, qx, qy, qz = q_k[0], q_k[1], q_k[2], q_k[3]
    sinp = 2.0 * (qw * qy - qz * qx)   # sin(pitch) for ZYX Euler
    R00  = 1 - 2*(qy**2 + qz**2)        # cos(pitch)*cos(yaw)
    R10  = 2*(qx*qy + qw*qz)            # cos(pitch)*sin(yaw)
    cosp = ca.sqrt(R00**2 + R10**2)     # |cos(pitch)|
    return ca.atan2(sinp, cosp)

if cfg.constraints.workspace_pz_2d:
    # 2D surface: z >= poly(x_rel, pitch) + surface_z
    # Polynomial is fit with feet at z=0, so outputs height above the contact surface.
    # We add the world-frame surface height (_stance_gz or _landing_gz) to get world z.
    # During flight there is no contact surface, so we use pz_min (above stance surface).
    x_stance_center = float(p0_L[0])   # p0_L is [0, hip_offset]; x=0
    x_land_center   = (p_L_land_W[0] + p_R_land_W[0]) / 2.0  # symbolic
    for k in range(N + 1):
        if k < stance_end:
            x_rel = X[0, k] - x_stance_center
            opti.subject_to(X[2, k] >= _pz_boundary_2d(x_rel, _pitch_from_quat(k)) + _stance_gz)
        elif k >= flight_end:
            x_rel = X[0, k] - x_land_center
            opti.subject_to(X[2, k] >= _pz_boundary_2d(x_rel, _pitch_from_quat(k)) + _landing_gz)
        else:
            opti.subject_to(X[2, k] >= pz_min + _stance_gz)
elif cfg.constraints.workspace_pz:
    # 1D surface with optional heuristic pitch coupling
    x_stance_center = float(p0_L[0])
    x_land_center   = (p_L_land_W[0] + p_R_land_W[0]) / 2.0
    for k in range(N + 1):
        if k < stance_end:
            floor_k = _pz_boundary(X[0, k] - x_stance_center) + _stance_gz
            if _c_pz > 0.0:
                floor_k = floor_k - _c_pz * _pitch_from_quat(k)
            opti.subject_to(X[2, k] >= floor_k)
        elif k >= flight_end:
            floor_k = _pz_boundary(X[0, k] - x_land_center) + _landing_gz
            if _c_pz > 0.0:
                floor_k = floor_k - _c_pz * _pitch_from_quat(k)
            opti.subject_to(X[2, k] >= floor_k)
        else:
            opti.subject_to(X[2, k] >= pz_min + _stance_gz)
else:
    for k in range(N + 1):
        # pz_min is relative to the stance surface; flight must stay above the box/platform.
        # Landing uses the landing surface as the reference.
        gz_k = _stance_gz if k < flight_end else _landing_gz
        if _c_pz > 0.0 and (k < stance_end or k >= flight_end):
            opti.subject_to(X[2, k] >= pz_min + gz_k - _c_pz * _pitch_from_quat(k))
        else:
            opti.subject_to(X[2, k] >= pz_min + gz_k)

# pz_max — scalar upper bound on CoM height during contact phases.
if cfg.constraints.pz_max is not None:
    for k in range(N + 1):
        if k < stance_end or k >= flight_end:
            opti.subject_to(X[2, k] <= cfg.constraints.pz_max)

# pz_touchdown_max — scalar upper bound on CoM z at the touchdown frame only.
if cfg.constraints.pz_touchdown_max is not None:
    opti.subject_to(X[2, flight_end] <= cfg.constraints.pz_touchdown_max)

# workspace_pz_2d_upper — IK-derived upper bound pz <= pz_max_poly(x_rel, pitch).
# Applied only at the liftoff and touchdown boundary frames where the IK is most
# constrained (near-full leg extension).  Later landing frames can have higher CoM
# as the robot stands up — IPOPTIK warm-starts fine from the previous frame there.
if cfg.constraints.workspace_pz_2d_upper:
    k = stance_end - 1   # liftoff
    x_rel = X[0, k] - x_stance_center
    opti.subject_to(X[2, k] <= _pz_ceiling_2d(x_rel, _pitch_from_quat(k)) + _stance_gz)

    k = flight_end        # touchdown
    x_rel = X[0, k] - x_land_center
    opti.subject_to(X[2, k] <= _pz_ceiling_2d(x_rel, _pitch_from_quat(k)) + _landing_gz)

print("[dbg] contact constraints...", flush=True)
for k in range(N):
    # STANCE
    if k < stance_end:
        opti.subject_to(A_friction @ F_L[:, k] <= b_friction)
        opti.subject_to(A_friction @ F_R[:, k] <= b_friction)
        opti.subject_to(ca.sumsqr(F_L[:, k]) <= F_leg_max**2)
        opti.subject_to(ca.sumsqr(F_R[:, k]) <= F_leg_max**2)
        opti.subject_to(opti.bounded(-M_ankle_x_max, M_L[0, k], M_ankle_x_max))
        opti.subject_to(opti.bounded(-M_ankle_y_max, M_L[1, k], M_ankle_y_max))
        opti.subject_to(opti.bounded(-M_ankle_z_max, M_L[2, k], M_ankle_z_max))
        opti.subject_to(opti.bounded(-M_ankle_x_max, M_R[0, k], M_ankle_x_max))
        opti.subject_to(opti.bounded(-M_ankle_y_max, M_R[1, k], M_ankle_y_max))
        opti.subject_to(opti.bounded(-M_ankle_z_max, M_R[2, k], M_ankle_z_max))

    # FLIGHT
    elif (k >= stance_end) and (k < flight_end):
        opti.subject_to(F_L[:, k] == 0)
        opti.subject_to(F_R[:, k] == 0)
        opti.subject_to(M_L[:, k] == 0)
        opti.subject_to(M_R[:, k] == 0)

    # LANDING
    else:
        opti.subject_to(A_friction @ F_L[:, k] <= b_friction)
        opti.subject_to(A_friction @ F_R[:, k] <= b_friction)
        opti.subject_to(ca.sumsqr(F_L[:, k]) <= F_leg_max**2)
        opti.subject_to(ca.sumsqr(F_R[:, k]) <= F_leg_max**2)
        opti.subject_to(opti.bounded(-M_ankle_x_max, M_L[0, k], M_ankle_x_max))
        opti.subject_to(opti.bounded(-M_ankle_y_max, M_L[1, k], M_ankle_y_max))
        opti.subject_to(opti.bounded(-M_ankle_z_max, M_L[2, k], M_ankle_z_max))
        opti.subject_to(opti.bounded(-M_ankle_x_max, M_R[0, k], M_ankle_x_max))
        opti.subject_to(opti.bounded(-M_ankle_y_max, M_R[1, k], M_ankle_y_max))
        opti.subject_to(opti.bounded(-M_ankle_z_max, M_R[2, k], M_ankle_z_max))

# landing foot placement constraint
landing_tol = cfg.constraints.landing_tol
opti.subject_to(ca.sumsqr(p_L_land - p_L_goal) <= landing_tol**2)
opti.subject_to(ca.sumsqr(p_R_land - p_R_goal) <= landing_tol**2)


# landing orientation constraint
# Three modes (mutually exclusive, in priority order):
#   workspace_pz_2d=True   → 2D surface handles it, no explicit orientation needed
#   pitch_pz_coupling > 0  → 1D surface + heuristic coupling handles it
#   otherwise              → explicit roll²+pitch² <= rp_max² + alpha*||p_err||²
if _c_pz == 0.0 and not cfg.constraints.workspace_pz_2d:
    rp_max   = cfg.constraints.touchdown_rp_max
    alpha_rp = cfg.constraints.landing_rp_alpha
    for k in range(flight_end, N + 1):
        q_k = X[3:7, k]
        roll_k = ca.atan2(2.0 * (q_k[0] * q_k[1] + q_k[2] * q_k[3]),
                          1.0 - 2.0 * (q_k[1]**2 + q_k[2]**2))
        sinp_k  = 2.0 * (q_k[0] * q_k[2] - q_k[3] * q_k[1])
        pitch_k = ca.asin(ca.fmin(ca.fmax(sinp_k, -1.0), 1.0))
        p_err_sq = ca.sumsqr(X[0:3, k] - x_goal_ca[0:3])
        rp_sq_allowed = rp_max**2 + alpha_rp * p_err_sq
        opti.subject_to(roll_k**2 + pitch_k**2 <= rp_sq_allowed)

print("[dbg] objective...", flush=True)
##############################################################
# Objective Function
##############################################################

J = 0
for k in range(N):
    if k < stance_end:
        dt_k = dt_stance
        phase_k = "stance"
    elif k < flight_end:
        dt_k = dt_flight
        phase_k = "flight"
    else:
        dt_k = dt_land
        phase_k = "landing"

    # stance stability: penalise CoM x deviation from feet (x=0)
    if k < stance_end and cfg.costs.Q_stance_px > 0:
        J += dt_k * 0.5 * cfg.costs.Q_stance_px * X[0, k]**2

    # stance alignment: penalise pitch/CoM-x misalignment (pitch should oppose px)
    if k < stance_end and cfg.costs.Q_stance_align > 0:
        q_k  = X[3:7, k]
        sinp = 2.0 * (q_k[0] * q_k[2] - q_k[3] * q_k[1])
        pitch_k = ca.asin(ca.fmin(ca.fmax(sinp, -1.0), 1.0))
        k_align = 1.0 / x0[2]   # 1/z_nom ≈ 1.3 rad/m
        J += dt_k * 0.5 * cfg.costs.Q_stance_align * (pitch_k + k_align * X[0, k])**2

    # phase-aware state objective
    if k < stance_end:
        x_ref_k = None
    elif k < flight_end:
        alpha = (k - stance_end + 1) / N_flight
        quat_k = interp.sample_piecewise_slerp(alpha, quat_slerp_keyframes)

        # body-frame angular velocity reference (scaled by N_flight / T_flight)
        j = k - stance_end
        omega_ref_k = ca.DM(omega_ref_body[j]) * N_flight / T_flight

        x_ref_k = ca.vertcat(
            alpha * x_goal[0] + (1 - alpha) * x0[0],
            alpha * x_goal[1] + (1 - alpha) * x0[1],
            x_goal[2],
            quat_k[0], quat_k[1], quat_k[2], quat_k[3],
            0.0, 0.0, 0.0,
            omega_ref_k[0], omega_ref_k[1], omega_ref_k[2],
        )
    else:
        x_ref_k = x_goal_ca

    # state cost only during flight + landing
    if x_ref_k is not None:
        J += dt_k * srb.state_cost(X[:, k], x_ref_k)

    # contact cost
    if phase_k in ("stance", "landing"):
        J += dt_k * srb.contact_cost(F_L[:, k], F_R[:, k], M_L[:, k], M_R[:, k])

    # landing velocity cost
    if phase_k == "landing" and cfg.costs.Q_landing_vel > 0:
        J += 0.5 * cfg.costs.Q_landing_vel * (
            ca.sumsqr(X[7:10, k]) + ca.sumsqr(X[10:13, k])
        )

    # stance foot-com x alignment: keep CoM x close to stance foot x (= 0)
    if phase_k == "stance" and cfg.costs.Q_stance_foot_com_x > 0:
        J += dt_k * 0.5 * cfg.costs.Q_stance_foot_com_x * X[0, k]**2

    # stance foot-com x alignment using actual foot positions
    if phase_k == "stance" and cfg.costs.Q_stance_feet_com_x > 0:
        foot_mid_x = (p0_L_ca[0] + p0_R_ca[0]) / 2
        J += dt_k * 0.5 * cfg.costs.Q_stance_feet_com_x * (X[0, k] - foot_mid_x)**2


    # force rate cost (skip across phase boundaries)
    if k < N - 1:
        if k + 1 < stance_end:
            phase_k1 = "stance"
        elif k + 1 < flight_end:
            phase_k1 = "flight"
        else:
            phase_k1 = "landing"

        if phase_k1 == phase_k:
            J += dt_k * srb.force_rate_cost(
                F_L[:, k], F_R[:, k], M_L[:, k], M_R[:, k],
                F_L[:, k+1], F_R[:, k+1], M_L[:, k+1], M_R[:, k+1],
                dt_k)

    # tuck rate cost — penalise rapid tuck changes during flight
    if _var_inertia and cfg.costs.Q_I_dot > 0 and stance_end <= k < flight_end:
        fi = k - stance_end
        J += dt_k * 0.5 * cfg.costs.Q_I_dot * (tuck_flight[fi + 1] - tuck_flight[fi])**2

# foot placement cost (body frame)
J += srb.foot_placement_cost(p_L_land, p_R_land, p_L_goal, p_R_goal)

# foot placement cost (world frame)
J += srb.world_foot_placement_cost(p_L_land_xy_W, p_R_land_xy_W, p_L_goal_W, p_R_goal_W)

# foot-com x alignment: penalise x offset of each landing foot from CoM x at touchdown
if cfg.costs.Q_foot_com_x > 0:
    com_td_x = X[0, flight_end]
    J += 0.5 * cfg.costs.Q_foot_com_x * (
        (p_L_land_xy_W[0] - com_td_x)**2 + (p_R_land_xy_W[0] - com_td_x)**2
    )

# terminal cost
J += srb.terminal_cost(X[:, N], x_goal_ca)

opti.minimize(J)

##############################################################
# Initial Guesses
##############################################################

# state trajectory
for k in range(N + 1):
    alpha = k / N

    # com xy: tanh S-curve — stays near start during stance, transitions during flight,
    # settles at goal during landing (better than linear for phased trajectories)
    _s = 2.5
    _tanh_lo = np.tanh(-_s)
    alpha_tanh = (np.tanh(_s * (2 * alpha - 1)) - _tanh_lo) / (np.tanh(_s) - _tanh_lo)
    p_xy_guess = (1 - alpha_tanh) * x0[:2] + alpha_tanh * x_goal[:2]

    if cfg.solver.smart_z_init:
        # Stance: hold at initial height (forces already balance there, zero dynamics violation).
        # Flight: linear descent from liftoff z to landing z (avoids large velocity
        #         discontinuity that a parabolic arc would introduce at the flight→landing
        #         boundary when there is a large height difference).
        # Landing: hold at goal z.
        if k < stance_end:
            p_z_guess = x0[2]
        elif k <= flight_end:
            alpha_fl = (k - stance_end) / N_flight
            p_z_guess = (1 - alpha_fl) * x0[2] + alpha_fl * x_goal[2]
        else:
            p_z_guess = x_goal[2]
        p_com_guess = np.array([p_xy_guess[0], p_xy_guess[1], p_z_guess])
    else:
        p_com_guess = (1 - alpha) * x0[:3] + alpha * x_goal[:3]

    opti.set_initial(X[0:3, k], p_com_guess)

    # quaternion initial guess aligned with phase
    if k < stance_end:
        quat_guess = quat0.copy()
    elif k < flight_end:
        alpha_f = (k - stance_end + 1) / N_flight
        quat_guess = interp.sample_piecewise_slerp(alpha_f, quat_slerp_keyframes)
    else:
        quat_guess = quat_goal.copy()
    opti.set_initial(X[3:7, k], quat_guess)

# landing foot positions
opti.set_initial(p_L_land, p_L_goal)
opti.set_initial(p_R_land, p_R_goal)
opti.set_initial(T_stance, T_stance_nom)
opti.set_initial(T_flight, T_flight_nom)
opti.set_initial(T_land, T_land_nom)

# wrenches: phase-aware
for k in range(N):
    if k < stance_end or k >= flight_end:
        opti.set_initial(F_L[:, k], [0, 0, srb.m * srb.g / 2])
        opti.set_initial(F_R[:, k], [0, 0, srb.m * srb.g / 2])
    else:
        opti.set_initial(F_L[:, k], [0, 0, 0])
        opti.set_initial(F_R[:, k], [0, 0, 0])

opti.set_initial(M_L, 0)
opti.set_initial(M_R, 0)

# tuck: bell-curve guess (0 at liftoff/touchdown, peak 0.5 at midpoint).
# L_var: I_nom @ ω_ref at each node (nominal inertia matches tuck=0).
if _var_inertia:
    I_nom_mat = np.array(srb.I)
    for fi in range(N_flight + 1):
        t_norm = fi / N_flight
        tuck_guess = 4.0 * 0.5 * t_norm * (1.0 - t_norm)  # ∈ [0, 0.5]
        opti.set_initial(tuck_flight[fi], tuck_guess)

    L_guess = np.zeros((3, N + 1))
    for k in range(N + 1):
        if stance_end <= k < flight_end:
            j = min(k - stance_end, N_flight - 1)
            w_ref = omega_ref_body[j] / float(T_flight_nom / N_flight)
            L_guess[:, k] = I_nom_mat @ w_ref
    opti.set_initial(L_var, L_guess)

##############################################################
# Solve
##############################################################

print("[dbg] calling opti.solve()...", flush=True)
opti.solver("ipopt", {"ipopt": {"max_iter": cfg.solver.max_iter}})
sol = opti.solve()

##############################################################
# Extract solutions
##############################################################

X_sol = sol.value(X)
pL_land = sol.value(p_L_land)
pR_land = sol.value(p_R_land)
FL_sol = sol.value(F_L)
FR_sol = sol.value(F_R)
ML_sol = sol.value(M_L)
MR_sol = sol.value(M_R)
if _var_inertia:
    tuck_sol  = np.array(sol.value(tuck_flight)).flatten()   # (N_flight+1,)
    L_var_sol = np.array(sol.value(L_var))                   # (3, N+1)

    # Reconstruct full-trajectory inertia from tuck (for pipeline compatibility).
    I_np = np.array(srb.I)
    I_var_sol = np.zeros((6, N + 1))
    for k in range(N + 1):
        t_k = float(tuck_sol[k - stance_end]) if stance_end <= k <= flight_end else 0.0
        I_var_sol[0, k] = float(I_np[0, 0]) + float(np.polyval(_poly_Ixx, t_k)) - _Ixx0
        I_var_sol[1, k] = float(I_np[1, 1]) + float(np.polyval(_poly_Iyy, t_k)) - _Iyy0
        I_var_sol[2, k] = float(I_np[2, 2]) + float(np.polyval(_poly_Izz, t_k)) - _Izz0
        # off-diagonals = 0 (diagonal-only; bilateral symmetry)
T_stance_sol = float(sol.value(T_stance))
T_flight_sol = float(sol.value(T_flight))
T_land_sol = float(sol.value(T_land))
T_sol = T_stance_sol + T_flight_sol + T_land_sol
dt_stance_sol = T_stance_sol / N_stance
dt_flight_sol = T_flight_sol / N_flight
dt_land_sol = T_land_sol / N_land

##############################################################
# Reconstruct wrench trajectory
##############################################################

F = (FL_sol + FR_sol).T  # (N, 3)

q_touchdown_sol = X_sol[3:7, flight_end]
yaw_touchdown_sol = kin.quat_to_yaw(q_touchdown_sol)
c_td_sol = np.cos(yaw_touchdown_sol)
s_td_sol = np.sin(yaw_touchdown_sol)
Rz_touchdown_sol = np.array([
    [c_td_sol, -s_td_sol],
    [s_td_sol,  c_td_sol],
])
p_com_touchdown_xy_sol = X_sol[0:2, flight_end]
pL_land_world_xy = p_com_touchdown_xy_sol + Rz_touchdown_sol @ pL_land
pR_land_world_xy = p_com_touchdown_xy_sol + Rz_touchdown_sol @ pR_land

M = np.zeros((N, 3))
for k in range(N):
    if k < stance_end:
        p_com = X_sol[0:3, k]
        p_L = np.array([float(p0_L_ca[0]), float(p0_L_ca[1]), float(cfg.constraints.stance_ground_z)])
        p_R = np.array([float(p0_R_ca[0]), float(p0_R_ca[1]), float(cfg.constraints.stance_ground_z)])
    elif k < flight_end:
        M[k, :] = 0.0
        continue
    else:
        p_com = X_sol[0:3, k]
        p_L = np.array([pL_land_world_xy[0], pL_land_world_xy[1], _landing_gz])
        p_R = np.array([pR_land_world_xy[0], pR_land_world_xy[1], _landing_gz])

    r_L = p_L - p_com
    r_R = p_R - p_com
    M[k, :] = (
          np.cross(r_L, FL_sol[:, k])
        + np.cross(r_R, FR_sol[:, k])
        + ML_sol[:, k] + MR_sol[:, k]
    )

U = np.hstack((F, M))  # (N, 6)

##############################################################
# Compute accelerations
##############################################################

X_sol = X_sol.T          # (N+1, nx)
q_opt = X_sol[:, 0:nq]
v_opt = X_sol[:, nq:nx]
a_opt = np.zeros_like(v_opt)

for k in range(N):
    xdot = np.array(srb.f_cont(X_sol[k, :], U[k, :])).squeeze()
    a_opt[k, :] = xdot[nq:nx]
a_opt[N, :] = a_opt[N-1, :]

##############################################################
# Print results
##############################################################

print(f"\nOptimal landing foot positions:")
print(f"  Left  foot (body): x={pL_land[0]:.3f}, y={pL_land[1]:.3f}")
print(f"  Right foot (body): x={pR_land[0]:.3f}, y={pR_land[1]:.3f}")
print(f"  Left  foot (world): x={pL_land_world_xy[0]:.3f}, y={pL_land_world_xy[1]:.3f}")
print(f"  Right foot (world): x={pR_land_world_xy[0]:.3f}, y={pR_land_world_xy[1]:.3f}")

print(f"\nOptimal phase durations:")
print(f"  Stance:  {T_stance_sol:.3f}s (dt={dt_stance_sol:.4f}s, nodes={N_stance})")
print(f"  Flight:  {T_flight_sol:.3f}s (dt={dt_flight_sol:.4f}s, nodes={N_flight})")
print(f"  Landing: {T_land_sol:.3f}s (dt={dt_land_sol:.4f}s, nodes={N_land})")
print(f"  Total:   {T_sol:.3f}s")

print(f"\nPhase summary:")
print(f"  Stance:  k=0  -> {stance_end-1}   (t=0.00 -> {T_stance_sol:.2f}s)")
print(f"  Flight:  k={stance_end} -> {flight_end-1}  (t={T_stance_sol:.2f} -> {T_stance_sol + T_flight_sol:.2f}s)")
print(f"  Landing: k={flight_end} -> {N-1}  (t={T_stance_sol + T_flight_sol:.2f} -> {T_sol:.2f}s)")

print(f"\nTerminal state:")
print(f"  p_com = {X_sol[N, 0:3]}")
print(f"  quat  = {X_sol[N, 3:7]}")
print(f"  v_com = {X_sol[N, 7:10]}")

##############################################################
# Save
##############################################################

dt_schedule = np.concatenate((
    np.full(N_stance, dt_stance_sol),
    np.full(N_flight, dt_flight_sol),
    np.full(N_land, dt_land_sol),
))
time = np.zeros(N + 1)
time[1:] = np.cumsum(dt_schedule)

feet = np.full((N, 4), np.nan)
feet[:stance_end, 0:2] = np.array([float(p0_L_ca[0]), float(p0_L_ca[1])])
feet[:stance_end, 2:4] = np.array([float(p0_R_ca[0]), float(p0_R_ca[1])])
feet[flight_end:, 0:2] = np.array([float(pL_land_world_xy[0]), float(pL_land_world_xy[1])])
feet[flight_end:, 2:4] = np.array([float(pR_land_world_xy[0]), float(pR_land_world_xy[1])])

save_dir = cfg.save_dir
os.makedirs(save_dir, exist_ok=True)

np.savetxt(save_dir + "time.csv",         time,          delimiter=",")
np.savetxt(save_dir + "q_opt.csv",        q_opt,         delimiter=",")
np.savetxt(save_dir + "v_opt.csv",        v_opt,         delimiter=",")
np.savetxt(save_dir + "a_opt.csv",        a_opt,         delimiter=",")
np.savetxt(save_dir + "tau_opt.csv",      U,             delimiter=",")
np.savetxt(save_dir + "feet.csv",         feet,          delimiter=",")
np.savetxt(save_dir + "force_left.csv",   FL_sol.T,      delimiter=",")
np.savetxt(save_dir + "force_right.csv",  FR_sol.T,      delimiter=",")
np.savetxt(save_dir + "moment_left.csv",  ML_sol.T,      delimiter=",")
np.savetxt(save_dir + "moment_right.csv", MR_sol.T,      delimiter=",")
if _var_inertia:
    np.savetxt(save_dir + "I_opt.csv", I_var_sol.T, delimiter=",",
               header="Ixx,Iyy,Izz,Ixy,Ixz,Iyz (N+1 rows, body-frame kg·m²)")
    np.savetxt(save_dir + "tuck_opt.csv", tuck_sol, delimiter=",",
               header="tuck parameter at each flight node (N_flight+1 values)")
    # Aerial foot positions in pelvis frame — for IK pipeline flight phase.
    foot_aerial = np.zeros((N_flight + 1, 6))
    for fi in range(N_flight + 1):
        t_k = float(tuck_sol[fi])
        foot_aerial[fi, 0] = float(np.polyval(_poly_fL[0], t_k))
        foot_aerial[fi, 1] = float(np.polyval(_poly_fL[1], t_k))
        foot_aerial[fi, 2] = float(np.polyval(_poly_fL[2], t_k))
        foot_aerial[fi, 3] = float(np.polyval(_poly_fR[0], t_k))
        foot_aerial[fi, 4] = float(np.polyval(_poly_fR[1], t_k))
        foot_aerial[fi, 5] = float(np.polyval(_poly_fR[2], t_k))
    np.savetxt(save_dir + "aerial_foot_pos.csv", foot_aerial, delimiter=",",
               header="fLx,fLy,fLz,fRx,fRy,fRz pelvis-frame (N_flight+1 rows)")

print(f"\nSaved results to {save_dir}")
