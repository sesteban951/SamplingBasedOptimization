##
#
# Dataclass config schema for srb_aerial.py
#
##

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class InitialStateConfig:
    p_com: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.77])
    rpy_deg: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    v_com: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    w_body: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])


@dataclass
class GoalStateConfig:
    p_com: List[float] = field(default_factory=lambda: [0.5, 0.0, 0.77])
    rpy_deg: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    v_com: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    w_body: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])


@dataclass
class ManeuverConfig:
    rpy_deg: List[float] = field(default_factory=lambda: [0.0, 0.0, 360.0])


@dataclass
class TimingConfig:
    dt_nom: float = 0.02
    T_stance_nom: float = 0.5
    T_flight_nom: float = 0.5
    T_land_nom: float = 0.5
    T_stance_bounds: List[float] = field(default_factory=lambda: [0.4, 1.0])
    T_flight_bounds: List[float] = field(default_factory=lambda: [0.2, 1.0])
    T_land_bounds: List[float] = field(default_factory=lambda: [0.2, 1.0])


@dataclass
class CostWeights:
    Qx_diag: List[float] = field(default_factory=lambda: [
        1.0, 1.0, 1.0,       # px, py, pz
        50.0, 50.0, 50.0,    # qx, qy, qz (log-space)
        1.0, 1.0, 1.0,       # vx, vy, vz
        1.0, 1.0, 1.0,       # wx, wy, wz
    ])
    Qx_terminal_scale: float = 100.0
    Q_foot: float = 50.0
    Q_foot_world: float = 150.0
    Q_force: float = 1e-4
    Q_moment: float = 1e-4
    Q_force_dot: float = 1e-4
    Q_moment_dot: float = 1e-4
    Q_stance_px: float = 0.0
    Q_stance_align: float = 0.0  # penalises pitch + k*px misalignment during stance
    Q_landing_vel: float = 0.0   # penalises ||v_com|| and ||w_body|| during landing
    Q_foot_com_x: float = 0.0        # penalises x-offset of each landing foot from CoM x at touchdown
    Q_stance_foot_com_x: float = 0.0 # penalises x-offset of CoM from stance feet (x=0) during pre-jump
    Q_stance_feet_com_x: float = 0.0 # penalises CoM x vs actual stance foot positions (p0_L, p0_R) during pre-jump
    Q_I_dot: float = 0.0         # rate penalty on inertia changes ||I(k+1)-I(k)||² / dt
    Q_flip_mid: float = 0.0      # orientation cost at flight midpoint — guides flip direction


@dataclass
class ConstraintConfig:
    terminal_epsilon: float = 0.01
    L_min: float = 0.45
    L_max: float = 0.8
    pz_min: float = 0.45
    pz_max: float = None  # upper bound on CoM z during contact phases; None = unconstrained
    mu: float = 1.0
    M_ankle_x_max: float = 50.0
    M_ankle_y_max: float = 50.0
    M_ankle_z_max: float = 10.0
    F_leg_max: float = 500.0
    landing_tol: float = 0.1
    stance_rotation_allow: float = 0.15
    stance_yaw_max: float = 0.0   # yaw limit during stance (rad)
    touchdown_rp_max: float = 0.15
    landing_rp_alpha: float = 1.0  # extra orientation budget per metre of CoM error during landing
    # Coupling between CoM pitch and minimum CoM height (m/rad).
    # Enforces pz >= pz_min + pitch_pz_coupling * |pitch| during contact phases,
    # reflecting that ankle ROM is shared between squat depth and forward tilt.
    pitch_pz_coupling: float = 0.0
    px_stance_max: Optional[float] = None
    # If True, replace the flat pz_min floor during contact phases with the
    # IK-derived x-dependent boundary pz >= poly(px) from squat_workspace.py.
    # The flat pz_min still applies during flight.
    workspace_pz: bool = False
    # If True, use the 2D IK-derived surface pz >= poly(x, pitch) from squat_workspace.py.
    # Supersedes workspace_pz and pitch_pz_coupling when active.
    workspace_pz_2d: bool = False
    # If True, add upper bound pz <= pz_max_poly(x, pitch) from squat_workspace_upper.py.
    # Prevents planning CoM heights the real robot cannot achieve at the given foot offset.
    workspace_pz_2d_upper: bool = False
    # Minimum leg length (m) enforced at the last stance node (liftoff) and first
    # landing node (touchdown). 0 = disabled. Ensures near-full extension at
    # phase boundaries for physical realism and IK smoothness.
    L_extension_min: float = 0.0
    # Height of the stance / landing ground surfaces above world z=0 (m).
    # Config values p_com[2] and pz_min are expressed RELATIVE to these surfaces;
    # srb_aerial.py adds the offsets when building world-frame states and constraints.
    # Example: box backflip with stance on 0.61m box, landing on ground →
    #   stance_ground_z=0.6096, landing_ground_z=0.0
    stance_ground_z: float = 0.0
    landing_ground_z: float = 0.0
    # Upper bound on CoM z at the touchdown frame only (k=flight_end).
    # Prevents planning near-full-extension landings where IK fails.
    # None = unconstrained.
    pz_touchdown_max: Optional[float] = None



@dataclass
class SolverConfig:
    max_iter: int = 5000
    # When True: stance z held at x0[2], flight z follows a ballistic parabolic arc,
    # landing z held at goal[2].  Better for elevated-surface configs where the linear
    # guess creates large dynamics violations.
    smart_z_init: bool = False
    # When True: centroidal inertia I(k) is added as a per-timestep decision variable.
    # Requires ik/results/inertia_bounds.csv (run ik/sample_inertia_workspace.py first).
    # The rotational dynamics change to d(I·ω)/dt = M_ext (angular momentum form),
    # so inertia variation during flight correctly alters angular velocity.
    variable_inertia: bool = False
    # When True (default): only the diagonal terms Ixx/Iyy/Izz vary; off-diagonals
    # Ixy/Ixz/Iyz are fixed at zero.  Correct for bilaterally symmetric maneuvers
    # (backflip, jump).  Set False only for asymmetric motions.
    variable_inertia_diagonal_only: bool = True
    # Hard upper bound on per-component inertia rate of change (kg·m²/s).
    # Prevents sharp oscillations that make IPOPT sensitive.  Applied as
    # |I_j(k+1) - I_j(k)| <= max_I_dot * dt_nom per component per step.
    # None = unconstrained (soft Q_I_dot penalty only).
    max_I_dot: Optional[float] = None
    # Hard upper bound on tuck rate (1/s) when variable_inertia=True.
    # Applied as |tuck(k+1) - tuck(k)| <= max_tuck_dot * dt_nom per flight step.
    # None = unconstrained (soft Q_I_dot penalty only).
    max_tuck_dot: Optional[float] = None
    # When True: each IK solution is checked for self-collision via pinocchio's
    # collision engine.  Frames in collision are flagged with a warning but kept
    # (the IK result is still used).  Costs extra time per frame due to geometry
    # loading on first use.
    reject_self_collision: bool = False


@dataclass
class AerialConfig:
    initial: InitialStateConfig = field(default_factory=InitialStateConfig)
    goal: GoalStateConfig = field(default_factory=GoalStateConfig)
    maneuver: ManeuverConfig = field(default_factory=ManeuverConfig)
    timing: TimingConfig = field(default_factory=TimingConfig)
    costs: CostWeights = field(default_factory=CostWeights)
    constraints: ConstraintConfig = field(default_factory=ConstraintConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    save_dir: str = "./results/srb/srb_aerial/"
