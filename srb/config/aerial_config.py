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


@dataclass
class ConstraintConfig:
    terminal_epsilon: float = 0.01
    L_min: float = 0.45
    L_max: float = 0.8
    pz_min: float = 0.45
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
    


@dataclass
class SolverConfig:
    max_iter: int = 5000


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
