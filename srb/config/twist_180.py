##
#
# Verification config: reproduces srb_twist.py parameters exactly.
#
##

from srb.config.aerial_config import (
    AerialConfig, InitialStateConfig, GoalStateConfig, ManeuverConfig,
    TimingConfig, CostWeights, ConstraintConfig, SolverConfig,
)

config = AerialConfig(
    initial=InitialStateConfig(
        p_com=[0.0, 0.0, 0.77],
        rpy_deg=[0.0, 1.0, 0.0],
        v_com=[0.0, 0.0, 0.0],
        w_body=[0.0, 0.0, 0.0],
    ),
    goal=GoalStateConfig(
        p_com=[0.3, 0.0, 0.77],
        rpy_deg=[0.0, 0.0, 180.0],
        v_com=[0.0, 0.0, 0.0],
        w_body=[0.0, 0.0, 0.0],
    ),
    maneuver=ManeuverConfig(
        rpy_deg=[0.0, 0.0, 180.0],
    ),
    timing=TimingConfig(
        dt_nom=0.02,
        T_stance_nom=0.5,
        T_flight_nom=0.5,
        T_land_nom=0.5,
        T_stance_bounds=[0.4, 1.0],
        T_flight_bounds=[0.5, 1.0],
        T_land_bounds=[0.2, 1.0],
    ),
    costs=CostWeights(
        Qx_diag=[
            1.0, 1.0, 1.0,
            50.0, 50.0, 50.0,
            1.0, 1.0, 1.0,
            1.0, 1.0, 1.0,
        ],
        Qx_terminal_scale=100.0,
        Q_foot=50.0,
        Q_foot_world=150.0,
        Q_force=1e-4,
        Q_moment=1e-4,
        Q_force_dot=1e-4,
        Q_moment_dot=1e-4,
    ),
    constraints=ConstraintConfig(
        terminal_epsilon=0.01,
        L_min=0.45,
        L_max=0.8,
        pz_min=0.45,
        mu=1.0,
        M_ankle_x_max=50.0,
        M_ankle_y_max=50.0,
        M_ankle_z_max=10.0,
        F_leg_max=500.0,
        landing_tol=0.1,
        stance_rotation_allow=1.0,
        stance_yaw_max=0.85,          # allow +/-~29deg stance yaw → counter-rotation windup
                                     # (0.0 default hard-pins yaw=0 and blocks any preload).
                                     # Total stance rotation still capped by stance_rotation_allow:
                                     # 0.8^2 >= pitch^2 + 0.5^2 → up to ~0.62 rad pitch alongside.
        touchdown_rp_max=0.15,
        stance_rp_max=0.1,
        touchdown_twist_frac=0.95,   # finish 85% of the twist in the air; only 15% settles on the ground
        # workspace_pz=True,
        # workspace_pz_2d=True,

        #IK Constraints not srb
        # free_foot_yaw=True,   # ONLY to let planted feet choose their yaw (within slack of the
                              # body).  Default (off) now FIXES the foot heading to the SRB-
                              # reference body yaw per ground phase — flat, sole-down, no skid,
                              # and generalises to any twist (180/360/...) on its own.
        free_waist_yaw=True,  # free waist_yaw for yaw momentum authority (kino IK)
        control_frame=True
    ),
    solver=SolverConfig(
        max_iter=5000,
        izz_modulation=True,        # free Izz +/-15% in flight → solver shapes a non-uniform twist
        izz_modulation_range=0.15,
    ),
    save_dir="./results/srb/srb_aerial/",
)
