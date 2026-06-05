##
#
# Backflip with variable centroidal inertia — tests the "diver tuck" effect.
# Run: python -m srb.srb_aerial srb.config.backflip_varinertia
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
        p_com=[-0.75, 0, 0.70],
        rpy_deg=[0.0, -330.0, 0.0],
        v_com=[0.0, 0.0, 0.0],
        w_body=[0.0, 0.0, 0.0],
    ),
    maneuver=ManeuverConfig(
        rpy_deg=[0.0, -330.0, 0.0],
    ),
    timing=TimingConfig(
        dt_nom=0.02,
        T_stance_nom=0.5,
        T_flight_nom=0.5,
        T_land_nom=0.5,
        T_stance_bounds=[0.4, 1.0],
        T_flight_bounds=[0.2, 1.0],
        T_land_bounds=[0.2, 1.0],
    ),
    costs=CostWeights(
        Qx_diag=[
            1.0, 1.0, 1.0,
            100.0, 100.0, 100.0,
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
        Q_I_dot=0.5,   # smoothness penalty on inertia rate-of-change
    ),
    constraints=ConstraintConfig(
        terminal_epsilon=0.1,
        L_min=0.45,
        L_max=0.8,
        pz_min=0.45,
        mu=1.0,
        M_ankle_x_max=50.0,
        M_ankle_y_max=50.0,
        M_ankle_z_max=10.0,
        F_leg_max=500.0,
        landing_tol=0.1,
        stance_rotation_allow=0.5,
        touchdown_rp_max=0.52,
        workspace_pz=True,
        workspace_pz_2d=True,
    ),
    solver=SolverConfig(
        max_iter=5000,
        variable_inertia=True,
        max_tuck_dot=4.0,  # 1/s — allows 0→1 tuck in 0.25 s at dt=0.02
        reject_self_collision=True,
    ),
    save_dir="./results/srb/backflip_varinertia/",
)
