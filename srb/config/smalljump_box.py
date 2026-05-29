##
#
# Small jump from a 20x30x24-inch box — robot starts and lands on the box.
# Geometrically identical to smalljump.py, just shifted up by the box height.
# Used to verify the stance_ground_z mechanism before attempting harder maneuvers.
#
##

from srb.config.aerial_config import (
    AerialConfig, InitialStateConfig, GoalStateConfig, ManeuverConfig,
    TimingConfig, CostWeights, ConstraintConfig, SolverConfig,
)

_BOX_HEIGHT = 24 * 0.0254   # 0.6096 m

config = AerialConfig(
    initial=InitialStateConfig(
        p_com=[0.0, 0.0, 0.77],   # CoM height above stance surface (box top)
        rpy_deg=[0.0, 1.0, 0.0],
        v_com=[0.0, 0.0, 0.0],
        w_body=[0.0, 0.0, 0.0],
    ),
    goal=GoalStateConfig(
        p_com=[0.0, 0.0, 0.77],   # CoM height above landing surface (box top, landing_ground_z=_BOX_HEIGHT)
        rpy_deg=[0.0, 0.0, 0.0],
        v_com=[0.0, 0.0, 0.0],
        w_body=[0.0, 0.0, 0.0],
    ),
    maneuver=ManeuverConfig(
        rpy_deg=[0.0, -15.0, 0.0],
    ),
    timing=TimingConfig(
        dt_nom=0.02,
        T_stance_nom=0.5,
        T_flight_nom=0.8,
        T_land_nom=0.5,
        T_stance_bounds=[0.4, 1.5],
        T_flight_bounds=[0.5, 1.0],
        T_land_bounds=[0.2, 1.5],
    ),
    costs=CostWeights(
        Qx_diag=[
            1.0, 1.0, 0.5,
            50.0, 50.0, 50.0,
            1.0, 1.0, 1.0,
            1.0, 1.0, 1.0,
        ],
        Qx_terminal_scale=100.0,
        Q_foot=50.0,
        Q_foot_world=150.0,
        Q_stance_align=30.0,
        Q_landing_vel=0.02,
        Q_force=1e-4,
        Q_moment=1e-4,
        Q_force_dot=1e-4,
        Q_moment_dot=1e-4,
    ),
    constraints=ConstraintConfig(
        terminal_epsilon=0.01,
        L_min=0.5,
        L_max=0.8,
        pz_min=0.2,
        mu=1.0,
        M_ankle_x_max=50.0,
        M_ankle_y_max=50.0,
        M_ankle_z_max=10.0,
        F_leg_max=350.0,
        landing_tol=0.1,
        stance_rotation_allow=0.4,
        workspace_pz=True,
        workspace_pz_2d=True,
        L_extension_min=0.69,
        stance_ground_z=_BOX_HEIGHT,
        landing_ground_z=_BOX_HEIGHT,   # lands back on the box
    ),
    solver=SolverConfig(
        max_iter=5000,
    ),
    save_dir="./results/srb/smalljump_box/",
)
