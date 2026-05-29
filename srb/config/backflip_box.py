##
#
# Backflip from a 20x30x24-inch box onto the ground.
#
# Box:  20 x 30 x 24 in  =  0.508 x 0.762 x 0.6096 m (W x D x H)
# Robot starts standing on top of the box (feet at z = 0.6096 m).
# After the backflip the robot lands on the ground (feet at z = 0).
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
        p_com=[-0.9, 0, 0.70],   # CoM height above landing surface (ground, landing_ground_z=0)
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
        T_flight_nom=0.65,
        T_land_nom=0.5,
        T_stance_bounds=[0.4, 1.0],
        T_flight_bounds=[0.45, 1.2],  # min ~0.45s needed to drop 0.68m + complete flip
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
    ),
    constraints=ConstraintConfig(
        terminal_epsilon=0.1,
        L_min=0.45,
        L_max=0.8,
        pz_min=0.04,  # above stance surface (box top); world pz_min = 0.04 + 0.6096 = 0.65 m
        mu=1.0,
        M_ankle_x_max=50.0,
        M_ankle_y_max=50.0,
        M_ankle_z_max=10.0,
        F_leg_max=500.0,
        landing_tol=0.15,
        stance_rotation_allow=0.5,
        touchdown_rp_max=0.52,
        workspace_pz=True,
        workspace_pz_2d=True,
        stance_ground_z=_BOX_HEIGHT,  # feet on top of box during stance
    ),
    solver=SolverConfig(
        max_iter=5000,
        smart_z_init=True,
    ),
    save_dir="./results/srb/backflip_box/",
)
