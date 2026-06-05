##
#
# Small jump with forward CoM pitch wind-up.
#
# Physical motivation:
#   When the SRB pitches forward during stance, the foot-to-CoM vector r_L
#   angles backward.  The cross product cross(r_L, F_L) then creates a
#   pitch-back angular impulse that is stored as rotational KE at liftoff.
#   During flight the body un-pitches (backward rotation), converting that
#   angular momentum into a slightly higher apex and a more dynamic takeoff.
#   The vanilla config prevents this with stance_rotation_allow=0.15 (~8.6 deg);
#   relaxing it to 0.4 rad (~23 deg) lets the optimizer discover the strategy.
#
#   The in-flight maneuver is set to a small backward pitch to help the
#   optimizer un-tilt the body before landing, keeping touchdown_rp_max tight.
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
        p_com=[0.0, 0.0, 0.77],
        rpy_deg=[0.0, 0.0, 0.0],
        v_com=[0.0, 0.0, 0.0],
        w_body=[0.0, 0.0, 0.0],
    ),
    maneuver=ManeuverConfig(
        # Small backward pitch in flight to un-tilt from the stance wind-up.
        # If the optimizer tilts ~15 deg forward at liftoff this brings it
        # back to ~0 deg by touchdown.
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
        # Relaxed from 0.15 → 0.4 rad to allow ~23 deg forward pitch wind-up
        # during stance.  The optimizer is free to use less if it doesn't help.
        stance_rotation_allow=0.4,
        # 2D workspace surface z >= poly(x, pitch) replaces both rp_max and the
        # heuristic pitch_pz_coupling.  Run ik/squat_workspace.py first.
        workspace_pz=True,
        workspace_pz_2d=True,
        L_extension_min=0.69,
    ),
    solver=SolverConfig(
        max_iter=5000,
        reject_self_collision=True,
    ),
    save_dir="./results/srb/srb_aerial/",
)
