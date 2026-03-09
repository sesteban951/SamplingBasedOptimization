##
#
# Single Rigid Body backflip trajectory optimization with planar leg contacts.
#
##

import os
import numpy as np
import casadi as ca

from srb.srb import SRBDynamics2D


class SRB_Backflip2D(SRBDynamics2D):

    def __init__(self):
        super().__init__()

        self.Qx = ca.diag(ca.vertcat(
            0.0, 0.0,   # px, pz
            40.0,       # theta
            0.0, 0.0,   # vx, vz
            8.0,        # w
        ))

        self.Q_foot = 5.0

        self.Q_force = 1e-4
        self.Q_moment = 1e-4
        self.Q_force_dot = 5e-5
        self.Q_moment_dot = 5e-5

        self.Qx_f = 25.0 * self.Qx

    def state_cost(self, x, x_ref):
        e = x - x_ref
        return 0.5 * e.T @ self.Qx @ e

    def contact_cost(self, F_L, F_R, M_L, M_R):
        return (
            0.5 * self.Q_force * (ca.sumsqr(F_L) + ca.sumsqr(F_R))
            + 0.5 * self.Q_moment * (ca.sumsqr(M_L) + ca.sumsqr(M_R))
        )

    def force_rate_cost(
        self,
        F_L_k,
        F_R_k,
        M_L_k,
        M_R_k,
        F_L_k1,
        F_R_k1,
        M_L_k1,
        M_R_k1,
        dt,
    ):
        dF_L = (F_L_k1 - F_L_k) / dt
        dF_R = (F_R_k1 - F_R_k) / dt
        dM_L = (M_L_k1 - M_L_k) / dt
        dM_R = (M_R_k1 - M_R_k) / dt
        return (
            0.5 * self.Q_force_dot * (ca.sumsqr(dF_L) + ca.sumsqr(dF_R))
            + 0.5 * self.Q_moment_dot * (ca.sumsqr(dM_L) + ca.sumsqr(dM_R))
        )

    def foot_placement_cost(self, p_L, p_R, p_L_des, p_R_des):
        return (
            0.5 * self.Q_foot * ca.sumsqr(p_L - p_L_des)
            + 0.5 * self.Q_foot * ca.sumsqr(p_R - p_R_des)
        )

    def terminal_cost(self, x, x_goal):
        e = x - x_goal
        return e.T @ self.Qx_f @ e


def cross2d(r, F):
    return r[0] * F[1] - r[1] * F[0]


def smoothstep(alpha):
    return 3.0 * alpha**2 - 2.0 * alpha**3


def phase_interp(start, end, alpha):
    return start + (end - start) * smoothstep(alpha)


def build_backflip_reference(times, x0, x_goal, stance_end, flight_end):
    dt = times[1] - times[0]
    N = len(times) - 1
    x_ref = np.zeros((x0.shape[0], N + 1))

    theta_takeoff = 0.45 * np.pi
    theta_pre_land = 1.85 * np.pi

    for k in range(N + 1):
        if k <= stance_end:
            alpha = 0.0 if stance_end == 0 else k / stance_end
            x_ref[2, k] = phase_interp(x0[2], theta_takeoff, alpha)
        elif k <= flight_end:
            alpha = (k - stance_end) / max(flight_end - stance_end, 1)
            x_ref[2, k] = phase_interp(theta_takeoff, theta_pre_land, alpha)
        else:
            alpha = (k - flight_end) / max(N - flight_end, 1)
            x_ref[2, k] = phase_interp(theta_pre_land, x_goal[2], alpha)

    x_ref[:, 0] = x0
    x_ref[:, -1] = x_goal

    x_ref[5, :] = np.gradient(x_ref[2, :], dt, edge_order=2)

    x_ref[0:2, :] = 0.0
    x_ref[3:5, :] = 0.0
    x_ref[3:, 0] = x0[3:]
    x_ref[5, -1] = x_goal[5]

    return x_ref


def build_seed_trajectory(times, x0, theta_ref, omega_ref, stance_end, flight_end, pz_nom, g):
    N = len(times) - 1
    x_seed = np.zeros((x0.shape[0], N + 1))
    x_seed[:, 0] = x0

    T_flight_seed = times[flight_end] - times[stance_end]
    vz_takeoff = g * T_flight_seed / 2.0

    for k in range(N + 1):
        x_seed[0, k] = x0[0]
        x_seed[2, k] = theta_ref[k]
        x_seed[5, k] = omega_ref[k]

        if k <= stance_end:
            alpha = 0.0 if stance_end == 0 else k / stance_end
            x_seed[1, k] = pz_nom
            x_seed[3, k] = 0.0
            x_seed[4, k] = phase_interp(0.0, vz_takeoff, alpha)
        elif k <= flight_end:
            tau = times[k] - times[stance_end]
            x_seed[1, k] = pz_nom + vz_takeoff * tau - 0.5 * g * tau**2
            x_seed[3, k] = 0.0
            x_seed[4, k] = vz_takeoff - g * tau
        else:
            alpha = (k - flight_end) / max(N - flight_end, 1)
            x_seed[1, k] = pz_nom
            x_seed[3, k] = 0.0
            x_seed[4, k] = phase_interp(-vz_takeoff, 0.0, alpha)

    x_seed[:, -1] = np.array([x0[0], pz_nom, theta_ref[-1], 0.0, 0.0, 0.0])
    return x_seed


srb = SRB_Backflip2D()
f = srb.f_disc
nx = srb.nq + srb.nv
nq = srb.nq
nv = srb.nv

dt = 0.01
T_stance = 0.70
T_flight = 0.65
T_land = 0.35
T = T_stance + T_flight + T_land

N_stance = int(T_stance / dt)
N_flight = int(T_flight / dt)
N_land = int(T_land / dt)
N = int(T / dt)

stance_end = N_stance
flight_end = N_stance + N_flight
opti = ca.Opti()

X = opti.variable(nx, N + 1)
p_L_land = opti.variable(1)
p_R_land = opti.variable(1)
F_L = opti.variable(2, N)
F_R = opti.variable(2, N)
M_L = opti.variable(1, N)
M_R = opti.variable(1, N)

pz_nom = 0.79
x0 = np.array([
    0.0, pz_nom,
    0.0,
    0.0, 0.0,
    0.0,
])
x0_ca = ca.DM(x0)

p0_L = ca.DM([x0[0]])
p0_R = ca.DM([x0[0]])

px_goal = 0.0
theta_goal = 2.0 * np.pi
x_goal = np.array([
    px_goal, pz_nom,
    theta_goal,
    0.0, 0.0,
    0.0,
])
x_goal_ca = ca.DM(x_goal)

p_L_goal = ca.DM([px_goal])
p_R_goal = ca.DM([px_goal])

opti.subject_to(p_L_land == px_goal)
opti.subject_to(p_R_land == px_goal)

times = np.linspace(0.0, T, N + 1)
x_ref = build_backflip_reference(times, x0, x_goal, stance_end, flight_end)
x_ref_ca = ca.DM(x_ref)
x_seed = build_seed_trajectory(
    times, x0, x_ref[2, :], x_ref[5, :], stance_end, flight_end, pz_nom, srb.g
)

opti.subject_to(X[:, 0] == x0_ca)

opti.subject_to(opti.bounded(theta_goal - 0.20, X[2, N], theta_goal + 0.20))
opti.subject_to(opti.bounded(pz_nom - 0.05, X[1, N], pz_nom + 0.05))
opti.subject_to(opti.bounded(-0.75, X[4, N], 0.75))
opti.subject_to(opti.bounded(-1.00, X[5, N], 1.00))

L_max = 0.80
L_min = 0.45

for k in range(N):
    if k < stance_end:
        p_com = X[0:2, k]
        p_L = ca.vertcat(p0_L, 0.0)
        p_R = ca.vertcat(p0_R, 0.0)

        r_L = p_L - p_com
        r_R = p_R - p_com

        F_total = F_L[:, k] + F_R[:, k]
        M_total = (
            cross2d(r_L, F_L[:, k])
            + cross2d(r_R, F_R[:, k])
            + M_L[:, k] + M_R[:, k]
        )

        opti.subject_to(ca.sumsqr(r_L) <= L_max**2)
        opti.subject_to(ca.sumsqr(r_R) <= L_max**2)
        opti.subject_to(ca.sumsqr(r_L) >= L_min**2)
        opti.subject_to(ca.sumsqr(r_R) >= L_min**2)

    elif k < flight_end:
        F_total = ca.DM.zeros(2)
        M_total = ca.DM.zeros(1)

    else:
        p_com = X[0:2, k]
        p_L = ca.vertcat(p_L_land, 0.0)
        p_R = ca.vertcat(p_R_land, 0.0)

        r_L = p_L - p_com
        r_R = p_R - p_com

        F_total = F_L[:, k] + F_R[:, k]
        M_total = (
            cross2d(r_L, F_L[:, k])
            + cross2d(r_R, F_R[:, k])
            + M_L[:, k] + M_R[:, k]
        )

        opti.subject_to(ca.sumsqr(r_L) <= L_max**2)
        opti.subject_to(ca.sumsqr(r_R) <= L_max**2)
        opti.subject_to(ca.sumsqr(r_L) >= L_min**2)
        opti.subject_to(ca.sumsqr(r_R) >= L_min**2)

    u = ca.vertcat(F_total, M_total)
    x_next = f(X[:, k], u, dt)
    opti.subject_to(X[:, k + 1] == x_next)

for k in range(N + 1):
    opti.subject_to(X[1, k] >= 0.5)

mu = 1.0
M_ankle_max = 80.0
F_leg_max = 500.0

A_friction, b_friction = srb.friction_cone_matrix(mu)

for k in range(N):
    if k < stance_end:
        opti.subject_to(A_friction @ F_L[:, k] <= b_friction)
        opti.subject_to(A_friction @ F_R[:, k] <= b_friction)
        opti.subject_to(ca.sumsqr(F_L[:, k]) <= F_leg_max**2)
        opti.subject_to(ca.sumsqr(F_R[:, k]) <= F_leg_max**2)
        opti.subject_to(opti.bounded(-M_ankle_max, M_L[:, k], M_ankle_max))
        opti.subject_to(opti.bounded(-M_ankle_max, M_R[:, k], M_ankle_max))
    elif k < flight_end:
        opti.subject_to(F_L[:, k] == 0)
        opti.subject_to(F_R[:, k] == 0)
        opti.subject_to(M_L[:, k] == 0)
        opti.subject_to(M_R[:, k] == 0)
    else:
        opti.subject_to(A_friction @ F_L[:, k] <= b_friction)
        opti.subject_to(A_friction @ F_R[:, k] <= b_friction)
        opti.subject_to(ca.sumsqr(F_L[:, k]) <= F_leg_max**2)
        opti.subject_to(ca.sumsqr(F_R[:, k]) <= F_leg_max**2)
        opti.subject_to(opti.bounded(-M_ankle_max, M_L[:, k], M_ankle_max))
        opti.subject_to(opti.bounded(-M_ankle_max, M_R[:, k], M_ankle_max))

J = 0
for k in range(N):
    J += srb.state_cost(X[:, k], x_ref_ca[:, k])

    if k < stance_end or k >= flight_end:
        J += srb.contact_cost(F_L[:, k], F_R[:, k], M_L[:, k], M_R[:, k])

    if k < N - 1:
        J += srb.force_rate_cost(
            F_L[:, k], F_R[:, k], M_L[:, k], M_R[:, k],
            F_L[:, k + 1], F_R[:, k + 1], M_L[:, k + 1], M_R[:, k + 1],
            dt,
        )

J += srb.terminal_cost(X[:, N], x_ref_ca[:, N])

opti.minimize(J)

for k in range(N + 1):
    opti.set_initial(X[:, k], x_seed[:, k])

opti.set_initial(p_L_land, p_L_goal)
opti.set_initial(p_R_land, p_R_goal)

for k in range(N):
    if k < stance_end:
        alpha = k / max(stance_end - 1, 1)
        lift_scale = 0.60 + 0.25 * np.sin(np.pi * alpha)
        brake_scale = np.sin(np.pi * alpha)
        opti.set_initial(F_L[:, k], [0.0, lift_scale * srb.m * srb.g])
        opti.set_initial(F_R[:, k], [0.0, lift_scale * srb.m * srb.g])
        opti.set_initial(M_L[:, k], 25.0 * brake_scale)
        opti.set_initial(M_R[:, k], 25.0 * brake_scale)
    elif k < flight_end:
        opti.set_initial(F_L[:, k], [0.0, 0.0])
        opti.set_initial(F_R[:, k], [0.0, 0.0])
        opti.set_initial(M_L[:, k], 0.0)
        opti.set_initial(M_R[:, k], 0.0)
    else:
        alpha = (k - flight_end) / max(N - flight_end - 1, 1)
        support_scale = 0.95 - 0.15 * alpha
        brake_scale = np.sin(np.pi * alpha)
        opti.set_initial(F_L[:, k], [0.0, support_scale * srb.m * srb.g])
        opti.set_initial(F_R[:, k], [0.0, support_scale * srb.m * srb.g])
        opti.set_initial(M_L[:, k], -20.0 * brake_scale)
        opti.set_initial(M_R[:, k], -20.0 * brake_scale)

opti.solver(
    "ipopt",
    {"expand": True},
    {
        "max_iter": 4000,
        "tol": 1e-5,
        "acceptable_tol": 1e-4,
        "acceptable_constr_viol_tol": 5e-4,
        "print_level": 0,
    },
)
try:
    sol = opti.solve()
except RuntimeError:
    print(f"\nSolver failed with status: {opti.return_status()}")
    print("Last terminal iterate:")
    print(opti.debug.value(X[:, N]))
    print("Constraint infeasibilities at last iterate:")
    opti.debug.show_infeasibilities(1e-4)
    raise

X_sol = sol.value(X)
pL_land = sol.value(p_L_land)
pR_land = sol.value(p_R_land)
FL_sol = sol.value(F_L)
FR_sol = sol.value(F_R)
ML_sol = np.asarray(sol.value(M_L)).reshape(-1)
MR_sol = np.asarray(sol.value(M_R)).reshape(-1)

F_traj = (FL_sol + FR_sol).T

M_traj = np.zeros((N, 1))
for k in range(N):
    if k < stance_end:
        p_com = X_sol[0:2, k]
        p_L = np.array([float(p0_L[0]), 0.0])
        p_R = np.array([float(p0_R[0]), 0.0])
    elif k < flight_end:
        M_traj[k] = 0.0
        continue
    else:
        p_com = X_sol[0:2, k]
        p_L = np.array([float(pL_land), 0.0])
        p_R = np.array([float(pR_land), 0.0])

    r_L = p_L - p_com
    r_R = p_R - p_com
    M_traj[k] = (
        cross2d(r_L, FL_sol[:, k])
        + cross2d(r_R, FR_sol[:, k])
        + ML_sol[k] + MR_sol[k]
    )

U = np.hstack((F_traj, M_traj))

X_sol = X_sol.T
q_opt = X_sol[:, :nq]
v_opt = X_sol[:, nq:]
a_opt = np.zeros_like(v_opt)

for k in range(N):
    xdot = np.array(srb.f_cont(X_sol[k, :], U[k, :])).squeeze()
    a_opt[k, :] = xdot[nq:]
a_opt[N, :] = a_opt[N - 1, :]

contact_schedule = np.zeros((N + 1,))
for k in range(N + 1):
    if k < stance_end:
        contact_schedule[k] = 1.0
    elif k < flight_end:
        contact_schedule[k] = 0.0
    else:
        contact_schedule[k] = 1.0

foot_positions = np.array([
    [float(p0_L[0]), 0.0, float(p0_R[0]), 0.0],
    [float(pL_land), 0.0, float(pR_land), 0.0],
])

print("\nOptimal landing foot positions:")
print(f"  Left  foot: x = {float(pL_land):.3f} m")
print(f"  Right foot: x = {float(pR_land):.3f} m")

print("\nPhase summary:")
print(f"  Stance:  k=0        -> {stance_end - 1:<3}  (t=0.00 -> {stance_end * dt:.2f}s)")
print(f"  Flight:  k={stance_end:<3}      -> {flight_end - 1:<3}  (t={stance_end * dt:.2f} -> {flight_end * dt:.2f}s)")
if N_land > 0:
    print(f"  Landing: k={flight_end:<3}      -> {N - 1:<3}  (t={flight_end * dt:.2f} -> {N * dt:.2f}s)")

print("\nTerminal state:")
print(f"  p_com = [{X_sol[N, 0]:.3f}, {X_sol[N, 1]:.3f}]")
print(f"  theta = {np.degrees(X_sol[N, 2]):.2f} deg")
print(f"  v_com = [{X_sol[N, 3]:.3f}, {X_sol[N, 4]:.3f}]")
print(f"  w     = {X_sol[N, 5]:.4f} rad/s")

time = np.linspace(0.0, T, N + 1)

feet = np.full((N, 4), np.nan)
feet[:stance_end, 0:2] = np.array([float(p0_L[0]), 0.0])
feet[:stance_end, 2:4] = np.array([float(p0_R[0]), 0.0])
feet[flight_end:, 0:2] = np.array([float(pL_land), 0.0])
feet[flight_end:, 2:4] = np.array([float(pR_land), 0.0])

save_dir = "./results/srb/srb_backflip_2d/"
os.makedirs(save_dir, exist_ok=True)

np.savetxt(save_dir + "time.csv", time, delimiter=",")
np.savetxt(save_dir + "q_opt.csv", q_opt, delimiter=",")
np.savetxt(save_dir + "v_opt.csv", v_opt, delimiter=",")
np.savetxt(save_dir + "a_opt.csv", a_opt, delimiter=",")
np.savetxt(save_dir + "tau_opt.csv", U, delimiter=",")
np.savetxt(save_dir + "contact_schedule.csv", contact_schedule, delimiter=",")
np.savetxt(save_dir + "foot_positions.csv", foot_positions, delimiter=",")
np.savetxt(save_dir + "feet.csv", feet, delimiter=",")

print(f"\nSaved results to {save_dir}")
