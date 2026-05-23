# Five-Link Walker Jump — Design Notes & Assumptions

Covers `five_link_jump.py` and the tuning decisions made to produce a dynamically
feasible vertical jump for the G1 5-link model.  Intended as a reference for
downstream use of the saved trajectory.

---

## Model

| Property | Value | Source |
|---|---|---|
| Total mass | 33.34 kg | `g1_5link_params.py` |
| State dim `nq` / `nv` | 13 / 12 | `FiveLinkDynamics` |
| Actuated joints `nj` | 6 (hip roll, hip pitch, knee × 2 legs) | `FiveLinkDynamics` |
| Neutral standing COM height | 0.689 m | measured via `neutral_standing()` |
| Max leg reach (hip → ground) | 0.72 m | `g1_5link_params.py::L_MAX` |
| Max reachable COM with feet on ground | ~0.73 m | `L_MAX + HIP_OFFSET_Z ≈ 0.726 m` |

Dynamics are evaluated via Pinocchio ABA (forward dynamics) and integrated with
**forward Euler**.  This is first-order accurate; trajectories should be treated
as warm starts for higher-fidelity integration if tracking accuracy matters.

---

## NLP Structure

Three fixed-node phases with per-phase time steps as decision variables:

| Phase | Nodes | Duration bounds |
|---|---|---|
| Stance (push-off) | 30 | [0.2, 2.5] s |
| Flight | 20 | [0.5, 1.5] s |
| Landing | 30 | [0.2, 2.5] s |

Phase durations hitting their upper bound in a solution means the optimizer
wants more time — a sign the push-off or landing is still impulsive.  Raise
`T_STANCE_MAX` / `T_LAND_MAX` or add more nodes if smoothness matters.

---

## Contact Model

- **Stance / landing**: both feet pinned to their neutral flat-ground positions
  (`p_foot_L0`, `p_foot_R0`).  Friction cone enforced per foot:
  `[fx, fy, fz]` must satisfy `μ = 0.5` pyramid approximation.
- **Flight**: contact forces set to zero.  Feet only constrained to stay above
  `z = 0`.
- No compliance, no impact dynamics.  The landing node `k = flight_end` snaps
  feet back to ground positions with non-zero contact forces in the same step —
  this is an abrupt transition, not a soft contact model.

---

## Key Kinematic Constraints

### Body orientation
- Roll (`qx`) and yaw (`qz`) penalised throughout with `w_no_twist = 50`.
- During flight all three rotations penalised at `w_no_twist + w_no_twist_flight = 350`.
- Hard pitch limit: `|qy| ≤ 0.18 rad (~10°)` during flight, `0.35 rad` on the ground.

### Flight height floor
Applied only to **interior** flight nodes `[stance_end+1, flight_end)` — **not**
at the phase boundary nodes.

> **Critical assumption**: the constraint `Q[2, k] ≥ pz_flight_min = 0.71 m`
> cannot be applied at `k = flight_end` (first landing node) because the maximum
> kinematically reachable COM height with feet on the ground is ~0.73 m.
> Setting `pz_flight_min > 0.73` at that node makes the NLP infeasible.

### Leg symmetry during flight
Left and right legs are forced to be mirror images during the entire flight phase:

```
Q[8,k]  == Q[11,k]   # hip pitch L == R
Q[9,k]  == Q[12,k]   # knee L == R
Q[7,k]  == -Q[10,k]  # hip roll antisymmetric
```

**Why**: without this, the optimizer finds asymmetric solutions (one knee fully
tucked, the other nearly straight) that induce ~27° of yaw rotation mid-flight.

### Flight joint bounds (self-collision avoidance)
The URDF joint limits are wide enough to allow the leg to swing behind the torso
or the knee to fold into the body.  Tighter bounds applied during flight:

| Joint | Ground limits | Flight limits |
|---|---|---|
| Hip pitch | [−2.53, 2.88] rad | [−0.30, 1.80] rad |
| Knee | [−0.09, 2.88] rad | [−0.09, 2.20] rad |
| Hip roll | ±0.50 rad | ±0.20 rad |

These are heuristic collision-avoidance proxies — the NLP has no explicit
geometry-based collision constraint.

---

## Objective Weights

| Weight | Value | Purpose |
|---|---|---|
| `w_tau` | 1e-3 | torque regularisation |
| `w_lam` | 5e-4 | contact force magnitude |
| `w_lam_rate` | 1e-4 | contact force rate of change (smoothness) |
| `w_vel` | 0.5 | velocity regularisation |
| `w_state` | 1.0 | joint angle tracking toward neutral during flight/landing |
| `w_term` | 20.0 | terminal state penalty |
| `w_no_twist` | 50.0 | roll + yaw penalty throughout |
| `w_no_twist_flight` | 300.0 | additional rotation penalty during flight |
| `w_height` | 2.0 | negative height reward during flight (prevents split degenerate solution) |

---

## Known Degenerate Solutions & How They Were Fixed

### 1. Split instead of jump
**Symptom**: robot spreads legs (split or pike) without leaving the ground.  
**Cause**: `pz_min = 0.45 m` was below standing height (0.689 m) so staying
low was feasible.  No cost rewarded height.  
**Fix**: `pz_flight_min = 0.71 m` on interior flight nodes + `−w_height * Q[2,k]`
in the flight objective.

### 2. Impulsive takeoff
**Symptom**: push-off looks like a single-frame impulse, not a continuous
extension.  
**Cause**: `w_lam = 1e-5` (near zero) — optimizer concentrated all force into
one or two nodes.  
**Fix**: raised to `w_lam = 5e-4` and added `w_lam_rate` penalising
step-to-step force changes.

### 3. Large body rotation on landing
**Symptom**: torso pitches forward heavily at touchdown.  
**Cause**: pitch (`qy`) was not included in the no-twist cost — only roll and
yaw were penalised.  
**Fix**: added `Q[4,k]^2` to the no-twist cost and added the flight-specific
`w_no_twist_flight` term.

### 4. NLP infeasibility after adding flight height floor
**Cause**: `pz_flight_min = 0.75 m` was applied to `k = flight_end` (first
landing node) where feet are pinned at `z = 0`.  Required COM height exceeded
kinematic maximum (~0.73 m).  
**Fix**: lowered to 0.71 m and excluded boundary nodes `stance_end` and
`flight_end` from the constraint.

### 5. Yaw spin mid-flight (~27°)
**Cause**: asymmetric leg motion (right knee fully tucked, left barely bent)
generated angular momentum about the vertical axis.  
**Fix**: added left-right symmetry hard constraints during flight.

---

## Observed Solution Quality (last converged run)

| Metric | Value | Notes |
|---|---|---|
| Apex COM height | ~1.22 m | +0.53 m above standing |
| Max pitch during flight | 0.16 rad | within 0.18 rad limit |
| Max landing Fz per foot | ~1140 N | ~7× bodyweight — high, consider softening |
| Max stance Fz per foot | ~650 N | ~4× bodyweight |
| Hip torque | 88 N·m | at limit — torque-saturated during push-off |
| Phase durations | 1.5s / 0.7s / 1.5s | stance+land at upper bound (prev run) |

The high landing force and torque saturation suggest the robot is torque-limited
during push-off; consider increasing `N_stance`/`N_land` or raising the time cap
further if a smoother trajectory is needed.

---

## Files

| File | Description |
|---|---|
| `five_link_jump.py` | trajectory optimisation |
| `five_link_dynamics.py` | Pinocchio ABA + foot kinematics |
| `g1_5link_params.py` | model parameters and joint limits |
| `view_five_link.py` | MuJoCo visualiser (reads from `results/srb_walker/five_link_jump/`) |
| `results/srb_walker/five_link_jump/` | saved CSVs: `time`, `q_opt`, `v_opt`, `u_opt`, `lam_L`, `lam_R`, `p_feet`, `c_sched` |
