##
# Numerical verification of the Wensing & Orin (2016) centroidal-dynamics
# factorization, adapted to pinocchio's [linear; angular] spatial convention.
#
# Wensing-Orin Table 1 (their [angular; linear] convention):
#   H11    = U1 H U1^T
#   I1^C   = Psi1^T H11 Psi1
#   M      = (I1^C)_{6,6}
#   1p_G   = (1/M) [ (I1^C)_{3,5}, (I1^C)_{1,6}, (I1^C)_{2,4} ]^T
#   iX_G^T = [[ 0R1 , 0R1 S(1p_G)^T ], [ 0 , 0R1 ]]
#   A_G    = iX_G^T Psi1^T U1 H
#   Adot_G qdot = iX_G^T Psi1^T U1 (C qdot)
#
# In pinocchio:
#   * the free-flyer base velocity is the body-frame spatial velocity in
#     [linear; angular] ordering, so the base motion subspace Psi1 = I_6 and the
#     CRBA block H11 = M[0:6,0:6] IS the locked (composite) inertia I1^C directly.
#   * pinocchio's spatial inertia in [v; w] ordering is
#         I1^C = [[ m I3,      m S(c)^T ],
#                 [ m S(c),    Ibar     ]]
#     so mass m = I1^C[0,0] and the CoM offset in base frame c = vee(I1^C[3:6,0:3])/m.
#   * the force/momentum transform base->centroid (world-aligned at CoM), in
#     [linear; angular] ordering, is
#         T = [[ R1,            0  ],
#              [ -S(R1 c) R1,   R1 ]]
#     (shown algebraically equal to the table's iX_G^T after the [w;v]->[v;w]
#      block permutation).
#
#   Then:   h      = A_G v          == pin.ccrba(q,v).hg
#           h_dot  = A_G a + Adot   == pin.computeCentroidalMomentumTimeVariation(q,v,a).dhg
#   where   A_G    = T @ M[0:6, :]
#           Adot   = T @ (C v)[0:6],   C v = rnea(q,v,0) - rnea(q,0,0)   (no gravity)
##

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pinocchio as pin

_REPO_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_URDF = os.path.join(_REPO_ROOT, "models", "g1", "g1_29dof_rev_1_0.urdf")


def _skew(u):
    return np.array([[0.0, -u[2], u[1]], [u[2], 0.0, -u[0]], [-u[1], u[0], 0.0]])


def _vee(S):
    return np.array([S[2, 1], S[0, 2], S[1, 0]])


def cmm_wensing_orin(model, data, q):
    """Return (A_G, m, c_base, R1) using the WO factorization in pin convention."""
    M = np.array(pin.crba(model, data, q))    # crba fills upper triangle only
    M = np.triu(M) + np.triu(M, 1).T          # mirror upper->lower (proper symmetric M)
    H11 = M[0:6, 0:6]                          # = I1^C  (Psi1 = I in pinocchio)
    m   = H11[0, 0]
    c_base = _vee(H11[3:6, 0:3]) / m           # CoM in base frame
    R1  = pin.Quaternion(q[3:7]).matrix()      # base orientation (body->world)
    d   = R1 @ c_base                          # world vector base->CoM
    T = np.zeros((6, 6))
    T[0:3, 0:3] = R1
    T[3:6, 0:3] = -_skew(d) @ R1
    T[3:6, 3:6] = R1
    A_G = T @ M[0:6, :]                         # (6 x nv)
    return A_G, T, m, c_base, R1


def coriolis_bias(model, data, q, v):
    """C(q,v) v  (no gravity) = rnea(q,v,0) - rnea(q,0,0)."""
    nv = model.nv
    full = pin.rnea(model, data, q, v, np.zeros(nv))     # C v + g
    grav = pin.rnea(model, data, q, np.zeros(nv), np.zeros(nv))  # g
    return full - grav


def main():
    np.random.seed(0)
    model = pin.buildModelFromUrdf(_DEFAULT_URDF, pin.JointModelFreeFlyer())
    data  = model.createData()
    data2 = model.createData()
    nq, nv = model.nq, model.nv
    print(f"[verify] model nq={nq} nv={nv}")

    max_h_err = max_hdot_err = max_com_err = 0.0
    for trial in range(50):
        q = pin.randomConfiguration(model)
        q[0:3] = np.random.randn(3)                      # random base translation
        q[3:7] /= np.linalg.norm(q[3:7])
        v = np.random.randn(nv)
        a = np.random.randn(nv)

        # --- Wensing-Orin factorization ---
        A_G, T, m, c_base, R1 = cmm_wensing_orin(model, data, q)
        Cv = coriolis_bias(model, data, q, v)
        h_wo    = A_G @ v
        Adot_v  = T @ Cv[0:6]
        hdot_wo = A_G @ a + Adot_v

        # --- pinocchio ground truth ---
        pin.ccrba(model, data2, q, v)
        h_pin = np.concatenate([np.array(data2.hg.linear), np.array(data2.hg.angular)])
        pin.computeCentroidalMomentumTimeVariation(model, data2, q, v, a)
        hdot_pin = np.concatenate([np.array(data2.dhg.linear), np.array(data2.dhg.angular)])

        # --- CoM cross-check ---
        com_pin = pin.centerOfMass(model, data2, q)
        com_wo  = q[0:3] + R1 @ c_base

        max_h_err    = max(max_h_err,    np.linalg.norm(h_wo - h_pin))
        max_hdot_err = max(max_hdot_err, np.linalg.norm(hdot_wo - hdot_pin))
        max_com_err  = max(max_com_err,  np.linalg.norm(com_wo - com_pin))

    print(f"[verify] max ||h_wo    - h_pin||    = {max_h_err:.3e}")
    print(f"[verify] max ||hdot_wo - hdot_pin|| = {max_hdot_err:.3e}")
    print(f"[verify] max ||com_wo  - com_pin||  = {max_com_err:.3e}")
    ok = max_h_err < 1e-8 and max_hdot_err < 1e-8 and max_com_err < 1e-9
    print(f"[verify] {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
