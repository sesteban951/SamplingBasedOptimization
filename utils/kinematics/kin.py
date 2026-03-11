##
#
# Assortment of kinematics utilities.
#
##

# standard imports
import numpy as np

# in case you want to use casadi for symbolic kinematics
import casadi as ca


def quat_normalize_ca(q, eps=1e-12):
    return q / (ca.norm_2(q) + eps)


def quat_conj(q):
    """
    Compute the conjugate of a quaternion in [qw, qx, qy, qz] format.

    Args:
        q: (np.array) Quaternion.
    Returns:
        q_conj: (np.array) Conjugate of the quaternion.
    """
    return np.array([q[0], -q[1], -q[2], -q[3]])

def quat_conj_ca(q):
    return ca.vertcat(q[0], -q[1], -q[2], -q[3])


def quat_mult(a, b):
    """
    Hamilton product of two quaternions in [qw, qx, qy, qz] format.
    c = a ⊗ b

    Args:
        a: (np.array) First quaternion.
        b: (np.array) Second quaternion.
    Returns:
        c: (np.array) Product of the two quaternions.
    """
    # take the components of the quaternions
    aw, ax, ay, az = a
    bw, bx, by, bz = b

    # Hamilton product
    c = np.array([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw
    ])
    return c

def quat_mult_ca(a, b):
    aw, ax, ay, az = a[0], a[1], a[2], a[3]
    bw, bx, by, bz = b[0], b[1], b[2], b[3]
    return ca.vertcat(
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw
    )


def quat_diff(q1, q2):
    """
    Compute the difference between two quaternions in [qw, qx, qy, qz] format.
    Equivalent of v_diff = v2 - v1 for quaternions. This is the rotation that
    when applied to q1 gives q2. In other words, q_diff ⊗ q1 = q2.

    Args:
        q1: (np.array) First quaternion.
        q2: (np.array) Second quaternion.
    Returns:
        q_diff: (np.array) Difference between the two quaternions.
    """
    # normalize the quaternions
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)

    # compute the relative rotation
    q_diff = quat_mult(q2, quat_conj(q1)) # equivalent of dv = v2 - v1 for quaternions
    q_diff = (q_diff / np.linalg.norm(q_diff))

    # take the shortest path if q_diff[0] < 0.0
    if q_diff[0] < 0.0:
        q_diff = -q_diff

    return q_diff

def quat_diff_ca(q1, q2, eps=1e-12):
    q1 = quat_normalize_ca(q1, eps)
    q2 = quat_normalize_ca(q2, eps)
    qd = quat_mult_ca(q2, quat_conj_ca(q1))
    qd = quat_normalize_ca(qd, eps)
    qd = ca.if_else(qd[0] < 0, -qd, qd)
    return qd


def quat_log(q_err):
    """
    Compute the logarithm of a quaternion in [qw, qx, qy, qz] format.
    The output is a 3D vector representing the rotation axis scaled by the rotation angle.
    Micro Lie Theory eq.(133): https://arxiv.org/pdf/1812.01537

    Args:
        q: (np.array) Quaternion in [qw, qx, qy, qz] format.
    Returns:
        log_q: (np.array) Logarithm of the quaternion, shape (3,).
    """
    # normalize the quaternion
    q_err = q_err / np.linalg.norm(q_err)

    # shortest path if q_err[0] < 0.0
    if q_err[0] < 0.0:
        q_err = -q_err

    # extract the vector part of the quaternion
    w = q_err[0]
    v = q_err[1:]
    v_norm = np.linalg.norm(v)

    # small enough to do linear approximation
    if v_norm < 1e-6:
        log_q = 2 * v
    # angle-axis representation to compute the logarithm of the quaternion
    else:
        angle = 2 * np.arctan2(v_norm, w)
        u = v / v_norm
        log_q = angle * u

    return log_q

def quat_log_ca(q_err, eps=1e-6, delta=1e-8):
    q_err = quat_normalize_ca(q_err)
    q_err = ca.if_else(q_err[0] < 0, -q_err, q_err)

    w = q_err[0]
    v = q_err[1:4]

    # Smooth norm to avoid undefined gradient at v=0
    v2 = ca.dot(v, v)
    v_norm = ca.sqrt(v2 + delta**2)

    # small-angle approx: log(q) ~ 2v  (works well near identity)
    log_small = 2 * v

    angle = 2 * ca.atan2(v_norm, w)
    log_full = angle * (v / v_norm)

    return ca.if_else(v2 < eps**2, log_small, log_full)


def quat_to_rot_matrix(q):
    """
    Convert a quaternion to a rotation matrix.
    Assumes q is describing the orientation of the body frame relative to the
    world frame, so that R transforms vectors from body frame to world frame.
    v_W = R @ v_B

    Args:
        q: (np.array) Quaternion in [qw, qx, qy, qz] format.
    Returns:
        R: (np.array) Rotation matrix corresponding to the quaternion.
    """
    # convert quaternion to rotation matrix
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    R = np.array([
        [1 - 2*(y**2 + z**2),     2*(x*y - z*w),       2*(x*z + y*w)],
        [    2*(x*y + z*w),   1 - 2*(x**2 + z**2),     2*(y*z - x*w)],
        [    2*(x*z - y*w),       2*(y*z + x*w),   1 - 2*(x**2 + y**2)]
    ])

    return R

def quat_to_rot_matrix_ca(q, eps=1e-12):
    q = quat_normalize_ca(q, eps)
    w, x, y, z = q[0], q[1], q[2], q[3]
    return ca.vertcat(
        ca.horzcat(1 - 2*(y*y + z*z),     2*(x*y - z*w),       2*(x*z + y*w)),
        ca.horzcat(    2*(x*y + z*w), 1 - 2*(x*x + z*z),       2*(y*z - x*w)),
        ca.horzcat(    2*(x*z - y*w),     2*(y*z + x*w),   1 - 2*(x*x + y*y))
    )


def quat_to_euler_ZYX(q):
    """
    Convert a quaternion to Euler angles (roll, pitch, yaw) in radians.
    Assumes q is describing the orientation of the body frame relative to the
    world frame, euler angles are ZYX intrinsic rotations.

    Args:
        q: (np.array) Quaternion in [qw, qx, qy, qz] format.
    Returns:
        euler: (np.array) Euler angles [roll, pitch, yaw] in radians
    """
    q = q / np.linalg.norm(q)
    w, x, y, z = q

    # pitch (y-axis rotation)
    sinp = 2.0 * (w*y - z*x)
    sinp = np.clip(sinp, -1.0, 1.0)  # numerical stability
    pitch = np.arcsin(sinp)

    # roll (x-axis rotation)
    roll = np.arctan2(2.0*(w*x + y*z), 1.0 - 2.0*(x*x + y*y))

    # yaw (z-axis rotation)
    yaw = np.arctan2(2.0*(w*z + x*y), 1.0 - 2.0*(y*y + z*z))

    euler = np.array([roll, pitch, yaw])
    return euler


def quat_to_yaw(q):
    """
    Extract yaw (about world z) from quaternion [qw, qx, qy, qz].
    """
    q = np.asarray(q, dtype=float)
    q = q / np.linalg.norm(q)
    return np.arctan2(
        2.0 * (q[0] * q[3] + q[1] * q[2]),
        1.0 - 2.0 * (q[2] * q[2] + q[3] * q[3]),
    )


def quat_to_yaw_ca(q):
    """
    Extract yaw (about world z) from CasADi quaternion [qw, qx, qy, qz].
    """
    return ca.atan2(
        2.0 * (q[0] * q[3] + q[1] * q[2]),
        1.0 - 2.0 * (q[2] * q[2] + q[3] * q[3]),
    )


def euler_ZYX_to_quat(roll, pitch, yaw):
    """
    Convert (roll, pitch, yaw) in radians to quaternion [qw, qx, qy, qz].
    ZYX intrinsic composition: q = qz(yaw) * qy(pitch) * qx(roll).
    """
    qx = np.array([np.cos(roll / 2), np.sin(roll / 2), 0.0, 0.0])
    qy = np.array([np.cos(pitch / 2), 0.0, np.sin(pitch / 2), 0.0])
    qz = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
    return quat_mult(qz, quat_mult(qy, qx))


def yaw_to_quat(yaw):
    """
    Construct yaw-only quaternion [qw, qx, qy, qz].
    """
    return np.array([np.cos(0.5 * yaw), 0.0, 0.0, np.sin(0.5 * yaw)], dtype=float)


def yaw_to_rot_matrix(yaw):
    """
    Build a world-frame rotation matrix for yaw-only rotation (flat on ground).
    """
    c = np.cos(yaw)
    s = np.sin(yaw)
    return np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=float)


def quat_rotate(q, v):
    """
    Rotate a 3D vector by a quaternion using:
        v' = q ⊗ v ⊗ q*, with v treated as [0, v].
    Assumes q describes the orientation of the body frame relative to the world frame.

    Args:
        q: (np.array) Quaternion [qw, qx, qy, qz].
        v: (np.array) 3D vector.

    Returns:
        v_rot: (np.array) Rotated 3D vector.
    """
    v = np.asarray(v, dtype=float).reshape(3,)
    q = np.asarray(q, dtype=float)
    q = q / np.linalg.norm(q)  # safety / normalization

    # Lift v to a pure quaternion
    v_quat = np.concatenate(([0.0], v))  # [0, vx, vy, vz]

    # q ⊗ v ⊗ q*
    q_conj = quat_conj(q)
    tmp = quat_mult(q, v_quat)
    res = quat_mult(tmp, q_conj)

    # return only the vector part
    v_rot = res[1:]

    return v_rot

def quat_rotate_ca(q, v, eps=1e-12):
    q = quat_normalize_ca(q, eps)
    v_quat = ca.vertcat(0, v)   # (4,)
    q_conj = quat_conj_ca(q)
    tmp = quat_mult_ca(q, v_quat)
    res = quat_mult_ca(tmp, q_conj)
    v_rot = res[1:4]
    return v_rot


def body_to_world(v_B, q):
    """
    Transform a vector from the body frame to the world frame using a quaternion.
    Assumes q describes the orientation of the body frame relative to the
    world frame, so that the corresponding rotation takes body → world.

    Args:
        v_B: (np.array) Vector in the body frame, shape (3,) or broadcastable to that.
        q:   (np.array) Quaternion [qw, qx, qy, qz] representing orientation of body in world.
    Returns:
        v_W: (np.array) The same vector expressed in the world frame, shape (3,).
    """
    return quat_rotate(q, v_B)

def body_to_world_ca(v_B, q, eps=1e-12):
    return quat_rotate_ca(q, v_B, eps)


def world_to_body(v_W, q):
    """
    Transform a vector from the world frame to the body frame using a quaternion.

    Assumes q describes the orientation of the body frame relative to the world frame,
    i.e. the corresponding rotation takes body → world.

    Then world → body is given by q*:
        v_B = q* ⊗ v_W ⊗ q

    Args:
        v_W: (np.array) Vector in the world frame, shape (3,) or broadcastable to that.
        q:   (np.array) Quaternion [qw, qx, qy, qz] representing orientation of body in world.
    Returns:
        v_B: (np.array) The same vector expressed in the body frame, shape (3,).
    """
    q = np.asarray(q, dtype=float)
    q_conj = quat_conj(q)  # world → body
    return quat_rotate(q_conj, v_W)

def world_to_body_ca(v_W, q, eps=1e-12):
    q_conj = quat_conj_ca(q)  # world → body
    return quat_rotate_ca(q_conj, v_W, eps)


#############################################################
# EXAMPLE USAGE
#############################################################

if __name__ == "__main__":
    
    q = np.array([np.cos(np.pi/4), 0, 0, np.sin(np.pi/4)])
    R = quat_to_rot_matrix(q)
    print(R)
