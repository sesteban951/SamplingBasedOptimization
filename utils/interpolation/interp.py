##
#
# Assortment of interpolation and finite difference utilities.
#
##

# standard imports
import numpy as np


def lerp(v1, v2, alpha):
    """
    Linear interpolation between two vectors.

    Args:
        v1: (np.array) First vector.
        v2: (np.array) Second vector.
        alpha: (float) Interpolation coefficient (between 0 and 1).
    Returns:
        v: (np.array) Interpolated vector.
    """
    # make sure that alpha is between 0 and 1
    assert 0 <= alpha <= 1, "Alpha must be between 0 and 1."

    # interpolate
    v = (1 - alpha) * v1 + alpha * v2

    return v


def slerp(q1, q2, alpha):
    """
    Spherical linear interpolation (SLERP) between two quaternions.
    In in [qw, qx, qy, qz] format.

    Args:
        q1: (np.array) First quaternion.
        q2: (np.array) Second quaternion.
        alpha: (float) Interpolation coefficient (between 0 and 1).
    Returns:
        v: (np.array) Interpolated quaternion.
    """
    # make sure that alpha is between 0 and 1
    assert 0 <= alpha <= 1, "Alpha must be between 0 and 1."

    # normalize the quaternions
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)

    # compute the dot product
    dot = np.dot(q1, q2)

    # take the shortest path if cos(theta) = dot < 0.0
    if dot < 0.0:
        q2 = -q2
        dot = -dot
    dot = np.clip(dot, -1.0, 1.0)  # clip to avoid numerical errors

    # compute the angle between the quaternions
    theta = np.arccos(dot)

    # if the angle is small, use linear interpolation
    if theta < 1e-6:
        q = lerp(q1, q2, alpha)
        q = q / np.linalg.norm(q)

    # use spherical linear interpolation (SLERP)
    else:
        sin_theta = np.sin(theta)
        term1 = np.sin((1 - alpha) * theta) / sin_theta
        term2 = np.sin(alpha * theta) / sin_theta
        q = term1 * q1 + term2 * q2
        q = q / np.linalg.norm(q)

    return q


def vec_finite_diff(v1, v2, dt):
    """
    Compute the finite difference between two vectors.

    Args:
        v1: (np.array) First vector.
        v2: (np.array) Second vector.
        dt: (float) Time step between the two vectors.
    Returns:
        dv: (np.array) Finite difference between the two vectors.
    """
    # make sure that dt is positive
    assert dt > 0, "Time step must be positive."

    return (v2 - v1) / dt

def quat_finite_diff(q1, q2, dt):
    """
    Compute angular vel from finite difference of two quaternions in [qw, qx, qy, qz] format.
    Micro Lie Theory eq.(133): https://arxiv.org/pdf/1812.01537
    
    Args:
        q1: (np.array) First quaternion.
        q2: (np.array) Second quaternion.
        dt: (float) Time step between the two quaternions.
    Returns:
        omega: (np.array) Finite difference between the two quaternions.
    """
    # make sure that dt is positive
    assert dt > 0, "Time step must be positive."

    # compute the relative rotation
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    qd = quat_mult(q2, quat_conj(q1))  # equivalent of dv = v2 - v1 for quaternions
    qd = (qd / np.linalg.norm(qd))

    # take the shortest path if qd[0] < 0.0
    if qd[0] < 0:
        qd = -qd

    # extract the vector part of the quaternion
    w = qd[0]
    v = qd[1:]
    v_norm = np.linalg.norm(v)

    # small enough to do linear approximation
    if v_norm < 1e-6:
        omega = 2 * v / dt
    # use the angle-axis representation to compute the angular velocity
    else:
        u = v / v_norm
        angle = 2 * np.arctan2(v_norm, w)
        omega = angle * u / dt

    return omega


def quat_conj(q):
    """
    Compute the conjugate of a quaternion in [qw, qx, qy, qz] format.

    Args:
        q: (np.array) Quaternion.
    Returns:
        q_conj: (np.array) Conjugate of the quaternion.
    """
    return np.array([q[0], -q[1], -q[2], -q[3]])

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
    
# quaternion to rotation matrix
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

# vector in body frame to world frame
def body_to_world(v_B, q):
    """
    Transform a vector from the body frame to the world frame using a quaternion.
    Assumes q is describing the orientation of the body frame relative to the
    world frame, so that R transforms vectors from body frame to world frame.

    Args:
        v_B: (np.array) Some vector in the body frame.
        q:   (np.array) Quaternion representing the orientation of the body frame in the world frame.
    Returns:
        v_W: (np.array) The same vector transformed to the world frame.
    """
    R = quat_to_rot_matrix(q)
    v_W = R @ v_B
    return v_W

#############################################################
# EXAMPLE USAGE
#############################################################

if __name__ == "__main__":

    q = np.array([np.cos(np.pi/4), 0, 0, np.sin(np.pi/4)])  # +90deg about z
    R = quat_to_rot_matrix(q)
    
    print(R)
