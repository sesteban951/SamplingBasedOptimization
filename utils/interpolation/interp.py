##
#
# Assortment of interpolation and finite difference utilities.
#
##

# standard imports
import numpy as np

# custom imports
from utils.kinematics import kin


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
    
    # compute quaternion error
    q_err = kin.quat_diff(q1, q2)

    # compute the logarithm of the quaternion error
    log_q_err = kin.quat_log(q_err)

    # compute the angular velocity
    omega = log_q_err / dt

    return omega



#############################################################
# EXAMPLE USAGE
#############################################################

if __name__ == "__main__":
    
    q = np.array([np.cos(np.pi/4), 0, 0, np.sin(np.pi/4)])
    R = kin.quat_to_rot_matrix(q)
    print(R)