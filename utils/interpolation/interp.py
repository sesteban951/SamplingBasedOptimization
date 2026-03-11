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


def build_yaw_slerp_keyframes(yaw_start, yaw_goal, max_step=0.95 * np.pi):
    """
    Build yaw-only quaternion keyframes with <= max_step angular increments.
    """
    dyaw = yaw_goal - yaw_start
    n_seg = max(1, int(np.ceil(abs(dyaw) / max_step)))
    yaw_keyframes = yaw_start + np.linspace(0.0, dyaw, n_seg + 1)
    quat_keyframes = np.array([kin.yaw_to_quat(yaw) for yaw in yaw_keyframes], dtype=float)

    # Keep a continuous quaternion sign convention across segments.
    for i in range(1, quat_keyframes.shape[0]):
        if np.dot(quat_keyframes[i - 1], quat_keyframes[i]) < 0.0:
            quat_keyframes[i] = -quat_keyframes[i]

    return quat_keyframes


def build_general_slerp_keyframes(quat_start, roll_total, pitch_total, yaw_total, max_step=0.95 * np.pi):
    """
    Build quaternion keyframes for an arbitrary in-air rotation maneuver.

    Args:
        quat_start: (np.array) Starting quaternion [qw, qx, qy, qz].
        roll_total, pitch_total, yaw_total: (float) Total relative rotation in radians.
        max_step: (float) Maximum angular step per segment.
    Returns:
        quat_keyframes: (np.array) Shape (n_seg+1, 4) quaternion keyframes.
    """
    total_angle = np.sqrt(roll_total**2 + pitch_total**2 + yaw_total**2)
    n_seg = max(1, int(np.ceil(total_angle / max_step)))

    quat_keyframes = np.zeros((n_seg + 1, 4))
    quat_keyframes[0] = quat_start / np.linalg.norm(quat_start)

    for i in range(1, n_seg + 1):
        frac = i / n_seg
        q_rel = kin.euler_ZYX_to_quat(frac * roll_total, frac * pitch_total, frac * yaw_total)
        quat_keyframes[i] = kin.quat_mult(q_rel, quat_start)
        quat_keyframes[i] /= np.linalg.norm(quat_keyframes[i])

    # Enforce continuous sign convention across keyframes
    for i in range(1, quat_keyframes.shape[0]):
        if np.dot(quat_keyframes[i - 1], quat_keyframes[i]) < 0.0:
            quat_keyframes[i] = -quat_keyframes[i]

    return quat_keyframes


def sample_piecewise_slerp(alpha, quat_keyframes):
    """
    Sample piecewise SLERP over keyframes for alpha in [0, 1].
    """
    alpha = float(np.clip(alpha, 0.0, 1.0))
    if alpha >= 1.0:
        return quat_keyframes[-1]

    s = alpha * (quat_keyframes.shape[0] - 1)
    i = min(int(np.floor(s)), quat_keyframes.shape[0] - 2)
    alpha_local = s - i
    return slerp(quat_keyframes[i], quat_keyframes[i + 1], alpha_local)


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
