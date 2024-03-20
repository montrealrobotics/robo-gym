#!/usr/bin/env python3

import math
import numpy as np
from scipy.spatial.transform import Rotation as R

# map axes strings to/from tuples of inner axis, parity, repetition, frame
_AXES2TUPLE = {
    'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
    'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
    'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
    'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
    'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
    'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
    'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
    'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}

_TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())

# axis sequences for Euler angles
_NEXT_AXIS = [1, 2, 0, 1]

# epsilon for testing whether a number is close to zero
_EPS = np.finfo(float).eps * 4.0

def normalize_angle_rad(a):
    """Normalize angle (in radians) to +-pi

    Args:
        a (float): Angle (rad).

    Returns:
        float: Normalized angle (rad).

    """

    return (a + math.pi) % (2 * math.pi) - math.pi

def point_inside_circle(x,y,center_x,center_y,radius):
    """Check if a point is inside a circle.

    Args:
        x (float): x coordinate of the point.
        y (float): y coordinate of the point.
        center_x (float): x coordinate of the center of the circle.
        center_y (float): y coordinate of the center of the circle.
        radius (float): radius of the circle (m).

    Returns:
        bool: True if the point is inside the circle.

    """

    dx = abs(x - center_x)
    dy = abs(y - center_y)
    if dx>radius:
        return False
    if dy>radius:
        return False
    if dx + dy <= radius:
        return True
    if dx**2 + dy**2 <= radius**2:
        return True
    else:
        return False

def rotate_point(x,y,theta):
    """Rotate a point around the origin by an angle theta.

    Args:
        x (float): x coordinate of point.
        y (float): y coordinate of point.
        theta (float): rotation angle (rad).

    Returns:
        list: [x,y] coordinates of rotated point.

    """
    """
    Rotation of a point by an angle theta(rads)
    """
    x_r = x * math.cos(theta) - y * math.sin(theta)
    y_r = x * math.sin(theta) + y * math.cos(theta)
    return [x_r, y_r]

def cartesian_to_polar_2d(x_target, y_target, x_origin = 0, y_origin = 0):
    """Transform 2D cartesian coordinates to 2D polar coordinates.

    Args:
        x_target (type): x coordinate of target point.
        y_target (type): y coordinate of target point.
        x_origin (type): x coordinate of origin of polar system. Defaults to 0.
        y_origin (type): y coordinate of origin of polar system. Defaults to 0.

    Returns:
        float, float: r,theta polard coordinates.

    """

    delta_x = x_target - x_origin
    delta_y = y_target - y_origin
    polar_r = np.sqrt(delta_x**2+delta_y**2)
    polar_theta = np.arctan2(delta_y,delta_x)

    return polar_r, polar_theta

def cartesian_to_polar_3d(cartesian_coordinates):
    """Transform 3D cartesian coordinates to 3D polar coordinates.

    Args:
        cartesian_coordinates (list): [x,y,z] coordinates of target point.

    Returns:
        list: [r,phi,theta] polar coordinates of point.

    """

    x = cartesian_coordinates[0]
    y = cartesian_coordinates[1]
    z = cartesian_coordinates[2]
    r =  np.sqrt(x**2+y**2+z**2)
    #? phi is defined in [-pi, +pi]
    phi = np.arctan2(y,x)
    #? theta is defined in [0, +pi]
    theta = np.arccos(z/r)

    return [r,theta,phi]

def downsample_list_to_len(data, output_len):
    """Downsample a list of values to a specific length.

    Args:
        data (list): Data to downsample.
        output_len (int): Length of the downsampled list.

    Returns:
        list: Downsampled list.

    """

    assert output_len > 0
    assert output_len <= len(data)

    temp = np.linspace(0, len(data)-1, num=output_len)
    temp = [int(round(x)) for x in temp]

    assert len(temp) == len(set(temp))

    ds_data = []
    for index in temp:
        ds_data.append(data[index])

    return ds_data

def change_reference_frame(point, translation, quaternion):
    """Transform a point from one reference frame to another, given
        the translation vector between the two frames and the quaternion
        between  the two frames.

    Args:
        point (array_like,shape(3,) or shape(N,3)): x,y,z coordinates of the point in the original frame
        translation (array_like,shape(3,)): translation vector from the original frame to the new frame 
        quaternion (array_like,shape(4,)): quaternion from the original frame to the new frame

    Returns:
        ndarray,shape(3,): x,y,z coordinates of the point in the new frame.
        
    """

    #point = [1,2,3]
    #point = np.array([1,2,3])
    #point = np.array([[11,12,13],[21,22,23]]) # point.shape = (2,3) # point (11,12,13)  and point (21,22,23)

    # Apply rotation
    r = R.from_quat(quaternion)
    rotated_point = r.apply(np.array(point))
    # Apply translation
    translated_point = np.add(rotated_point, np.array(translation))

    return translated_point

# Inverts a homogeneous transformation matrix
def trans_inv(T):
    R, p = T[:3, :3], T[:3, 3]
    Rt = np.array(R).T
    return np.r_[np.c_[Rt, -np.dot(Rt, p)], [[0, 0, 0, 1]]]


# Calculates 2D Rotation Matrix given a desired yaw angle
def yaw_to_rotation_matrix(yaw):
    R_z = np.array([[math.cos(yaw), -math.sin(yaw)],
                    [math.sin(yaw), math.cos(yaw)],
                    ])
    return R_z


# Transform a Six Element Pose vector to a Transformation Matrix
def pose_to_transformation_matrix(pose):
    mat = np.identity(4)
    mat[:3, :3] = euler_angles_to_rotation_matrix(pose[3:])
    mat[:3, 3] = pose[:3]
    return mat


def euler_angles_to_rotation_matrix(theta):
    """Calculates rotation matrix given euler angles in 'xyz' sequence

    :param theta: list of 3 euler angles
    :return: 3x3 rotation matrix equivalent to the given euler angles
    """
    return euler_matrix(theta[0], theta[1], theta[2], axes="sxyz")[:3, :3]


def rotation_matrix_to_euler_angles(R):
    """Calculates euler angles given rotation matrix in 'xyz' sequence

    :param R: 3x3 rotation matrix
    :return: list of three euler angles equivalent to the given rotation matrix
    """
    return list(euler_from_matrix(R, axes="sxyz"))


def euler_matrix(ai, aj, ak, axes='sxyz'):
    """Return homogeneous rotation matrix from Euler angles and axis sequence.

    ai, aj, ak : Euler's roll, pitch and yaw angles
    axes : One of 24 axis sequences as string or encoded tuple
    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes]
    except (AttributeError, KeyError):
        _ = _TUPLE2AXES[axes]
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i+parity]
    k = _NEXT_AXIS[i-parity+1]

    if frame:
        ai, ak = ak, ai
    if parity:
        ai, aj, ak = -ai, -aj, -ak

    si, sj, sk = math.sin(ai), math.sin(aj), math.sin(ak)
    ci, cj, ck = math.cos(ai), math.cos(aj), math.cos(ak)
    cc, cs = ci*ck, ci*sk
    sc, ss = si*ck, si*sk

    M = np.identity(4)
    if repetition:
        M[i, i] = cj
        M[i, j] = sj*si
        M[i, k] = sj*ci
        M[j, i] = sj*sk
        M[j, j] = -cj*ss+cc
        M[j, k] = -cj*cs-sc
        M[k, i] = -sj*ck
        M[k, j] = cj*sc+cs
        M[k, k] = cj*cc-ss
    else:
        M[i, i] = cj*ck
        M[i, j] = sj*sc-cs
        M[i, k] = sj*cc+ss
        M[j, i] = cj*sk
        M[j, j] = sj*ss+cc
        M[j, k] = sj*cs-sc
        M[k, i] = -sj
        M[k, j] = cj*si
        M[k, k] = cj*ci
    return M


def euler_from_matrix(matrix, axes='sxyz'):
    """Return Euler angles from rotation matrix for specified axis sequence.

    axes : One of 24 axis sequences as string or encoded tuple

    Note that many Euler angle triplets can describe one matrix.

    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _ = _TUPLE2AXES[axes]
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i+parity]
    k = _NEXT_AXIS[i-parity+1]

    M = np.array(matrix, dtype=np.float64, copy=False)[:3, :3]
    if repetition:
        sy = math.sqrt(M[i, j]*M[i, j] + M[i, k]*M[i, k])
        if sy > _EPS:
            ax = math.atan2( M[i, j],  M[i, k])
            ay = math.atan2( sy,       M[i, i])
            az = math.atan2( M[j, i], -M[k, i])
        else:
            ax = math.atan2(-M[j, k],  M[j, j])
            ay = math.atan2( sy,       M[i, i])
            az = 0.0
    else:
        cy = math.sqrt(M[i, i]*M[i, i] + M[j, i]*M[j, i])
        if cy > _EPS:
            ax = math.atan2( M[k, j],  M[k, k])
            ay = math.atan2(-M[k, i],  cy)
            az = math.atan2( M[j, i],  M[i, i])
        else:
            ax = math.atan2(-M[j, k],  M[j, j])
            ay = math.atan2(-M[k, i],  cy)
            az = 0.0

    if parity:
        ax, ay, az = -ax, -ay, -az
    if frame:
        ax, az = az, ax
    return ax, ay, az