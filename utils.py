"""
utils.py
File for mathematical helper functions for NDT SLAM
Author: Ashwin Kanhere
Date Created: 15th April 2019
"""
import numpy as np


def transform_pts(points, affine_trans):
    """
    Function to transform all given points by the given affine transform T
    :param points: Set of points forming a point cloud Nx3
    :param affine_trans: Affine transformation matrix generated using transforms3d
    :return: transformed_points: Transformed points
    """
    N = points.shape[0]
    test1 = np.ones([1, N])
    homogeneous_points = np.vstack((points.T, np.ones([1, N])))
    transform_homogeneous_points = np.transpose(np.matmul(affine_trans, homogeneous_points))
    transformed_points = transform_homogeneous_points[:, :3]
    return transformed_points


# Sending these functions to the end of the file as they're deprecated by using transforms3d.py
def rotation_matrix(angle, axis='z'):
    """
    Function yields rotation matrix that will rotate point about the coordinate axis by given angle and axis
    :param angle: Angle in degrees
    :param axis: Axis of rotation ('x', 'y' or 'z')
    :return: rot_matrix: The resultant rotation matrix
    """
    angle = np.deg2rad(angle)
    if axis == 'x':
        c = np.cos(angle)
        s = np.sin(angle)
        rot_matrix = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    elif axis == 'y':
        c = np.cos(angle)
        s = np.sin(angle)
        rot_matrix = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    elif axis == 'z':
        c = np.cos(angle)
        s = np.sin(angle)
        rot_matrix = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    else:
        raise Exception('Wrong value for axis input.')
    return rot_matrix


def eul2dcm(euler_angle):
    """
    Function to convert given euler angles (in degrees) to a DCM
    :param euler_angle: Euler angles (in degrees)
    :return: dcm: Required DCM
    """
    phi = euler_angle[0]
    theta = euler_angle[1]
    psi = euler_angle[2]
    rot_mat_x = rotation_matrix(phi, axis='x')
    rot_mat_y = rotation_matrix(theta, axis='y')
    rot_mat_z = rotation_matrix(psi, axis='z')
    dcm = np.matmul(rot_mat_z, np.matmul(rot_mat_y, rot_mat_x))
    return dcm


def dcm2eul(dcm):
    """
    Function to convert given direction cosine matrix into euler angles
    :param dcm: Given DCM
    :return: euler_angle: Corresponding euler angles (returned in degrees)
    """
    A11 = dcm[0][0]
    A12 = dcm[0][1]
    A13 = dcm[0][2]
    A21 = dcm[1][0]
    A22 = dcm[1][1]
    A23 = dcm[1][2]
    A31 = dcm[2][0]
    A32 = dcm[2][1]
    A33 = dcm[2][2]
    euler_angle = np.zeros([3, 1])
    # Implementing formula from Wikipedia now
    euler_angle[0] = -1*np.arctan2(A32, A33)
    euler_angle[1] = -np.arcsin(A31)
    euler_angle[2] = -1*np.arctan2(A12, A11)
    """
    Implementation from Wolfram Alpha
    denom = np.sqrt(A11**2 + A21**2)
    if not denom < 1e-6:
        euler_angle[0] = np.arctan2(A31, A33)
        euler_angle[1] = np.arctan2(-A31, denom)
        euler_angle[2] = np.arctan2(A21, A11)
    else:
        euler_angle[0] = np.arctan2(-A23, A22)
        euler_angle[1] = np.arctan2(-A31, denom)
        euler_angle[2] = 0
    """
    euler_angle = np.rad2deg(euler_angle)
    return euler_angle
