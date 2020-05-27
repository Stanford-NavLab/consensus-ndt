"""
utils.py
File for mathematical helper functions for NDT SLAM
Author: Ashwin Kanhere
Date Created: 15th April 2019
"""
import numpy as np
import transforms3d


def transform_pts(points, affine_trans):
    """
    Function to transform all given points by the given affine transform T
    :param points: Set of points forming a point cloud Nx3
    :param affine_trans: Affine transformation matrix generated using transforms3d
    :return: transformed_points: Transformed points
    """
    N = points.shape[0]
    homogeneous_points = np.vstack((points.T, np.ones([1, N])))
    transform_homogeneous_points = np.transpose(np.matmul(affine_trans, homogeneous_points))
    transformed_points = transform_homogeneous_points[:, :3]
    return transformed_points


def transform_pc(odometry_vector, original_pc):
    """
    Function to transform a point cloud according to the given odometry vector
    :param odometry_vector: [tx, ty, tz, phi, theta, psi] (angles in degrees)
    :param original_pc: original point cloud that is to be transformed
    :return:
    """
    if original_pc.ndim == 1:
        original_pc = np.atleast_2d(original_pc)
    phi = np.deg2rad(odometry_vector[3])
    theta = np.deg2rad(odometry_vector[4])
    psi = np.deg2rad(odometry_vector[5])
    R = transforms3d.euler.euler2mat(phi, theta, psi, 'rxyz')  # 'rxyz' convention for this project
    T = odometry_vector[:3]
    Z = np.ones([3])
    A = transforms3d.affines.compose(T, R, Z)
    transformed_pc = transform_pts(original_pc, A)
    return transformed_pc


def odometry_difference(odom_1_ref, odom_2_ref):
    """
    Returns the odometry vector of the second input from the first input (both inputs share same reference)
    :param odom_1_ref: The odometry of the first point cloud. This point cloud is the reference for the delta odometry
    :param odom_2_ref: The odometry of the second point cloud.
    :return: delta_odom: The odometry that transforms points in pc1 to the origin of pc2
    """
    phi_1 = np.deg2rad(odom_1_ref[3])
    theta_1 = np.deg2rad(odom_1_ref[4])
    psi_1 = np.deg2rad(odom_1_ref[5])
    R_1 = transforms3d.euler.euler2mat(phi_1, theta_1, psi_1, 'rxyz')
    T_1 = odom_1_ref[:3]
    Z = np.ones([3])
    A_1 = transforms3d.affines.compose(T_1, R_1, Z)
    phi_2 = np.deg2rad(odom_2_ref[3])
    theta_2 = np.deg2rad(odom_2_ref[4])
    psi_2 = np.deg2rad(odom_2_ref[5])
    R_2 = transforms3d.euler.euler2mat(phi_2, theta_2, psi_2, 'rxyz')
    T_2 = odom_2_ref[:3]
    A_2 = transforms3d.affines.compose(T_2, R_2, Z)
    # Since for the previous one, it was A_2 = A_delta*A_1
    A_delta = np.matmul(A_2, np.linalg.inv(A_1))
    delta_odom = affine_to_odometry(A_delta)
    return delta_odom


def combine_odometry(odom_1_ref, odom_12_delta):
    """
    Combines odom_vector_1 and odom_vector_2 and returns a odom_vector for the second pc that is from the same
     reference as the first one
    :param odom_1_ref: The odometry vector of the first pc that is referenced to the common reference
    :param odom_12_delta: The odometry vector of the second pc that is referenced to the first pc
    :return: odom_2_ref: The odometry vector of the second point cloud that has the same reference as the first one
    """
    phi_1 = np.deg2rad(odom_1_ref[3])
    theta_1 = np.deg2rad(odom_1_ref[4])
    psi_1 = np.deg2rad(odom_1_ref[5])
    R_1 = transforms3d.euler.euler2mat(phi_1, theta_1, psi_1, 'rxyz')
    T_1 = odom_1_ref[:3]
    Z = np.ones([3])
    A_1 = transforms3d.affines.compose(T_1, R_1, Z)
    phi_2 = np.deg2rad(odom_12_delta[3])
    theta_2 = np.deg2rad(odom_12_delta[4])
    psi_2 = np.deg2rad(odom_12_delta[5])
    R_2 = transforms3d.euler.euler2mat(phi_2, theta_2, psi_2, 'rxyz')
    T_2 = odom_12_delta[:3]
    A_2 = transforms3d.affines.compose(T_2, R_2, Z)
    A_2_ref = np.matmul(A_2, A_1)
    odom_2_ref = affine_to_odometry(A_2_ref)
    return odom_2_ref


def affine_to_odometry(A):
    """
    Convert Affine transformation matrix to equivalent odometry vector
    :param A: Affine transformation matrix, following transforms3d convention
    :return odometry_vector: Vector of [x, y, z, phi, theta, psi] with the angles in degrees
    """
    T, R, _, _ = transforms3d.affines.decompose44(A)
    phi, theta, psi = transforms3d.euler.mat2euler(R, 'rxyz')
    odometry_vector = np.zeros(6)
    odometry_vector[:3] = T
    odometry_vector[3:6] = np.rad2deg(np.array([phi, theta, psi]))
    return odometry_vector


def invert_odom_transfer(odom):
    """
    Given odometry vector matching PC A to PC B, return odometry vector matching PC B to PC A
    :param odom: Odometry vector matching PC A to PC B
    :return inverted_odom: Odometry vector matching PC B to PC A
    """
    T = odom[:3]
    phi = np.deg2rad(odom[3])
    theta = np.deg2rad(odom[4])
    psi = np.deg2rad(odom[4])
    R = transforms3d.euler.euler2mat(phi, theta, psi, 'rxyz')
    Z = np.ones([3])
    A = transforms3d.affines.compose(T, R, Z)
    A_inv = np.linalg.inv(A)
    inverted_odom = affine_to_odometry(A_inv)
    return inverted_odom


def plot_averaged(run_stats):
    """
    Average results from multiple runs for plotting
    :param run_stats: Results from multiple runs
    :return : Averaged results from runs
    """
    assert (run_stats.ndim == 3)
    return np.mean(np.mean(run_stats, axis=2), axis=1)

