"""
odometry.py
Functions to calculate odometry given a point cloud and corresponding ndt_clouds
Author: Ashwin Kanhere
Date created: 9th June 2019
Last modified: 9th June 2019
"""

import numpy as np
import utils
# TODO: Remove all debugging print statements from this file


def objective(odometry_vector, ndt_cloud, test_pc):
    """
    A function that combines functions for transformation of a point cloud and the computation of likelihood (score)
    :param odometry_vector: Candidate odometry vector
    :param ndt_cloud: The NDT cloud with respect to which the odometry is required
    :param test_pc: Input point cloud
    :return: objective_val: Maximization objective which is the likelihood of transformed point cloud for the given NDT
    """
    transformed_pc = utils.transform_pc(odometry_vector, test_pc)
    obj_value = -1 * ndt_cloud.find_likelihood(transformed_pc)
    """
    print('Objective: the objective value is ', -1 * ndt_cloud.find_likelihood(transformed_pc), ' for the odometry vector ',
          odometry_vector)
    """
    return obj_value


def jacobian(odometry_vector, ndt_cloud, test_pc):
    """
    Function to return the Jacobian of the likelihood objective for the odometry calculation
    :param odometry_vector: The point at which the Jacobian is to be evaluated
    :param ndt_cloud: The NDT cloud with respect to which the odometry is required
    :param test_pc: The point cloud for which the optimization is being performed
    :return: jacobian_val: The Jacobian matrix (of the objective w.r.t the odometry vector)
    """
    N = test_pc.shape[0]
    jacobian_val = np.zeros(6)
    transformed_pc = utils.transform_pc(odometry_vector, test_pc)
    transform_centers = ndt_cloud.find_voxel_center(transformed_pc)
    print('Woo hoo')
    for pt_num in range(N):
        # The output from find_voxel_center matches this implementation and the next line should yield the dict key
        center_key = tuple(transform_centers[pt_num][:])
        if center_key in ndt_cloud.stats:
            mu = ndt_cloud.stats[center_key]['mu']
            sigma = ndt_cloud.stats[center_key]['sigma']
            sigma_inv = np.linalg.inv(sigma)
            qx = test_pc[pt_num][0] - mu[0]
            qy = test_pc[pt_num][1] - mu[1]
            qz = test_pc[pt_num][2] - mu[2]
            q = np.array([[qx], [qy], [qz]])
            delq_delt = find_delqdelt(odometry_vector, np.atleast_2d(test_pc[pt_num, :]))
            g = np.zeros(6)
            for i in range(6):
                g[i] = np.matmul(q.T, np.matmul(sigma_inv, np.atleast_2d(delq_delt[:, i]).T)) * np.exp(
                    -0.5 * np.matmul(q.T, np.matmul(sigma_inv, q)))
            jacobian_val += g
    # print('The Jacobian has run')
    return jacobian_val


def jacobian_vect(odometry_vector, ndt_cloud, test_pc):
    """
    Function to return the Jacobian of the likelihood objective for the odometry calculation
    :param odometry_vector: The point at which the Jacobian is to be evaluated
    :param ndt_cloud: The NDT cloud with respect to which the odometry is required
    :param test_pc: The point cloud for which the optimization is being performed
    :return: jacobian_val: The Jacobian matrix (of the objective w.r.t the odometry vector)
    """
    jacobian_val = np.zeros(6)
    transformed_pc = utils.transform_pc(odometry_vector, test_pc)
    points_in_voxels = ndt_cloud.bin_in_voxels(transformed_pc)
    for key, value in points_in_voxels.items():
        if key in ndt_cloud.stats:
            # Vectorized implementation
            mu = ndt_cloud.stats[key]['mu']
            print(mu.shape)
            sigma = ndt_cloud.stats[key]['sigma']
            sigma_inv = np.linalg.inv(sigma)
            g = np.zeros(6)
            q = value - mu
            delq_delt = find_delqdelt_vect(odometry_vector, value)
            for i in range(6):
                g[i] = np.sum(np.diag(np.matmul(q, np.matmul(sigma_inv, delq_delt[:, :, i].T))) *
                              np.exp(-0.5 * np.diag(np.matmul(q, np.matmul(sigma_inv, q.T)))))  # Check this out
            jacobian_val += g
    return jacobian_val


def hessian(odometry_vector, ndt_cloud, test_pc):
    """
    Function to return an approximation of the Hessian of the likelihood objective for the odometry calculation
    :param odometry_vector: The point at which the Hessian is evaluated
    :param ndt_cloud: The NDT cloud with respect to which the odometry is required
    :param test_pc: The point cloud for which the optimization is being carried out
    :return: hessian_val: The Hessian matrix of the objective w.r.t. the odometry vector
    """
    # TODO: Vectorize this function
    N = test_pc.shape[0]
    hessian_val = np.zeros([6, 6])
    transformed_pc = utils.transform_pc(odometry_vector, test_pc)
    transform_centers = ndt_cloud.find_voxel_center(transformed_pc)
    for pt_num in range(N):
        center_key = tuple(transform_centers[pt_num, :])
        if center_key in ndt_cloud.stats:
            mu = ndt_cloud.stats[center_key]['mu']
            sigma = ndt_cloud.stats[center_key]['sigma']
            sigma_inv = np.linalg.inv(sigma)
            qx = test_pc[pt_num][0] - mu[0]
            qy = test_pc[pt_num][1] - mu[1]
            qz = test_pc[pt_num][2] - mu[2]
            q = np.array([[qx], [qy], [qz]])
            temp_hess = np.zeros([6, 6])
            delq_delt = find_delqdelt(odometry_vector, np.atleast_2d(test_pc[pt_num, :]))
            del2q_deltnm = find_del2q_deltnm(odometry_vector, np.atleast_2d(test_pc[pt_num, :]))
            for i in range(6):
                for j in range(6):
                    temp_hess[i, j] = -np.exp(-0.5 * np.matmul(q.T, np.matmul(sigma_inv, q))) * (
                            (-np.matmul(q.T, np.matmul(sigma_inv, np.atleast_2d(delq_delt[:, i]).T)) * (
                                np.matmul(q.T, np.matmul(sigma_inv, np.atleast_2d(delq_delt[:, j]).T)))) - (
                                np.matmul(q.T, np.matmul(sigma_inv, np.atleast_2d(del2q_deltnm[:, i, j]).T))) - (
                                np.matmul(np.atleast_2d(delq_delt[:, j]),
                                          np.matmul(sigma_inv, np.atleast_2d(delq_delt[:, i]).T))))
            hessian_val += temp_hess
    return hessian_val


def callback(odometry_vector, test_pc):
    """

    :param odometry_vector:
    :param test_pc:
    :return:
    """
    # TODO: Write and test this function
    global num_eval
    print('Callback: the objective value is ', objective(odometry_vector, test_pc),
          ' for the odometry vector ',
          odometry_vector)
    num_eval += 1
    return None

################################################################
# Helper functions for odometry calculation follow
################################################################


def find_pc_limits(pointcloud):
    """
    Function to find cartesian coordinate limits of given point cloud
    :param pointcloud: Given point cloud as an np.array of shape Nx3
    :return: xlim, ylim, zlim: Corresponding maximum absolute coordinate values
    """
    xlim = np.max(np.abs(pointcloud[:, 0]))
    ylim = np.max(np.abs(pointcloud[:, 1]))
    zlim = np.max(np.abs(pointcloud[:, 2]))
    return xlim, ylim, zlim


def find_delqdelt_vect(odometry_vector, points):
    """
    Vectorized implementation of the Jacobian function
    :param odometry_vector: Odometry vector at which Jacobian is needed
    :param points: The original set of points
    :return: delq_delt: A Nx3x6 matrix for the required partial derivative
    """
    N = points.shape[0]
    phi = np.deg2rad(odometry_vector[0])
    theta = np.deg2rad(odometry_vector[1])
    psi = np.deg2rad(odometry_vector[2])
    c1 = np.cos(phi)
    s1 = np.sin(phi)
    c2 = np.cos(theta)
    s2 = np.sin(theta)
    c3 = np.cos(psi)
    s3 = np.sin(psi)
    param_mat = np.zeros([6, 3, 3])
    param_mat[3, :, :] = np.array([[0, (-s1 * s3 + c3 * c1 * s2), (c1 * s3 + s1 * c3 * s2)],
                                   [0, - (s1 * c3 + c1 * s2 * s3), (c1 * c3 - s1 * s2 * s3)],
                                   [0, - c2 * c1, - s1 * c2]])
    param_mat[4, :, :] = np.array([[-s2 * c3,  c3 * s1 * c2, -c1 * c2 * c3],
                                   [s2 * s3, - s1 * c2 * s3, c1 * c2 * s3],
                                   [c2, s1 * s2, - c1 * s2]])
    param_mat[5, :, :] = np.array([[-c2 * s3, (c1 * c3 - s1 * s2 * s3), (s1 * c3 + c1 * s2 * s3)],
                                   [- c2 * c3, - (c1 * s3 + s1 * s2 * c3), (-s1 * s3 + c1 * s2 * c3)],
                                   [0, 0, 0]])
    delq_delt = np.zeros([N, 3, 6])  # Indexing in the first dimension based on experience
    delq_delt[:, :3, :3] = np.broadcast_to(np.eye(3), [N, 3, 3])
    for i in range(3, 6):
        delq_delt[:, :, i] = np.pi / 180.0 * np.matmul(points, param_mat[i, :, :])
    return delq_delt


def find_del2q_deltnm_vect(odometry_vector, points):
    """
    Vectorized implementation of function to return double partial derivative of point w.r.t. odometry vector parameters
    :param odometry_vector: Vector at which Hessian is being calculated
    :param q: Points which are being compared to the NDT Cloud
    :return: del2q_deltnm:
    """
    return del2q_deltnm


def find_delqdelt(odometry_vector, q):
    """
    Return a 3x6 matrix that captures partial q/ partial t_n
    :param odometry_vector:
    :param q: The original point which is then transformed
    :return: delq_delt: A 3x6 matrix for the required partial derivative
    """
    phi = np.deg2rad(odometry_vector[0])
    theta = np.deg2rad(odometry_vector[1])
    psi = np.deg2rad(odometry_vector[2])
    c1 = np.cos(phi)
    s1 = np.sin(phi)
    c2 = np.cos(theta)
    s2 = np.sin(theta)
    c3 = np.cos(psi)
    s3 = np.sin(psi)
    qx = q[0, 0]  # q is a 2D vector, the second indexing is to make values a scalar and fix following broadcast issues
    qy = q[0, 1]
    qz = q[0, 2]
    delq_delt = np.zeros([3, 6])
    delq_delt[:, 0] = np.array([1, 0, 0])
    delq_delt[:, 1] = np.array([0, 1, 0])
    delq_delt[:, 2] = np.array([0, 0, 1])
    delq_delt[:, 3] = np.pi / 180.0 * np.array([0,
                                                (-s1 * s3 + c3 * c1 * s2) * qx - (s1 * c3 + c1 * s2 * s3) * qy - c2 * c1 * qz,
                                                (c1 * s3 + s1 * c3 * s2) * qx + (c1 * c3 - s1 * s2 * s3) * qy - s1 * c2 * qz])
    delq_delt[:, 4] = np.pi / 180.0 * np.array([-s2 * c3 * qx + s2 * s3 * qy + c2 * qz,
                                                c3 * s1 * c2 * qx - s1 * c2 * s3 * qy + s1 * s2 * qz,
                                                -c1 * c2 * c3 * qx + c1 * c2 * s3 * qy - c1 * s2 * qz])
    delq_delt[:, 5] = np.pi / 180.0 * np.array([-c2 * s3 * qx - c2 * c3 * qy,
                                                (c1 * c3 - s1 * s2 * s3) * qx - (c1 * s3 + s1 * s2 * c3) * qy,
                      (s1 * c3 + c1 * s2 * s3) * qx + (-s1 * s3 + c1 * s2 * c3) * qy])
    return delq_delt


def find_del2q_deltnm(odometry_vector, q):
    """
    Function to return double partial derivative of point w.r.t. odometry vector parameters
    :param odometry_vector:
    :param q: Initial point
    :return:
    """
    del2q_deltnm = np.zeros([3, 6, 6])
    delta = 1.5e-08
    for i in range(6):
        odometry_new = np.zeros(6)
        odometry_new[i] = odometry_new[i] + delta
        q_new = np.transpose(utils.transform_pc(odometry_new, q))
        # Assuming that the incremental change allows us to directly add a rotation instead of an incremental change
        odometry_new += odometry_vector
        del2q_deltnm[:, :, i] = (find_delqdelt(odometry_new, q_new.T) - find_delqdelt(odometry_vector, q))/delta
    return del2q_deltnm
