"""
odometry.py
Functions to calculate odometry given a point cloud and corresponding ndt_clouds
Author: Ashwin Kanhere
Date created: 9th June 2019
Last modified: 10th June 2019
"""

import numpy as np
import utils
from scipy.optimize import minimize


def odometry(ndt_cloud, test_pc):
    """
    Function to find the best traansformation (in the form of a translation, Euler angle vector)
    :param test_pc: Point cloud which has to be matched to the existing NDT approximation
    :return: odom_vector: The resultant odometry vector (Euler angles in degrees)
    """
    global obj_neval
    obj_neval = 0
    global jacob_neval
    jacob_neval = 0
    global hess_neval
    hess_neval = 0
    test_xyz = test_pc[:, :3]
    initial_odom = np.array([0, 0, 0, 0, 0, 0])
    # xlim, ylim, zlim = find_pc_limits(test_xyz)
    # odometry_bounds = Bounds([-xlim, -ylim, -zlim, -180.0, -90.0, -180.0], [xlim, ylim, zlim, 180.0, 90.0, 180.0])
    # TODO: Any way to implement bounds on the final solution?
    res = minimize(objective, initial_odom, method='Newton-CG', jac=jacobian_vect,
                   hess=hessian_vect, args=(ndt_cloud, test_xyz), options={'disp': True, 'maxiter': 10})
    odom_vector = res.x
    # Return odometry in navigational frame of reference
    return odom_vector


def objective(odometry_vector, ndt_cloud, test_pc):
    """
    A function that combines functions for transformation of a point cloud and the computation of likelihood (score)
    :param odometry_vector: Candidate odometry vector
    :param ndt_cloud: The NDT cloud with respect to which the odometry is required
    :param test_pc: Input point cloud
    :return: objective_val: Maximization objective which is the likelihood of transformed point cloud for the given NDT
    """
    global obj_neval
    global jacob_neval
    global hess_neval
    transformed_pc = utils.transform_pc(odometry_vector, test_pc)
    obj_value = -1 * ndt_cloud.find_likelihood(transformed_pc)
    print('Objective iteration: {:4d}'.format(obj_neval), 'Jacobian iteration: {:4d}'.format(jacob_neval),
          'Hessian iteration: {:4d}'.format(hess_neval), '\n Objective Value: {:10.4f}'.format(obj_value), ' Odometry:',
          ' x: {:2.5f}'.format(odometry_vector[0]), ' y: {:2.5f}'.format(odometry_vector[1]),
          ' z: {:2.5f}'.format(odometry_vector[2]), ' Phi: {:2.5f}'.format(odometry_vector[3]),
          ' Theta:{:2.5f}'.format(odometry_vector[4]), ' Psi: {:2.5f}'.format(odometry_vector[5]))
    obj_neval += 1
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
            sigma = ndt_cloud.stats[key]['sigma']
            sigma_inv = np.linalg.inv(sigma)
            g = np.zeros(6)
            q = value - mu
            if q.ndim < 2:
                q = np.atleast_2d(q)
            delq_delt = find_delqdelt_vect(odometry_vector, value)
            for i in range(6):
                g[i] = np.sum(np.diag(np.matmul(q, np.matmul(sigma_inv, delq_delt[:, :, i].T))) *
                              np.exp(-0.5 * np.diag(np.matmul(q, np.matmul(sigma_inv, q.T)))))  # Check this out
            jacobian_val += g
    global jacob_neval
    jacob_neval += 1
    return jacobian_val


def hessian_vect(odometry_vector, ndt_cloud, test_pc):
    """
    Vectorized implementation of the function to return an approximation of the Hessian of the likelihood objective
    for the odometry calculation
    :param odometry_vector: The point at which the Hessian is evaluated
    :param ndt_cloud: The NDT cloud with respect to which the odometry is required
    :param test_pc: The point cloud for which the optimization is being carried out
    :return: hessian_val: The Hessian matrix of the objective w.r.t. the odometry vector
    """
    hessian_val = np.zeros([6, 6])
    transformed_pc = utils.transform_pc(odometry_vector, test_pc)
    points_in_voxels = ndt_cloud.bin_in_voxels(transformed_pc)
    for key, value in points_in_voxels.items():
        if key in ndt_cloud.stats:
            if value.ndim == 1:
                value = np.atleast_2d(value)
            mu = ndt_cloud.stats[key]['mu']
            sigma = ndt_cloud.stats[key]['sigma']
            sigma_inv = np.linalg.inv(sigma)
            q = value - mu
            delq_delt = find_delqdelt_vect(odometry_vector, value)
            del2q_deltnm = find_del2q_deltnm_vect(odometry_vector, value)
            temp_hess = np.zeros([6, 6])
            for i in range(6):
                for j in range(6):
                    # Terms written out separately for increased readability in Hessian calculation
                    term1 = np.diag(np.exp(-0.5 * np.matmul(q, np.matmul(sigma_inv, q.T))))
                    term2 = np.diag(np.matmul(delq_delt[:, :, j], np.matmul(sigma_inv, delq_delt[:, :, i].T)))
                    term3 = np.diag(np.matmul(q, np.matmul(sigma_inv, del2q_deltnm[:, :, i, j].T)))
                    term4 = np.diag(-np.matmul(q, np.matmul(sigma_inv, delq_delt[:, :, i].T)))
                    term5 = np.diag(np.matmul(q, np.matmul(sigma_inv, delq_delt[:, :, j].T)))
                    temp_hess[i, j] = np.sum(term1*(term2 + term3 - term4*term5))
            hessian_val += temp_hess
    global hess_neval
    hess_neval += 1
    return hessian_val


def hessian(odometry_vector, ndt_cloud, test_pc):
    """
    Function to return an approximation of the Hessian of the likelihood objective for the odometry calculation
    :param odometry_vector: The point at which the Hessian is evaluated
    :param ndt_cloud: The NDT cloud with respect to which the odometry is required
    :param test_pc: The point cloud for which the optimization is being carried out
    :return: hessian_val: The Hessian matrix of the objective w.r.t. the odometry vector
    """
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


################################################################
# Helper functions for odometry calculation follow
################################################################


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
    :return: del2q_deltnm: Nx3x6x6 matrix of del2/deltndeltm
    """
    N = points.shape[0]
    del2q_deltnm = np.zeros([N, 3, 6, 6])
    delta = 1.5e-08
    for i in range(6):
        odometry_new = np.zeros(6)
        odometry_new[i] = odometry_new[i] + delta
        points_new = utils.transform_pc(odometry_new, points)
        # Assuming that the incremental change allows us to directly add a rotation instead of an incremental change
        odometry_new += odometry_vector
        del2q_deltnm[:, :, :, i] = (find_delqdelt(odometry_new, points_new) - find_delqdelt(odometry_vector, points)) / delta
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
