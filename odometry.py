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
import time
import pptk

global obj_neval
obj_neval = 0
global jacob_neval
jacob_neval = 0
global hess_neval
hess_neval = 0


def odometry(ndt_cloud, test_pc, max_iter_pre=15, max_iter_post=10, integrity_filter=0.7):
    """
    Function to find the best traansformation (in the form of a translation, Euler angle vector)
    """
    global obj_neval
    obj_neval = 0
    global jacob_neval
    jacob_neval = 0
    global hess_neval
    hess_neval = 0
    test_xyz = test_pc[:, :3]
    initial_odom = np.zeros(6)
    # initial_odom = search_initial(ndt_cloud, test_xyz)
    res1 = minimize(objective, initial_odom, method='Newton-CG', jac=jacobian_vect,
                    hess=hessian_vect, args=(ndt_cloud, test_xyz), options={'disp': True, 'maxiter': max_iter_pre})
    # res = minimize(objective, initial_odom, method='BFGS', args=(ndt_cloud, test_xyz), options={'disp' : True})
    temp_odom_vector = res1.x
    transformed_xyz = utils.transform_pc(temp_odom_vector, test_xyz)
    ndt_cloud.find_integrity(transformed_xyz)
    ndt_cloud.filter_voxels_integrity(integrity_limit=integrity_filter)
    if max_iter_post != 0:
        res2 = minimize(objective, temp_odom_vector, method='Newton-CG', jac=jacobian_vect,
                        hess=hessian_vect, args=(ndt_cloud, test_xyz), options={'disp': True, 'maxiter': max_iter_post})
        odom_vector = res2.x
    else:
        odom_vector = np.copy(temp_odom_vector)
    # Return odometry in navigational frame of reference
    return odom_vector


def search_initial(ndt_cloud, test_pc, limit=0.5, case_num=10):
    """
    Returns an initial x, y guess based on a global grid evaluation of the objective function
    :param ndt_cloud:
    :param test_pc:
    :param limit: The search limit for the grid search (about the origin)
    :param case_num: Number of x and y coordinates (respectively) for which the grid search is performec
    :return: initial_odom: Initial guess of the minimizing point
    """
    t0 = time.time()
    print('Searching for initial point')
    x_search = np.linspace(-limit, limit, case_num)
    grid_objective = np.zeros([case_num, case_num])
    y_search = np.linspace(-limit, limit, case_num)
    for i in range(case_num):
        for j in range(case_num):
            odom = np.zeros(6)
            odom[0] = x_search[i]
            odom[1] = y_search[j]
            transformed_pc = utils.transform_pc(odom, test_pc)
            grid_objective[i, j] = ndt_cloud.find_likelihood(transformed_pc)
    x_ind, y_ind = np.unravel_index(np.argmax(grid_objective), [case_num, case_num])
    initial_odom = np.array([x_search[x_ind], y_search[y_ind], 0, 0, 0, 0])
    print('The search time is ', time.time() - t0)
    print('The initial odometry is ', initial_odom)
    return initial_odom


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
    if hess_neval % 5 == 0:
        print('Objective iteration: {:4d}'.format(obj_neval), 'Jacobian iteration: {:4d}'.format(jacob_neval),
                'Hessian iteration: {:4d}'.format(hess_neval), 'Objective Value: {:10.4f}'.format(obj_value))
    """
    , ' Odometry:',
          ' x: {:2.5f}'.format(odometry_vector[0]), ' y: {:2.5f}'.format(odometry_vector[1]),
          ' z: {:2.5f}'.format(odometry_vector[2]), ' Phi: {:2.5f}'.format(odometry_vector[3]),
          ' Theta:{:2.5f}'.format(odometry_vector[4]), ' Psi: {:2.5f}'.format(odometry_vector[5])
    """
    obj_neval += 1
    return obj_value


def jacobian(odometry_vector, ndt_cloud, test_pc):
    # FUNCTION DEPRECATED. VECTORIZED JACOBIAN SHOWS SUPERIOR PERFORMANCE
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
    _, transform_centers = ndt_cloud.find_voxel_center(transformed_pc)
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
            q = value - mu
            if q.ndim < 2:
                q = np.atleast_2d(q)
            delq_delt = find_delqdelt_vect(odometry_vector, value)
            first_term = np.matmul(q, np.matmul(sigma_inv, delq_delt.T))
            exp_term = np.exp(-0.5 * np.diag(np.matmul(q, np.matmul(sigma_inv, q.T))))
            g = np.sum(np.diagonal(first_term, axis1=1, axis2=2) * exp_term, axis=1)
            """
            old_g = np.zeros(6)
            for i in range(6):
                old_g[i] = np.sum(np.diag(np.matmul(q, np.matmul(sigma_inv, delq_delt[:, :, i].T))) *
                              np.exp(-0.5 * np.diag(np.matmul(q, np.matmul(sigma_inv, q.T)))))  # Check this out
            # The following print statement gave all 0s, the vectorization works
            error += np.max(np.abs(g - old_g))
            """
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
    # t0 = time.time()
    # total_main_loop_time = 0
    # total_sub_loop_time = 0
    # main_loop_neval = 0
    for key, value in points_in_voxels.items():
        # t1 = time.time()
        if key in ndt_cloud.stats:
            if value.ndim == 1:
                value = np.atleast_2d(value)
            mu = ndt_cloud.stats[key]['mu']
            sigma = ndt_cloud.stats[key]['sigma']
            sigma_inv = np.linalg.inv(sigma)
            # normal_factor = 1e-8
            q = value - mu
            delq_delt = find_delqdelt_vect(odometry_vector, value)
            del2q_deltnm = find_del2q_deltnm_vect(odometry_vector, value)
            # TODO: Use new found knowledge of matrix multiplication to make this more stable (replicable too)
            # t2 = time.time()
            exp_term = np.diag(np.exp(-0.5 * np.matmul(q, np.matmul(sigma_inv, q.T))))
            temp1 = np.einsum('abi,jbc->aijc', delq_delt, np.matmul(sigma_inv, delq_delt.T))
            term1 = np.sum(np.diagonal(temp1, axis1=0, axis2=3)*exp_term, axis=2)
            term2 = np.sum(np.diagonal(np.matmul(q, np.matmul(sigma_inv, del2q_deltnm.T)), axis1=2, axis2=3)*exp_term,
                           axis=2)
            temp3 = np.diagonal(-np.matmul(q, np.matmul(sigma_inv, delq_delt.T)), axis1=1, axis2=2)*exp_term
            temp4 = np.diagonal(np.matmul(q, np.matmul(sigma_inv, delq_delt.T)), axis1=1, axis2=2)
            term3 = np.matmul(temp3, temp4.T)
            # test should match temp_check_3
            # temp_term1 = np.diag(np.matmul(delq_delt, np.matmul(sigma_inv, delq_delt.T)))
            temp_hess = term1 + term2 + term3
            hessian_val += temp_hess
            """
            # WHAT FOLLOWS IS THE UNVECTORIZED IMPLEMENTATION OF HESSIAN CALCULATION
            temp_hess = np.zeros([6, 6])
            temp_check_1 = np.zeros([6, 6])
            temp_check_2 = np.zeros([6, 6])
            temp_check_3 = np.zeros([6, 6])
            for i in range(6):
                for j in range(6):
                    # Terms written out separately for increased readability in Hessian calculation
                    term1 = np.diag(np.exp(-0.5 * np.matmul(q, np.matmul(sigma_inv, q.T))))
                    term2 = np.diag(np.matmul(delq_delt[:, :, j], np.matmul(sigma_inv, delq_delt[:, :, i].T)))
                    term3 = np.diag(np.matmul(q, np.matmul(sigma_inv, del2q_deltnm[:, :, i, j].T)))
                    term4 = np.diag(-np.matmul(q, np.matmul(sigma_inv, delq_delt[:, :, i].T)))
                    term5 = np.diag(np.matmul(q, np.matmul(sigma_inv, delq_delt[:, :, j].T)))
                    temp_hess[i, j] = np.sum(term1*(term2 + term3 - term4*term5))
                    temp_check_1[i, j] = np.sum(term2)
                    temp_check_2[i, j] = np.sum(term3)
                    temp_check_3[i, j] = np.sum(term4*term5) # MATCHES TEST
            """
            """
            exp_term = np.diag(np.exp(-0.5 * np.matmul(q, np.matmul(sigma_inv, q.T))))
            test1 = np.matmul(sigma_inv, delq_delt.T)
            test2 = np.einsum('abi,jbc->aijc', delq_delt, test1)
            test3 = np.diagonal(test2, axis1=0, axis2=3)
            test4 = np.sum(test3*exp_term, axis=2)
            testing1 = np.matmul(sigma_inv, del2q_deltnm.T)
            testing2 = np.matmul(q, testing1)
            testing3 = np.diagonal(testing2, axis1=2, axis2=3)
            testing3_alter = np.diagonal(testing2, axis1=3, axis2=2)
            testing4 = np.sum(testing3*exp_term, axis=2)
            temp_term3 = np.diagonal(-np.matmul(q, np.matmul(sigma_inv, delq_delt.T)), axis1=1, axis2=2)*exp_term 
            temp_term4 = np.diagonal(np.matmul(q, np.matmul(sigma_inv, delq_delt.T)), axis1=1, axis2=2) 
            test = np.matmul(temp_term3, temp_term4.T)
            # test should match temp_check_3
            # temp_term1 = np.diag(np.matmul(delq_delt, np.matmul(sigma_inv, delq_delt.T)))
            temp_hess = testing4.T + test4 - test
            hessian_val += temp_hess
            total_sub_loop_time += time.time() - t2
        main_loop_neval += 1
        total_main_loop_time += time.time() - t1
    print('The average main loop run time was ', total_main_loop_time/main_loop_neval)
    print('The total main loop run time was ', total_main_loop_time)
    print('The number of sub loop runs is', main_loop_neval*36)
    print('The total sub loop run time was ', total_sub_loop_time)
    print('The total Hessian run time was ', time.time() - t0)
    """
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
    # TODO: Verify vectorized hessian and remove this function
    N = test_pc.shape[0]
    hessian_val = np.zeros([6, 6])
    transformed_pc = utils.transform_pc(odometry_vector, test_pc)
    _, transform_centers = ndt_cloud.find_voxel_center(transformed_pc)
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


def interp_odometry(ndt_cloud, test_pc, max_iter_pre=15, max_iter_post=10, integrity_filter=0.7):
    global obj_neval
    obj_neval = 0
    global jacob_neval
    jacob_neval = 0
    global hess_neval
    hess_neval = 0
    test_xyz = test_pc[:, :3]
    initial_odom = np.zeros(6)
    # initial_odom = search_initial(ndt_cloud, test_xyz)
    res1 = minimize(interp_objective, initial_odom, method='Newton-CG', jac=interp_jacobian, hess=interp_hessian,
                    args=(ndt_cloud, test_xyz), options={'disp': True, 'maxiter': max_iter_pre})
    # res = minimize(objective, initial_odom, method='BFGS', args=(ndt_cloud, test_xyz), options={'disp' : True})
    temp_odom = res1.x
    transformed_xyz = utils.transform_pc(temp_odom, test_xyz)
    ndt_cloud.find_integrity(transformed_xyz)
    ndt_cloud.filter_voxels_integrity(integrity_limit=integrity_filter)
    if max_iter_post != 0:
        res2 = minimize(interp_objective, temp_odom, method='Newton-CG', jac=interp_jacobian, hess=interp_hessian,
                        args=(ndt_cloud, test_xyz), options={'disp': True, 'maxiter': max_iter_post})
        odom_vector = res2.x
    else:
        odom_vector = np.copy(temp_odom)
    # Return odometry in navigational frame of reference
    return odom_vector


def interp_objective(odometry_vector, ndt_cloud, test_pc):
    assert (ndt_cloud.cloud_type == 'interpolate')
    global obj_neval
    global jacob_neval
    global hess_neval
    transformed_pc = utils.transform_pc(odometry_vector, test_pc)
    obj_value = -1 * ndt_cloud.find_likelihood(transformed_pc)
    if hess_neval % 5 == 0:
        print('Objective iteration: {:4d}'.format(obj_neval), 'Jacobian iteration: {:4d}'.format(jacob_neval),
              'Hessian iteration: {:4d}'.format(hess_neval), 'Objective Value: {:10.4f}'.format(obj_value))
    """
    , ' Odometry:',
          ' x: {:2.5f}'.format(odometry_vector[0]), ' y: {:2.5f}'.format(odometry_vector[1]),
          ' z: {:2.5f}'.format(odometry_vector[2]), ' Phi: {:2.5f}'.format(odometry_vector[3]),
          ' Theta:{:2.5f}'.format(odometry_vector[4]), ' Psi: {:2.5f}'.format(odometry_vector[5])
    """
    obj_neval += 1
    return obj_value


def interp_jacobian(odometry_vector, ndt_cloud, test_pc):
    assert(ndt_cloud.cloud_type == 'interpolate')
    N = test_pc.shape[0]
    transformed_xyz = utils.transform_pc(odometry_vector, test_pc[:, :3])
    neighbours = ndt_cloud.find_neighbours(transformed_xyz)
    weights = ndt_cloud.find_interp_weights(transformed_xyz, neighbours)[:, :N]

    vect_nearby_init = np.array(np.hsplit(neighbours, 8))
    vert_stack = np.reshape(vect_nearby_init, [N * 8, 3])
    vert_idx = ndt_cloud.pairing_cent2int(vert_stack)
    vect_nearby_idx = np.reshape(vert_idx.T, [8, N]).T
    vect_mus = np.zeros([8, N, 3, 1])
    vect_inv_sigmas = 10000*np.ones([8, N, 3, 3])

    delq_delt = find_delqdelt_vect(odometry_vector, transformed_xyz)
    # shape N, 3, 6

    for key in ndt_cloud.stats:
        indices = vect_nearby_idx == ndt_cloud.stats[key]['idx']
        mu = ndt_cloud.stats[key]['mu']
        inv_sigma = np.linalg.inv(ndt_cloud.stats[key]['sigma'])
        vect_mus[indices.T, :, 0] = mu
        vect_inv_sigmas[indices.T, :, :] = inv_sigma
    # q_i.T*inv_sigma*delq_delt*likelihood (original formulation)
    diff = np.zeros_like(vect_mus)
    diff[:, :, :, 0] = -vect_mus[:, :, :, 0] + transformed_xyz[:N, :]  # shape 8, N, 3, 1
    diff_transpose = np.transpose(diff, [0, 1, 3, 2])  # shape 8, N, 1, 3
    lkds = np.exp(-0.5*np.matmul(np.matmul(diff_transpose, vect_inv_sigmas), diff))[:, :, 0, 0]  # shape 8, N
    wgt_lkd = weights * lkds  # shape 8, N
    vect_wgt_lkd = np.repeat(wgt_lkd[:, :, np.newaxis, np.newaxis], 6, axis=3)
    # shape 8, N, 1, 6 (likelihood value repeated along last axis)
    vect_delq_delt = np.transpose(np.repeat(delq_delt[:, :, :, None], 8, axis=3), (3, 0, 1, 2)) # shape 8, N, 3, 6
    vect_jacob = vect_wgt_lkd*np.matmul(np.matmul(diff_transpose, vect_inv_sigmas), vect_delq_delt)
    jacob_val = np.sum(np.sum(vect_jacob[:, :, 0, :], axis=0), axis=0)
    return jacob_val


def interp_hessian(odometry_vector, ndt_cloud, test_pc):
    # TODO: Write interpolated hessian function
    assert (ndt_cloud.cloud_type == 'interpolate')
    return None

################################################################
# Consensus odometry function follow. Don't really work
################################################################


def consensus_odometry(ndt_cloud, test_pc):
    """
    Function to find the best traansformation (in the form of a translation, Euler angle vector)
    :param ndt_cloud: NDT approximation of the prior representation
    :return: test_pc: Point cloud which has to be matched to the existing NDT approximation
    """
    test_xyz = test_pc[:, :3]
    initial_odom = np.zeros(6)
    #initial_odom = search_initial(ndt_cloud, test_xyz)
    # initial_odom = np.array([[0.30756262, 0.02702107, 0.01152367, 0.03415157, 0.6494663, 0.72756484]])
    # xlim, ylim, zlim = find_pc_limits(test_xyz)
    # odometry_bounds = Bounds([-xlim, -ylim, -zlim, -180.0, -90.0, -180.0], [xlim, ylim, zlim, 180.0, 90.0, 180.0])
    # TODO: Any way to implement bounds on the final solution?
    res1 = minimize(consensus_objective, initial_odom, method='Newton-CG', jac=consensus_jacobian,
                    args=(ndt_cloud, test_xyz), options={'disp': True, 'maxiter': 15})
    # res = minimize(objective, initial_odom, method='BFGS', args=(ndt_cloud, test_xyz), options={'disp' : True})
    temp_odom_vector = res1.x
    transformed_xyz = utils.transform_pc(temp_odom_vector, test_xyz)
    ndt_cloud.find_integrity(transformed_xyz)
    ndt_cloud.filter_voxels_integrity(integrity_limit=0.7)
    res2 = minimize(consensus_objective, temp_odom_vector, method='Newton-CG', jac=consensus_jacobian,
                    args=(ndt_cloud, test_xyz), options={'disp': True, 'maxiter': 10})
    # Return odometry in navigational frame of reference
    odom_vector = res2.x
    return odom_vector


def consensus_objective(odometry_vector, ndt_cloud, test_pc):
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
    _, iscore_sum = ndt_cloud.find_integrity(transformed_pc)
    obj_value = -1 * iscore_sum
    # print(obj_value)
    print('Objective iteration: {:4d}'.format(obj_neval), 'Jacobian iteration: {:4d}'.format(jacob_neval),
          'Hessian iteration: {:4d}'.format(hess_neval), 'Objective Value: {:10.4f}'.format(obj_value))
    """
    , ' Odometry:',
          ' x: {:2.5f}'.format(odometry_vector[0]), ' y: {:2.5f}'.format(odometry_vector[1]),
          ' z: {:2.5f}'.format(odometry_vector[2]), ' Phi: {:2.5f}'.format(odometry_vector[3]),
          ' Theta:{:2.5f}'.format(odometry_vector[4]), ' Psi: {:2.5f}'.format(odometry_vector[5])
    """
    obj_neval += 1
    return obj_value


def consensus_jacobian(odometry_vector, ndt_cloud, test_pc):
    """
    Function to return the Jacobian of the likelihood objective for the odometry calculation
    :param odometry_vector: The point at which the Jacobian is to be evaluated
    :param ndt_cloud: The NDT cloud with respect to which the odometry is required
    :param test_pc: The point cloud for which the optimization is being performed
    :return: jacobian_val: The Jacobian matrix (of the objective w.r.t the odometry vector)
    """
    # TODO: Does Consensus Jacobian work?
    jacobian_val = np.zeros(6)
    transformed_pc = utils.transform_pc(odometry_vector, test_pc)
    points_in_voxels = ndt_cloud.bin_in_voxels(transformed_pc)
    iscore_dict, rbar_dict, k_dict = ndt_cloud.optimization_integrity(transformed_pc)
    for key, value in points_in_voxels.items():
        if key in ndt_cloud.stats and key in iscore_dict:
            # Vectorized implementation
            mu = ndt_cloud.stats[key]['mu']
            sigma = ndt_cloud.stats[key]['sigma']
            sigma_inv = np.linalg.inv(sigma)
            r_bar = rbar_dict[key]
            if r_bar > 6:
                r_bar = 6
            elif r_bar < -6:
                r_bar = -6
            Cv = iscore_dict[key]
            k = k_dict[key]
            # TODO: Review sigma_inv_det
            # sigma_inv_det = np.linalg.det(sigma_inv)
            # normal_factor = 1e-8
            q = value - mu
            if q.ndim < 2:
                q = np.atleast_2d(q)
            delq_delt = find_delqdelt_vect(odometry_vector, value)
            constant_term = Cv*np.exp(-r_bar)*k
            first_term = np.matmul(q, np.matmul(sigma_inv, delq_delt.T))
            # g = (sigma_inv_det*normal_factor)*np.sum(np.diagonal(first_term, axis1=1, axis2=2)*exp_term, axis=1)
            g = np.sum(np.diagonal(first_term, axis1=1, axis2=2) * constant_term, axis=1)
            jacobian_val += g
    global jacob_neval
    jacob_neval += 1
    return jacobian_val


def consensus_hessian(odometry_vector, ndt_cloud, test_pc):
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
    iscore_dict, rbar_dict, k_dict = ndt_cloud.optimization_integrity(transformed_pc)
    # t0 = time.time()
    # total_main_loop_time = 0
    # total_sub_loop_time = 0
    # main_loop_neval = 0
    for key, value in points_in_voxels.items():
        # t1 = time.time()
        if key in ndt_cloud.stats and key in iscore_dict:
            if value.ndim == 1:
                value = np.atleast_2d(value)
            mu = ndt_cloud.stats[key]['mu']
            sigma = ndt_cloud.stats[key]['sigma']
            sigma_inv = np.linalg.inv(sigma)
            Cv = iscore_dict[key]
            r_bar = rbar_dict[key]
            if r_bar > 6:
                r_bar = 6
            elif r_bar < -6:
                r_bar = -6
            k = k_dict[key]
            # TODO: Review sigma_inv_det
            # sigma_inv_det = np.linalg.det(sigma_inv)
            # normal_factor = 1e-8
            q = value - mu
            delq_delt = find_delqdelt_vect(odometry_vector, value)
            del2q_deltnm = find_del2q_deltnm_vect(odometry_vector, value)
            # t2 = time.time()
            # TODO: sigma_inv_det used here for hessian_vect
            # exp_term = (sigma_inv_det*normal_factor)*np.diag(np.exp(-0.5 * np.matmul(q, np.matmul(sigma_inv, q.T))))
            constant_term1 = Cv*k*np.exp(-r_bar)
            temp1 = np.einsum('abi,jbc->aijc', delq_delt, np.matmul(sigma_inv, delq_delt.T))
            term1 = np.sum(np.diagonal(temp1, axis1=0, axis2=3), axis=2)
            term2 = np.sum(np.diagonal(np.matmul(q, np.matmul(sigma_inv, del2q_deltnm.T)), axis1=2, axis2=3), axis=2)
            temp3 = k*np.diagonal(np.matmul(q, np.matmul(sigma_inv, delq_delt.T)), axis1=1, axis2=2)
            temp4 = (-np.exp(-r_bar)/(1 - np.exp(-r_bar)**2))*np.diagonal(np.matmul(q, np.matmul(sigma_inv, delq_delt.T)), axis1=1, axis2=2)
            term3 = np.matmul(temp3, temp4.T)
            # test should match temp_check_3
            # temp_term1 = np.diag(np.matmul(delq_delt, np.matmul(sigma_inv, delq_delt.T)))
            temp_hess = constant_term1*(term1 + term2) + term3
            hessian_val += temp_hess
    global hess_neval
    hess_neval += 1
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
    # TODO: Vectorize this function completely. odometry should be an input matrix and the function should return the
    #  first order derivative for each row. Spread out by a fourth index. That should speed up computation of the
    #  second order derivative
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
    param_mat = np.zeros([3, 3, 3])
    param_mat[:, :, 0] = np.array([[0, (-s1 * s3 + c3 * c1 * s2), (c1 * s3 + s1 * c3 * s2)],
                                   [0, - (s1 * c3 + c1 * s2 * s3), (c1 * c3 - s1 * s2 * s3)],
                                   [0, - c2 * c1, - s1 * c2]])
    param_mat[:, :, 1] = np.array([[-s2 * c3, c3 * s1 * c2, -c1 * c2 * c3],
                                   [s2 * s3, - s1 * c2 * s3, c1 * c2 * s3],
                                   [c2, s1 * s2, - c1 * s2]])
    param_mat[:, :, 2] = np.array([[-c2 * s3, (c1 * c3 - s1 * s2 * s3), (s1 * c3 + c1 * s2 * s3)],
                                   [- c2 * c3, - (c1 * s3 + s1 * s2 * c3), (-s1 * s3 + c1 * s2 * c3)],
                                   [0, 0, 0]])

    delq_delt = np.zeros([N, 3, 6])  # Indexing in the first dimension based on experience
    delq_delt[:, :3, :3] = np.broadcast_to(np.eye(3), [N, 3, 3])
    delq_delt[:, :, 3:] = np.transpose(np.pi / 180.0 * np.matmul(param_mat.T, points.T))
    """
    # WHAT FOLLOWS IS THE UNVECTORIZED IMPLEMENTATION OF FIRST DERIVATIVE CALCULATION
    param_mat = np.zeros([3, 3, 3])
    param_mat[0, :, :] = np.array([[0, (-s1 * s3 + c3 * c1 * s2), (c1 * s3 + s1 * c3 * s2)],
                                   [0, - (s1 * c3 + c1 * s2 * s3), (c1 * c3 - s1 * s2 * s3)],
                                   [0, - c2 * c1, - s1 * c2]])
    param_mat[1, :, :] = np.array([[-s2 * c3,  c3 * s1 * c2, -c1 * c2 * c3],
                                   [s2 * s3, - s1 * c2 * s3, c1 * c2 * s3],
                                   [c2, s1 * s2, - c1 * s2]])
    param_mat[2, :, :] = np.array([[-c2 * s3, (c1 * c3 - s1 * s2 * s3), (s1 * c3 + c1 * s2 * s3)],
                                   [- c2 * c3, - (c1 * s3 + s1 * s2 * c3), (-s1 * s3 + c1 * s2 * c3)],
                                   [0, 0, 0]])
                                       delq_delt = np.zeros([N, 3, 6])  # Indexing in the first dimension based on experience
    delq_delt[:, :3, :3] = np.broadcast_to(np.eye(3), [N, 3, 3])
        for i in range(3, 6):
        delq_delt[:, :, i] = np.pi / 180.0 * np.matmul(points, param_mat[i-3, :, :])
    """
    return delq_delt


def find_del2q_deltnm_vect(odometry_vector, points):
    """
    Vectorized implementation of function to return double partial derivative of point w.r.t. odometry vector parameters
    :param odometry_vector: Vector at which Hessian is being calculated
    :param q: Points which are being compared to the NDT Cloud
    :return: del2q_deltnm: Nx3x6x6 matrix of del2/deltndeltm
    """
    N = points.shape[0]
    del2q_deltnm_old = np.zeros([N, 3, 6, 6])
    delta = 1.5e-08
    original_delq_delt = find_delqdelt_vect(odometry_vector, points)
    # TODO: Vectorize computation of second derivative for different odometry vectors to happen at once
    for i in range(6):
        odometry_new = np.zeros(6)
        odometry_new[i] = odometry_new[i] + delta
        points_new = utils.transform_pc(odometry_new, points)
        # Assuming that the incremental change allows us to directly add a rotation instead of an incremental change
        odometry_new += odometry_vector
        del2q_deltnm_old[:, :, :, i] = (find_delqdelt_vect(odometry_new, points_new) - original_delq_delt) / delta
    del2q_deltnm_new = np.zeros([N, 3, 6, 6])
    delta = 1.5e-8
    return del2q_deltnm_old


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
