"""
function_tests.py
File containing all functions that test code functionality
Author: Ashwin Kanhere
Date created: 13th November 2019
Last modified: 13th November 2019
"""
import pykitti
import numpy as np
import transforms3d
import ndt
import time
import data_utils
import utils
from matplotlib import pyplot as plt
import diagnostics
import odometry

"""
Helper functions for tests
"""


def extract_data():
    basedir = 'D:\\Users\\kanhe\\Box Sync\\Research Projects\\Consensus NDT SLAM\\Dataset'
    date = '2011_09_26'
    drive = '0005'

    data = pykitti.raw(basedir, date, drive, frames=range(0, 5, 1))

    points_lidar = data.velo
    return data

"""
Actually test functions
"""


def transforms_test():
    angles = [0, np.pi/4, np.pi/3]
    for i in range(3):
        for j in range(3):
            for k in range(3):
                R = transforms3d.euler.euler2mat(angles[i], angles[j], angles[k], 'rxyz')
                c1 = np.cos(angles[i])
                s1 = np.sin(angles[i])
                c2 = np.cos(angles[j])
                s2 = np.sin(angles[j])
                c3 = np.cos(angles[k])
                s3 = np.sin(angles[k])
                test_R = np.array([[c2*c3, -c2*s3, s2], [c1*s3 + c3*s1*s2, c1*c3 - s1*s2*s3, -c2*s1],
                                   [s1*s3 - c1*c3*s2, c3*s1 + c1*s2*s3, c1*c2]])
                error_R = R - test_R
                mag_R = np.max(np.abs(error_R))
                print('Case 1: i:', i, ' j:', j, ' k:', k, ' Value:', mag_R)
                c1 = np.cos(angles[k])
                s1 = np.sin(angles[k])
                c2 = np.cos(angles[j])
                s2 = np.sin(angles[j])
                c3 = np.cos(angles[i])
                s3 = np.sin(angles[i])
                test_R_2 = np.array([[c1*c2, c1*s2*s3 - c3*s1, s1*s3 + c1*c3*s2],
                                     [c2*s1, c1*c3+s1*s2*s3, c3*s1*s2 - c1*s3],[-s2, c2*s3, c2*c3]])
                error_R_2 = R - test_R_2
                mag_R_2 = np.max(np.abs(error_R_2))
                # print('Case 2: i:', i, ' j:', j, ' k:', k, ' Value:', mag_R_2)
    return None


def new_ndt_test():
    xlim = 10
    ylim = 10
    zlim = 5
    input_horiz_grid_size = 2
    input_vert_grid_size = 2
    # test_cloud1 = ndt.NDTCloudBase(xlim, ylim, zlim, input_horiz_grid_size, input_vert_grid_size)
    # test_cloud2 = ndt.NDTCloudNoOverLap(xlim, ylim, zlim, input_horiz_grid_size, input_vert_grid_size)
    # test_cloud3 = ndt.NDTCloudOverLap(xlim, ylim, zlim, input_horiz_grid_size, input_vert_grid_size)
    # test_cloud4 = ndt.NDTCloudInterpolated(xlim, ylim, zlim, input_horiz_grid_size, input_vert_grid_size)
    data = extract_data()
    test_lidar = data.get_velo(0) # LiDAR point cloud is a Nx4 numpy array
    test_lidar = test_lidar[:, :3]
    # test_cloud = ndt_approx.ndt_approx(test_lidar, horiz_grid_size=0.25, vert_grid_size=0.25)
    name = ['overlapping', 'interpolate', 'nooverlap', 'overlapping']
    for i in range(len(name)):
        testing = ndt.ndt_approx(test_lidar, horiz_grid_size=1, vert_grid_size=1, type=name[i])
        print('Creating NDT approximation for ', name[i])
        print('The likelihood for NDT approximation of type ', name[i], 'is ', testing.find_likelihood(test_lidar))
    return None


def interpolate_likelihood_test():
    num_pt = 100000
    data = extract_data()
    test_lidar = data.get_velo(0) # LiDAR point cloud is a Nx4 numpy array
    test_lidar = test_lidar[:, :3]
    test_approximation = ndt.ndt_approx(test_lidar, horiz_grid_size=1, vert_grid_size=1, type='interpolate')
    t0 = time.time()
    test_likelihood = test_approximation.find_likelihood(test_lidar[:, :])
    print('The time taken to run the interpolated likelihood code is: ', time.time() - t0)
    print('Interpolated likelihood is:', test_likelihood)
    test_approximation = ndt.ndt_approx(test_lidar, horiz_grid_size=1, vert_grid_size=1, type='overlapping')
    t0 = time.time()
    test_likelihood2 = test_approximation.find_likelihood(test_lidar[:, :])
    print('The time taken to run the overlapping likelihood code is: ', time.time() - t0)
    print('Overlapping likelihood is: ', test_likelihood2)
    print('Eighths of the overlapping likelihood is: ', test_likelihood2/8.)
    test_approximation = ndt.ndt_approx(test_lidar, horiz_grid_size=1, vert_grid_size=1, type='nooverlap')
    t0 = time.time()
    test_likelihood3 = test_approximation.find_likelihood(test_lidar[:, :])
    print('The time taken to run the no overlap likelihood code is: ', time.time() - t0)
    print('No overlap likelihood is: ', test_likelihood3)
    return None


def verify_interp_lkd():
    pcs = data_utils.load_kitti_pcs(0, 10)
    poses = data_utils.kitti_sequence_poses(0, 10, diff=1)
    ref_lidar = pcs[0]
    test_lidar = pcs[1]
    cand_pos = poses[1]
    ref_ndt = ndt.ndt_approx(ref_lidar, horiz_grid_size=0.5, vert_grid_size=0.5, type='overlapping')
    print('Untransformed likelihood is ', ref_ndt.find_likelihood(test_lidar))
    trans_pc = utils.transform_pc(cand_pos, test_lidar)
    print('First transformed likelihood is ', ref_ndt.find_likelihood(trans_pc))
    print('Raw likelihood is ', ref_ndt.find_likelihood(ref_lidar))
    print('random transformation likelihood is ', ref_ndt.find_likelihood(utils.transform_pc(cand_pos, ref_lidar)))
    print('Running axis variation test')
    ref_ndt = ndt.ndt_approx(ref_lidar, horiz_grid_size=1, vert_grid_size=1, type='interpolate')

    compare_lidar = test_lidar
    vectorized = True
    test_against = 'self'
    #test_against = 'mwahahaha'
    if test_against == 'self':
        cand_pos = np.zeros_like(cand_pos)
        compare_lidar = ref_lidar
    for j in range(6):
        print('Running variation for axis ', j)
        axis = j
        limit = 20
        num_vals = 11
        objective_value = np.zeros(num_vals)
        dim_variation = np.linspace(-limit, limit, num_vals)
        print(dim_variation)
        for i in range(num_vals):
            # print(inverse_odom_value)
            print('Running case ', i)
            odom = np.copy(cand_pos)
            odom[axis] += dim_variation[i]
            trans_pc = utils.transform_pc(odom, compare_lidar)
            if vectorized:
                tic = time.time()
                objective_value[i] = ref_ndt.find_likelihood(trans_pc)
                print('Time taken by vectorized version is ', time.time() - tic)
            else:
                tic = time.time()
                objective_value[i] = ref_ndt.find_likelihood_non_vect(trans_pc)
                print('Time taken by non-vectorized version is ', time.time() - tic)
        plt.plot(dim_variation, objective_value)
        plt.show()
        print('Required variation on axis', j, ': Done')
    #diagnostics.plot_consec_pc(trans_pc, ref_lidar)
    #diagnostics.plot_consec_pc(compare_lidar, ref_lidar)
    return None


def test_interp_weights():
    num_pt = 100000
    data = extract_data()
    ref_lidar = data.get_velo(0) # LiDAR point cloud is a Nx4 numpy array
    ref_lidar = ref_lidar[:num_pt, :3]
    ref_ndt = ndt.ndt_approx(ref_lidar, horiz_grid_size=1, vert_grid_size=1, type='interpolate')
    nearby = ref_ndt.find_neighbours(ref_lidar)
    point = np.array([[0.25, 0.5, 0.75], [0.1, 0.5, 0.75]])
    nearby = ref_ndt.find_neighbours(ref_lidar)
    weights = ref_ndt.find_interp_weights(ref_lidar, nearby)
    ref_ndt.pair_check()
    print(np.shape(weights))
    print('Printing stuff in the test function')
    """
    for i in range(8):
        print(nearby[:, i:3*i + 3])
    """
    return None


def jacobian_vect_test():
    # Verified for test case of same point cloud at origin
    pcs = data_utils.load_kitti_pcs(0, 10)
    ref_lidar = pcs[0]
    N = 10000
    cand_pos = np.array([0., 0., 0., 0., 0., 0.])
    ref_ndt = ndt.ndt_approx(ref_lidar, horiz_grid_size=1., vert_grid_size=1., type='overlapping')

    _, jacob_error_norm = diagnostics.check_gradient(odometry.objective, odometry.jacobian, ref_ndt, ref_lidar[:N, :],
                                                     cand_pos)
    _, jacob_vect_error_norm = diagnostics.check_gradient(odometry.objective, odometry.jacobian_vect, ref_ndt,
                                                          ref_lidar[:N, :], cand_pos)
    print('The jacobian error is ', jacob_error_norm)
    print('The vectorized jacobian error is', jacob_vect_error_norm)
    return None


def demo_hessian(odom):
    hessian = np.diag([2., 2., 2., 2., 2., 2.])
    return hessian


def demo_jacobian(odom):
    jacobian = 2*odom
    return jacobian


def demo_hessian_check(odom=np.array([2., 3., 4., 5., 6., 7.]), print_output=True):
    # Same function as diagnostics.check_hessian. Applied to x1^2 + ... + x6^2 to check the math
    delta = 1.5e-8
    hess_val = np.zeros([6, 6])
    analytical_hess = demo_hessian(odom)
    for cidx in range(6):
        for ridx in range(6):
            new_odometry = np.zeros(6)
            new_odometry += odom
            new_odometry[ridx] += delta
            hess_val[ridx, cidx] = (demo_jacobian(new_odometry)[cidx] - demo_jacobian(odom)[cidx]) / delta
    hessian_error = hess_val - analytical_hess
    hess_error_norm = np.linalg.norm(hessian_error)
    if print_output:
        print('The analytical Hessian is ', analytical_hess)
        print('The numerical Hessian vector is', hess_val)
        print('The Hessian vector error is ', hessian_error)
        print('The magnitude of the Hessian error is', hess_error_norm)
    return None


def hessian_vect_test():
    pcs = data_utils.load_kitti_pcs(0, 10)
    poses = data_utils.kitti_sequence_poses(0, 10, diff=1)
    ref_lidar = pcs[0]
    test_lidar = pcs[1]
    N = 10000
    cand_pos = np.array([0., 0., 0., 0., 0., 0.])
    ref_ndt = ndt.ndt_approx(ref_lidar, horiz_grid_size=1., vert_grid_size=1., type='overlapping')
    tic = time.time()
    original_hessian = odometry.hessian(cand_pos, ref_ndt, ref_lidar[:N, :])
    print('Time taken to run original Hessian is ', time.time() - tic)
    tic = time.time()
    vect_hessian = odometry.hessian_vect(cand_pos, ref_ndt, ref_lidar[:N, :])
    print('Time taken to run new and surprisingly fancy Hessian is ', time.time() - tic)

    # Checking the Hessian: Check the row of the Hessian that corresponds to each jacobian element (using grad check)
    #_, hess_error_norm = diagnostics.check_hessian(odometry.jacobian_vect, odometry.hessian, ref_ndt, ref_lidar[:N, :],
    #                                               cand_pos)
    _, hess_vect_error_norm = diagnostics.check_hessian(odometry.jacobian_vect, odometry.hessian_vect, ref_ndt,
                                                        ref_lidar[:N, :], cand_pos)
    #print('The Hessian error is ', hess_error_norm)
    print('The vectorized Hessian error is', hess_vect_error_norm)

    return None


# TODO: Add all testing functions to this file
#verify_interp_lkd()
#interpolate_likelihood_test()
#test_interp_weights()
#demo_hessian_check()
hessian_vect_test()
