"""
demo_results.py
Functions used to generate example results and illustrations, primarily for presentations and papers
Author: Ashwin Kanhere
Date Created: 13th November 2019
Date Modified: 22nd June 2020
"""

import numpy as np
import ndt
import data_utils
import utils
import pptk
import diagnostics
from matplotlib import pyplot as plt
import pykitti
import lidar_mods
import time
import odometry


def ndt_resolution():
    """
    Visually compare impact of different NDT voxel lengths
    """
    kitti_data = data_utils.load_uiuc_pcs(0, 10, 1)
    pc = kitti_data[0]
    view_pc = pptk.viewer(pc)
    view_pc.set(lookat=[0.0, 0.0, 0.0])
    ndt_1 = ndt.ndt_approx(pc, horiz_grid_size=2.0, vert_grid_size=2.0)
    ndt.display_ndt_cloud(ndt_1, point_density=0.8)
    ndt_2 = ndt.ndt_approx(pc, horiz_grid_size=1.0, vert_grid_size=1.0)
    ndt.display_ndt_cloud(ndt_2, point_density=0.4)
    ndt_3 = ndt.ndt_approx(pc, horiz_grid_size=0.5, vert_grid_size=0.5)
    ndt.display_ndt_cloud(ndt_3, point_density=0.2)
    return None
    
    
def c_v_presentation_plot():
    """
    Plot variation in voxel consensus metric with chi-square residual
    """
    N = 6
    high = 13.816
    low = 0.01
    r = np.linspace(0, 20, 100)
    bar_r = 3 - 6.*((r - low)/(high - low))
    C_v = 1 / (1 + np.exp(-bar_r))
    good_voxels = 5
    bad_voxels = 8.87
    plt.plot(r, C_v)
    plt.xlim([0, 20])
    plt.ylim([0, 1])
    plt.show()
    return None
    
    
def paper_total_con():
    """
    Effect of different arrangement of voxels on the localization consensus metric
    """
    points = np.array([[1, 0],
                       [0.707, 0.707],
                       [0, 1],
                       [-0.707, 0.707],
                       [-1, 0],
                       [-0.707, -0.707],
                       ])
    original_points = np.array([[1, 0],
                                [0.707, 0.707],
                                [0, 1],
                                [-0.707, 0.707],
                                [-1, 0],
                                [-0.707, -0.707],
                                [0, -1],
                                [0.707, -0.707],
                       ])
    metric_1 = np.array([1.0, 1.0, 1.0, 0.5, 0.5, 0.5])
    metric_2 = np.array([1.0, 0.5, 1.0, 0.5, 1.0, 0.5])
    H_inv_1 = np.matmul(points.T, np.matmul(np.diag(metric_1**2), points))
    H_inv_2 = np.matmul(points.T, np.matmul(np.diag(metric_2**2), points))
    H_mat_1 = np.linalg.inv(H_inv_1)
    H_mat_2 = np.linalg.inv(H_inv_2)
    H_1 = np.sqrt(np.sum(np.diag(H_mat_1)))
    H_2 = np.sqrt(np.sum(np.diag(H_mat_2)))
    DOP = np.sqrt(np.sum(np.diag(np.linalg.inv(np.matmul(original_points.T, original_points)))))
    # print(H_1)
    # print(H_2)
    print(DOP)
    print(DOP/H_1)
    print(DOP/H_2)
    w1, v1 = np.linalg.eigh(H_mat_1)
    w2, v2 = np.linalg.eigh(H_mat_2)
    wDOP, _ = np.linalg.eigh(np.linalg.inv(np.matmul(points.T, points)))
    w_inv_1, _ = np.linalg.eigh(H_inv_1)
    w_inv_2, _ = np.linalg.eigh(H_inv_2)
    return None
    
    
def paper_vox_con_2():
    """
    Generate qualitative examples of perfect match, outliers, occlusions and feature mismatches and their impact on voxel consensus metric
    """
    case = 3
    N_plot = 3
    plot_cov = 0.01
    points = np.array([[0.3, 0.3],
                       [0.3, 0.35],
                       [0.5, 0.45],
                       [0.7, 0.66],
                       [0.7, 0.74]])
    mu = np.mean(points, axis=0)
    cov = np.cov(points.T)
    print(mu)
    print(cov)
    plot_points = np.random.multivariate_normal(mu, cov, 1000)
    test_match = np.array([[0.4, 0.4],
                           [0.5, 0.5],
                           [0.55, 0.55],
                           [0.6, 0.6],
                           [0.65, 0.65]])
    test_outliers = np.array([[0.2, 0.3],
                              [0.5, 0.7],
                              [0.55, 0.15],
                              [0.7, 0.6],
                              [0.8, 0.85]])
    test_mismatch = np.array([[0.2, 0.4],
                              [0.3, 0.55],
                              [0.55, 0.75],
                              [0.6, 0.76],
                              [0.8, 0.94]])
    test_occlusion = np.array([[0.9, 0.8],
                               [0.9, 0.75],
                               [0.8, 0.7],
                               [0.8, 0.9],
                               [0.8, 0.65]])
    if case == 0:
        test_points = test_match
    elif case == 1:
        test_points = test_outliers
    elif case == 2:
        test_points = test_mismatch
    elif case == 3:
        test_points = test_occlusion
    N_match = np.shape(test_points)[0]
    plot_test = np.reshape(test_points[0, :], [1, -1])
    for test_num in range(N_match):
        temp_x = np.random.normal(test_points[test_num, 0], plot_cov, N_plot)
        temp_y = np.random.normal(test_points[test_num, 1], plot_cov, N_plot)
        temp_pts = np.hstack((np.reshape(temp_x, [-1, 1]), np.reshape(temp_y, [-1, 1])))
        plot_test = np.vstack((plot_test, temp_pts))
    fig = plt.figure()
    plt.scatter(plot_points[:, 0], plot_points[:, 1], s=1, c='b')
    plt.scatter(plot_test[:, 0], plot_test[:, 1], s=36, c='r')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    print(voxel_integrity(mu, cov, plot_test))
    plt.show()

    return None


def plot_cv():
    kitti_pcs = data_utils.load_kitti_pcs(0, 60, pc_mode='laptop')
    kitti_ground_truth = data_utils.kitti_sequence_poses(0, 60, seq_input_mode='laptop')
    pc_index = np.array([0])
    idx = pc_index[0]
    prev_pc = kitti_pcs[idx]
    curr_pc = kitti_pcs[idx + 1]
    print('NDT Approximation')
    prev_ndt = ndt.ndt_approx(prev_pc, horiz_grid_size=1.0, vert_grid_size=1.0)
    inv_odom = utils.invert_odom_transfer(kitti_ground_truth[idx + 1])
    trans_pc = utils.transform_pc(inv_odom, curr_pc)
    print('Displaying original point cloud')
    view_new = pptk.viewer(prev_pc)
    view_new.set(lookat=[0.0, 0.0, 0.0])
    print('Finding integrity')
    prev_ndt.find_integrity(trans_pc)
    print('Displaying integrity filtered NDT')
    ndt.display_ndt_cloud(prev_ndt, point_density=2)
    print('Plotting consecutive point clouds')
    diagnostics.plot_consec_pc(prev_pc, trans_pc)
    return None


def presentation_run():
    # Extract data for run
    kitti_pcs = data_utils.load_kitti_pcs(0, 60, pc_mode='server')
    # Generating plots for localization consensus metric
    prev_pc = kitti_pcs[0]
    test_pc = kitti_pcs[1]
    num_test = 100
    test_axis = 0
    Cm = np.zeros(num_test)
    delta_poses = np.linspace(0, 5, num_test)
    ground_truths = data_utils.kitti_sequence_poses(0, 60, seq_input_mode='server')
    true_pose = ground_truths[1]
    inv_truth = utils.invert_odom_transfer(true_pose)
    for idx in range(num_test):
        prev_ndt = ndt.ndt_approx(prev_pc, horiz_grid_size=0.5, vert_grid_size=0.5)
        true_trans_pc = utils.transform_pc(inv_truth, test_pc)
        prev_ndt.find_integrity(true_trans_pc)
        prev_ndt.filter_voxels_integrity(integrity_limit=0.7)
        new_pose = true_pose
        new_pose[test_axis] += delta_poses[idx]
        inv_odom = utils.invert_odom_transfer(new_pose)
        trans_pc = utils.transform_pc(inv_odom, test_pc)
        Cm[idx] = prev_ndt.find_integrity(trans_pc)
    plt.figure()
    plt.plot(delta_poses, Cm)
    plt.show()
    return None


def fault_consensus():
    kitti_pcs = data_utils.load_kitti_pcs(0, 60, pc_mode='server')
    kitti_ground_truth = data_utils.kitti_sequence_poses(0, 60, seq_input_mode='server')
    pc_index = np.array([5, 30, 50])
    delta_r = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 2.0, 2.5, 5.0, 10.0])
    for idx in pc_index:
        for cand_r in delta_r:
            #print(len(kitti_pcs))
            #print(len(kitti_ground_truth))
            prev_pc = kitti_pcs[idx]
            curr_pc = kitti_pcs[idx+1]
            #print('Calculating NDT approximation')
            prev_ndt = ndt.ndt_approx(prev_pc, horiz_grid_size=1.0, vert_grid_size=1.0)
            inv_odom = utils.invert_odom_transfer(kitti_ground_truth[idx + 1])
            #print('Transforming point cloud by ground truth')
            trans_pc = utils.transform_pc(inv_odom, curr_pc)
            prev_ndt.find_integrity(trans_pc)
            prev_ndt.filter_voxels_integrity()
            #print('Modifying the transformed point cloud with the additive laser bias')
            lidar_mod_trans_pc = lidar_mods.modify_ranging(trans_pc, cand_r)
            #print('Calculating integrity of biased point cloud')
            Cm, _ = prev_ndt.find_integrity(lidar_mod_trans_pc)
            print('For candidate number ', idx, 'and ranging error ', cand_r, 'C_m is ', Cm)
            #print('Plotting both point clouds')
            #diagnostics.plot_consec_pc(prev_pc, lidar_mod_trans_pc)
    return None


def check_parameter_sizes():
    pcs = data_utils.load_kitti_pcs(0, 10)
    ref_pc = pcs[0]
    pc_sizes = np.size(ref_pc)
    print('Computing NDT with no overlap')
    no_overlap_ndt = ndt.ndt_approx(ref_pc, horiz_grid_size=0.5, vert_grid_size=0.5, type='nooverlap')
    print('Computing NDT for overlapping case')
    overlap_ndt = ndt.ndt_approx(ref_pc, horiz_grid_size=0.5, vert_grid_size=0.5, type='overlapping')
    print('Computing NDT for interpolated case')
    interpolate_ndt = ndt.ndt_approx(ref_pc, horiz_grid_size=0.5, vert_grid_size=0.5, type='interpolate')
    no_overlap_ndt.find_integrity(ref_pc)
    no_overlap_ndt.filter_voxels_integrity(integrity_limit=0.7)

    overlap_ndt.find_integrity(ref_pc)
    overlap_ndt.filter_voxels_integrity(integrity_limit=0.7)

    interpolate_ndt.find_integrity(ref_pc)
    interpolate_ndt.filter_voxels_integrity(integrity_limit=0.7)

    no_overlap_sizes = 9*len(no_overlap_ndt.stats)
    overlap_sizes = 9*len(overlap_ndt.stats)
    interpolate_sizes = 9*len(interpolate_ndt.stats)
    print('Point cloud size is ', pc_sizes)
    print('No overlap size is ', no_overlap_sizes)
    print('Overlapping size is ', overlap_sizes)
    print('Interpolated size is ', interpolate_sizes)
    return None


def group_meeting_demo():

    #run_mode = 'server'
    run_mode = 'laptop'
    total_iters = 20
    iter1 = 10
    iter2 = 10

    print('Loading dataset')
    pcs = data_utils.load_uiuc_pcs(0, 10, mode=run_mode)

    assert(total_iters == iter1 + iter2)

    integrity_filters = np.array([0.5, 0.6, 0.7, 0.8])
    ref_lidar = pcs[0]
    ref_ndt = ndt.ndt_approx(ref_lidar)
    perturb = np.array([0.2, 0.2, 0., 0., 0., 0.])
    trans_pc = utils.transform_pc(perturb, ref_lidar)
    ground_truth = utils.invert_odom_transfer(perturb)
    error_consensus = np.zeros([np.size(integrity_filters), 1])
    time_consensus = np.zeros_like(error_consensus)

    print('Running baseline case')
    tic = time.time()
    new_odom = odometry.odometry(ref_ndt, trans_pc, max_iter_pre=total_iters, max_iter_post=0)
    toc = time.time()
    error_vanilla = np.linalg.norm(ground_truth - new_odom)
    time_vanilla = toc - tic
    # Save error and time values
    print('The vanilla run error is ', error_vanilla)
    print('The vanilla time taken is', time_vanilla)

    for idx, filter_value in enumerate(integrity_filters):
        print('Running case ', idx)
        tic = time.time()
        test_odom = odometry.odometry(ref_ndt, trans_pc, max_iter_pre=iter1, max_iter_post=iter2,
                                      integrity_filter=filter_value)
        toc = time.time()
        error_consensus[idx] = np.linalg.norm(ground_truth - test_odom)
        time_consensus[idx] = toc - tic
        print('Error in run ', idx, ' is ', error_consensus[idx])
        print('Time taken in run ', idx, ' is ', time_consensus[idx])

    print('The consensus run errors are ', error_consensus)
    print('The consensus run times are ', time_consensus)
    print('The integrity filter values are ', integrity_filters)

    return 0


def old_run():
    folder_location = '/home/kanhere2/ion-gnss-19/uiuc_dataset/'
    filename = 'pc_'
    ext = '.csv'

    icp_odom = np.load('saved_icp_odom.npy')
    icp_odom = -icp_odom
    ndt_odom = np.load('saved_ndt_odom.npy')
    diff_ndt = np.zeros_like(ndt_odom)
    diff_icp = np.zeros_like(icp_odom)
    cand_transform = np.array([0.3, 0.3, 0.001, 0.25, 0.25, 0.5])
    for i in range(icp_odom.shape[0]):
        diff_ndt[i, :] = -ndt_odom[i, :] - cand_transform
        diff_icp[i, :] = icp_odom[i, :] - cand_transform
    print(np.mean(np.abs(diff_ndt), axis=0))
    print(np.mean(np.abs(diff_icp), axis=0))
    """
    test = np.loadtxt(folder_location+filename+str(676)+ext, delimiter=',')
    test2 = np.loadtxt(folder_location+filename+str(675)+ext, delimiter=',')
    pptk.viewer(test)
    pptk.viewer(test2)
    test_cloud = ndt.ndt_approx(test)
    ndt.display_ndt_cloud(test_cloud)
    test_cloud2 = ndt.ndt_approx(test2)
    ndt.display_ndt_cloud(test_cloud2)
    result_odom = odometry.odometry(test_cloud2, test)
    update_points = test_cloud2.points_in_filled_voxels(test)
    pptk.viewer(update_points)
    ndt.display_ndt_cloud(test_cloud2)
    result_pc = utils.transform_pc(utils.invert_odom_transfer(result_odom), update_points)
    test_cloud2 = ndt.ndt_approx(test2)
    print(test_cloud2.find_integrity(result_pc))
    print(test_cloud2.find_integrity(test))
    """
    """
    first_idx = 500
    last_idx = 675
    num_run = last_idx - first_idx
    ndt_odom = np.zeros([num_run, 6])
    icp_odom = np.zeros([num_run, 6])
    cand_transform = np.array([0.3, 0.3, 0.001, 0.25, 0.25, 0.5])
    for idx in range(first_idx, last_idx):
        t0 = time.time()
        print('Loading point cloud number: ', idx)
        curr_pc = np.loadtxt(folder_location+filename+str(idx)+ext, delimiter=',')
        trans_pc = utils.transform_pc(cand_transform, curr_pc)
        trans_ndt = ndt.ndt_approx(trans_pc)
        print('Calculating odometry:', idx)
        curr_ndt_odom_inv = odometry.odometry(trans_ndt, curr_pc)
        curr_icp_odom = diagnostics.ind_lidar_odom(curr_pc, trans_pc)
        print('NDT ODOMETRY: ', curr_ndt_odom_inv)
        print('ICP ODOMETRY: ', curr_icp_odom)
        ndt_odom[idx - first_idx, :] = utils.invert_odom_transfer(curr_ndt_odom_inv)
        icp_odom[idx - first_idx, :] = curr_icp_odom
        print('PC: ', idx, 'Run Time: ', time.time() - t0)
    np.save('saved_ndt_odom', ndt_odom)
    np.save('saved_icp_odom', icp_odom)
    """
    """
    for idx in range(first_idx, last_idx):
        t0 = time.time()
        print('Loading point cloud number: ', idx)
        curr_pc = np.loadtxt(folder_location+filename+str(idx)+ext, delimiter=',')
        prev_ndt = ndt.ndt_approx(prev_pc)
        print('Calculating odometry:', idx)
        curr_ndt_odom_inv = odometry.odometry(prev_ndt, curr_pc)
        N1 = prev_pc.shape[0]
        N2 = curr_pc.shape[0]
        if N1 <= N2:
            curr_icp_odom = diagnostics.ind_lidar_odom(curr_pc[:N1, :], prev_pc)
        else:
            curr_icp_odom = diagnostics.ind_lidar_odom(curr_pc, prev_pc[:N2, :])
        ndt_odom[idx - first_idx, :] = utils.invert_odom_transfer(curr_ndt_odom_inv)
        icp_odom[idx - first_idx, :] = curr_icp_odom
        state = utils.combine_odometry(state, ndt_odom[idx - first_idx, :])
        update_points = prev_ndt.points_in_filled_voxels(curr_pc)
        trans_update_points = utils.transform_pc(state, update_points)
        map_ndt.update_cloud(trans_update_points)
        prev_pc = curr_pc
        print('PC: ', idx, 'Run Time: ', time.time() - t0)
    np.save('saved_ndt_odom', ndt_odom)
    np.save('saved_icp_odom', icp_odom)
    ndt.display_ndt_cloud(map_ndt)
    """
    """
    ndt_odom = np.load('saved_ndt_odom_4.npy')
    icp_odom = np.load('saved_icp_odom_4.npy')
    N = ndt_odom.shape[0]
    total_ndt_odom = np.zeros_like(ndt_odom)
    total_icp_odom = np.zeros_like(icp_odom)
    for i in range(1, N):
        total_ndt_odom[i, :] = utils.combine_odometry(total_ndt_odom[i-1, :], ndt_odom[i, :])
        total_icp_odom[i, :] = utils.combine_odometry(total_icp_odom[i-1, :], icp_odom[i, :])
    print(np.mean(np.abs(total_ndt_odom - total_icp_odom)))
    print(np.mean(np.abs(ndt_odom - icp_odom)))
    plt.interactive(False)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(total_ndt_odom[:, 0], total_ndt_odom[:, 1], total_ndt_odom[:, 2], s=4)
    #ax.scatter(total_icp_odom[:, 0], total_icp_odom[:, 1], total_icp_odom[:, 2], s=4)
    plt.show()
    """
    return None
