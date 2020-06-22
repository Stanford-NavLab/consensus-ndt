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
    """
    Match two PCs, find voxel consensus metric for each point in second point cloud and shade points in plots using voxel consensus metric
    """
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

