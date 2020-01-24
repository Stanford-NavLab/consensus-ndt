"""
demo_results.py
Functions used to generate small examples and illustrations, primarily for presentations and papers
Author: Ashwin Kanhere
Date Created: 13th November 2019
Date Modified: 13th November 2019
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
