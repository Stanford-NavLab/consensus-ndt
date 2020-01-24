"""
function_verification.py
File containing test cases to verify whether functions are working as they're supposed to or not
Author: Ashwin Kanhere
Date created: 30th May 2019
"""
import numpy as np
from utils import combine_odometry
from utils import odometry_difference
import utils
import mapping
import data_utils
import ndt
import diagnostics
# TODO: Remove deprecated functions and unify with general testing file


def verify_combine_odometry():
    test_pt = np.array([[1, 1, 1]])
    odom_1 = np.array([0.5, 0.5, 0.5, 0, 0, 0])
    comb_1 = combine_odometry(odom_1, odom_1)
    temp_1 = utils.transform_pc(odom_1, test_pt)
    check_11 = utils.transform_pc(odom_1, temp_1)
    check_12 = utils.transform_pc(comb_1, test_pt)
    print(check_11 - check_12, check_11, check_12)
    odom_2 = np.array([0, 0, 0, 90, 0, 0])
    comb_2 = combine_odometry(odom_1, odom_2)
    check_21 = utils.transform_pc(odom_2, temp_1)
    check_22 = utils.transform_pc(comb_2, test_pt)
    print(check_21 - check_22, check_21, check_22)
    odom_3 = np.array([0.5, 0, 0, 90, 0, 0])
    comb_3 = combine_odometry(odom_1, odom_3)
    check_31 = utils.transform_pc(odom_3, temp_1)
    check_32 = utils.transform_pc(comb_3, test_pt)
    print(check_31 - check_32, check_31, check_32)
    print('IT WORKS!!!')
    return None


def verify_odometry_difference():
    print('Verifying odometry_difference')
    test_pt = np.array([[1, 1, 1]])
    odom_1 = np.array([0.5, 0.5, 0.5, 0, 0, 0])
    temp_1 = utils.transform_pc(odom_1, test_pt)
    comb_1 = np.array([1, 1, 1, 0, 0, 0])
    odom_21 = odometry_difference(odom_1, comb_1)
    check_11 = utils.transform_pc(odom_21, temp_1)
    check_12 = utils.transform_pc(comb_1, test_pt)
    print(check_11 - check_12, check_11, check_12)
    print(odom_21)
    comb_2 = np.array([0.5, -0.5, 0.5, 90, 0, 0])
    odom_2 = odometry_difference(odom_1, comb_2)
    check_21 = utils.transform_pc(odom_2, temp_1)
    check_22 = utils.transform_pc(comb_2, test_pt)
    print(check_21 - check_22, check_21, check_22)
    print(odom_2)
    comb_3 = np.array([1, -0.5, 0.5, 90, 0, 0])
    odom_3 = odometry_difference(odom_1, comb_3)
    check_31 = utils.transform_pc(odom_3, temp_1)
    check_32 = utils.transform_pc(comb_3, test_pt)
    print(check_31 - check_32, check_31, check_32)
    print(odom_3)
    print('THIS WORKS TOO!!!!!')
    return None

########################################################################################################################
# Testing functions in mapping.py #
########################################################################################################################


def test_map_stitch():
    kitti_pcs = data_utils.load_kitti_pcs(0, 5, pc_diff=5)
    kitti_truth = data_utils.kitti_sequence_poses(0, 5, diff=5)
    inv_kitti_truth = utils.invert_odom_transfer(kitti_truth[1, :])

    diagnostics.plot_consec_pc(kitti_pcs[0], kitti_pcs[1])
    inv_pc = utils.transform_pc(inv_kitti_truth, kitti_pcs[1])
    trans_pc = utils.transform_pc(kitti_truth[1, :], kitti_pcs[1])
    diagnostics.plot_consec_pc(kitti_pcs[0], inv_pc)
    diagnostics.plot_consec_pc(kitti_pcs[0], trans_pc)

    init_map_ndt = ndt.ndt_approx(kitti_pcs[0], horiz_grid_size=1.0, vert_grid_size=1.0)
    ndt.display_ndt_cloud(init_map_ndt)
    final_map_ndt = mapping.combine_pc_for_map([kitti_pcs[1]], inv_kitti_truth, init_map_ndt)
    ndt.display_ndt_cloud(final_map_ndt)
    print('It works!')
    return None


def pc_sim_test():
    kitti_pcs = data_utils.load_kitti_pcs(0, 70, pc_diff=70)
    kitti_truth = data_utils.kitti_sequence_poses(0, 70, diff=70)
    inv_kitti_truth = utils.invert_odom_transfer(kitti_truth[1, :])
    inv_pc = utils.transform_pc(inv_kitti_truth, kitti_pcs[1])
    init_map_ndt = ndt.ndt_approx(kitti_pcs[0], horiz_grid_size=1.0, vert_grid_size=1.0)
    diagnostics.plot_consec_pc(kitti_pcs[0], inv_pc)
    print(mapping.pc_similarity(init_map_ndt, inv_pc))
    return None


#test_map_stitch()
pc_sim_test()
#verify_combine_odometry()
#verify_odometry_difference()

"""
def verify_rot_matrix():
    test_angles = np.array([90, 45])
    test_axes = ['x', 'y', 'z']
    for ang in range(len(test_angles)):
        for ax in range(len(test_axes)):
            output = utils.rotation_matrix(test_angles[ang], test_axes[ax])
            print('Press any key')
    print('verify_rot_matrix: Test function complete')
    return None


def verify_dcm2eul():
    test_matrix_1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    output_1 = utils.dcm2eul(test_matrix_1)
    print('Press any key')
    test_matrix_2 = np.array([[1/np.sqrt(2), -1/np.sqrt(2), 0], [1/np.sqrt(2), 1/np.sqrt(2), 0], [0, 0, 1]])
    output_2 = utils.dcm2eul(test_matrix_2)
    print('Press any key')
    test_matrix_3 = np.array([[np.cos(np.pi/6), 0, np.sin(np.pi/6)], [0, 1, 0], [-np.sin(np.pi/6), 0, np.cos(np.pi/6)]])
    output_3 = utils.dcm2eul(test_matrix_3)
    print('Press any key')
    test_matrix_4 = np.matmul(test_matrix_2, test_matrix_3)
    output_4 = utils.dcm2eul(test_matrix_4)
    print('Press any key')
    return None


def verify_eul2dcm():
    test_angle_1 = np.array([45, 0, 0])
    output_1 = utils.eul2dcm(test_angle_1)
    print('Go for Ashwin')
    test_angle_2 = np.array([0, 0, 0])
    output_1 = utils.eul2dcm(test_angle_2)
    print('Go for Ashwin')
    test_angle_3 = np.array([0, 45, 0])
    output_1 = utils.eul2dcm(test_angle_3)
    print('Go for Ashwin')
    test_angle_4 = np.array([90, 0, 0])
    output_1 = utils.eul2dcm(test_angle_4)
    print('Go for Ashwin')
    test_angle_5 = np.array([90, 90, 45])
    output_1 = utils.eul2dcm(test_angle_5)
    print('Go for Ashwin')
    return None


def verify_transform_pts():
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 1]])
    euler_angle_1 = np.array([0, 0, 0])
    dcm_1 = utils.eul2dcm(euler_angle_1)
    translation_1 = np.array([0, 0, 0])
    T_1 = np.vstack((np.hstack((dcm_1, translation_1.T)), np.array([[0, 0, 0, 1]])))
    transformed_pts_1 = utils.transform_pts(points, T_1)
    print('review_transform_pts 1 done.')
    euler_angle_2 = np.array([30, 0, 90])
    dcm_2 = utils.eul2dcm(euler_angle_2)
    translation_2 = np.array([0, 0, 0])
    T_2 = np.vstack((np.hstack((dcm_2, translation_2.T)), np.array([[0, 0, 0, 1]])))
    transformed_pts_2 = utils.transform_pts(points, T_2)
    print('review_transform_pts 2 done.')
    translation_3 = np.array([0.5, 0.5, 0.5])
    T_3 = np.vstack((np.hstack((dcm_2, translation_3.T)), np.array([[0, 0, 0, 1]])))
    transformed_pts_3 = utils.eul2dcm(points, T_3)
    print('review_transform_pts 3 done.')
    return None


# Running functions to test utils module


verify_rot_matrix()
verify_eul2dcm()
verify_dcm2eul()
verify_transform_pts()
"""