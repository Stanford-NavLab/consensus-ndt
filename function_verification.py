"""
function_verification.py
File containing test cases to verify whether functions are working as they're supposed to or not
Author: Ashwin Kanhere
Date created: 30th May 2019
"""
import numpy as np
import utils


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