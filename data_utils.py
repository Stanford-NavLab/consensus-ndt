"""
data_utils.py
File for NDT SLAM helper functions that deal exclusively with data handling
Author: Ashwin Kanhere
Date Created: 15th April 2019
"""
import numpy as np
import transforms3d

from utils import affine_to_odometry, odometry_difference


def raw_kitti_poses(data, num_frames=100):
    R_imu_to_velo = np.array([[9.999976e-01, 7.553071e-04, -2.035826e-03], [-7.854027e-04, 9.998898e-01, -1.482298e-02],
                  [2.024406e-03, 1.482454e-02, 9.998881e-01]])
    T_imu_to_velo = -8.086759e-01, 3.195559e-01, -7.997231e-01
    A_imu_to_velo = transforms3d.affines.compose(T_imu_to_velo, R_imu_to_velo, np.ones([3]))
    map_odometry = np.zeros([num_frames, 6])
    for i in range(num_frames):
        oxts_data = data.oxts[i]
        A_w_to_imu = oxts_data[1]
        A_w_to_velo = np.matmul(A_imu_to_velo, A_w_to_imu)
        map_odometry[i, :] = affine_to_odometry(A_w_to_velo)
    return map_odometry


def kitti_sequence_poses(data, num_frames=100):
    map_odometry = raw_kitti_poses(data, num_frames)
    num_frames = map_odometry.shape[0]
    kitti_odom = np.zeros([num_frames, 6])
    for i in range(1, num_frames):
        kitti_odom[i, :] = odometry_difference(map_odometry[i-1, :], map_odometry[i, :])
    return kitti_odom
