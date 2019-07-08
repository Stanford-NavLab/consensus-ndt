"""
data_utils.py
File for NDT SLAM helper functions that deal exclusively with data handling
Author: Ashwin Kanhere
Date Created: 15th April 2019
"""
import numpy as np
import transforms3d
import pykitti


from utils import affine_to_odometry, odometry_difference


def raw_kitti_poses(num_frames=100, raw_pose_mode='laptop'):
    data = load_kitti_data(0, num_frames, 1, mode=raw_pose_mode)
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


def kitti_sequence_poses(num_frames=100, seq_input_mode='laptop'):
    map_odometry = raw_kitti_poses(num_frames, raw_pose_mode=seq_input_mode)
    num_frames = map_odometry.shape[0]
    kitti_odom = np.zeros([num_frames, 6])
    for i in range(1, num_frames):
        kitti_odom[i, :] = odometry_difference(map_odometry[i-1, :], map_odometry[i, :])
    return kitti_odom


def load_kitti_data(start, end, diff=1, mode='laptop'):
    if mode == 'laptop':
        basedir = 'D:\\Users\\kanhe\\Box Sync\\RA Work\\ION GNSS 19\\Implementation\\Dataset'
        date = '2011_09_26'
        drive = '0005'
    elif mode == 'server':
        basedir = '/home/kanhere2/ion-gnss-19/bleagh'
        date = '2011_09_26'
        drive = '0005'
    else:
        raise ValueError('Wrong value entered for mode in load_kitti_data')
    data = pykitti.raw(basedir, date, drive, frames=range(start, end, diff))
    return data


def load_uiuc_pcs(start, end, diff=1, mode='laptop'):
    uiuc_pcs = []
    if mode == 'laptop':
        folder_loc = 'D:\\Users\\kanhe\\Box Sync\\RA Work\\ION GNSS 19\\Implementation\\Dataset\\uiuc_pointclouds\\'
        filename = 'pc_'
        ext = '.csv'
    elif mode == 'server':
        folder_loc = '/home/kanhere2/ion-gnss-19/uiuc_dataset/'
        filename = 'pc_'
        ext = '.csv'
    else:
        raise ValueError('Wrong value entered for mode in load_uiuc_pcs')
    for idx in range(start, end, diff):
        pc = np.loadtxt(folder_loc + filename + str(idx) + ext, delimiter=',')
        uiuc_pcs.append(pc)
    return uiuc_pcs


def load_kitti_pcs(start, end, pc_diff=1, pc_mode='laptop'):
    kitti_pcs = []
    data = load_kitti_data(start, end, diff=pc_diff, mode=pc_mode)
    for idx in range(start, end, pc_diff):
        kitti_pcs.append(data.get_velo(idx))
    return kitti_pcs
