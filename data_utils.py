"""
data_utils.py
File for NDT SLAM helper functions that deal exclusively with data handling
Author: Ashwin Kanhere
Date Created: 15th April 2019
Date Modified: 26th May 2020
"""
import numpy as np
import transforms3d
import pykitti


from utils import affine_to_odometry, odometry_difference


def raw_kitti_poses(start, end, pose_diff, raw_pose_mode='laptop'):
    """
    Function to extract KITTI pose data with respect to the origin (map frame of reference)
    :param start: Starting PC index
    :param end: Ending PC index
    :param pose_diff: Difference in consecutive indices
    :param raw_pose_mode: Data extraction file path to use
    :return map_odometry: Poses wrt map reference
    """
    data, num_frames = load_kitti_data(start, end, diff=pose_diff, mode=raw_pose_mode)
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


def kitti_sequence_poses(start, end, diff=1, seq_input_mode='laptop'):
    """
    Function to extract odometry (consecutive pose differences) for KITTI data
    :param start: Starting PC index
    :param end: Ending PC index
    :param diff: Difference in consecutive indices
    :param seq_input_mode: Data extraction file path to use
    :return kitti_odom: 6DOF odometry between consecutive PCs
    """
    map_odometry = raw_kitti_poses(start, end, pose_diff=diff, raw_pose_mode=seq_input_mode)
    num_frames = map_odometry.shape[0]
    kitti_odom = np.zeros([num_frames, 6])
    for i in range(1, num_frames):
        kitti_odom[i, :] = odometry_difference(map_odometry[i-1, :], map_odometry[i, :])
    return kitti_odom


def load_kitti_data(start, end, diff=1, mode='laptop'):
    """
    Function to extract KITTI measurements
    :param start: Starting PC index
    :param end: Ending PC index
    :param diff: Difference in consecutive indices
    :param mode: Data extraction file path to use
    :return data: KITTI data (containing PCs, poses etc.)
    :return num_frames: Number of frames in extracted data
    """
    end += 1
    if mode == 'laptop':
        basedir = 'C:\Users\kanhe\Documents\Consensus-NDT\ion-gnss-19\example-dataset'
        date = '2011_09_26'
        drive = '0005'
    elif mode == 'server':
        basedir = '/home/kanhere2/ion-gnss-19/test_dataset/'
        date = '2011_09_26'
        drive = '0005'
    else:
        raise ValueError('Wrong value entered for mode in load_kitti_data')
    data = pykitti.raw(basedir, date, drive, frames=range(start, end, diff))
    num_frames = np.int(np.ceil((end - start)/diff))
    return data, num_frames


def load_uiuc_pcs(start, end, diff=1, mode='laptop'):
    """
    Function to extract UIUC PC 
    :param start: Starting PC index
    :param end: Ending PC index
    :param diff: Difference in consecutive indices
    :param mode: Data extraction file path to use
    :return uiuc_pcs: List containig numpy array of PCs
    """
    end += 1  # To include the end index
    uiuc_pcs = []
    if mode == 'laptop':
        folder_loc = 'C:\Users\kanhe\Documents\Consensus-NDT\ion-gnss-19\example-dataset\uiuc_pointclouds\\'
        filename = 'pc_'
        ext = '.csv'
    elif mode == 'server':
        folder_loc = '/home/navlab-admin/consensus-ndt/datasets/uiuc_pointclouds/'
        filename = 'pc_'
        ext = '.csv'
    else:
        raise ValueError('Wrong value entered for mode in load_uiuc_pcs')
    for idx in range(start, end, diff):
        pc = np.loadtxt(folder_loc + filename + str(idx) + ext, delimiter=',')
        uiuc_pcs.append(pc)
    return uiuc_pcs


def load_kitti_pcs(start, end, pc_diff=1, pc_mode='laptop'):
    """
    Function to extract KITTI PC 
    :param start: Starting PC index
    :param end: Ending PC index
    :param diff: Difference in consecutive indices
    :param mode: Data extraction file path to use
    :return uiuc_pcs: List containig numpy array of PCs
    """
    kitti_pcs = []
    data, num_frames = load_kitti_data(start, end, diff=pc_diff, mode=pc_mode)
    for idx in range(num_frames):
        kitti_pcs.append(data.get_velo(idx)[:, :3])
    return kitti_pcs
