"""
mapping.py
Functions to update the map and perform global optimization given for a keyframe of NDT Clouds
Author: Ashwin Kanhere
Date created: 13th June 2019
Last modified: 13th June 2019
"""

import ndt
import numpy as np
from ndt import ndt_approx
import odometry
from scipy.optimize import minimize
import utils


def pc_similarity(ndt_cloud, pc):
    """
    Computing common area between two point clouds
    :param ndt_cloud: NDT Cloud representing the map
    :param pc: Point cloud transformed with the odometry vector
    :return: sim: The degree of similarity between the pointcloud and map
    """
    xlim_pc, ylim_pc, _ = ndt.find_pc_limits(pc)
    base_area = 2*xlim_pc*2*ylim_pc
    bin_in_voxels = ndt_cloud.bin_in_voxels(pc)
    voxel_xmin = 0
    voxel_xmax = 0
    voxel_ymin = 0
    voxel_ymax = 0
    for key, _ in bin_in_voxels.items():
        current_center = np.array(key)
        if not np.sum(np.isnan(current_center)):
            if current_center[0] > voxel_xmax:
                voxel_xmax = current_center[0]
            elif current_center[0] < voxel_xmin:
                voxel_xmin = current_center[0]
            if current_center[1] > voxel_ymax:
                voxel_ymax = current_center[1]
            elif current_center[1] < voxel_ymin:
                voxel_ymin = current_center[1]
    cover_area = (voxel_xmax - voxel_xmin)*(voxel_ymax - voxel_ymin)
    sim = cover_area/base_area
    return sim


def mapping(map_ndt, keyframe_pcs, sequential_odometry):
    """
    Implements the mapping step of the function
    :param map_ndt: NDT approximation of the map created prior to current keyframe
    :param keyframe_pcs: Point clouds that belong to the current keyframe
    :param sequential_odometry: Laser odometry between sequential point clouds as calculated by the odometry process
    :return: map_odom_solution: Transformation of PCs to map NDT reference
    """
    keyframe_ndts = []
    horiz_grid_size = map_ndt.horiz_grid_size
    vert_grid_size = map_ndt.vert_grid_size
    initial_map_odometry_matrix = odometry_from_map(sequential_odometry)
    initial_map_odometry = np.reshape(initial_map_odometry_matrix.T, [1, -1])[0, :]
    for pc in keyframe_pcs:
        keyframe_ndts.append(ndt_approx(pc, horiz_grid_size, vert_grid_size))
    res = minimize(objective, initial_map_odometry, method='Nelder-Mead', args=(map_ndt, keyframe_pcs, keyframe_ndts),
                   options={'disp': True})
    map_odom_solution_vector = res.x
    map_odom_solution = np.reshape(map_odom_solution_vector, [-1, 6])
    return map_odom_solution


def objective(map_odometry, map_ndt, keyframe_pcs, keyframe_ndts):
    """
    Objective function for mapping optimization
    """
    # Since this is the objective function, the pcs and ndts will not be transformed for every iteration
    obj_val = 0
    pc_num = len(keyframe_pcs)
    for i in range(pc_num):
        obj_val += odometry.objective(map_odometry[6*i: 6*(i+1)], map_ndt, keyframe_pcs[i])
        for j in range(i+1, pc_num):
            pc_delta_odom = utils.odometry_difference(map_odometry[6*i:6*(i+1)], map_odometry[6*j:6*(j+1)])
            obj_val += odometry.objective(pc_delta_odom, keyframe_ndts[i], keyframe_pcs[j])
    return obj_val


def odometry_from_map(sequential_odometry):
    N = sequential_odometry.shape[0]
    map_odometry = np.zeros_like(sequential_odometry)
    map_odometry[0, :] = sequential_odometry[0, :]
    for i in range(1, N):
        map_odometry[i, :] = utils.combine_odometry(map_odometry[i-1, :], sequential_odometry[i, :])
    return map_odometry


def combine_pc_for_map(keyframe_pcs, mapping_odom, map_ndt):
    """
    Function to update the map using the mapping solution
    :param keyframe_pcs: PCs belonging to the current keyframe
    :param mapping_odom: Solution to the mapping objective function
    :param map_ndt: NDT approximation for the map upto the previous keyframe
    :return map_ndt: Updated parameter map_ndt with the given odometry function
    """
    mapping_odom = np.atleast_2d(mapping_odom)
    for idx, pc in enumerate(keyframe_pcs):
        # Transform points to the LiDAR's reference when the measurements were taken
        pc = pc[:, :3]  # To trim the pointcloud incase it isn't trimmed
        inv_odom = utils.invert_odom_transfer(mapping_odom[idx, :])
        trans_pc = utils.transform_pc(inv_odom, pc)
        map_ndt.update_cloud(trans_pc)
    return map_ndt


