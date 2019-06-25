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
import transforms3d


# TODO: Check rotation and displacement conventions for the map. Does the optimizer return the distance of the pc from
#  the map, or the distance of the map from the pc?

# TODO: Populate docstring for all functions once they're tested and working

# TODO: Fix broken function references


def pc_similarity(ndt_cloud, pc):
    """
    Function
    :param ndt_cloud: NDT Cloud representing the map
    :param pc: Point cloud transformed with the odometry vector
    :return: sim: The degree of similarity between the pointcloud and map
    """
    # TODO: Check this function across multiple point clouds
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
    # TODO: Check the transfom convention as mentioned in the header of mapping.py
    pc_num = len(keyframe_pcs)
    keyframe_ndts = []
    horiz_grid_size = map_ndt.horiz_grid_size
    vert_grid_size = map_ndt.vert_grid_size
    initial_map_odometry = odometry_from_map(sequential_odometry)
    for pc in keyframe_pcs:
        keyframe_ndts.append(ndt_approx(pc, horiz_grid_size, vert_grid_size))
    res = minimize(objective, initial_map_odometry, method='BFGS', jac=jacobian,
                   args=(map_ndt, keyframe_pcs, keyframe_ndts), options={'disp': True})
    map_odom_solution = res.x
    for i in range(pc_num):
        pc = keyframe_pcs[i]
        pc_map_odom = map_odom_solution[i, :]
        # TODO: Check if this is the right vector to be transforming the pc by
        transformed_pc = utils.transform_pc(pc_map_odom, pc)
        map_ndt.update_cloud(transformed_pc)
    map_ndt.update_displacement(map_odom_solution[-1, :])
    return map_odom_solution, map_ndt


def objective(map_odometry, map_ndt, keyframe_pcs, keyframe_ndts):
    # TODO: Change map_odometry to a vector from a matrix
    # TODO: Check the value of the objective for some test cases
    # Since this is the objective function, the pcs and ndts will not be transformed for every iteration
    obj_val = 0
    pc_num = len(keyframe_pcs)
    for i in range(pc_num):
        obj_val += odometry.objective(map_odometry[i, :], map_ndt, keyframe_pcs[i])
        for j in range(i+1, pc_num):
            pc_delta_odom = utils.odometry_difference(map_odometry[i, :], map_odometry[j, :])
            obj_val += odometry.objective(pc_delta_odom, keyframe_ndts[i], keyframe_pcs[j])
    return obj_val


def jacobian(map_odometry, map_ndt, keyframe_pcs, keyframe_ndts):
    # TODO: Jacobian format and values are wrong. The Jacobian should be a vector length of all parameters 6*pc_num here
    # TODO: Check the value of the gradient vector returned by this function
    jacob_val = np.zeros(6)
    pc_num = len(keyframe_pcs)
    for i in range(pc_num):
        jacob_val += odometry.jacobian(map_odometry[i, :], map_ndt, keyframe_pcs[i])
        for j in range(i + 1, pc_num):
            pc_delta_odom = utils.odometry_difference(map_odometry[i, :], map_odometry[j, :])
            jacob_val += odometry.jacobian(pc_delta_odom, keyframe_ndts[i], keyframe_pcs[j])
    return jacob_val


def odometry_from_map(sequential_odometry):
    # TODO: Check odometry_from_map function implementation
    N = sequential_odometry.shape[0]
    map_odometry = np.zeros_like(sequential_odometry)
    map_odometry[0, :] = sequential_odometry[0, :]
    for i in range(1, N):
        map_odometry[i, :] = utils.combine_odometry(map_odometry[i-1, :], sequential_odometry[i, :])
    return map_odometry


