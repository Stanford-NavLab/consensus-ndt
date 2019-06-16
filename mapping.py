"""
mapping.py
Functions to update the map and perform global optimization given for a keyframe of NDT Clouds
Author: Ashwin Kanhere
Date created: 13th June 2019
Last modified: 13th June 2019
"""

import ndt
import numpy as np

integrity_limit = 0.7


def pc_similarity(ndt_cloud, pc):
    """

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


def filter_voxels_integrity(ndt_cloud):
    delete_index = []
    for key in ndt_cloud.stats.keys():
        if ndt_cloud.stats[key]['integrity'] < integrity_limit:
            delete_index.append(key)
    for del_key in delete_index:
        del ndt_cloud.stats[del_key]
    return ndt_cloud
