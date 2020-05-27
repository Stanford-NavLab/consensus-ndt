"""
lidar_mods.py
Functions to modify LiDAR point clouds for faulty cases
Author: Ashwin Kanhere
Date created: 8th July 2019
Date modified: 18th July 2019
"""

import numpy as np


def modify_ranging(pc, delta_r):
    """
    Add ranging errors to PC. \Delta is proportional to distance of point from LiDAR
    param pc: PC before modification
    param delta_r: Bias per m ranging to add to point cloud
    return mod_pc: Modified point cloud with added bias
    """
    r = np.atleast_2d(np.linalg.norm(pc, axis=1)).T
    delta_coord = pc/r*delta_r
    mod_pc = pc + delta_coord
    return mod_pc