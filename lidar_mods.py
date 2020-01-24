"""
lidar_mods.py
Functions to modify LiDAR point clouds for faulty cases
Author: Ashwin Kanhere
Date created: 8th July 2019
"""

import numpy as np


def modify_ranging(pc, delta_r):
    r = np.atleast_2d(np.linalg.norm(pc, axis=1)).T
    delta_coord = pc/r*delta_r
    mod_pc = pc + delta_coord
    return mod_pc