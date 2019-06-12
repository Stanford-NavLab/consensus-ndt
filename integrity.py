"""
integrity.py
Functions to calculate the integrity of a particular voxel/ navigation solution
Author: Ashwin Kanhere
Date created: 10th June 2019
Last modified: 10th June 2019
"""

import numpy as np


def calculate_dop(points, iscore):
    """
    Function to calculate the DOP of a group of points, weighted by their integrity score
    :param points: The points for which the IDOP (Integrity DOP) is to be calculated
    :param iscore: The corresponding integrity scores of each point
    :return IDOP: The Integrity weighted DOP of the points
    :return DOP: The traditional DOP of the points (for reference or normalization)
    """
    N = points.shape[0]
    row_norm = np.linalg.norm(points, axis=1)
    coord_norm = np.broadcast_to(np.atleast_2d(row_norm).T, [N, 3])
    unit_vect = points/coord_norm
    G_norm = np.hstack((unit_vect, np.ones([N, 1])))
    G_dash = np.atleast_2d(iscore).T*G_norm
    H_dash = np.linalg.inv(np.matmul(G_dash.T, G_dash))
    IDOP = np.sqrt(np.sum(np.diag(H_dash)))
    H = np.linalg.inv(np.matmul(G_norm.T, G_norm))
    DOP = np.sqrt(np.sum(np.diag(H)))
    return IDOP, DOP


def solution_score(points, point_iscore):
    """
    Function to return the integrity score of a navigation solution
    :param points: Points that were used to obtain the navigation solution
    :param point_iscore: The integrity score corresponding to these points
    :return: sol_iscore: The final solution's integrity score
    """
    IDOP, DOP = calculate_dop(points, point_iscore)
    sol_iscore = DOP/IDOP
    return sol_iscore