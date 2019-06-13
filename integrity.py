"""
integrity.py
Functions to calculate the integrity of a particular voxel/ navigation solution
Author: Ashwin Kanhere
Date created: 10th June 2019
Last modified: 13th June 2019
"""

import numpy as np
from scipy.stats import chi2


def sigmoid(x):
    s = 1/(1 + np.exp(-x))
    return s


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


def voxel_integrity(voxel_dict, points):
    """
    Function to calculate an integrity score for the points distributed inside a voxel
    :param voxel_dict: Dictionary containing the mean and covariance for the voxel in question
    :param points: Points that lie inside the voxel in question
    :return: r: The integrity score for that voxel
    """
    N = points.shape[0]  # Number of points
    mu = voxel_dict['mu']
    sigma_inv = np.linalg.inv(voxel_dict['sigma'])
    q = points[:, :3] - mu
    r = np.sum(np.diag(np.matmul(q, np.matmul(sigma_inv, q.T))))
    r = r / (N - 4)
    T_upper = chi2.ppf(0.999, N - 4)
    T_lower = chi2.ppf(0.001, N - 4)
    scale_limit = 3
    r_scaled = (2*scale_limit)*T_lower/(T_upper - T_lower) - (2*scale_limit)*r/(T_upper - T_lower) + scale_limit
    Iv = sigmoid(r_scaled)
    return Iv


