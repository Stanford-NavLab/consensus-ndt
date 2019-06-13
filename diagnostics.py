"""
diagnostics.py
Functions to perform diagnostics and testing
Author: Ashwin Kanhere
Date created: 10th June 2019
Last modified: 12th June 2019
"""

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import integrity


def display_voxel_points(key, voxel_dict, points=np.array([]), density=1.0, horiz_size=1.0, vert_size=1.0):
    plt.interactive(False)
    center = np.array(key)
    mu = voxel_dict['mu']
    sigma = voxel_dict['sigma']
    plot_number = np.int(density*100)
    plot_dist = np.random.multivariate_normal(mu, sigma, plot_number)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(plot_dist[:,0], plot_dist[:, 1], plot_dist[:, 2], s=4)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=100)
    ax.set_xlim([center[0] - horiz_size/2, center[0] + horiz_size/2])
    ax.set_ylim([center[1] - horiz_size/2, center[1] + horiz_size/2])
    ax.set_zlim([center[2] - vert_size/2, center[2] + vert_size/2])
    plt.show()
    print('Bleagh')
    return None


def find_integrity(ndt_cloud, points):
    test_xyz = points[:, :3]
    binned_points = ndt_cloud.bin_in_voxels(test_xyz)
    N = len(ndt_cloud.stats)
    iscore= np.zeros(N)
    loop_index = 0
    mu_points = np.zeros([N, 3])
    for key, val in ndt_cloud.stats.items():
        mu_points[loop_index, :] = val['mu']
        iscore[loop_index] = integrity.voxel_integrity(val, binned_points[key])
        loop_index += 1
    avg_iscore = np.mean(iscore)
    Im = integrity.solution_score(mu_points, iscore)
    return Im