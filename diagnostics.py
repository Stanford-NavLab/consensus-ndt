"""
diagnostics.py
Functions to perform diagnostics and testing
Author: Ashwin Kanhere
Date created: 10th June 2019
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


def check_gradient(objective, jacobian, ndt_cloud, test_pc):
    delta = 1.5e-08
    odometry = np.zeros(6)
    jacob_val = np.zeros(6)
    analytical_jacob = jacobian(odometry, ndt_cloud, test_pc)
    for i in range(6):
        new_odometry = np.zeros(6)
        new_odometry[i] = delta
        jacob_val[i] = (objective(new_odometry, ndt_cloud, test_pc) - objective(odometry, ndt_cloud, test_pc))/delta
    jacobian_error = jacob_val - analytical_jacob
    print('The jacobian vector error is ', jacobian_error)
    return jacobian_error