"""
diagnostics.py
Functions to perform diagnostics and testing
Author: Ashwin Kanhere
Date created: 10th June 2019
"""

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import odometry
import utils
import pptk
from sklearn.neighbors import NearestNeighbors


def display_voxel_points(key, voxel_stats, points=np.array([]), density=1.0, horiz_size=1.0, vert_size=1.0):
    """
    Function to display points containted in an NDT-voxel and the corresponding Gaussian
    :param key: Identifier for NDT-voxel
    :param voxel_stats: Statistics of the NDT-voxel
    :param points: Points binned inside said NDT voxel
    :param density: The plot density for the Gaussian distribution
    :param horiz_size: Horizontal size of the voxel
    :param vert_size: Vertical size of the voxel
    :return: None
    """
    plt.interactive(False)
    center = np.array(key)
    mu = voxel_stats['mu']
    sigma = voxel_stats['sigma']
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


def check_gradient(objective, jacobian, ndt_cloud, test_pc, odometry_vector, print_output=False):
    """
    Manual function to check analytical gradient with the numerical approximation
    :param objective: Target objective function (for which the gradient is to be checked)
    :param jacobian: Analytical Jacobian function for the given objective function
    :param ndt_cloud: NDT Cloud Parameter for objective and gradient
    :param test_pc: PC parameter for functions
    :param odometry_vector: Point at which the Jacobian needs to be checked
    :param print_output: Flag to activate printing of values in the function
    :return: jacobian_error: Vector containing the difference between analytical and numerical Jacobian
    :return: jacob_error_norm: Magnitude of jacobian error
    """
    delta = 1.5e-08
    odom = odometry_vector
    jacob_val = np.zeros(6)
    analytical_jacob = jacobian(odometry, ndt_cloud, test_pc)
    for i in range(6):
        new_odometry = np.zeros(6)
        for j in range(6):
            new_odometry[j] += odometry_vector[j]
        new_odometry[i] += delta
        jacob_val[i] = (objective(new_odometry, ndt_cloud, test_pc) - objective(odom, ndt_cloud, test_pc))/delta
    jacobian_error = jacob_val - analytical_jacob
    jacob_error_norm = np.linalg.norm(jacobian_error)
    if print_output:
        print('The analytical jacobian is ', analytical_jacob)
        print('The numerical jacobian vector is', jacob_val)
        print('The jacobian vector error is ', jacobian_error)
        print('The magnitude of the jacobian error is', jacob_error_norm)
    return jacobian_error, jacob_error_norm


def check_hessian(jacobian, hessian, ndt_cloud, test_pc, odometry_vector, print_output=False):
    # First column of hessian is derivative of first element of jacobian with respect to all the variables
    delta = 1.5e-8
    odom = odometry_vector
    hess_val = np.zeros([6, 6])
    analytical_hess = hessian(odom, ndt_cloud, test_pc)
    for cidx in range(6):
        for ridx in range(6):
            new_odometry = np.zeros(6)
            new_odometry += odometry_vector
            new_odometry[ridx] += delta
            hess_val[ridx, cidx] = (jacobian(new_odometry, ndt_cloud, test_pc)[cidx] - jacobian(odom, ndt_cloud,
                                                                                                test_pc)[cidx]) / delta
            print('Checking Hessian for ', [ridx, cidx])
    hessian_error = hess_val - analytical_hess
    hess_error_norm = np.linalg.norm(hessian_error)
    if print_output:
        print('The analytical jacobian is ', analytical_hess)
        print('The numerical jacobian vector is', hess_val)
        print('The jacobian vector error is ', hessian_error)
        print('The magnitude of the jacobian error is', hess_error_norm)
    return hessian_error, hess_error_norm


def objective_variation(ndt_cloud, test_pc, axis=0, limit=0.5, num_vals=40):
    """
    Function to plot the variation in the objective function about a particular axis of the objective function
    :param ndt_cloud: NDT Cloud Parameter for objective and gradient
    :param test_pc: PC parameter for functions
    :param axis: Axis about which the variation in the objective is being checked
    :param limit: Limit of the variation about origin for which the objective is evaluated
    :param num_vals: The number of points at which the objective is evaluated
    :return:
    """
    odom_dim = axis
    objective_value = np.zeros(num_vals)
    dim_variation = np.linspace(-limit, limit, num_vals)
    for i in range(num_vals):
        inverse_odom_value = utils.invert_odom_transfer(np.array([0.11192455, - 0.31511185,  0.01197815,  0.24025847,
                                                                  0.06658534,  0.90697335]))
        # print(inverse_odom_value)
        odom_value = inverse_odom_value
        odom_value[odom_dim] += dim_variation[i]
        objective_value[i] = odometry.objective(odom_value, ndt_cloud, test_pc)
    plt.plot(dim_variation, objective_value)
    plt.show()
    print('Required Variation Done')
    inverse_odom_value = utils.invert_odom_transfer(np.array([0.11192455, - 0.31511185, 0.01197815, 0.24025847,
                                                              0.06658534, 0.90697335]))

    print(inverse_odom_value)
    return None


def ind_lidar_odom(test_pc, ref_pc):
    """
    Function to find the ICP odometry between two point clouds
    :param test_pc:
    :param ref_pc:
    :return: icp_odometry : Transform that maps points in test_pc to ref_pc (similar to the existing NDT implementation)
    """
    A = icp(ref_pc[:, :3], test_pc[:, :3])
    icp_odometry = utils.affine_to_odometry(A[0])
    return icp_odometry


def plot_consec_pc(pc_0, pc_1):
    """
    Tool to plot two pointclouds on the same pptk viewer
    :param pc_0:
    :param pc_1:
    :return:
    """
    color_index = np.hstack((np.ones(pc_0.shape[0]), np.zeros(pc_1.shape[0])))
    stacked_pc = np.vstack((pc_0, pc_1))
    view = pptk.viewer(stacked_pc, color_index)
    view.color_map('cool')
    return None

########################################################
# ICP Code taken from https://github.com/ClayFlannigan/icp/blob/master/icp.py
########################################################


def best_fit_transform(A, B):
    """
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    """

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
       Vt[m-1,:] *= -1
       R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R,centroid_A.T)

    # homogeneous transformation
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t


def nearest_neighbor(src, dst):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''

    assert src.shape == dst.shape

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()


def icp(A, B, init_pose=None, max_iterations=20, tolerance=0.001):
    '''
    The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
    Input:
        A: Nxm numpy array of source mD points
        B: Nxm numpy array of destination mD point
        init_pose: (m+1)x(m+1) homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
        T: final homogeneous transformation that maps A on to B
        distances: Euclidean distances (errors) of the nearest neighbor
        i: number of iterations to converge
    '''

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m+1,A.shape[0]))
    dst = np.ones((m+1,B.shape[0]))
    src[:m,:] = np.copy(A.T)
    dst[:m,:] = np.copy(B.T)

    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 0

    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination points
        distances, indices = nearest_neighbor(src[:m,:].T, dst[:m,:].T)

        # compute the transformation between the current source and nearest destination points
        T,_,_ = best_fit_transform(src[:m,:].T, dst[:m,indices].T)

        # update the current source
        src = np.dot(T, src)

        # check error
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # calculate final transformation
    T,_,_ = best_fit_transform(A, src[:m,:].T)

    return T, distances, i