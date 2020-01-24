"""
ashwin-playground.py
File to test different modules and functions created for the offline implementation of NDT-SLAM
Author: Ashwin Kanhere
Date Modified: 15th April 2019
"""
import numpy as np
import pykitti
import pptk
import transforms3d

import data_utils
import integrity
import time
from matplotlib import pyplot as plt
from scipy.stats import chi2
import ndt
import data_utils


def sigmoid(x):
    s = 1/(1 + np.exp(-x))
    return s


def extract_data():
    basedir = 'D:\\Users\\kanhe\\Box Sync\\Research Projects\\Consensus NDT SLAM\\Dataset'
    date = '2011_09_26'
    drive = '0005'

    data = pykitti.raw(basedir, date, drive, frames=range(0, 5, 1))

    points_lidar = data.velo
    return data


def ndt_test():
    data = extract_data()
    test_lidar = data.get_velo(0) # LiDAR point cloud is a Nx4 numpy array
    # test_cloud = ndt_approx.ndt_approx(test_lidar, horiz_grid_size=0.25, vert_grid_size=0.25)
    testing = ndt.ndt_approx(test_lidar, horiz_grid_size=1, vert_grid_size=1)
    return None


def integrity_test():
    points1 = np.array([[1, 0, 0],
                        [-1, 0, 0],
                        [0, 1, 0],
                        [0, -1, 0],
                        [0, 0, 1],
                        [0, 0, -1],
                        ])
    score1 = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9])
    score2 = np.array([0.9, 0.2, 0.9, 0.2, 0.9, 0.2])
    IDOP1, DOP1 = integrity.calculate_dop(points1, score1)
    IDOP2, _ = integrity.calculate_dop(points1, score2)
    print('IDOP1:', IDOP1)
    print('DOP1:', DOP1)
    print('IDOP2:', IDOP2)
    score3 = np.array([0.9, 0.9, 0.9, 0.9, 0.1, 0.1])
    score4 = np.array([0.9, 0.9, 0.9, 0.9, 0.001, 0.001])
    IDOP3, _ = integrity.calculate_dop(points1, score3)
    print('IDOP3:', IDOP3)
    IDOP4, _ = integrity.calculate_dop(points1, score4)
    print('IDOP4:', IDOP4)
    score5 = np.sqrt(DOP1/IDOP3)
    score6 = np.sqrt(DOP1/IDOP2)
    score7 = np.sqrt(DOP1/IDOP4)
    print('score6', score6)
    print('score5', score5)
    print('score7', score7)
    score8 = np.array([0.9, 0.9, 0.001, 0.001, 0.001, 0.001])
    IDOP5, _ = integrity.calculate_dop(points1, score8)
    print('IDOP5', IDOP5)
    print(1/IDOP1)
    print(1/IDOP2)
    print(1/IDOP4)
    print(1/IDOP5)
    data = extract_data()
    test_lidar = data.get_velo(0)
    test_xyz = test_lidar[:, :3]
    N = test_xyz.shape[0]
    print(N)
    print(np.int(N/2))
    score_lidar = np.ones(N)
    IDOP_lidar, DOP_lidar = integrity.calculate_dop(test_xyz, score_lidar)
    print('LiDAR IDOP', IDOP_lidar)
    print('LiDAR DOP', DOP_lidar)
    print('Integrity Score', DOP_lidar, IDOP_lidar)
    score_lidar1 = np.concatenate((0.9*np.ones(np.int(N/2)), 0.2*np.ones(np.int(N/2)+1)))
    print('\n')
    IDOP_lidar1, _ = integrity.calculate_dop(test_xyz, score_lidar1)
    print('Case 1: LiDAR IDOP', IDOP_lidar1)
    print('Case 1: LiDAR DOP', DOP_lidar)
    print('Case 1: Integrity Score ', DOP_lidar/IDOP_lidar1)
    print('\n')
    score_lidar2 = np.concatenate((0.2*np.ones(np.int(N/2)), 0.9*np.ones(np.int(N/2)+1)))
    IDOP_lidar2, DOP_lidar2 = integrity.calculate_dop(test_xyz, score_lidar2)
    print('Case 2: LiDAR IDOP', IDOP_lidar2)
    print('Case 2: LiDAR DOP', DOP_lidar2)
    print('Case 2: Integrity Score ', DOP_lidar/IDOP_lidar2)
    print('\n')
    score_lidar3 = np.concatenate((np.ones(np.int(N/2)), np.ones(np.int(N/2)+1)))
    IDOP_lidar3, DOP_lidar3 = integrity.calculate_dop(test_xyz, score_lidar)
    print('Case 3: LiDAR IDOP', IDOP_lidar3)
    print('Case 3: LiDAR DOP', DOP_lidar3)
    print('Case 3: Integrity Score ', DOP_lidar/IDOP_lidar3)
    print('Ratio of volume of LiDAR vs volume sphere:', np.deg2rad(26.9)/(2*np.pi))
    print(0.55)
    timing = 0
    for i in range(1000):
        t0 = time.time()
        IDOP_lidar, DOP_lidar = integrity.calculate_dop(test_xyz, np.ones(N))
        timing += time.time() - t0
    print(timing/1000)
    timing = 0
    t0 = time.time()
    for i in range(1000):
        IDOP, DOP = integrity.calculate_dop(points1, score1)
    timing = (time.time() - t0)/1000
    print(timing)
    return None
# TODO: Move all plot and example generating code to its own file


def calculate_2d_dop(points, iscore):
    N = points.shape[0]
    row_norm = np.linalg.norm(points, axis=1)
    coord_norm = np.broadcast_to(np.atleast_2d(row_norm).T, [N, 2])
    unit_vect = points/coord_norm
    G_norm = np.hstack((unit_vect, np.ones([N, 1])))
    G_dash = np.atleast_2d(iscore).T*G_norm
    H_dash = np.linalg.inv(np.matmul(G_dash.T, G_dash))
    IDOP = np.sqrt(np.sum(np.diag(H_dash)))
    H = np.linalg.inv(np.matmul(G_norm.T, G_norm))
    DOP = np.sqrt(np.sum(np.diag(H)))
    return IDOP, DOP


def new_integrity_example():
    # Clustered arrangement
    N = 100
    theta_low = np.random.uniform(0, np.pi, N)
    theta_high = np.random.uniform(np.pi, 2*np.pi, N)
    low_coordinates = np.hstack((np.reshape(np.cos(theta_low), [-1, 1]), np.reshape(np.sin(theta_low), [-1, 1])))
    high_coordinates = np.hstack((np.reshape(np.cos(theta_high), [-1, 1]), np.reshape(np.sin(theta_high), [-1, 1])))
    # plt.scatter(low_coordinates[:, 0], low_coordinates[:, 1], color='blue')
    #plt.scatter(high_coordinates[:, 0], high_coordinates[:, 1], color='red')
    u =np.zeros(N)
    v = np.zeros(N)
    plt.quiver(u, v, low_coordinates[:, 0], low_coordinates[:, 1], color='red', units='xy', scale=1.0)
    plt.quiver(u, v, high_coordinates[:, 0], high_coordinates[:, 1], color='blue', units='xy', scale=1.0)
    plt.axis('equal')
    plt.xlim([-1.5, 1.5])
    plt.ylim([-1.5, 1.5])
    plt.show()
    iscore = np.hstack((0.05*np.ones(N), 0.95*np.ones(N)))
    points1 = np.vstack((low_coordinates, high_coordinates))
    IDOP1, DOP = calculate_2d_dop(points1, iscore)
    print(IDOP1)
    print(DOP)
    print(DOP/IDOP1)
    # Non clustered arrangement
    theta_low = np.random.uniform(0, 2*np.pi, N)
    theta_high = np.random.uniform(0, 2*np.pi, N)
    low_coordinates = np.hstack((np.reshape(np.cos(theta_low), [-1, 1]), np.reshape(np.sin(theta_low), [-1, 1])))
    high_coordinates = np.hstack((np.reshape(np.cos(theta_high), [-1, 1]), np.reshape(np.sin(theta_high), [-1, 1])))
    plt.quiver(u, v, low_coordinates[:, 0], low_coordinates[:, 1], color='red', units='xy', scale=1.0)
    plt.quiver(u, v, high_coordinates[:, 0], high_coordinates[:, 1], color='blue', units='xy', scale=1.0)
    plt.axis('equal')
    plt.xlim([-1.5, 1.5])
    plt.ylim([-1.5, 1.5])
    plt.show()
    points2 = np.vstack((low_coordinates, high_coordinates))
    IDOP2, DOP = calculate_2d_dop(points2, iscore)
    print(IDOP2)
    print(DOP)
    print(DOP/IDOP2)
    return None





def paper_total_con():
    points = np.array([[1, 0],
                       [0.707, 0.707],
                       [0, 1],
                       [-0.707, 0.707],
                       [-1, 0],
                       [-0.707, -0.707],
                       ])
    original_points = np.array([[1, 0],
                                [0.707, 0.707],
                                [0, 1],
                                [-0.707, 0.707],
                                [-1, 0],
                                [-0.707, -0.707],
                                [0, -1],
                                [0.707, -0.707],
                       ])
    metric_1 = np.array([1.0, 1.0, 1.0, 0.5, 0.5, 0.5])
    metric_2 = np.array([1.0, 0.5, 1.0, 0.5, 1.0, 0.5])
    H_inv_1 = np.matmul(points.T, np.matmul(np.diag(metric_1**2), points))
    H_inv_2 = np.matmul(points.T, np.matmul(np.diag(metric_2**2), points))
    H_mat_1 = np.linalg.inv(H_inv_1)
    H_mat_2 = np.linalg.inv(H_inv_2)
    H_1 = np.sqrt(np.sum(np.diag(H_mat_1)))
    H_2 = np.sqrt(np.sum(np.diag(H_mat_2)))
    DOP = np.sqrt(np.sum(np.diag(np.linalg.inv(np.matmul(original_points.T, original_points)))))
    # print(H_1)
    # print(H_2)
    print(DOP)
    print(DOP/H_1)
    print(DOP/H_2)
    w1, v1 = np.linalg.eigh(H_mat_1)
    w2, v2 = np.linalg.eigh(H_mat_2)
    wDOP, _ = np.linalg.eigh(np.linalg.inv(np.matmul(points.T, points)))
    w_inv_1, _ = np.linalg.eigh(H_inv_1)
    w_inv_2, _ = np.linalg.eigh(H_inv_2)
    return None


def voxel_integrity(mu, sigma, points):
    N = points.shape[0]  # Number of points
    sigma_inv = np.linalg.inv(sigma)
    if points.ndim == 1:
        q = np.atleast_2d(points[:3]) - mu
    else:
        q = points[:, :3] - mu
    r = np.sum(np.diag(np.matmul(q, np.matmul(sigma_inv, q.T))))
    if N > 4:
        T_upper = chi2.ppf(0.999, N - 4)
        T_lower = chi2.ppf(0.001, N - 4)
        r = r / (N - 4)
        scale_limit = 3
        r_scaled = (2 * scale_limit) * T_lower / (T_upper - T_lower) - (2 * scale_limit) * r / (
                T_upper - T_lower) + scale_limit
        Iv = sigmoid(r_scaled)
    else:
        Iv = 0
    if np.isnan(Iv):
        print('Yet another Nan!')
    return Iv


def paper_vox_con_2():
    case = 3
    N_plot = 3
    plot_cov = 0.01
    points = np.array([[0.3, 0.3],
                       [0.3, 0.35],
                       [0.5, 0.45],
                       [0.7, 0.66],
                       [0.7, 0.74]])
    mu = np.mean(points, axis=0)
    cov = np.cov(points.T)
    print(mu)
    print(cov)
    plot_points = np.random.multivariate_normal(mu, cov, 1000)
    test_match = np.array([[0.4, 0.4],
                           [0.5, 0.5],
                           [0.55, 0.55],
                           [0.6, 0.6],
                           [0.65, 0.65]])
    test_outliers = np.array([[0.2, 0.3],
                              [0.5, 0.7],
                              [0.55, 0.15],
                              [0.7, 0.6],
                              [0.8, 0.85]])
    test_mismatch = np.array([[0.2, 0.4],
                              [0.3, 0.55],
                              [0.55, 0.75],
                              [0.6, 0.76],
                              [0.8, 0.94]])
    test_occlusion = np.array([[0.9, 0.8],
                               [0.9, 0.75],
                               [0.8, 0.7],
                               [0.8, 0.9],
                               [0.8, 0.65]])
    if case == 0:
        test_points = test_match
    elif case == 1:
        test_points = test_outliers
    elif case == 2:
        test_points = test_mismatch
    elif case == 3:
        test_points = test_occlusion
    N_match = np.shape(test_points)[0]
    plot_test = np.reshape(test_points[0, :], [1, -1])
    for test_num in range(N_match):
        temp_x = np.random.normal(test_points[test_num, 0], plot_cov, N_plot)
        temp_y = np.random.normal(test_points[test_num, 1], plot_cov, N_plot)
        temp_pts = np.hstack((np.reshape(temp_x, [-1, 1]), np.reshape(temp_y, [-1, 1])))
        plot_test = np.vstack((plot_test, temp_pts))
    fig = plt.figure()
    plt.scatter(plot_points[:, 0], plot_points[:, 1], s=1, c='b')
    plt.scatter(plot_test[:, 0], plot_test[:, 1], s=36, c='r')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    print(voxel_integrity(mu, cov, plot_test))
    plt.show()

    return None


def vox_con():
    points = np.array([[0.3, 0.3],
                       [0.3, 0.35],
                       [0.5, 0.45],
                       [0.7, 0.66],
                       [0.7, 0.74]])
    mu = np.mean(points, axis=0)
    cov = np.cov(points.T)
    print(mu)
    print(cov)
    plot_points = np.random.multivariate_normal(mu, cov, 1000)
    test_points = np.array([[0.8, 0.8],
                       [0.8, 0.75],
                       [0.7, 0.75],
                       [0.7, 0.66],
                       [0.7, 0.74]])
    fig = plt.figure()
    plt.scatter(plot_points[:, 0], plot_points[:, 1], s=1, c='b')
    plt.scatter(test_points[:, 0], test_points[:, 1], s=36, c='r')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    print(voxel_integrity(mu, cov, test_points))
    plt.show()

    return None


def test_mapping_func():
    data = extract_data()
    sequence_ground_truth = data_utils.kitti_sequence_poses(data)
    mapping_ground_truth = np.zeros_like(sequence_ground_truth)
    N = sequence_ground_truth.shape[0]
    for i in range(N):
        print(i)
    index_list = [0, 30, 60]
    return None


def total_metric_test():
    points = np.array([[1, 0],
                       [0.707, 0.707],
                       [0, 1],
                       [-0.707, 0.707],
                       [-1, 0],
                       [-0.707, -0.707]
                      ])
    metric_1 = np.array([1.0, 1.0, 1.0, 0.5, 0.5, 0.5])
    metric_2 = np.array([1.0, 0.5, 1.0, 0.5, 1.0, 0.5])
    H_inv_1 = np.matmul(points.T, np.matmul(np.diag(metric_1**2), points))
    H_inv_2 = np.matmul(points.T, np.matmul(np.diag(metric_2**2), points))
    H_mat_1 = np.linalg.inv(H_inv_1)
    H_mat_2 = np.linalg.inv(H_inv_2)
    H_1 = np.sqrt(np.sum(np.diag(H_mat_1)))
    H_2 = np.sqrt(np.sum(np.diag(H_mat_2)))
    DOP = np.sqrt(np.sum(np.diag(np.linalg.inv(np.matmul(points.T, points)))))
    DOP_H = np.linalg.inv(np.matmul(points.T, points))
    DOP_original = np.matmul(points.T, points)
    # print(H_1)
    # print(H_2)
    print(DOP)
    print(DOP/H_1)
    print(DOP/H_2)
    # Extracting eigenvalues of the system
    print('Printing the values for the DOP matrix')
    w_DOP, v_DOP = np.linalg.eigh(np.linalg.inv(np.matmul(points.T, points)))
    print('DOP eigenvalues', w_DOP)
    print('DOP eigenvectors', v_DOP)
    w_inv_1, v_inv_1 = np.linalg.eigh(H_mat_1)
    w_inv_2, v_inv_2 = np.linalg.eigh(H_mat_2)
    print('Printing the values for the original matrix')
    print('First H eigenvalues', w_inv_1)
    print('First H eigenvectors', v_inv_1)
    print('Second H eigenvalues', w_inv_2)
    print('Second H eigenvectors', v_inv_2)
    print('Printing the values for the matrices before inversion')
    w_1, v_1 = np.linalg.eigh(H_inv_1)
    w_2, v_2 = np.linalg.eigh(H_inv_2)
    print('Printing the values for the original matrix')
    print('First H eigenvalues', w_1)
    print('First H eigenvectors', v_1)
    print('Second H eigenvalues', w_2)
    print('Second H eigenvectors', v_2)
    return None


def find_pc_DOP(pointcloud):
    pc_DOP = integrity.calculate_dop(pointcloud)
    print(pc_DOP)
    return pc_DOP


def test_data_loader():
    uiuc_pcs = data_utils.load_uiuc_pcs(0, 10, 1)
    #for pc in uiuc_pcs:
        #print(np.shape(pc))
    kitti_data = data_utils.load_kitti_pcs(0, 10, 10)
    for pc in kitti_data:
        print(np.shape(pc))
    return 0


def c_v_presentation_plot():
    N = 6
    high = 13.816
    low = 0.01
    r = np.linspace(0, 20, 100)
    bar_r = 3 - 6.*((r - low)/(high - low))
    C_v = 1 / (1 + np.exp(-bar_r))
    good_voxels = 5
    bad_voxels = 8.87
    plt.plot(r, C_v)
    plt.xlim([0, 20])
    plt.ylim([0, 1])
    plt.show()
    return None


def ndt_resolution():
    kitti_data = data_utils.load_uiuc_pcs(0, 10, 1)
    pc = kitti_data[0]
    view_pc = pptk.viewer(pc)
    view_pc.set(lookat=[0.0, 0.0, 0.0])
    ndt_1 = ndt.ndt_approx(pc, horiz_grid_size=2.0, vert_grid_size=2.0)
    ndt.display_ndt_cloud(ndt_1, point_density=0.8)
    ndt_2 = ndt.ndt_approx(pc, horiz_grid_size=1.0, vert_grid_size=1.0)
    ndt.display_ndt_cloud(ndt_2, point_density=0.4)
    ndt_3 = ndt.ndt_approx(pc)
    ndt.display_ndt_cloud(ndt_3, point_density=0.2)
    return None




# vox_con()
# paper_total_con()
# total_metric_test()
# transforms_test()
# ndt_test()
# integrity_test()
# test_data_loader()
# new_integrity_example()
# paper_vox_con_2()
# c_v_presentation_plot()
# ndt_resolution()
test_new_ndt()