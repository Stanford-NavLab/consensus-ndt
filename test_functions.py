"""
test_functions.py
File containing all functions that test code functionality
Author: Ashwin Kanhere
Date created: 13th November 2019
Last modified: 13th Novermber 2019
"""
import pykitti
import numpy as np
import transforms3d
import ndt

"""
Helper functions for tests
"""


def extract_data():
    basedir = 'D:\\Users\\kanhe\\Box Sync\\Research Projects\\Consensus NDT SLAM\\Dataset'
    date = '2011_09_26'
    drive = '0005'

    data = pykitti.raw(basedir, date, drive, frames=range(0, 5, 1))

    points_lidar = data.velo
    return data

"""
Actually test functions
"""


def transforms_test():
    angles = [0, np.pi/4, np.pi/3]
    for i in range(3):
        for j in range(3):
            for k in range(3):
                R = transforms3d.euler.euler2mat(angles[i], angles[j], angles[k], 'rxyz')
                c1 = np.cos(angles[i])
                s1 = np.sin(angles[i])
                c2 = np.cos(angles[j])
                s2 = np.sin(angles[j])
                c3 = np.cos(angles[k])
                s3 = np.sin(angles[k])
                test_R = np.array([[c2*c3, -c2*s3, s2], [c1*s3 + c3*s1*s2, c1*c3 - s1*s2*s3, -c2*s1],
                                   [s1*s3 - c1*c3*s2, c3*s1 + c1*s2*s3, c1*c2]])
                error_R = R - test_R
                mag_R = np.max(np.abs(error_R))
                print('Case 1: i:', i, ' j:', j, ' k:', k, ' Value:', mag_R)
                c1 = np.cos(angles[k])
                s1 = np.sin(angles[k])
                c2 = np.cos(angles[j])
                s2 = np.sin(angles[j])
                c3 = np.cos(angles[i])
                s3 = np.sin(angles[i])
                test_R_2 = np.array([[c1*c2, c1*s2*s3 - c3*s1, s1*s3 + c1*c3*s2],
                                     [c2*s1, c1*c3+s1*s2*s3, c3*s1*s2 - c1*s3],[-s2, c2*s3, c2*c3]])
                error_R_2 = R - test_R_2
                mag_R_2 = np.max(np.abs(error_R_2))
                # print('Case 2: i:', i, ' j:', j, ' k:', k, ' Value:', mag_R_2)
    return None


def test_new_ndt():
    xlim = 10
    ylim = 10
    zlim = 5
    input_horiz_grid_size = 2
    input_vert_grid_size = 2
    # test_cloud1 = ndt.NDTCloudBase(xlim, ylim, zlim, input_horiz_grid_size, input_vert_grid_size)
    # test_cloud2 = ndt.NDTCloudNoOverLap(xlim, ylim, zlim, input_horiz_grid_size, input_vert_grid_size)
    # test_cloud3 = ndt.NDTCloudOverLap(xlim, ylim, zlim, input_horiz_grid_size, input_vert_grid_size)
    # test_cloud4 = ndt.NDTCloudInterpolated(xlim, ylim, zlim, input_horiz_grid_size, input_vert_grid_size)
    data = extract_data()
    test_lidar = data.get_velo(0) # LiDAR point cloud is a Nx4 numpy array
    test_lidar = test_lidar[:, :3]
    # test_cloud = ndt_approx.ndt_approx(test_lidar, horiz_grid_size=0.25, vert_grid_size=0.25)
    name = ['overlapping', 'interpolate', 'nooverlap', 'overlapping']
    for i in range(len(name)):
        testing = ndt.ndt_approx(test_lidar, horiz_grid_size=1, vert_grid_size=1, type=name[i])
        print('Creating NDT approximation for ', name[i])
        print('The likelihood for NDT approximation of type ', name[i], 'is ', testing.find_likelihood(test_lidar))
    return None


def test_neighbour_finding():
    data = extract_data()
    test_lidar = data.get_velo(0) # LiDAR point cloud is a Nx4 numpy array
    test_lidar = test_lidar[:, :3]
    test_approximation = ndt.ndt_approx(test_lidar, horiz_grid_size=1, vert_grid_size=1, type='interpolate')
    test_likelihood = test_approximation.find_likelihood(test_lidar[:10, :])
    return None
