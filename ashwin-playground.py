"""
ashwin-playground.py
File to test different modules and functions created for the offline implementation of NDT-SLAM
Author: Ashwin Kanhere
Date Modified: 15th April 2019
"""
import numpy as np
import pykitti
import pptk
import ndt
import transforms3d
import integrity
import time


def extract_data():
    basedir = 'D:\\Users\\kanhe\\Box Sync\\RA Work\\ION GNSS 19\\Implementation\\Dataset'
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
                #print('Case 2: i:', i, ' j:', j, ' k:', k, ' Value:', mag_R_2)
    return None


#transforms_test()
ndt_test()
#integrity_test()