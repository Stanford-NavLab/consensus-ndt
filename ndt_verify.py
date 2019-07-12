"""
ndt_verify.py
File for the testing of ndt odometry and mapping modules
Author: Ashwin Kanhere
Date Created: 16th June 2019
Date Modified: 16th June, 2019
"""

import numpy as np
import pykitti
import pptk

import data_utils
import ndt
import odometry
import mapping
import utils
import diagnostics
import time

basedir = '/home/kanhere2/ion-gnss-19/test_dataset'
date = '2011_09_26'
drive = '0005'

data = pykitti.raw(basedir, date, drive)

indices = np.array([2, 20, 30])

print('Extracting map odometry')

num_frames = 40

sequence_odometry = data_utils.kitti_sequence_poses(data, num_frames=40)

ndt_odom_save = np.zeros([3, 6])
icp_odom_save = np.zeros([3, 6])
kitti_odom_save = np.zeros([3, 6])
disturbance = np.array([0.25, 0.5, 1.0, 1.5, 2.0])
for i in range(5):
    curr_pc = data.get_velo(0)[:, :3]

translate = np.array([5., 5., 0.5, 10, 10, 30])

for i in range(3):
    for j in range(6):
        idx = indices[i]
        print('Loading PCs')
        prev_pc = data.get_velo(idx)[:, :3]
        curr_pc = data.get_velo(idx+1)[:, :3]
        print('Finding NDT approximation')
        prev_ndt = ndt.ndt_approx(prev_pc, horiz_grid_size=1.0, vert_grid_size=1.0)
        kitti_truth = sequence_odometry[idx + 1, :]
        kitti_pc = utils.transform_pc(kitti_truth, curr_pc)
        prev_ndt.find_integrity(kitti_pc)
        prev_ndt.filter_voxels_integrity(integrity_limit=0.7)
        new_transform = kitti_truth
        new_transform[j] += translate[j]
        translate_pc = utils.transform_pc(new_transform, curr_pc)
        print('The original integrity metric is', prev_ndt.find_integrity(kitti_pc))
        print('The modification was along axis', j, 'giving the new integrity as ', prev_ndt.find_integrity(translate_pc))
print('Task complete boss')


"""
for i in range(3):
    curr_pc = data.get_velo(indices[i])[:, :3]
    curr_ndt = ndt.ndt_approx(curr_pc, horiz_grid_size=1.0, vert_grid_size=1.0)
    pptk.viewer(curr_pc)
    ndt.display_ndt_cloud(curr_ndt)
for i in range(1, 3):
    idx = indices[i]
    print('Loading PCs')
    prev_pc = data.get_velo(idx)[:, :3]
    curr_pc = data.get_velo(idx+1)[:, :3]
    print('Finding NDT approximation')
    prev_ndt = ndt.ndt_approx(prev_pc, horiz_grid_size=1.0, vert_grid_size=1.0)
    print('Finding odometry')
    result_odom_inv = odometry.odometry(prev_ndt, curr_pc)
    #result_odom_inv = utils.invert_odom_transfer(np.array([-0.08868455, 0.01848244, 0.01338164, 0.14718466, -0.14054559, -0.13982452]))
    N1 = prev_pc.shape[0]
    N2 = curr_pc.shape[0]
    if N1 <= N2:
        icp_odom_inv = diagnostics.ind_lidar_odom(curr_pc[:N1, :], prev_pc)
    else:
        icp_odom_inv = diagnostics.ind_lidar_odom(curr_pc, prev_pc[:N2, :])
    kitti_truth = sequence_odometry[idx+1, :]
    result_odom = utils.invert_odom_transfer(result_odom_inv)
    icp_odom = utils.invert_odom_transfer(icp_odom_inv)
    icp_odom_save[i, :] = icp_odom
    ndt_odom_save[i, :] = result_odom
    kitti_odom_save[i, :] = kitti_truth
    print('The result odometry is ', result_odom)
    print('The ICP odometry is ', icp_odom)
    print('The KITTI odom is ', kitti_truth)
    prev_ndt = ndt.ndt_approx(prev_pc, horiz_grid_size=1.0, vert_grid_size=1.0)
    kitti_truth_inv = utils.invert_odom_transfer(kitti_truth)
    kitti_pc = utils.transform_pc(kitti_truth, curr_pc)
    prev_ndt.find_integrity(kitti_pc)
    prev_ndt.filter_voxels_integrity(integrity_limit=0.7)
    print('The consensus of the KITTI truth is ', prev_ndt.find_integrity(kitti_pc))
    #diagnostics.plot_consec_pc(kitti_pc, prev_pc)
    result_pc = utils.transform_pc(result_odom_inv, curr_pc)
    print('The consensus of the result is ', prev_ndt.find_integrity(result_pc))
    #print('The consensus of the result is ', prev_ndt.find_integrity(result_pc))
    #diagnostics.plot_consec_pc(result_pc, prev_pc)
    icp_pc = utils.transform_pc(icp_odom, curr_pc)
    print('The consensus of the ICP is ', prev_ndt.find_integrity(icp_pc))
    #diagnostics.plot_consec_pc(icp_pc, prev_pc)
    print('The consensus of the untransformed PC is', prev_ndt.find_integrity(curr_pc))
    #ndt.display_ndt_cloud(prev_ndt)
"""
"""
test_1 = np.array([-0.14301, -0.24005, -0.00007, -0.51058, -0.48959, -0.42584])
test_pt = np.array([[0, 0, 0]])
blah = utils.transform_pc(test_1, test_pt)
test_2 = utils.invert_odom_transfer(test_1)
data = pykitti.raw(basedir, date, drive)
# , frames=(0, num_frames, 1)
print('Extracting map odometry')
map_odometry = np.zeros([num_frames, 6])
for i in range(num_frames):
    oxts_data = data.oxts[i]
    T_w_imu = oxts_data[1]
    map_odometry[i, :] = utils.affine_to_odometry(T_w_imu)

test_value = utils.odometry_difference(map_odometry[0, :], map_odometry[1, :])
print(test_value)

sequence_odom = utils.kitti_sequence_poses(data)
print('Loading pointcloud data')
test_pc_0 = data.get_velo(0)[:, :3]
test_pc_1 = data.get_velo(1)[:, :3]
#diagnostics.plot_consec_pc(test_pc_0, test_pc_1)
test_pc_2 = data.get_velo(2)[:, :3]
test_pc_3 = data.get_velo(3)[:, :3]
test_pc_10 = data.get_velo(10)[:, :3]
pptk.viewer(test_pc_10)
pptk.viewer(test_pc_1)
"""
"""
pptk.viewer(test_pc_0)
pptk.viewer(test_pc_1)
pptk.viewer(test_pc_2)
pptk.viewer(test_pc_10)
"""
"""
icp_odometry = diagnostics.ind_lidar_odom(test_pc_1[:test_pc_0.shape[0], :], test_pc_0)

icp_odometry_2 = diagnostics.ind_lidar_odom(test_pc_0, test_pc_0)

print('Transforming pointcloud data')
transformed_pc_0 = utils.transform_pc(map_odometry[0, :], test_pc_0)
transformed_pc_1 = utils.transform_pc(map_odometry[1, :], test_pc_1)
# diagnostics.plot_consec_pc(transformed_pc_0, transformed_pc_1)
transformed_pc_2 = utils.transform_pc(map_odometry[2, :], test_pc_2)
# pptk.viewer(test_pc[:, :3])

dummy_odom = np.array([0.34561, 0.01779, 0.00471, -0.00407, 0.27465, 0.84315])
dummy_transform = utils.transform_pc(utils.invert_odom_transfer(dummy_odom), test_pc_1)

# Testing larger odometry solver
"""
"""
print('Approximating pointcloud by NDT')
test_ndt = ndt.ndt_approx(test_pc, horiz_grid_size=1.0, vert_grid_size=1.0)
print('Finding odometry vector')
test_odom = odometry.odometry(test_ndt, test_pc[:10000, :])

print(test_odom)
"""
"""
test_value = utils.odometry_difference(map_odometry[1, :], map_odometry[2, :])
t0 = time.time()
print('Optimizing to find transform between first two point clouds')
ndt_cloud = ndt.ndt_approx(test_pc_2, horiz_grid_size=1.0, vert_grid_size=1.0)
result_odom = odometry.odometry(ndt_cloud, test_pc_3)
#result_odom = np.array([-0.35000, -0.00681, -0.00706, 0.00201, -0.27428, -0.84816])
#icp_odometry = utils.invert_odom_transfer(diagnostics.ind_lidar_odom(test_pc_1[:test_pc_0.shape[0], :], test_pc_0))
icp_odometry = diagnostics.ind_lidar_odom(test_pc_3, test_pc_2[:test_pc_3.shape[0], :])
invert_icp_odometry = utils.invert_odom_transfer(icp_odometry)
# result_odom = np.array([0.34622, 0.01808, 0.00503, -0.00373, 0.27911, 0.84272])
print('KITTI odometry is ', test_value)
invert_result_odom = utils.invert_odom_transfer(result_odom)
#invert_result_odom = result_odom
print('The original result is ', result_odom)
print('The original ICP result is', icp_odometry)
print('Odometry mapping ref_pc to test_pc is ', invert_result_odom)
print('ICP odometry for the same mapping is is ', invert_icp_odometry)
print(time.time() - t0)
result_pc = utils.transform_pc(invert_result_odom, test_pc_3)
icp_pc = utils.transform_pc(invert_icp_odometry, test_pc_3)
kitti_pc = utils.transform_pc(test_value, test_pc_3)
print('Result Error RMS: ', np.mean(np.abs(kitti_pc - result_pc)))
print('ICP Error RMS: ', np.mean(np.abs(kitti_pc - icp_pc)))
diagnostics.plot_consec_pc(kitti_pc, result_pc)
diagnostics.plot_consec_pc(kitti_pc, icp_pc)
"""
"""
print('NDT Approximation 1')
ndt_cloud = ndt.ndt_approx(test_pc_0, horiz_grid_size=1.0, vert_grid_size=1.0)
t0 = time.time()
print(ndt_cloud.find_likelihood(test_pc_1))
print(ndt_cloud.find_likelihood(test_pc_1[:1, :]))
print(time.time() - t0)
result_odom = np.array([0.34622535, 0.01837177,  0.00505789, -0.00360152,  0.27945714,  0.84356299])
inverse_result = utils.invert_odom_transfer(result_odom)
result_pc = utils.transform_pc(inverse_result, test_pc_1)
print(ndt_cloud.find_integrity(result_pc))
ndt.display_ndt_cloud(ndt_cloud)
diagnostics.plot_consec_pc(test_pc_0, result_pc)
print('NDT Approximation 2')
ndt_cloud = ndt.ndt_approx(test_pc_0, horiz_grid_size=1.0, vert_grid_size=1.0)
kitti_pc = utils.transform_pc(test_value, test_pc_1)
print(ndt_cloud.find_integrity(kitti_pc))
ndt.display_ndt_cloud(ndt_cloud)
print('This should be done')
transform_diff = np.linalg.norm(utils.transform_pc(inverse_result, test_pc_1) - utils.transform_pc(test_value, test_pc_1), axis=1)
filtered_pc = test_pc_1[transform_diff>0.5, :]
pptk.viewer(filtered_pc)
print('You shall not pass!')
#plot_points, _ = ndt_cloud.display(plot_density=0.2)
#diagnostics.plot_consec_pc(plot_points, test_pc_0)
#diagnostics.objective_variation(ndt_cloud, test_pc_1, axis=1)
#diagnostics.objective_variation(ndt_cloud, test_pc_1, axis=2)
# odometry.search_initial(ndt_cloud, test_pc_1)
#diagnostics.check_gradient(odometry.objective, odometry.jacobian_vect, ndt_cloud, test_pc_1, utils.invert_odom_transfer(test_value))
"""
"""
print('NDT Approximation 3')
ndt_cloud = ndt.ndt_approx(test_pc_0, horiz_grid_size=0.5, vert_grid_size=0.5)
result_odom = np.array([0.34622535, 0.01837177,  0.00505789, -0.00360152,  0.27945714,  0.84356299])
result_pc = utils.transform_pc(utils.invert_odom_transfer(result_odom), test_pc_1)
ndt_cloud.find_integrity(result_pc)
ndt.display_ndt_cloud(ndt_cloud)
print('NDT Approximation 4')
ndt_cloud = ndt.ndt_approx(test_pc_0, horiz_grid_size=0.5, vert_grid_size=0.5)
kitti_pc = utils.transform_pc(test_value, test_pc_1)
ndt_cloud.find_integrity(kitti_pc)
ndt.display_ndt_cloud(ndt_cloud)
print('This should be done')
"""
"""
print('Should be done now')

# Testing odometry solver for consecutive point clouds
# print('Approximating first point cloud into NDT')
# dummy_odom = np.array([0.18102, -0.03025, -0.00115, 0.02562, 0.18534, 0.07479])
# t0 = time.time()
# test_ndt = ndt.ndt_approx(test_pc_0, horiz_grid_size=1.0, vert_grid_size=1.0)
# print('NDT Approximation Time Elapsed: ', time.time() - t0)
# t0 = time.time()
# binned_points = test_ndt.old_bin_in_voxels(test_pc_1)
# print('Old Point Binning Time Elapsed: ', time.time() - t0)
# t0 = time.time()
# new_binned_points = test_ndt.bin_in_voxels(test_pc_1)
# print('New Point Binning Time Elapsed: ', time.time() - t0)
# t0 = time.time()
# print(odometry.objective(np.zeros(6), test_ndt, test_pc_1))
# print('Time Elapsed: ', time.time() - t0)
# t0 = time.time()
# print(odometry.jacobian_vect(np.zeros(6), test_ndt, test_pc_1))
# print('Jacobian Time Elasped: ', time.time() - t0)
# t0 = time.time()
# odometry.hessian_vect(np.zeros(6), test_ndt, test_pc_1)
# print('Hessian TIme Elapsed: ', time.time() - t0)

# print(odometry.objective(utils.invert_odom_transfer(test_value), test_ndt, test_pc_1))
# print(odometry.objective(dummy_odom, test_ndt, test_pc_1))
# print(test_ndt.find_integrity(test_pc_0))
# print(test_ndt.find_integrity(test_pc_1))
# ntegrity_trans_pc_1 = utils.transform_pc(utils.invert_odom_transfer(test_value), test_pc_1)
# integrity_trans_pc_2 = utils.transform_pc(dummy_odom, test_pc_1)
# print('KITTI says', test_ndt.find_integrity(integrity_trans_pc_1))
# print('NDT says', test_ndt.find_integrity(integrity_trans_pc_2))
print('Transforming the original point cloud by a typical odometry (typical to KITTI)')
# dummy_odom = np.array([0.15, 0.25, 0.001, 0.5, 0.5, 1.5])
# dummy_transform_pc = utils.transform_pc(dummy_odom, test_pc_0)
"""
"""
print('Calculating odometry between first point cloud and its NDT')
test_odom = odometry.odometry(test_ndt, test_pc_1)
print(test_odom)
print(utils.invert_odom_transfer(test_odom))
print(blah)
"""
"""
# Testing mapping functions

# test_ndt_0 = ndt.ndt_approx(test_pc_0, horiz_grid_size=1.0, vert_grid_size=1.0)
# test_ndt_1 = ndt.ndt_approx(test_pc_1, horiz_grid_size=1.0, vert_grid_size=1.0)
# test_ndt_2 = ndt.ndt_approx(test_pc_2, horiz_grid_size=1.0, vert_grid_size=1.0)



# sim_12 = mapping.pc_similarity()
# TODO: Test similarity function after obtaining KITTI ground truth



diagnostics.objective_variation(test_ndt, test_pc_0, limit=0.5)

check_initial_odom = np.array([5, 5, 0, 0, 0, 45])
transformed_pc_0_test = utils.transform_pc(check_initial_odom, test_pc_0)
pptk.viewer(transformed_pc_0_test)
pptk.viewer(test_pc_0)
diagnostics.check_gradient(odometry.objective, odometry.jacobian_vect, test_ndt, test_pc_0, check_initial_odom)

"""
