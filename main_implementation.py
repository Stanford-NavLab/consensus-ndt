"""
main_implementation.py
File for the offline implementation of NDT-SLAM on the Talbot 330 resident powerhouse machine
Author: Ashwin Kanhere
Date Created: 16th June 2019
Date Modified: 16th June, 2019
"""
import numpy as np
import pptk
import ndt
import odometry
import diagnostics
import time
import utils
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import data_utils
import lidar_mods


def main():

    run_mode = 'server'
    total_iters = 30
    iter1 = 15
    iter2 = 15

    print('Loading dataset')
    pcs = data_utils.load_uiuc_pcs(0, 10, mode=run_mode)

    assert(total_iters == iter1 + iter2)

    integrity_filters = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    ref_lidar = pcs[0]
    ref_ndt = ndt.ndt_approx(ref_lidar)
    perturb = np.array([0.5, 0.5, 0., 0., 0., 0.])
    trans_pc = utils.transform_pc(perturb, ref_lidar)
    ground_truth = utils.invert_odom_transfer(perturb)
    error_consensus = np.zeros([np.size(integrity_filters), 1])
    time_consensus = np.zeros_like(error_consensus)
    for idx, filter in enumerate(integrity_filters):
        print('Running case ', idx)
        tic = time.time()
        test_odom = odometry.odometry(ref_ndt, trans_pc, max_iter_pre=iter1, max_iter_post=iter2,
                                      integrity_filter=filter)
        toc = time.time()
        error_consensus[idx] = np.linalg.norm(ground_truth - test_odom)
        time_consensus[idx] = toc - tic
    print('Running baseline case')
    tic = time.time()
    new_odom = odometry.odometry(ref_ndt, trans_pc, max_iter_pre=total_iters, max_iter_post=0)
    toc = time.time()
    error_vanilla = np.linalg.norm(ground_truth - new_odom)
    time_vanilla = toc - tic
    # Save error and time values
    print('The vanilla run error is ', error_vanilla)
    print('The vanilla time taken is', time_vanilla)

    print('The consensus run errors are ', error_consensus)
    print('The consensus run times are ', time_consensus)
    print('The integrity filter values are ', integrity_filters)



    return 0


main()
