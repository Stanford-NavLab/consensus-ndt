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

basedir = 'D:\\Users\\kanhe\\Box Sync\\RA Work\\ION GNSS 19\\Implementation\\Dataset'
date = '2011_09_26'
drive = '0005'

data = pykitti.raw(basedir, date, drive, frames=range(0, 5, 1))

points_lidar = data.velo
test_lidar = data.get_velo(0) # LiDAR point cloud is a Nx4 numpy array
#test_cloud = ndt_approx.ndt_approx(test_lidar)
testing = ndt.pc_to_ndt(test_lidar)