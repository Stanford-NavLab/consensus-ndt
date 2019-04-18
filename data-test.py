import numpy as np
import pykitti
import pptk

basedir = 'D:\\Users\\kanhe\\Box Sync\\RA Work\\ION GNSS 19\\Implementation\\Dataset'
date = '2011_09_26'
drive = '0005'

data = pykitti.raw(basedir, date, drive, frames= range(0, 5, 1))

points_lidar = data.velo
test_lidar = data.get_velo(0) # LiDAR point cloud is a Nx4 numpy array
print(np.shape(test_lidar))
test_lidar = test_lidar[:,:3]
print(np.shape(points_lidar))
display_lidar = test_lidar.reshape((-1, 3))
pptk.viewer(display_lidar)
print('Test Line')
