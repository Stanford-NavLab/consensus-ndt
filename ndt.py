"""
ndt.py
File containing functions for NDT-based point cloud function approximation functions
Author: Ashwin Kanhere
Date created: 15ht April 2019
Last modified: 19th April 2019
"""
import numpy as np
"""
Importing base libraries
"""


class NDTCloud:
    """
    A class to store the sparse grid center points, means and covariances for grid points that are full.
    This class will be the de facto default for working with NDT point clouds
    """
    # TODO: Finish class definition and change other functions to work with this class
    def __init__(self, xlim, ylim, zlim, input_horiz_grid_size, input_vert_grid_size):
        """
        A method to initialize a member of the NDTCloud class. When initializing a member of the class, grid limits are
        given along with the grid sizes. Using these values a sparse grid is created and corresponding zero mean and
        covariance lists are also created.
        Since the first grid is highly dependent on the user it is for, there is no default initialization
        :param xlim: Limit of the grid along the x-axis
        :param ylim: Limit of the grid along the y-axis
        :param zlim: Limit of the grid along the z-axis
        :param input_horiz_grid_size: User entered
        :param input_vert_grid_size:
        """
        # Don't really need to store the limits of the space spanned by the NDT cloud. They will be needed to find if
        # the origin is a grid center though
        # When initializing the cloud, the origin is either going to be a grid center or not.
        self.horiz_grid_size = np.float(input_horiz_grid_size)
        self.vert_grid_size = np.float(input_vert_grid_size)
        first_center_x = np.mod(2*xlim/self.horiz_grid_size + 1, 2)*self.horiz_grid_size/2.0
        first_center_y = np.mod(2*ylim/self.horiz_grid_size + 1, 2)*self.horiz_grid_size/2.0
        first_center_z = np.mod(2*zlim/self.vert_grid_size + 1, 2)*self.vert_grid_size/2.0
        self.first_center = np.array([first_center_x, first_center_y, first_center_z])
        # Create NDT map for reference grid
        # Initialize empty lists to store means and covariance matrices
        self.stats = {}  # Create an empty dictionary for mu and sigma corresponding to each voxel
        """
        Dictionary structure is {<key = center point>, {<key = 'mu'>, [mu value], <key='sigma'>, [sigma_value]
        , <key='no_points'>, int}, ...}
        NOTE: key must be a tuple not a ndarray
        """
        self.local_to_global = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        # Store an estimate of the transformation of the current scan origin to global origin

    def update_displacement(self, affine_transform):
        """
        A function to update the displacement of the current local frame of reference from the global reference
        :param affine_transform: A vector of [x, y, z, phi, theta, psi] measuring the  affine transformation of the
        current local frame of reference (LiDAR origin) to the global frame of reference (map origin)
        :return: None
        """
        # TODO: Convert this so that coordinate transformations are used to update the frame of reference
        self.local_to_global = np.reshape(affine_transform, [-1, 6])

    def find_voxel_center(self, ref_pointcloud):
        """
        A function to return grid indices for a given set of 3D points. The input may be a set of (x, y, z) Nx3 or
        (x, y, z, int) Nx4. This function is written to be agnostic to either form of the array
        :param ref_pointcloud: Nx3 or Nx4 numpy array for which binning is required
        :return: grid_centers: Matrix containing center coordinates corresponding to the given points Nx3
        """
        # Used an array over a tuple as there is a small possibility that the coordinates might change
        ref_points = np.array(ref_pointcloud[:, :3])  # to remove intensity if it has been passed accidentally
        # int(ref_x/horizontal grid size)* horizontal grid size + first grid center (same sign that the quotient is)
        # first_center is the offset from the origin of the first grid center (in either direction)
        grid_size = np.array([self.horiz_grid_size, self.horiz_grid_size, self.vert_grid_size])
        pre_voxel_number = ref_points/grid_size
        pre_voxel_center = pre_voxel_number.astype(int)*grid_size
        no_points = ref_pointcloud.shape[0]
        voxel_centers = np.multiply(np.sign(ref_points), np.abs(pre_voxel_center) + np.broadcast_to(
            self.first_center, (no_points, 3)))
        return voxel_centers

    def find_likelihood(self, transformed_pc):
        """
        Function to return likelihood for a given transformed point cloud w.r.t NDT point cloud
        :param transformed_pc: Point cloud that has been passed through a candidate affine transformation
        :return: likelihood: Scalar value representing the likelihood of the given
        """
        likelihood = 0
        return likelihood

    def display(self):
        """
        Function to display the single NDT approximation
        :return: None
        """
        display_ndt_cloud(self)
        return None

    def update_stats(self, points_in_voxels):
        """
        Function to update the statistics of the NDT cloud given points and the center of the grid that they belong to
        :param points_in_voxels: A dictionary indexed by the center of the grid and containing corresponding values
        :return: None
        """
        for k, v in points_in_voxels.items():
            no_in_voxel = v.size/3  # to prevent a single row vector from being counted as 3
            if k in self.stats:
                # Use update methodology from 3D NDT Scan Matching Paper Eq 4 and 5
                m_old = self.stats[k]['no_points']*self.stats[k]['mu']  # row vector
                s_old = self.stats[k]['no_points']*self.stats[k]['sigma'] + np.matmul(self.stats[k]['mu'].T, m_old)
                m_new = m_old + np.sum(v, axis=0)
                s_new = s_old + np.matmul(v.T, v)
                self.stats[k]['no_points'] += no_in_voxel
                self.stats[k]['mu']= m_new/self.stats[k]['no_points']
                self.stats[k]['sigma'] = (s_new - np.matmul(self.stats[k]['mu'].T, m_new))/self.stats[k]['no_points']
            else:
                if no_in_voxel >= 4:
                    test1 = np.mean(v, axis=0)
                    test2 = np.cov(v, rowvar=False)
                    test3 = np.linalg.det(test2)
                    self.stats[k] = {}  # Initialize empty dictionary before populating with values
                    self.stats[k]['mu'] = np.mean(v, axis=0)
                    self.stats[k]['sigma'] = np.cov(v, rowvar=False)
                    self.stats[k]['no_points'] = no_in_voxel
        return None

    def eig_check(self):
        """
        Function to perform an eigenvalue based consistency check on the covariance matrix and adjust values accordingly
        :return: None
        """
        # TODO: Write eig_check fucntion
        return None

    def update_cloud(self, pc_points):
        """
        Function to add points to current NDT approximation. This function adds both, new centers and points to
        existing grid points.
        :param pc_points: The points that are to be added to the NDT approximation. Might be Nx3 or Nx4.
        Function agnostic to that
        :return: None
        """
        # This function should be used to update an empty NDT cloud as well using the given points
        # Find grid centers corresponding to given points
        update_points = pc_points[:, :3]
        no_points = update_points.shape[0]
        voxel_centers = self.find_voxel_center(update_points)
        # Dictionary approach here as well
        points_in_voxels = {}
        for i in range(no_points):
            voxel_key = tuple(voxel_centers[i, :])
            if voxel_key in points_in_voxels:
                points_in_voxels[voxel_key] = np.vstack((points_in_voxels[voxel_key], update_points[i, :]))
            else:
                points_in_voxels[voxel_key] = update_points[i, :]
        # Update the NDT approximation with these binned points
        self.update_stats(points_in_voxels)
        self.eig_check()
        return None


def ndt_approx(ref_pointcloud, horiz_grid_size=0.5, vert_grid_size=0.5, offset_axis=np.array([0, 0, 0])):
    """
    Function to create single NDT approximation for given offset and point cloud
    :param ref_pointcloud: [x, y, z, int] Nx4 numpy array of the reference point cloud
    :param horiz_grid_size: Float describing required horizontal grid size (in m)
    :param vert_grid_size: Float describing required vertical grid size. LiDAR span will be significantly shorter ...
    ... vertically with different concentrations, hence the different sizes. Same as horiz_grid_size by default
    :param offset_axis: 1x3 np array containing booleans for which axis to offset on
    :return: ndt_cloud: NDT approximated cloud for the given point cloud and grid size
    """
    ref_pointcloud = ref_pointcloud.reshape([-1, 4])
    # Extract the size of the grid
    xlim = np.ceil(np.max(np.absolute(ref_pointcloud[:, 1]))) + 2*horiz_grid_size + 0.5*horiz_grid_size*offset_axis[0]
    ylim = np.ceil(np.max(np.absolute(ref_pointcloud[:, 1]))) + 2*horiz_grid_size + 0.5*horiz_grid_size*offset_axis[1]
    zlim = np.ceil(np.max(np.absolute(ref_pointcloud[:, 1]))) + 2*vert_grid_size + 0.5*vert_grid_size*offset_axis[2]

    # Create NDT map for reference grid

    ndt_cloud = NDTCloud(xlim, ylim, zlim, input_horiz_grid_size=horiz_grid_size, input_vert_grid_size=vert_grid_size)
    ref_pointcloud_test = ref_pointcloud[:10, :]
    ndt_cloud.update_cloud(ref_pointcloud_test)
    ref_pointcloud_test2 = ref_pointcloud[10:14, :]
    ndt_cloud.update_cloud(ref_pointcloud_test2)
    print('Huzzah')

    """
    OLD BINNING CODE
    # Find which points lie in a particular grid
    # TODO: Convert binning to a function
    for i in range(2):  # range(ref_pointcloud.shape[1]):
        print(i)
        x_index = np.int(np.floor(ref_pointcloud[i, 0]/horiz_grid_size) + 0.5*xrange.size)
        # 0.5*xrange.size added to account for -ve coordinates and their impact on indexing
        y_index = np.int(np.floor(ref_pointcloud[i, 1]/horiz_grid_size) + 0.5*yrange.size)
        z_index = np.int(np.floor(ref_pointcloud[i, 1]/vert_grid_size) + 0.5*zrange.size)
        grid_number = x_index + y_index*xgrid.shape[0] + z_index*xgrid.shape[0]*xgrid.shape[1]
        if not points_in_grid:
            points_in_grid.append([[grid_number], [ref_pointcloud[i, :3]]])
        else:
            # Didn't use a list comprehension in what follows because two different lists were needed
            list_indices = []
            grid_indices = []
            for index, item in enumerate(points_in_grid[:]):
                list_indices.append(index)  # Will result in a list of numbers based on indices of points_in_grid
                grid_indices.append(item[0])  # Will result in a list of lists for grid points that have points
            if [grid_number] in grid_indices:
                list_index = list_indices[grid_indices.index([grid_number])]
                points_in_grid[list_index].append(ref_pointcloud[i, :3])
            else:
                points_in_grid.append([[grid_number], [ref_pointcloud[i, :3]]])
                """
    """
    OLD CODE FOR FINDING MEAN AND COVARIANCE
    # Approximate corresponding points into a gaussian for the grid
    for i in range(len(points_in_grid)):
        grid_number = points_in_grid[i][0][0]
        z_index = np.int(grid_number/(xgrid.shape[0]*xgrid.shape[1]))
        y_index = np.int(np.mod(grid_number, xgrid.shape[0]*xgrid.shape[1])/xgrid.shape[0])
        x_index = np.mod(grid_number, xgrid.shape[0])
        points_in_grid_array = np.array([np.array(point) for point in points_in_grid[i][1]])
        if points_in_grid_array.shape[0] >= 4:
            # At least 4 points are needed for 3D NDT to make sense 1 - point, 2- line, 3 - plane, min 4 for a volume
            grid_mu[x_index, y_index, z_index, :] = np.mean(points_in_grid_array, axis=1)
            grid_sigma[x_index, y_index, z_index, :, :] = np.cov(points_in_grid_array.T)
    # Use grid_mu and grid_sigma to create/define ndt_cloud object
    print(np.min(grid_mu))
    print(np.max(grid_mu))
    """

    return ndt_cloud


def pc_to_ndt(ref_pointcloud, horiz_grid_size=1, vert_grid_size=1):
    """
    Function to convert given point cloud into a NDT based approximation
    :param ref_pointcloud: Point cloud that needs to be converted to NDT reference
    :param horiz_grid_size: Parameter for horizontal grid sizing
    :param vert_grid_size: Parameter for vertical grid sizing
    :return: pc_ndt_approx: An object containing a collection (8) of NDT clouds that makes up the total reference ...
    ... for the given point cloud
    """
    pc_ndt_approx = []  # Initializing the cloud object
    for i in range(1):  # range(8):
        # TODO: Check effects of offset and whether it is working as it is supposed to
        offset = np.array([np.mod(i, 2), np.mod(np.int(i/2), 2), np.int(i/4)])
        pc_ndt_approx.append(ndt_approx(ref_pointcloud, horiz_grid_size, vert_grid_size, offset_axis=offset))
    pc_ndt_approx = []
    return pc_ndt_approx


def display_ndt_cloud(pc_ndt_approx):
    """
    Function to display average NDT clouds from a collection of NDT clouds
    :param pc_ndt_approx: Collection of 8 NDT point clouds (representing the offset NDT approximations)
    :return: None
    """
    return None



