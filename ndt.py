"""
ndt.py
File containing class definitions of NDT approximation for Consensus NDT SLAM
Also contains helper NDT functions
Author: Ashwin Kanhere
Date created: 15th April 2019
Last modified: 13th November 2019
"""
import numpy as np
import pptk
import utils
import transforms3d
from scipy.optimize import check_grad
from scipy.optimize import minimize
import odometry
import diagnostics
import integrity
import numpy_indexed
import itertools
import time
from scipy.interpolate import RegularGridInterpolator as RGI

"""
Importing base libraries
"""


class NDTCloudBase:
    """
    A class to store the sparse grid center points, means and covariances for grid points that are full.
    This class will be the de facto default for working with NDT point clouds.
    After refactoring for multiscale NDT with different methods, has become parent class for all NDT approximation methods
    """
    def __init__(self, xlim, ylim, zlim, input_horiz_grid_size, input_vert_grid_size, cloud_type):
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
        # Create NDT map for reference grid
        # Initialize empty lists to store means and covariance matrices
        self.stats = {}  # Create an empty dictionary for mu and sigma corresponding to each voxel
        """
        Dictionary structure is {<key = center point>, {<key = 'mu'>, [mu value], <key='sigma'>, [sigma_value]
        , <key='no_points'>, int, <key='integrity'>, float}, ...}
        NOTE: key must be a tuple not a ndarray
        """
        self.local_to_global = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.max_no_points = 0
        self.first_center = np.empty([1, 3])
        self.max_no_voxels = -1
        self.cloud_type = cloud_type

    def update_displacement(self, odometry_vector):
        """
        A function to update the displacement of the current local frame of reference from the global reference
        :param odometry_vector: A vector of [x, y, z, phi, theta, psi] measuring the  affine transformation of the
        current local frame of reference (LiDAR origin) to the global frame of reference (map origin)
        :return: None
        """
        # TODO: 
        # Update translation vector
        self.local_to_global[:3] += odometry_vector[:3]
        # Update euler angle vector
        phi_local = np.deg2rad(self.local_to_global[3])
        theta_local = np.deg2rad(self.local_to_global[4])
        psi_local = np.deg2rad(self.local_to_global[5])
        R_local = transforms3d.euler.euler2mat(phi_local, theta_local, psi_local)
        phi_delta = np.deg2rad(odometry_vector[3])
        theta_delta = np.deg2rad(odometry_vector[4])
        psi_delta = np.deg2rad(odometry_vector[5])
        R_delta = transforms3d.euler.euler2mat(phi_delta, theta_delta, psi_delta, 'rxyz')
        R_new = np.matmul(R_delta, R_local)
        phi_rad, theta_rad, psi_rad = transforms3d.euler.mat2euler(R_new, 'rxyz')
        angle_new = np.rad2deg(np.array([phi_rad, theta_rad, psi_rad]))
        self.local_to_global[3:] = angle_new
        return None

    def find_voxel_center(self, ref_pointcloud, tol=1.0e-7):
        """
        A function to return grid indices for a given set of 3D points. The input may be a set of (x, y, z) Nx3 or
        (x, y, z, int) Nx4. This function is written to be agnostic to either form of the array
        This function also checks if points on a edge of the grid upto a tolerance level. If they are, it assigns them
        a value to ensure that no calculations involve that point
        :param ref_pointcloud: Nx3 or Nx4 numpy array for which binning is required
        :param tol: Tolerance for picking center. Default values used for overlapping
        :return: grid_centers: Matrix containing center coordinates corresponding to the given points Nx3
        """
        # Used an array over a tuple as there is a small possibility that the coordinates might change
        ref_points = np.array(ref_pointcloud[:, :3])  # to remove intensity if it has been passed accidentally
        grid_size = np.array([self.horiz_grid_size, self.horiz_grid_size, self.vert_grid_size])
        number_row = np.shape(self.first_center)[0]
        points_repeated = np.tile(ref_points, (number_row, 1))
        N = ref_points.shape[0]
        voxel_centers = np.zeros_like(points_repeated)
        for i in range(number_row):
            pre_voxel_number = (ref_points + self.first_center[i, :]) / grid_size
            pre_voxel_center = np.round(pre_voxel_number).astype(int) * grid_size
            first_grid_edge = self.first_center[i, :] - 0.5*np.array([self.horiz_grid_size, self.horiz_grid_size,
                                                                      self.vert_grid_size])
            line_check = np.abs(np.mod(ref_points, grid_size) + first_grid_edge)
            pre_voxel_center[line_check < tol] = np.nan
            pre_voxel_center[np.abs(line_check - 1) < tol] = np.nan
            voxel_centers[i*N:(i+1)*N, :] = np.multiply(np.sign(ref_points), np.abs(pre_voxel_center) -
                                                        np.sign(ref_points)*np.broadcast_to(self.first_center[i, :],
                                                                                            (N, 3)))
        return points_repeated, voxel_centers

    def bin_in_voxels(self, points_to_bin):
        """
        Function to bin given points into voxels in a dictionary approach
        :param points_to_bin: The points that are to be binned into the voxel clusters indexed by the voxel center tuple
        :return: points_in_voxel: A dictionary indexed by the tuple of the center of the bin
        """
        points_repeated, voxel_centers = self.find_voxel_center(points_to_bin)
        dummy = numpy_indexed.group_by(voxel_centers, points_repeated)
        points_in_voxels = {}
        for i in range(np.shape(dummy[0])[0]):
            voxel_key = tuple(dummy[0][i])
            points_in_voxels[voxel_key] = dummy[1][i]
        return points_in_voxels

    def find_likelihood(self, transformed_pc):
        """
        Function to return likelihood for a given transformed point cloud w.r.t NDT point cloud
        Slightly different from reference papers in that 1/2det(sigma) is also included while calculating the likelihood
        The likelihood is increased if a corresponding Gaussian is found. If not, 0 is added
        :param transformed_pc: Point cloud that has been passed through a candidate affine transformation
        :return: likelihood: Scalar value representing the likelihood of the given
        """
        transformed_xyz = transformed_pc[:, :3]
        likelihood = 0
        points_in_voxels = self.bin_in_voxels(transformed_xyz)
        for key, val in points_in_voxels.items():
            if key in self.stats:
                sigma = self.stats[key]['sigma']
                sigma_inv = np.linalg.inv(sigma)
                diff = np.atleast_2d(val - self.stats[key]['mu']) 
                likelihood += np.sum(np.exp(-0.5 * np.diag(np.matmul(np.matmul(diff, sigma_inv), diff.T))))
        return likelihood

    def display(self, plot_density=1.0):
        """
        Function to display the single NDT approximation
        :param fig: The figure object on which the probability function has to be plotted
        :param plot_density: The density of the plot (as a int scalar) the higher the density, the more points per grid
        :return: plot_points: The points sampled from the distribution that are to be plotted like any other PC
        """
        base_num_pts = 48  # 3 points per vertical and 4 per horizontal
        plot_points = np.empty([3, 0])
        plot_integrity = np.empty(0)
        for key, value in self.stats.items():
            sigma = self.stats[key]['sigma']
            mu = self.stats[key]['mu']
            measure_num = self.stats[key]['no_points']
            num_pts = np.int(3 * measure_num / self.max_no_points * plot_density * base_num_pts )
            if num_pts < 2:
                num_pts = 2
            if 'integrity' in self.stats[key]:
                voxel_score = self.stats[key]['integrity'] * np.ones(num_pts)
            else:
                voxel_score = np.ones(num_pts)
            center_pt = np.array(key)
            grid_lim = np.zeros([2, 3])
            grid_lim[0, 0] = center_pt[0] - self.horiz_grid_size
            grid_lim[1, 0] = center_pt[0] + self.horiz_grid_size
            grid_lim[0, 1] = center_pt[1] - self.horiz_grid_size
            grid_lim[1, 1] = center_pt[1] + self.horiz_grid_size
            grid_lim[0, 2] = center_pt[2] - self.vert_grid_size
            grid_lim[1, 2] = center_pt[2] + self.vert_grid_size
            grid_plot_points = np.random.multivariate_normal(mu, sigma, num_pts)
            # Ensure that all selected points are inside the grid
            for i in range(3):
                grid_plot_points[grid_plot_points[:, i] < grid_lim[0, i], i] = grid_lim[0, i]
                grid_plot_points[grid_plot_points[:, i] > grid_lim[1, i], i] = grid_lim[1, i]
            plot_points = np.hstack((plot_points, grid_plot_points.T))
            plot_integrity = np.append(plot_integrity, voxel_score)
        print('The maximum number of points per voxel is ', self.max_no_points)
        return plot_points.T, plot_integrity

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
                s_old = self.stats[k]['no_points']*self.stats[k]['sigma'] + \
                        np.matmul(np.reshape(self.stats[k]['mu'], [3, 1]), np.reshape(m_old, [1, 3]))
                m_new = m_old + np.sum(v, axis=0)
                s_new = s_old + np.matmul(v.T, v)
                self.stats[k]['no_points'] += no_in_voxel
                self.stats[k]['mu'] = m_new/self.stats[k]['no_points']
                self.stats[k]['sigma'] = (s_new - np.matmul(np.reshape(self.stats[k]['mu'], [3, 1]),
                                                            np.reshape(m_new, [1, 3])))/self.stats[k]['no_points']
                if self.stats[k]['no_points'] > self.max_no_points:
                    self.max_no_points = self.stats[k]['no_points']
            else:
                if no_in_voxel >= 5 and np.sum(np.isnan(np.array(k))) == 0:
                    self.stats[k] = {}  # Initialize empty dictionary before populating with values
                    self.stats[k]['mu'] = np.mean(v, axis=0)
                    self.stats[k]['sigma'] = np.cov(v, rowvar=False)
                    self.stats[k]['no_points'] = no_in_voxel
                    self.stats[k]['idx'] = self.pairing_cent2int(np.atleast_2d(np.array(k)))
                    self.max_no_voxels += 1
                    if self.stats[k]['no_points'] > self.max_no_points:
                        self.max_no_points = self.stats[k]['no_points']
        return None

    def eig_check(self):
        """
        Function to perform an eigenvalue based consistency check on the covariance matrix and adjust values accordingly
        Algorithm based on 3d NDT Scan Matching and Biber's NDT paper
        Using an SVD approach here. For covariance matrices, SVD and eigen decomposition should be the same. SVD
        implementations are often more stable
        :return: None
        """
        scale_param = 0.0001
        for key, val in self.stats.items():
            u, s_diag, v = np.linalg.svd(val['sigma'])  # np.svd naturally returns a diagonal
            s_diag[s_diag < scale_param*s_diag.max()] = scale_param*s_diag.max()
            val['sigma'] = np.matmul(np.matmul(u, np.diag(s_diag)), v)
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
        # Dictionary approach here as well
        points_in_voxels = self.bin_in_voxels(update_points)
        # Update the NDT approximation with these binned points
        self.update_stats(points_in_voxels)
        self.eig_check()
        return None

    def find_integrity(self, points):
        """
        Given a set of points and the underlying NDT Cloud, find the integrity of each voxel and the combined navigation
        solution
        :param points: Transformed points for which the integrity is required
        :return: Im: The integrity of the navigation solution obtained using the transformed points given
        :return: iscore: Voxel integrity score corresponding to the voxel center
        """
        test_xyz = points[:, :3]
        binned_points = self.bin_in_voxels(test_xyz)
        N = len(self.stats)
        iscore_array = np.zeros(N)
        loop_index = 0
        mu_points = np.zeros([N, 3])
        for key, val in self.stats.items():
            if key in binned_points:
                mu_points[loop_index, :] = val['mu']
                iscore_array[loop_index] = integrity.voxel_integrity(val, binned_points[key])
                self.stats[key]['integrity'] = iscore_array[loop_index]
                if np.isnan(iscore_array[loop_index]):
                    print('NaN detected!')
                loop_index += 1
            else:
                self.stats[key]['integrity'] = 0
        iscore_array[iscore_array == 0] = 1e-9
        Im, iscore_sum = integrity.solution_score(mu_points[:loop_index, :], iscore_array[:loop_index], points)
        # The loop index is added to ensure that only points that have a corresponding voxel are used for IDOP
        return Im, iscore_sum

    def optimization_integrity(self, points):
        """
        Given a set of points and the underlying NDT Cloud, find the integrity of each voxel and the combined navigation
        solution
        :param points: Transformed points for which the integrity is required
        :return: Im: The integrity of the navigation solution obtained using the transformed points given
        :return: iscore: Voxel integrity score corresponding to the voxel center
        """
        test_xyz = points[:, :3]
        binned_points = self.bin_in_voxels(test_xyz)
        N = len(self.stats)
        iscore_dict = {}
        rbar_dict = {}
        k_dict = {}
        loop_index = 0
        mu_points = np.zeros([N, 3])
        for key, val in self.stats.items():
            if key in binned_points:
                mu_points[loop_index, :] = val['mu']
                iscore_dict[key], rbar_dict[key], k_dict[key] = integrity.voxel_int_opt(val, binned_points[key])
                if np.isnan(iscore_dict[key]):
                    print('NaN detected!')
                loop_index += 1
        iscore_dict[iscore_dict == 0] = 1e-9
        # The loop index is added to ensure that only points that have a corresponding voxel are used for IDOP
        return iscore_dict, rbar_dict, k_dict

    def filter_voxels_integrity(self, integrity_limit=0.7):
        """
        Function to trim an ndt_cloud based on the integrity values of its voxels
        :param self: The NDT approximation to be trimmed
        :param integrity_limit: The minimum valid voxel integrity value
        :return: ndt_cloud: The same NDT approximation, but now with all voxels below an integrity limit removed
        """
        delete_index = []
        for key in self.stats.keys():
            if self.stats[key]['integrity'] < integrity_limit:
                delete_index.append(key)
        for del_key in delete_index:
            del self.stats[del_key]
        return None

    def pairing_cent2int(self, point_centers):
        """

        :param point_centers: Nx3 numpy array containing coordinates under consideration
        :return:
        """
        """
        1. Using voxel size, convert each center to a coordinate with only integer values
        2. Implement a standard pairing function to bind said coordinate to an index
        """
        assert(point_centers.shape[1] == 3)  # Checking that the matrix is all row vectors
        # Assign unique positive value to each integer
        pt_centers_temp = np.copy(point_centers)
        pt_centers_temp = (pt_centers_temp + self.first_center[0, :])/np.array([self.horiz_grid_size, self.horiz_grid_size, self.vert_grid_size])
        pt_centers_temp[pt_centers_temp > 0] = 2*pt_centers_temp[pt_centers_temp > 0]
        pt_centers_temp[pt_centers_temp < 0] = -2*pt_centers_temp[pt_centers_temp < 0] - 1
        x = np.atleast_2d(pt_centers_temp[:, 0])
        y = np.atleast_2d(pt_centers_temp[:, 1])
        z = np.atleast_2d(pt_centers_temp[:, 2])
        assert(np.min(x) > -1)
        assert(np.min(y) > -1)
        assert(np.min(z) > -1)
        pair_1 = np.atleast_2d(0.5*(x + y)*(x + y + 1) + y)
        int_pairing = np.atleast_2d(0.5*(pair_1 + z)*(pair_1 + z + 1) + z)
        int_pairing = np.reshape(int_pairing, [-1, 1])
        assert(int_pairing.shape == (point_centers.shape[0], 1))
        return int_pairing

    def pair_check(self):
        """
        Checking that the number of voxels and the number of unique index assignments is the same
        :return: None
        """
        voxels = []
        number = 0
        for key in self.stats:
            voxels.append(self.stats[key]['idx'][0][0])
            number += 1
        voxels = np.array(voxels)
        unique_voxels, unique_counts, case_counts = np.unique(voxels, return_index=True, return_counts=True)
        unique_no = np.size(unique_voxels)
        print('The number of voxels is ', number)
        print('The number of maximum voxels is ', self.max_no_voxels)
        print('The number of unique voxels is ', unique_no)
        assert(np.size(unique_voxels) == self.max_no_voxels)
        return None

    def prune_pc(self, pc):
        """
        Remove all points that don't overlap with NDT Cloud
        :param pc: Point cloud
        :return pruned_pc: Unique points that overlap with NDT Cloud
        """
        pruned_pc = np.zeros([0, 3])
        center_dict = self.bin_in_voxels(pc)
        keys = np.zeros([0, 3])
        binned_keys = np.zeros([0, 3])
        original_keys = np.zeros([0, 3])
        for key in self.stats:
            original_keys = np.vstack((original_keys, key))
        for key in center_dict:
            binned_keys = np.vstack((binned_keys, key))
            if key in self.stats:
                keys = np.vstack((keys, key))
                pruned_pc = np.vstack((pruned_pc, center_dict[key]))
        return np.unique(pruned_pc, axis=0)


class NDTCloudNoOverLap(NDTCloudBase):
    """
    Class inherited from NDTCloudBase parent for NDT approximation with no overlapping voxels
    """
    def __init__(self, xlim, ylim, zlim, input_horiz_grid_size, input_vert_grid_size, cloud_type):
        super(NDTCloudNoOverLap, self).__init__(xlim, ylim, zlim, input_horiz_grid_size, input_vert_grid_size
                                                , cloud_type)
        self.first_center = np.zeros([1, 3])
        first_center_x = np.mod(2 * xlim / self.horiz_grid_size + 1, 2) * self.horiz_grid_size / 2.0
        first_center_y = np.mod(2 * ylim / self.horiz_grid_size + 1, 2) * self.horiz_grid_size / 2.0
        first_center_z = np.mod(2 * zlim / self.vert_grid_size + 1, 2) * self.vert_grid_size / 2.0
        self.first_center[0, :] = np.array([first_center_x, first_center_y, first_center_z])


class NDTCloudOverLap(NDTCloudBase):
    """
    Class inherited from NDTCloudBase parent for NDT approximation with overlapping voxels. Overlap ensure smoother likelihood computation
    """
    def __init__(self, xlim, ylim, zlim, input_horiz_grid_size, input_vert_grid_size, cloud_type):
        super(NDTCloudOverLap, self).__init__(xlim, ylim, zlim, input_horiz_grid_size, input_vert_grid_size
                                              , cloud_type)
        self.first_center = np.empty([8, 3])
        for i in range(8):
            offset = np.array([np.mod(i, 2), np.mod(np.int(i / 2), 2), np.int(i / 4)])
            first_center_x = np.mod(2 * xlim / self.horiz_grid_size + offset[0] + 1, 2) * self.horiz_grid_size / 2.0
            first_center_y = np.mod(2 * ylim / self.horiz_grid_size + offset[1] + 1, 2) * self.horiz_grid_size / 2.0
            first_center_z = np.mod(2 * zlim / self.vert_grid_size + offset[2] + 1, 2) * self.vert_grid_size / 2.0
            self.first_center[i, :] = np.array([first_center_x, first_center_y, first_center_z])


class NDTCloudInterpolated(NDTCloudBase):
    """
    Class inherited from NDTCloudBase parent for NDT approximation with non-overlapping voxels with interpolated likelihood calculation.
    Method of objective, Jacobian and Hessian computations is different from base class.
    """
    def __init__(self, xlim, ylim, zlim, input_horiz_grid_size, input_vert_grid_size, cloud_type):
        super(NDTCloudInterpolated, self).__init__(xlim, ylim, zlim, input_horiz_grid_size, input_vert_grid_size
                                                   , cloud_type)
        self.first_center = np.zeros([1, 3])
        first_center_x = np.mod(2 * xlim / self.horiz_grid_size + 1, 2) * self.horiz_grid_size / 2.0
        first_center_y = np.mod(2 * ylim / self.horiz_grid_size + 1, 2) * self.horiz_grid_size / 2.0
        first_center_z = np.mod(2 * zlim / self.vert_grid_size + 1, 2) * self.vert_grid_size / 2.0
        self.first_center[0, :] = np.array([first_center_x, first_center_y, first_center_z])

    def find_octant(self, transformed_xyz):
        """
        Find which octant of divided voxel point lies in. Used to find centers of neighbouring voxels
        :param transformed_xyz: PC after transformation
        :return octant: Octants for each point in the PC
        """
        _, point_centers = self.find_voxel_center(transformed_xyz)
        diff_sign = np.sign(transformed_xyz - point_centers)
        diff_sign[np.isnan(diff_sign)] = -1.0
        octant_index = np.zeros([2, 2, 2])
        octant_index[1, 1, 1] = 1
        octant_index[0, 1, 1] = 2
        octant_index[0, 0, 1] = 3
        octant_index[1, 0, 1] = 4
        octant_index[1, 1, 0] = 5
        octant_index[0, 1, 0] = 6
        octant_index[0, 0, 0] = 7
        octant_index[1, 0, 0] = 8
        diff_sign[diff_sign == -1] = 0
        diff_sign = diff_sign.astype(np.int, copy=False)
        octant = octant_index[diff_sign[:, 0], diff_sign[:, 1], diff_sign[:, 2]]
        return octant

    def find_neighbours(self, transformed_xyz, no_neighbours=8):
        """
        Return centers of voxels neighbouring points in given point cloud
        :param transformed_xyz: PC after transformation
        :param no_neighbours: Number of neighbours to find, 8 by default
        :return nearby: neighbours of each point in transformed_xyz
        """
        if no_neighbours == 8:
            octants = self.find_octant(transformed_xyz)
        else:
            octants = 0
            ValueError("Input number of neighbours not supported")
        diff_index = self.octant2diff()
        oct_index = octants.astype(np.int) - 1
        grid_disp = np.array([self.horiz_grid_size, self.horiz_grid_size, self.vert_grid_size])
        _, point_centers = self.find_voxel_center(transformed_xyz, tol=0.0)
        nearby = np.tile(point_centers, 8) + np.tile(grid_disp, no_neighbours)*diff_index[oct_index, :]
        assert(not np.any(np.isnan(nearby)))
        return nearby

    @staticmethod
    def octant2diff():
        """
        Map octant to coordinate changes in center of voxel
        return diff_index: Octant to difference mapping
        """
        # Defining a reverse octant index
        oct_rev = np.zeros([8, 2, 3])
        oct_rev[0, 1, :] = [1, 1, 1]
        oct_rev[1, 1, :] = [-1, 1, 1]
        oct_rev[2, 1, :] = [-1, -1, 1]
        oct_rev[3, 1, :] = [1, -1, 1]
        oct_rev[4, 1, :] = [1, 1, -1]
        oct_rev[5, 1, :] = [-1, 1, -1]
        oct_rev[6, 1, :] = [-1, -1, -1]
        oct_rev[7, 1, :] = [1, -1, -1]
        # Creating a displacement vector pertaining to each octant
        diff_index = np.zeros([8, 24])
        for vidx in range(8):
            counter = 0
            for xidx in range(2):
                for yidx in range(2):
                    for zidx in range(2):
                        diff_index[vidx, 3*counter:3*counter+3] = [oct_rev[vidx, xidx, 0], oct_rev[vidx, yidx, 1],
                                                                   oct_rev[vidx, zidx, 2]]
                        counter += 1
        return diff_index

    def find_interp_weights(self, pts, neighbours):
        """
        Function debugged by checking sum of all values and that the max and min do not leave the unit interval
        :param pts: Points for which interpolated weights are required
        :param neighbours: Neighbours of points for pts and NDT cloud
        :return weights: Weight of each neighbour for corresponding point
        """
        norm_neighbours = (neighbours - np.tile(neighbours[:, :3], 8)) / \
                          np.tile(np.array([self.horiz_grid_size, self.horiz_grid_size, self.vert_grid_size]), 8)
        norm_neighbours[norm_neighbours == -1] = 1
        diff_pts_mask = np.abs(np.tile(pts - neighbours[:, :3], 8))
        diff_pts_minus_mask = np.ones_like(diff_pts_mask) - diff_pts_mask
        weight_fact = diff_pts_mask*norm_neighbours + diff_pts_minus_mask*(1 - norm_neighbours)
        testing = np.hsplit(weight_fact, 8)
        weights = np.prod(testing, axis=2) 
        return weights

    def find_likelihood(self, transformed_pc):
        """
        Likelihood calculated on a per point basis. First, calculate the 8 closest grid centers/ or means to a point
        Use expectation of point for each Gaussian as the value at the mean of that Gaussian.
        Vectorized implementation
        Finally, calculate an interpolation for the actual value of the mean with the associated weights
        :param transformed_pc: PC for which likelihood calculation is required
        :return likelihhod: Calculated interpolated likelihood
        """
        transformed_xyz = transformed_pc[:, :3]
        neighbours = self.find_neighbours(transformed_xyz)
        N = transformed_xyz.shape[0]
        vect_nearby_init = np.array(np.hsplit(neighbours, 8))
        vert_stack = np.reshape(vect_nearby_init, [N*8, 3])
        vert_idx = self.pairing_cent2int(vert_stack)
        vect_nearby_idx = np.reshape(vert_idx.T, [8, N]).T
        vect_mus = np.empty([8, N, 3, 1])
        vect_inv_sigmas = 10000*np.ones([8, N, 3, 3])
        for key in self.stats:
            indices = vect_nearby_idx == self.stats[key]['idx']
            mean = self.stats[key]['mu']
            inv_sigma = np.linalg.inv(self.stats[key]['sigma'])
            vect_mus[indices.T, :, 0] = mean
            vect_inv_sigmas[indices.T, :, :] = inv_sigma
        diff = np.empty_like(vect_mus)
        diff[:, :, :, 0] = -vect_mus[:, :, :, 0] + transformed_xyz[:N, :]
        diff_transpose = np.transpose(diff, [0, 1, 3, 2])
        lkds = np.exp(-0.5*np.matmul(np.matmul(diff_transpose, vect_inv_sigmas), diff))[:, :, 0, 0]
        weights = self.find_interp_weights(transformed_xyz, neighbours)[:, :N]
        wgt_lkd = weights*lkds
        likelihood = np.sum(wgt_lkd)
        return likelihood


def ndt_approx(ref_pointcloud, horiz_grid_size=0.5, vert_grid_size=0.5, type='overlapping'):
    """
    Function to create single NDT approximation for given offset and point cloud
    :param ref_pointcloud: [x, y, z, int] Nx4 or Nx3 numpy array of the reference point cloud
    :param horiz_grid_size: Float describing required horizontal grid size (in m)
    :param vert_grid_size: Float describing required vertical grid size. LiDAR span will be significantly shorter ...
    ... vertically with different concentrations, hence the different sizes. Same as horiz_grid_size by default
    :param type: Input to control overlap, and type of objective function used for calculation
    :return: ndt_cloud: NDT approximated cloud for the given point cloud and grid size
    """
    if ref_pointcloud.shape[1] == 4:
        ref_pointcloud = ref_pointcloud[:, :3]
    # Extract the size of the grid
    xlim = np.ceil(np.max(np.absolute(ref_pointcloud[:, 0]))) + 2*horiz_grid_size
    ylim = np.ceil(np.max(np.absolute(ref_pointcloud[:, 1]))) + 2*horiz_grid_size
    zlim = np.ceil(np.max(np.absolute(ref_pointcloud[:, 2]))) + 2*vert_grid_size
    # Create NDT map for reference grid
    if type == 'overlapping':
        ndt_cloud = NDTCloudOverLap(xlim, ylim, zlim, input_horiz_grid_size=horiz_grid_size,
                                    input_vert_grid_size=vert_grid_size, cloud_type=type)
    elif type == 'nooverlap':
        ndt_cloud = NDTCloudNoOverLap(xlim, ylim, zlim, input_horiz_grid_size=horiz_grid_size,
                                      input_vert_grid_size=vert_grid_size, cloud_type=type)
    elif type == 'interpolate':
        ndt_cloud = NDTCloudInterpolated(xlim, ylim, zlim, input_horiz_grid_size=horiz_grid_size,
                                         input_vert_grid_size=vert_grid_size, cloud_type=type)
    else:
        print('Wrong type of NDT Cloud specified in input. Defaulting to overlapping cloud')
        ndt_cloud = NDTCloudOverLap(xlim, ylim, zlim, input_horiz_grid_size=horiz_grid_size,
                                    input_vert_grid_size=vert_grid_size, cloud_type='overlapping')
    ndt_cloud.update_cloud(ref_pointcloud)
    return ndt_cloud


def display_ndt_cloud(ndt_cloud, point_density=0.1):
    """
    Function to display NDT approximation from a collection of NDT clouds
    :param ndt_cloud: NDT point cloud approximation
    :return: None
    """
    points_to_plot, pt_integrity = ndt_cloud.display(plot_density=point_density)
    pt_integrity = (np.max(pt_integrity) - pt_integrity)/(np.max(pt_integrity) - np.min(pt_integrity))
    ndt_viewer = pptk.viewer(points_to_plot, pt_integrity)
    ndt_viewer.color_map('hot')
    ndt_viewer.set(lookat=[0.0, 0.0, 0.0])
    return None


def find_pc_limits(pointcloud):
    """
    Function to find cartesian coordinate limits of given point cloud
    :param pointcloud: Given point cloud as an np.array of shape Nx3
    :return: xlim, ylim, zlim: Corresponding maximum absolute coordinate values
    """
    xlim = np.max(np.abs(pointcloud[:, 0]))
    ylim = np.max(np.abs(pointcloud[:, 1]))
    zlim = np.max(np.abs(pointcloud[:, 2]))
    return xlim, ylim, zlim


def multi_scale_ndt_odom(ref_pc, test_pc, scale_vect, filter_cv, test_mode, iters1, iters2):
    """
    find odometry mapping test PC to reference PC using multiscale NDT approximation
    :param ref_pc: Reference PC
    :param test_pc: Test PC
    :param scale_vect: Vector with NDT voxel lengths
    :param filter_cv: Voxel consensus metric below which voxels are removed
    :param test_mode: Type of NDT approximation, overlapping, nooverlap, interpolate
    :param iters1: Maximum number of iterations before removing low consensus voxels
    :param iters2: Maximum number of iterations after removing low consensus voxels
    """
    use_ref_pc = np.copy(ref_pc)
    use_test_pc = np.copy(test_pc)
    odom = np.zeros([np.size(scale_vect), 6])
    tic = time.time()
    for scale_idx, scale in enumerate(scale_vect):
        #print('DEBUG: Optimizing for voxel size ', scale)
        ref_ndt = ndt_approx(use_ref_pc, horiz_grid_size=scale, vert_grid_size=scale, type=test_mode)
        odom[scale_idx, :] = odometry.odometry(ref_ndt, use_test_pc, max_iter_pre=iters1, max_iter_post=iters2,
                                               integrity_filter=filter_cv)
        if len(ref_ndt.stats)!= 0:
            use_ref_pc = ref_ndt.prune_pc(ref_pc)
            #print('DEBUG: Size of ref_ndt is ', len(ref_ndt.stats))
            #print('DEBUG: Size of the pruned point cloud is ', np.shape(use_ref_pc)[0])
            use_test_pc = utils.transform_pc(odom[scale_idx, :], test_pc)
        else:
            break
    # Transform the test point cloud by the obtained odometry
    toc = time.time()
    return odom[scale_idx, :], toc - tic, odom
