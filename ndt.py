"""
ndt.py
File containing functions for NDT-based point cloud function approximation functions
Author: Ashwin Kanhere
Date created: 15th April 2019
Last modified: 13th June 2019
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
import time


"""
Importing base libraries
"""


class NDTCloud:
    """
    A class to store the sparse grid center points, means and covariances for grid points that are full.
    This class will be the de facto default for working with NDT point clouds
    """
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
        self.first_center = np.empty([8, 3])
        for i in range(8):
            offset = np.array([np.mod(i, 2), np.mod(np.int(i/2), 2), np.int(i/4)])
            first_center_x = np.mod(2*xlim/self.horiz_grid_size + offset[0] + 1, 2)*self.horiz_grid_size/2.0
            first_center_y = np.mod(2*ylim/self.horiz_grid_size + offset[1] + 1, 2)*self.horiz_grid_size/2.0
            first_center_z = np.mod(2*zlim/self.vert_grid_size + offset[2] + 1, 2) *self.vert_grid_size/2.0
            self.first_center[i, :] = np.array([first_center_x, first_center_y, first_center_z])
            # xlim = np.ceil(np.max(np.absolute(ref_pointcloud[:, 0]))) + 2*horiz_grid_size + 0.5*horiz_grid_size*offset_axis[0]
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
        # Store an estimate of the transformation of the current scan origin to global origin

    def update_displacement(self, odometry_vector):
        """
        A function to update the displacement of the current local frame of reference from the global reference
        :param odometry_vector: A vector of [x, y, z, phi, theta, psi] measuring the  affine transformation of the
        current local frame of reference (LiDAR origin) to the global frame of reference (map origin)
        :return: None
        """
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

    def find_voxel_center(self, ref_pointcloud):
        """
        A function to return grid indices for a given set of 3D points. The input may be a set of (x, y, z) Nx3 or
        (x, y, z, int) Nx4. This function is written to be agnostic to either form of the array
        This function also checks if points on a edge of the grid upto a tolerance level. If they are, it assigns them
        a value to ensure that no calculations involve that point
        :param ref_pointcloud: Nx3 or Nx4 numpy array for which binning is required
        :return: grid_centers: Matrix containing center coordinates corresponding to the given points Nx3
        """
        # Used an array over a tuple as there is a small possibility that the coordinates might change
        ref_points = np.array(ref_pointcloud[:, :3])  # to remove intensity if it has been passed accidentally
        # int(ref_x/horizontal grid size)* horizontal grid size + first grid center (same sign that the quotient is)
        # first_center is the offset from the origin of the first grid center (in either direction)
        grid_size = np.array([self.horiz_grid_size, self.horiz_grid_size, self.vert_grid_size])
        """
        Main idea: First add the center. Things that were getting rounded to 23 while they should've gone down to 22, 
        22.6 for example will for sure get rounded up now. Then remove the added center in the final project to bring 
        them back down.
        0.03 second case. Adding center will give 0.53. Rounding will give 1 and removing the center will give 0.5
        """
        # Check if the point lies on the edge of the grid. If it does, provide it np. nan as a center so that that
        # particular point doesn't get considered for likelihood or jacobian, thus preventing gradient jumps.
        tol = 1.0e-7  # the maximum translation (with 3 safety margin)caused by a rotation of 1.45e-8 degrees
        points_repeated = np.tile(ref_points, (8, 1))
        N = ref_points.shape[0]
        voxel_centers = np.zeros_like(points_repeated)
        for i in range(8):
            pre_voxel_number = (ref_points + self.first_center[i, :]) / grid_size
            pre_voxel_center = np.round(pre_voxel_number).astype(int) * grid_size
            first_grid_edge = self.first_center[i, :] - 0.5*np.array([self.horiz_grid_size, self.horiz_grid_size,
                                                                      self.vert_grid_size])
            # first_grid_edge[first_grid_edge == 0] = 1.0
            # Taking the mod with the grid size should make the above statement redundant
            # In this case, points that are literally in the middle of the voxel will get ignored
            line_check = np.abs(np.mod(ref_points, grid_size) + first_grid_edge)
            pre_voxel_center[line_check < tol] = np.nan
            pre_voxel_center[np.abs(line_check - 1) < tol] = np.nan
            voxel_centers[i*N:(i+1)*N, :] = np.multiply(np.sign(ref_points), np.abs(pre_voxel_center) - np.broadcast_to(
                self.first_center[i, :], (N, 3)))
        return points_repeated, voxel_centers

    def old_bin_in_voxels(self, points_to_bin):
        """
        Function to bin given points into voxels in a dictionary approach
        :param points_to_bin: The points that are to be binned into the voxel clusters indexed by the voxel center tuple
        :return: points_in_voxel: A dictionary indexed by the tuple of the center of the bin
        """
        points_repeated, voxel_centers = self.find_voxel_center(points_to_bin)
        points_in_voxels = {}
        for i in range(points_repeated.shape[0]):
            voxel_key = tuple(voxel_centers[i, :])
            if voxel_key in points_in_voxels:
                points_in_voxels[voxel_key] = np.vstack((points_in_voxels[voxel_key], points_repeated[i, :]))
            else:
                points_in_voxels[voxel_key] = points_repeated[i, :]
        return points_in_voxels

    def bin_in_voxels(self, points_to_bin):
        """
        Function to bin given points into voxels in a dictionary approach
        :param points_to_bin: The points that are to be binned into the voxel clusters indexed by the voxel center tuple
        :return: points_in_voxel: A dictionary indexed by the tuple of the center of the bin
        """
        points_repeated, voxel_centers = self.find_voxel_center(points_to_bin)
        dummy = numpy_indexed.group_by(voxel_centers, points_repeated)
        points_in_voxels = {}
        # TODO: Verify the results from the modified code that follows
        for i in range(np.shape(dummy[0])[0]):
            voxel_key = tuple(dummy[0][i])
            points_in_voxels[voxel_key] = dummy[1][i]
        """
        for i in range(points_repeated.shape[0]):
            voxel_key = tuple(voxel_centers[i, :])
            if voxel_key in points_in_voxels:
                points_in_voxels[voxel_key] = np.vstack((points_in_voxels[voxel_key], points_repeated[i, :]))
            else:
                points_in_voxels[voxel_key] = points_repeated[i, :]
        """
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
                # sigma_inv_det = np.linalg.det(sigma_inv)
                # normal_factor = 1e-8
                diff = np.atleast_2d(val - self.stats[key]['mu']) # It's a coincidence that the dimensions work out
                # likelihood += (sigma_inv_det*normal_factor)*np.sum(np.exp(-0.5*np.diag(np.matmul(np.matmul(diff, sigma_inv), diff.T))))
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
            u, s_diag, v = np.linalg.svd(val['sigma'])  # np.svd naturally returns a diagonals
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
        # avg_iscore = np.mean(iscore)
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
        # avg_iscore = np.mean(iscore)
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

    def points_in_filled_voxels(self, test_pc):
        points_repeated, voxel_centers = self.find_voxel_center(test_pc)
        dummy = numpy_indexed.group_by(voxel_centers, points_repeated)
        repeated_voxel_points = np.empty([0, 3])
        for i in range(np.shape(dummy[0])[0]):
            voxel_key = tuple(dummy[0][i])
            if voxel_key in self.stats:
                repeated_voxel_points = np.vstack((repeated_voxel_points, dummy[1][i]))
        voxel_points = np.unique(repeated_voxel_points, axis=0)
        return voxel_points


def ndt_approx(ref_pointcloud, horiz_grid_size=0.5, vert_grid_size=0.5):
    """
    Function to create single NDT approximation for given offset and point cloud
    :param ref_pointcloud: [x, y, z, int] Nx4 or Nx3 numpy array of the reference point cloud
    :param horiz_grid_size: Float describing required horizontal grid size (in m)
    :param vert_grid_size: Float describing required vertical grid size. LiDAR span will be significantly shorter ...
    ... vertically with different concentrations, hence the different sizes. Same as horiz_grid_size by default
    :return: ndt_cloud: NDT approximated cloud for the given point cloud and grid size
    """
    if ref_pointcloud.shape[1] == 4:
        ref_pointcloud = ref_pointcloud[:, :3]
    # Extract the size of the grid
    xlim = np.ceil(np.max(np.absolute(ref_pointcloud[:, 0]))) + 2*horiz_grid_size
    ylim = np.ceil(np.max(np.absolute(ref_pointcloud[:, 1]))) + 2*horiz_grid_size
    zlim = np.ceil(np.max(np.absolute(ref_pointcloud[:, 2]))) + 2*vert_grid_size
    # Create NDT map for reference grid
    ndt_cloud = NDTCloud(xlim, ylim, zlim, input_horiz_grid_size=horiz_grid_size, input_vert_grid_size=vert_grid_size)
    ndt_cloud.update_cloud(ref_pointcloud)
    return ndt_cloud


def display_ndt_cloud(ndt_cloud, point_density = 0.1):
    """
    Function to display NDT approximation from a collection of NDT clouds
    :param ndt_cloud: NDT point cloud approximation
    :return: None
    """
    points_to_plot, pt_integrity = ndt_cloud.display(plot_density= point_density)
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
