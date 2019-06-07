"""
ndt.py
File containing functions for NDT-based point cloud function approximation functions
Author: Ashwin Kanhere
Date created: 15ht April 2019
Last modified: 30th May 2019
"""
import numpy as np
import pptk
import utils
import transforms3d
from scipy.optimize import minimize
from scipy.optimize import Bounds
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

    def update_displacement(self, odometry_vector):
        """
        A function to update the displacement of the current local frame of reference from the global reference
        :param odometry_vector: A vector of [x, y, z, phi, theta, psi] measuring the  affine transformation of the
        current local frame of reference (LiDAR origin) to the global frame of reference (map origin)
        :return: None
        """
        # TODO: Check update_displacement
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
        R_delta = transforms3d.euler.euler2mat(phi_delta, theta_delta, psi_delta)
        R_new = np.matmul(R_delta, R_local)
        phi_rad, theta_rad, psi_rad = transforms3d.euler.mat2euler(R_new)
        angle_new = np.rad2deg(np.array([phi_rad, theta_rad, psi_rad]))
        self.local_to_global[3:] = angle_new
        return None

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

    def bin_in_voxels(self, points_to_bin):
        """
        Function to bin given points into voxels in a dictionary approach
        :param points_to_bin: The points that are to be binned into the voxel clusters indexed by the voxel center tuple
        :return: points_in_voxel: A dictionary indexed by the tuple of the center of the bin
        """
        no_points = points_to_bin.shape[0]
        voxel_centers = self.find_voxel_center(points_to_bin)
        points_in_voxels = {}
        for i in range(no_points):
            voxel_key = tuple(voxel_centers[i, :])
            if voxel_key in points_in_voxels:
                points_in_voxels[voxel_key] = np.vstack((points_in_voxels[voxel_key], points_to_bin[i, :]))
            else:
                points_in_voxels[voxel_key] = points_to_bin[i, :]
        return points_in_voxels

    def find_likelihood(self, transformed_pc):
        """
        Function to return likelihood for a given transformed point cloud w.r.t NDT point cloud
        Slightly different from reference papers in that 1/2det(sigma) is also included while calculating the likelihood
        The likelihood is increased if a corresponding Gaussian is found. If not, 0 is added
        :param transformed_pc: Point cloud that has been passed through a candidate affine transformation
        :return: likelihood: Scalar value representing the likelihood of the given
        """
        # TODO: Test likelihood computation using a number of points that are easy to evaluate on a calculator/MATLAB.
        #  This should guard against obvious calculative problems
        transformed_xyz = transformed_pc[:, :3]
        likelihood = 0
        points_in_voxels = self.bin_in_voxels(transformed_xyz)
        for key, val in points_in_voxels.items():
            if key in self.stats:
                sigma = self.stats[key]['sigma']
                sigma_inv = np.linalg.inv(sigma)
                diff = val - self.stats[key]['mu']
                likelihood += np.sum(np.exp(-np.diag(np.matmul(np.matmul(diff, sigma_inv), diff.T))))
        print("Random statement")
        return likelihood

    def display(self, plot_density=1.0):
        """
        Function to display the single NDT approximation
        :param fig: The figure object on which the probability function has to be plotted
        :param plot_density: The density of the plot (as a int scalar) the higher the density, the more points per grid
        :return: plot_points: The points sampled from the distribution that are to be plotted like any other PC
        """
        # TODO: Display points are off center (when compared to the original point cloud. FIX THIS
        base_num_pts = 48  # 3 points per vertical and 4 per horizontal
        num_pts = np.int(plot_density * base_num_pts)
        plot_points = np.empty([3, 0])
        for key, value in self.stats.items():
            sigma = self.stats[key]['sigma']
            mu = self.stats[key]['mu']
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
        # TODO: The results of display seem to be off center. Check possible issues in point/ center computations both
        #  here and in the part where they're first being approximated
        return plot_points

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
            else:
                if no_in_voxel >= 4:
                    self.stats[k] = {}  # Initialize empty dictionary before populating with values
                    self.stats[k]['mu'] = np.mean(v, axis=0)
                    self.stats[k]['sigma'] = np.cov(v, rowvar=False)
                    self.stats[k]['no_points'] = no_in_voxel
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

    def odometry_objective(self, odometry_vector, original_pc):
        """
        A function that combines functions for transformation of a point cloud and the computation of likelihood (score)
        :param odometry_vector: Candidate odometry vector
        :param original_pc: Input point cloud
        :return: objective: Maximization objective which is the likelihood of transformed point cloud for the given NDT
        """
        transformed_pc = transform_pc(odometry_vector, original_pc)
        objective = -1*self.find_likelihood(transformed_pc)
        return objective

    def odometry_jacobian(self, odometry_vector, test_pc):
        """
        Function to return the Jacobian of the likelihood objective for the odometry calculation
        :param odometry_vector: The point at which the Jacobian is to be evaluated
        :param test_pc: The point cloud for which the optimization is being performed
        :return: jacobian: The Jacobian matrix (of the objective w.r.t the odometry vector)
        """
        N = test_pc.shape[0]
        jacobian = np.zeros([6, 1])
        transformed_pc = transform_pc(odometry_vector, test_pc)
        transform_centers = self.find_voxel_center(transformed_pc)
        # TODO: Re-test the output from bin in voxels and see if it the centers can be indexed in the following for loop
        print('Woo hoo')
        for pt_num in range(N):
            center_key = tuple(transform_centers[pt_num][:])
            # TODO: Check if this condition is being correctly triggered
            if center_key in self.stats:
                mu = self.stats[center_key]['mu']
                sigma = self.stats[center_key]['sigma']
                sigma_inv = np.linalg.inv(sigma)
                qx = test_pc[pt_num][0] - mu[0]
                qy = test_pc[pt_num][1] - mu[1]
                qz = test_pc[pt_num][2] - mu[2]
                q = np.array([[qx], [qy], [qz]])
                delq_delt = find_delqdelt(odometry_vector, q)
                g = np.zeros([6, 1])
                for i in range(6):
                    g[i] = np.matmul(q.T, np.matmul(sigma_inv, np.atleast_2d(delq_delt[:, i]).T)) * np.exp(
                    -0.5*np.matmul(q.T, np.matmul(sigma_inv, q)))
                jacobian += g
        return jacobian

    def odometry_hessian(self, odometry_vector, test_pc):
        """
        Function to return an approximation of the Hessian of the likelihood objective for the odometry calculation
        :param odometry_vector: The point at which the Hessian is evaluated
        :param test_pc: The point cloud for which the optimization is being carried out
        :return: hessian: The Hessian matrix of the objective w.r.t. the odometry vector
        """
        # TODO: Test implementation of the Hessian
        N = test_pc.shape[0]
        hessian = np.zeros([6, 6])
        transformed_pc = transform_pc(odometry_vector, test_pc)
        transform_centers = self.find_voxel_center(transformed_pc)
        for pt_num in range(N):
            center_key = tuple(transform_centers[i][:])
            # TODO: Check if this condition is being correctly triggered
            if center_key in self.stats:
                mu = self.stats[center_key]['mu']
                sigma = self.stats[center_key]['sigma']
                sigma_inv = np.linalg.inv(sigma)
                qx = test_pc[pt_num][0] - mu[0]
                qy = test_pc[pt_num][1] - mu[1]
                qz = test_pc[pt_num][2] - mu[2]
                q = np.array([[qx], [qy], [qz]])
                temp_hess = np.zeros([6, 6])
                delq_delt = find_delqdelt(odometry_vector, q)
                del2q_deltnm = find_del2q_deltnm(odometry_vector, q)
                for i in range(6):
                    for j in range(6):
                        temp_hess[i, j] = -np.exp(-0.5*np.matmul(q.T, np.matmul(sigma_inv, q)))*(
                                (-np.matmul(q.T, np.matmul(sigma_inv, np.atleast_2d(delq_delt[:, i])))*(
                            np.matmul(q.T, np.matmul(sigma_inv, np.atleast_2d(delq_delt[:, j]))))) - (
                            np.matmul(q.T, np.matmul(sigma_inv, np.atleast_2d(del2q_deltnm[:, i, j]))))-(
                            np.matmul(np.atleast_2d(delq_delt[:, j]).T, np.matmul(sigma_inv, np.atleast_2d(delq_delt[:, i])))))
                hessian += temp_hess
        return hessian

    def calculate_odometry(self, test_pc):
        """
        Function to find the best traansformation (in the form of a translation, Euler angle vector)
        :param test_pc: Point cloud which has to be matched to the existing NDT approximation
        :return: odom_vector: The resultant odometry vector (Euler angles in degrees)
        """
        # TODO: Provide the Jacobian for the objective function
        # TODO: Provide the Hessian for the objective function
        # TODO: Define the optimization solver and run it

        initial_odom = np.array([0, 0, 0, 0, 0, 0])
        xlim, ylim, zlim = find_pc_limits(test_pc)
        odometry_bounds = Bounds([-xlim, xlim], [-ylim, ylim], [-zlim, zlim], [-180.0, 180.0], [-90.0, 90.0],
                                 [-180.0, 180.0])
        res = minimize(self.odometry_objective, initial_odom, method='Newton-CG', jac=self.odometry_jacobian,
                       hessp=self.odometry_hessian, bounds=odometry_bounds, args=(test_pc,))
        odom_vector = res.x
        return odom_vector


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
    # TODO: Clean up this function and move all testing functionality to a separate file (ashwin-playground for example)
    ref_pointcloud = ref_pointcloud.reshape([-1, 4])
    # Extract the size of the grid
    xlim = np.ceil(np.max(np.absolute(ref_pointcloud[:, 1]))) + 2*horiz_grid_size + 0.5*horiz_grid_size*offset_axis[0]
    ylim = np.ceil(np.max(np.absolute(ref_pointcloud[:, 1]))) + 2*horiz_grid_size + 0.5*horiz_grid_size*offset_axis[1]
    zlim = np.ceil(np.max(np.absolute(ref_pointcloud[:, 1]))) + 2*vert_grid_size + 0.5*vert_grid_size*offset_axis[2]

    # Create NDT map for reference grid
    ndt_cloud = NDTCloud(xlim, ylim, zlim, input_horiz_grid_size=horiz_grid_size, input_vert_grid_size=vert_grid_size)
    ref_pointcloud_test = ref_pointcloud
    ndt_cloud.update_cloud(ref_pointcloud_test)
    test_point_1 = ref_pointcloud[12:14, :3]
    ndt_cloud.find_likelihood(test_point_1)
    points_to_plot = ndt_cloud.display(plot_density=0.5)
    pptk.viewer(points_to_plot.T)
    pptk.viewer(ref_pointcloud[:, :3])
    input("Press any key to finish program")
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
    # TODO: Move pptk viewer here (as opposed to its current location in the class defintion)
    return None


def find_pc_limits(pointcloud):
    """
    Function to find cartesian coordinate limits of given point cloud
    :param pointcloud: Given point cloud as an np.array of shape Nx3
    :return: xlim, ylim, zlim: Corresponding maximum absolute coordinate values
    """
    # TODO: Test this function and it's output
    xlim = np.max(np.abs(pointcloud[:, 0]))
    ylim = np.max(np.abs(pointcloud[:, 1]))
    zlim = np.max(np.abs(pointcloud[:, 2]))
    return xlim, ylim, zlim


def transform_pc(odometry_vector, original_pc):
    """
    Function to transform a point cloud according to the given odometry vector
    :param odometry_vector: [tx, ty, tz, phi, theta, psi] (angles in degrees)
    :param orignal_pc: original point cloud that is to be transformed
    :return:
    """
    phi = np.deg2rad(odometry_vector[3])
    theta = np.deg2rad(odometry_vector[4])
    psi = np.deg2rad(odometry_vector[5])
    R = transforms3d.euler.euler2mat(phi, theta, psi)  # Using default rotation as the convention for this project
    T = odometry_vector[:3]
    Z = np.eye(3)
    A = transforms3d.affines.compose(T, R, Z)
    transformed_pc = utils.transform_pts(original_pc, A)
    return transformed_pc


def find_delqdelt(odometry_vector, q):
    """
    Return a 3x6 matrix that captures partial q/ partial t_n
    :param odometry_vector:
    :param q:
    :return: delq_delt: A 3x6 matrix for the required partial derivative
    """
    phi = np.deg2rad(odometry_vector[0])
    theta = np.deg2rad(odometry_vector[1])
    psi = np.deg2rad(odometry_vector[2])
    c1 = np.cos(phi)
    s1 = np.sin(phi)
    c2 = np.cos(theta)
    s2 = np.sin(theta)
    c3 = np.cos(psi)
    s3 = np.sin(psi)
    qx = q[0]
    qy = q[1]
    qz = q[2]
    delq_delt = np.zeros([3, 6])
    delq_delt[:, 0] = np.array([[1], [0], [0]])
    delq_delt[:, 1] = np.array([[0], [1], [0]])
    delq_delt[:, 2] = np.array([[0], [0], [1]])
    delq_delt[:, 3] = np.pi/180*np.array([[0], [-s2 * c3 * qx + s2 * s3 * qy + c2 * qz], [-c2 * s3 * qx - c2 * c3 * qy]])
    delq_delt[:, 4] = np.pi/180*np.array([[(-s1 * s3 + c3 * c1 * s2) * qx - (s1 * c3 + c1 * s2 * s3) * qy - c2 * c1 * qz],
                                [c3 * s1 * c2 * qx - s1 * c2 * s3 * qy + s1 * s2 * qz],
                                [(c1 * c3 - s1 * s2 * s3) * qx - (c1 * s3 + s1 * s2 * c3) * qy]])
    delq_delt[:, 5] = np.pi/180*np.array([[(c1 * s3 + s1 * c3 * s2) * qx + (c1 * c3 - s1 * s2 * s3) * qy - s1 * c2 * qz],
                                [-c1 * c2 * c3 * qx + c1 * c2 * s3 * qy - c1 * s2 * qz],
                                [(s1 * c3 + c1 * s2 * s3) * qx + (-s1 * s3 + c1 * s2 * c3) * qy]])
    return delq_delt


def find_del2q_deltnm(odometry_vector, q):
    del2q_deltnm = np.zeros([3, 6, 6])
    delta = 0.001
    for i in range(6):
        odometry_new = np.zeros([6, ])
        odometry_new[i] = odometry_new[i] + delta
        q_new = transform_pc(odometry_new, q)
        odometry_new += odometry_vector
        del2q_deltnm[:, :, i] = (find_delqdelt(odometry_new, q_new) - find_delqdelt(odometry_vector, q))/delta
    return del2q_deltnm
