"""
ndt.py
File containing functions for NDT-based point cloud function approximation functions
Author: Ashwin Kanhere
Date created: 15ht April 2019
Last modified: 18th April 2019
"""
import numpy as np
"""
Importing base libraries
"""


def ndt_approx(ref_pointcloud, horiz_grid_size=0.5, vert_grid_size=0.5, offset_axis=np.array([0, 0, 0])):
    """
    Function to create single NDT approximation for given offset and point cloud
    :param ref_pointcloud: [x, y, z, int] Nx4 numpy array of the reference point cloud
    :param horiz_grid_size: Float describing required horizontal grid size (in m)
    :param vert_grid_size: Float describing required vertical grid size. LiDAR span will be significantly shorter vertically with ...
    ...different concentrations, hence the different sizes. Same as horiz_grid_size by default
    :param offset_axis: 1x3 np array containing booleans for which axis to offset on
    :return: ndt_cloud: NDT approximated cloud for the given point cloud and grid size
    """
    ref_pointcloud = ref_pointcloud.reshape([-1, 4])
    # Extract the size of the grid
    xlim = np.ceil(np.max(np.absolute(ref_pointcloud[:, 1]))) + 2*horiz_grid_size - 0.5*horiz_grid_size*(1 - offset_axis[0])
    ylim = np.ceil(np.max(np.absolute(ref_pointcloud[:, 1]))) + 2*horiz_grid_size - 0.5*horiz_grid_size*(1 - offset_axis[1])
    zlim = np.ceil(np.max(np.absolute(ref_pointcloud[:, 1]))) + 2*vert_grid_size - 0.5*vert_grid_size*(1 - offset_axis[2])
    xrange = np.arange(-xlim, xlim + horiz_grid_size, horiz_grid_size) # np.arange does not include right hand limit
    yrange = np.arange(-ylim, ylim + horiz_grid_size, horiz_grid_size)
    zrange = np.arange(-zlim, zlim + vert_grid_size, vert_grid_size)

    # Create a grid for NDT approximation. Each index must represent the center of the grid cell
    xgrid, ygrid, zgrid = np.meshgrid(xrange, yrange, zrange, sparse=False, indexing='xy')
    # Create NDT map for reference grid
    ndt_cloud = np.array(xgrid.shape)
    grid_mu = np.zeros([xgrid.shape[0], xgrid.shape[1], xgrid.shape[2], 3])
    # Better to index in the first variable, even thought definition is counter intuitive
    grid_sigma = np.zeros([xgrid.shape[0], xgrid.shape[1], xgrid.shape[2], 3, 3])

    # Initialize list of lists for the points per grid
    # points_in_grid = [[] for i in range(xgrid.size)]
    points_in_grid = []  # Initialize empty list to build on later
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
    return ndt_cloud


def pc_to_ndt(ref_pointcloud, horiz_grid_size=0.5, vert_grid_size=0.5):
    """
    Function to convert given point cloud into a NDT based approximation
    :param ref_pointcloud: Point cloud that needs to be converted to NDT reference
    :param horiz_grid_size: Parameter for horizontal grid sizing
    :param vert_grid_size: Parameter for vertical grid sizing
    :return: pc_ndt_approx: An object containing a collection (8) of NDT clouds that makes up the total reference ...
    ... for the given point cloud
    """
    pc_ndt_approx = []  # Initializing the cloud object
    for i in range(1):#range(8):
        offset = np.array([np.mod(i, 2), np.mod(np.int(i/2), 2), np.int(i/4)])
        pc_ndt_approx.append(ndt_approx(ref_pointcloud, horiz_grid_size, vert_grid_size, offset_axis=offset))
    pc_ndt_approx = []
    return pc_ndt_approx