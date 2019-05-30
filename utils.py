"""
utils.py
File for mathematical helper functions for NDT SLAM
Author: Ashwin Kanhere
Date Created: 15th April 2019
"""
import numpy as np


def rotation_matrix(angle, axis='z'):
    if axis == 'x':
        c = np.cos(angle)
        s = np.sin(angle)
        rot_matrix = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    elif axis == 'y':
        c = np.cos(angle)
        s = np.sin(angle)
        rot_matrix = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    elif axis == 'z':
        c = np.cos(angle)
        s = np.sin(angle)
        rot_matrix = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    else:
        raise Exception('Wrong value for axis input. Check your input')
    return rot_matrix


def dcm2eul(euler_angle):
    phi = euler_angle[0]
    theta = euler_angle[1]
    psi = euler_angle[2]
    rot_mat_x = rotation_matrix(phi, axis='x')
    rot_mat_y = rotation_matrix(theta, axis='y')
    rot_mat_z = rotation_matrix(psi, axis='z')
    dcm = np.matmul(rot_mat_z, np.matmul(rot_mat_y, rot_mat_x))
    return dcm
