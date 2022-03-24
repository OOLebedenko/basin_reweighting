import numpy as np


def rotate_vector_clockwise(vector,
                            axis,
                            theta_rad
                            ):
    # The Rodrigues' clockwise rotation
    cos_theta = np.cos(theta_rad)
    sin_theta = np.sin(theta_rad)
    axis = axis / np.linalg.norm(axis, axis=0)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    R = np.eye(3, 3) + sin_theta * K + (1 - cos_theta) * np.dot(K, K)
    rotated_vector = np.dot(vector, R)
    return rotated_vector


def extract_csa_c_z_axis(CO_vectors,
                         C_CA_vectors
                         ):
    # definition of the z-axis: perpendicular to peptide plane
    C_CA_vectors /= np.linalg.norm(C_CA_vectors, axis=1)[:, np.newaxis]
    CO_vectors /= np.linalg.norm(CO_vectors, axis=1)[:, np.newaxis]
    z_axis = np.cross(CO_vectors, C_CA_vectors, axis=1)
    return z_axis


def extract_csa_c_x_axis(CO_vectors,
                         csa_c_z_axis
                         ):
    # # definition of the x-axis: 38 degrees from CN to CO (rotation around z-axis)
    # # it's equvalent clockwise rotation CO on 82 degrees around z-axis
    CO_vectors /= np.linalg.norm(CO_vectors, axis=1)[:, np.newaxis]
    theta = np.deg2rad(82)
    x_axis = np.array(
        [rotate_vector_clockwise(vector, z_axis, theta) for vector, z_axis in zip(CO_vectors, csa_c_z_axis)])
    return x_axis


def extract_csa_c_y_axis(CO_vectors,
                         csa_c_z_axis
                         ):
    # # definition of the y-axis: 128 degrees from CN to CO(rotation around z-axis)]
    # # it's equvalent clockwise rotation CO on -8 degrees around z-axis
    CO_vectors /= np.linalg.norm(CO_vectors, axis=1)[:, np.newaxis]
    theta = np.deg2rad(-8)
    y_axis = np.array(
        [rotate_vector_clockwise(vector, z_axis, theta) for vector, z_axis in zip(CO_vectors, csa_c_z_axis)])
    return y_axis
