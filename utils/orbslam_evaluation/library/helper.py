import numpy as np
import pandas as pd
import argparse

# evo toolbox
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def create_summary_dataframe(num_of_runs, boxplot_distances):
    """
    Creates a panda dataframe with column names corresponding to the run number, camera parameters, the appd metric
    and trajectory evaluations
    :param num_of_runs (int): Number of runs for the experiments
    :param boxplot_distances (list): the relative-distance ranges for trajectory evaluation
    :return:
    """
    indices = ["run_%d" % i for i in range(num_of_runs)]
    columns = ["run_name", "fx", "fy", "cx", "cy", "k1", "k2", "k3", "p1", "p2", "appd", 'abs_e_rot_stats_mean',
               'abs_e_rot_stats_median',
               'abs_e_rot_stats_rmse', 'abs_e_scale_stats_mean', 'abs_e_scale_stats_median', 'abs_e_scale_stats_rmse',
               'abs_e_trans_stats_mean', 'abs_e_trans_stats_median', 'abs_e_trans_stats_rmse']
    for dist in boxplot_distances:
        dist = float(dist)
        relative_columns = ['rel_rot_stats_mean_trajlen_' + str(dist),
                            'rel_rot_stats_median_trajlen_' + str(dist),
                            'rel_rot_stats_rmse_trajlen_' + str(dist),
                            'rel_trans_perc_stats_mean_trajlen_' + str(dist),
                            'rel_trans_perc_stats_median_trajlen_' + str(dist),
                            'rel_trans_perc_stats_rmse_trajlen_' + str(dist),
                            'rel_trans_stats_mean_trajlen_' + str(dist),
                            'rel_trans_stats_median_trajlen_' + str(dist),
                            'rel_trans_stats_rmse_trajlen_' + str(dist)
                            ]
        columns = columns + relative_columns

    dataframe = pd.DataFrame(index=indices, columns=columns)
    dataframe["run_name"] = ["Run %d" % i for i in range(num_of_runs)]
    return dataframe


def update_dataframe_with_rectification_stats(df, row_name, K, D, appd=None):
    """
    Update panda dataframe with current information
    :param df: panda dataframe
    :param row_name: row name in the dataframe
    :param K: Intrinsic camera matrix (3, 3)
    :param D: Distortion parameters (1, 5)
    :param appd: Statistic on new perturbed image
    :return: updated panda dataframe
    """
    # store camera intrinsic matrix
    df.at[row_name, "fx"] = K[0, 0]
    df.at[row_name, "fy"] = K[1, 1]
    df.at[row_name, "cx"] = K[0, 2]
    df.at[row_name, "cy"] = K[1, 2]
    # store camera distortion parameters
    df.at[row_name, "k1"] = D[0]
    df.at[row_name, "k2"] = D[1]
    df.at[row_name, "k3"] = D[4]
    df.at[row_name, "p1"] = D[2]
    df.at[row_name, "p2"] = D[3]
    # store perturbed image statistics
    df.at[row_name, "appd"] = appd

    return df


def get_range_perturbed_camera_parameters(K, D, settings, perturb_parameters):
    # default ranges without any perturbation
    ranges = {'fx': (K[0, 0], K[0, 0]),
              'fy': (K[1, 1], K[1, 1]),
              'cx': (K[0, 2], K[0, 2]),
              'cy': (K[1, 2], K[1, 2]),
              'k1': (D[0], D[0]),
              'k2': (D[1], D[1]),
              'p1': (D[2], D[2]),
              'p2': (D[3], D[3]),
              'k3': (D[4], D[4])}

    for param in perturb_parameters:
        # focal length parameters
        if param == 'f':
            ranges['fx'] = (K[0, 0] * settings["fx_range"][0], K[0, 0] * settings["fx_range"][1])
            ranges['fy'] = (K[1, 1] * settings["fy_range"][0], K[1, 1] * settings["fy_range"][1])
        # principle point parameters
        if param == 'c':
            ranges['cx'] = (K[0, 2] * settings["cx_range"][0], K[0, 2] * settings["cx_range"][1])
            ranges['cy'] = (K[1, 2] * settings["cy_range"][0], K[1, 2] * settings["cy_range"][1])
        # distortion parameters
        if param == 'd':
            ranges['k1'] = (D[0] * settings["k1_range"][0], D[0] * settings["k1_range"][1])
            ranges['k2'] = (D[1] * settings["k2_range"][0], D[1] * settings["k2_range"][1])
            ranges['p1'] = (D[2] * settings["p1_range"][0], D[2] * settings["p1_range"][1])
            ranges['p2'] = (D[3] * settings["p2_range"][0], D[3] * settings["p2_range"][1])
            ranges['k3'] = (D[4] * settings["k3_range"][0], D[4] * settings["k3_range"][1])

    return ranges


def kitti_poses_and_timestamps_to_trajectory(poses_file, timestamp_file):
    pose_path = file_interface.read_kitti_poses_file(poses_file)
    raw_timestamps_mat = file_interface.csv_read_matrix(timestamp_file)
    error_msg = "timestamp file must have one column of timestamps and same number of rows as the KITTI poses file"
    if len(raw_timestamps_mat) > 0 and len(raw_timestamps_mat[0]) != 1 or \
            len(raw_timestamps_mat) != pose_path.num_poses:
        raise file_interface.FileInterfaceException(error_msg)
    try:
        timestamps_mat = np.array(raw_timestamps_mat).astype(float)
    except ValueError:
        raise file_interface.FileInterfaceException(error_msg)
    return PoseTrajectory3D(poses_se3=pose_path.poses_se3, timestamps=timestamps_mat)


def modify_orbslam_settings(orbslam_settings, K, D):
    """
    Modify the orbslam configuration settings with the perturbed parameters
    :param settings: (dict) original orbslam settings
    :param K: perturbed camera intrinsic matrix
    :param D: perturbed camera distortion parameters
    :return:  (dict) modified orbslam settings
    """
    orbslam_settings['Camera.fx'] = float(K[0, 0])
    orbslam_settings['Camera.fy'] = float(K[1, 1])
    orbslam_settings['Camera.cx'] = float(K[0, 2])
    orbslam_settings['Camera.cy'] = float(K[1, 2])
    orbslam_settings['Camera.k1'] = float(D[0])
    orbslam_settings['Camera.k2'] = float(D[1])
    orbslam_settings['Camera.k3'] = float(D[4])
    orbslam_settings['Camera.p1'] = float(D[2])
    orbslam_settings['Camera.p2'] = float(D[3])

    return orbslam_settings

# EOF
