#!/usr/bin/env python

import os
import yaml
import subprocess
import cv2
import copy
import sys
import numpy as np
import pandas as pd
import argparse

# evo toolbox
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface

# uzh trajectory toolbox
from library.trajectory import Trajectory

# Aleks's image rectification toolbox
import carnivalmirror as cm

def create_summary_dataframe(num_of_runs, boxplot_distances):
    indices = ["run_%d" % i for i in range(num_of_runs)]
    columns = ["run_name", "fx", "fy", "cx", "cy", "k1", "k2", "k3", "p1", "p2", "d_q0.05", "d_q0.25", "d_q0.50",
               "d_q0.75", "d_q0.95", "d_avg", "d_std", 'abs_e_rot_stats_mean', 'abs_e_rot_stats_median',
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


def get_range_perturbed_camera_matrices(camera, settings, perturb_parameters, num_of_runs, run_num):
    """
    Retrive perturbed camera calibration matrices
    :param camera: dictionary containing default camera parameters
    :param settings: dictionary containing perturbation settings
    :param perturb_parameters: list specifies which parameters to perturb
    :param num_of_runs: number of runs to perform
    :param run_num: current run number
    :return: Camera Matrix (3, 3), Distortion Matrix (1, 5)
    """
    K = np.array(camera["K"]).reshape((3, 3))
    D = np.array(camera["D"])

    for param in perturb_parameters:
        if param == 'f':
            K[0, 0] = camera["K"][0] * (1 + np.linspace(settings["fx_range"][0], settings["fx_range"][1],
                                                        num=num_of_runs, endpoint=True)[run_num])
            K[1, 1] = camera["K"][4] * (1 + np.linspace(settings["fy_range"][0], settings["fy_range"][1],
                                                        num=num_of_runs, endpoint=True)[run_num])
        if param == 'c':
            K[0, 2] = camera["K"][2] * (1 + np.linspace(settings["cx_maxoffset"][0], settings["cx_maxoffset"][1],
                                                        num=num_of_runs, endpoint=True)[run_num])
            K[1, 2] = camera["K"][5] * (1 + np.linspace(settings["cy_maxoffset"][0], settings["cy_maxoffset"][1],
                                                        num=num_of_runs, endpoint=True)[run_num])
        if param == 'd':
            D[0] = camera["D"][0] * (1 + np.linspace(-settings["k1_maxoffset"], settings["k1_maxoffset"],
                                                     num=num_of_runs, endpoint=True)[run_num])
            D[1] = camera["D"][1] * (1 + np.linspace(-settings["k2_maxoffset"], settings["k2_maxoffset"],
                                                     num=num_of_runs, endpoint=True)[run_num])
            D[4] = camera["D"][4] * (1 + np.linspace(-settings["k3_maxoffset"], settings["k3_maxoffset"],
                                                     num=num_of_runs, endpoint=True)[run_num])
            D[2] = camera["D"][2] * (1 + np.linspace(-settings["p1_maxoffset"], settings["p1_maxoffset"],
                                                     num=num_of_runs, endpoint=True)[run_num])
            D[3] = camera["D"][3] * (1 + np.linspace(-settings["p2_maxoffset"], settings["p2_maxoffset"],
                                                     num=num_of_runs, endpoint=True)[run_num])

    return K, D


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


def process_correct_image(im, K, D):
    """
    Process raw sequence image with default parameters
    :param im: raw input image
    :param K: camera intrinsic matrix (3, 3)
    :param D: camera distortion parameters (1, 5)
    :return:
    """
    # Apply the rectification
    ir = ImageRectifier(im, K, D)
    rect_image, _ = ir.rectify(im)
    ir_correct = copy.deepcopy(ir)
    return ir_correct, rect_image


def process_perturbed_image(im, K, D, sampling_settings, ir_correct, output_im_file, diag_im_file):
    """
    Process raw sequence image with perturbed parameters
    :param im: raw input image
    :param K: camera intrinsic matrix (3, 3)
    :param D: camera distortion parameters (1, 5)
    :param sampling_settings: dictionary with settings for sampling images
    :param output_im_file: output image name
    :param ir_correct: correct rectified image struct
    :return: computed statistics of the perturbed rectification process
    """

    # Calculate the output shape:
    if "output_resolution" not in sampling_settings or sampling_settings["output_resolution"] == None or sampling_settings["output_resolution"] == "None":
        output_resolution = (im.shape[1], im.shape[0])
    else:
        output_resolution = (sampling_settings["output_resolution"][0], sampling_settings["output_resolution"][1])

    # Apply the rectification
    ir = ImageRectifier(im, K, D)
    rect_image, _ = ir.rectify(im)

    # Check if the ROI is valid
    if not (ir.validPixROI[0] > 0 and ir.validPixROI[1] > 0 and ir.validPixROI[2] > 0 and ir.validPixROI[3] > 0):
        if sampling_settings["loud_ugliness_guard"]:
            print("The original ROI is not valid (only zeros).")
            sys.stdout.flush()
            return False

    # Find how to crop the new image in order to use only the region of correct
    # rectification while maintaining the original aspect ratio
    try:
        target_aspect_ratio = float(rect_image.shape[0]) / float(rect_image.shape[1])
        validROI = {'x': (ir.validPixROI[0] + 2, ir.validPixROI[0] + ir.validPixROI[2] - 2),
                    'y': (ir.validPixROI[1] + 2, ir.validPixROI[1] + ir.validPixROI[3] - 2)}
        valid_aspect_ratio = float(validROI['y'][1] - validROI['y'][0]) / (validROI['x'][1] - validROI['x'][0])

        # If the image is taller than it should, cut its legs and head
        if valid_aspect_ratio > target_aspect_ratio:
            desired_number_of_rows = int(round(target_aspect_ratio * (validROI['x'][1] - validROI['x'][0])))
            cut_top = int(((validROI['y'][1] - validROI['y'][0]) - desired_number_of_rows) / 2)
            cropped_ROI = {'x': validROI['x'],
                           'y': (validROI['y'][0] + cut_top, validROI['y'][0] + cut_top + desired_number_of_rows)}
        elif valid_aspect_ratio < target_aspect_ratio:
            desired_number_of_cols = int(round((validROI['y'][1] - validROI['y'][0]) / target_aspect_ratio))
            cut_left = int(((validROI['x'][1] - validROI['x'][0]) - desired_number_of_cols) / 2)
            cropped_ROI = {'x': (validROI['x'][0] + cut_left, validROI['x'][0] + cut_left + desired_number_of_cols),
                           'y': validROI['y']}
    except:
        return False

    # [Ugliness guard] Check if the cropped ROI is valid
    if cropped_ROI['x'][1] - cropped_ROI['x'][0] < sampling_settings["min_cropped_width"] \
            or cropped_ROI['y'][1] - cropped_ROI['y'][0] < sampling_settings["min_cropped_height"]:
        if sampling_settings["loud_ugliness_guard"]:
            return False

    # Crop and resize the new image based on the previously calculated cropped_ROI
    cropped_aspect_ratio = float(cropped_ROI['x'][1] - cropped_ROI['x'][0]) / (
            cropped_ROI['y'][1] - cropped_ROI['y'][0])

    cropped_image = rect_image[cropped_ROI['y'][0]:cropped_ROI['y'][1], cropped_ROI['x'][0]:cropped_ROI['x'][1]]
    resized_image = cv2.resize(cropped_image, output_resolution, interpolation=cv2.INTER_LANCZOS4)

    # Calculate the cropped maps (need that for the px movement metric calculation)
    correct_map1 = copy.deepcopy(ir_correct.map1)
    correct_map2 = copy.deepcopy(ir_correct.map2)
    correct_map1 = correct_map1[cropped_ROI['y'][0]:cropped_ROI['y'][1], cropped_ROI['x'][0]:cropped_ROI['x'][1]]
    correct_map1 = cv2.resize(correct_map1, output_resolution, interpolation=cv2.INTER_LANCZOS4)
    correct_map2 = correct_map2[cropped_ROI['y'][0]:cropped_ROI['y'][1], cropped_ROI['x'][0]:cropped_ROI['x'][1]]
    correct_map2 = cv2.resize(correct_map2, output_resolution, interpolation=cv2.INTER_LANCZOS4)

    map1 = copy.deepcopy(ir.map1)
    map2 = copy.deepcopy(ir.map2)
    map1 = map1[cropped_ROI['y'][0]:cropped_ROI['y'][1], cropped_ROI['x'][0]:cropped_ROI['x'][1]]
    map1 = cv2.resize(map1, output_resolution, interpolation=cv2.INTER_LANCZOS4)
    map2 = map2[cropped_ROI['y'][0]:cropped_ROI['y'][1], cropped_ROI['x'][0]:cropped_ROI['x'][1]]
    map2 = cv2.resize(map2, output_resolution, interpolation=cv2.INTER_LANCZOS4)

    # Calculate the map difference statistics
    map_diff_stats = dist_map_diff_statistics(correct_map1, correct_map2, map1, map2, cropped_ROI)

    # Crop and resize the new image based on the prevoously calculated cropped_ROI
    scale_factor = float(correct_map1.shape[0]) / correct_map1.shape[0]
    validROI['x'] = (int(scale_factor * validROI['x'][0]), int(scale_factor * validROI['x'][1]))
    validROI['y'] = (int(scale_factor * validROI['y'][0]), int(scale_factor * validROI['y'][1]))
    cropped_ROI['x'] = (int(scale_factor * cropped_ROI['x'][0]), int(scale_factor * cropped_ROI['x'][1]))
    cropped_ROI['y'] = (int(scale_factor * cropped_ROI['y'][0]), int(scale_factor * cropped_ROI['y'][1]))

    # [Ugliness guard] Check if the cropped ROI is valid and if the average pixel movement is not too high
    if map_diff_stats["average"] > sampling_settings["max_px_movement"]:
        if sampling_settings["loud_ugliness_guard"]:
            return False

    # Show diagnostic information if required:
    if sampling_settings["diagnostic_images"] == 1:
        diag_image = create_diagnostic_image(ir, validROI, cropped_ROI,
                                             cv2.resize(rect_image, output_resolution, interpolation=cv2.INTER_LANCZOS4),
                                             map_diff_stats, write_text=True)
        cv2.imwrite(diag_im_file, diag_image)

    # Save the new image
    cv2.imwrite(output_im_file, resized_image)

    return map_diff_stats


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


def run_orbslam(orbslam_applet, vocab_path, orbslam_settings_path, dataset_folder, output_file):
    # define the command to run
    command = [orbslam_applet,
               '--sequenceType', 'kitti',
               '--vocabularyPath', vocab_path,
               '--settingsPath', orbslam_settings_path,
               '--sequencePath', dataset_folder,
               '--outputFilePath', output_file]
    # execute the command
    process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=False)
    if process.wait():
        print("Releasing command!")


def perform_traj_evaluation(results_dir, run_name, gt_traj_file, estimated_traj_file, align_type='sim3',
                            align_num_frames=-1, plot_results=True, boxplot_distances=[]):
    """
    Uses UZH Trajectory Evaluation toolbox to evaluate accuracy of the algorithm
    :param results_dir: directory to save the results in
    :param run_name: name of the experiment run
    :param gt_traj_file: Stamped groundtruth poses in TUM format
    :param estimated_traj_file: Stamped estimated poses in TUM format
    :param align_type: alignment type to perform
    :param align_num_frames: number of frames to align
    :return: dictionary with the evaluated statistics
    """
    # create directory to save evaluation results
    os.makedirs(results_dir, exist_ok=True)
    # compute trajectory errors
    traj = Trajectory(results_dir, run_name, gt_traj_file, estimated_traj_file, align_type=align_type,
                      align_num_frames=align_num_frames, preset_boxplot_distances=boxplot_distances)
    traj.compute_absolute_error()
    traj.compute_relative_errors()
    traj.cache_current_error()
    traj.write_errors_to_yaml()
    if plot_results:
        traj.plot_trajectory('.png')
        traj.plot_rel_traj_error('.png')

    error_dict = traj.get_abs_errors_stats()
    error_dict.update(traj.get_rel_errors_stats())

    return error_dict


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    """
    Parses CLI arguments applicable for this helper script
    """
    # Create parser instance
    parser = argparse.ArgumentParser(description='Evaluate ORBSLAM with different camera parameters')
    # Define arguments
    parser.add_argument('--orbslam_root', '-b', metavar='', dest='orbslam_root', help="path to ORBSLAM2 directory")
    parser.add_argument('--dataset_folder', '-k', metavar='', dest='dataset_folder', help='path to a directory containing KITTI sequence dataset')
    parser.add_argument('--sequence_name', '-e', metavar='', dest='sequence_name', help='name of the KITTI sequence')
    parser.add_argument('--sampling_settings', '-s', metavar='', dest='sampling_settings', help='path to a file with the sampling settings',
                        default="config/sampling_settings.yaml")
    parser.add_argument('--num_of_runs', '-n', metavar='', type=int, dest='num_of_runs', help='number of runs to perform', default=6)
    parser.add_argument('--output_dir', '-o', metavar='', dest='output_dir', help='path of the directory to save experiment resutls in',
                        default='eval')
    parser.add_argument('--experiment_type', '-t', metavar='', dest='experiment_type',
                        help='type of experiment to perform: random/range', default='range')
    parser.add_argument('--perturb_focal', '-f', metavar='', type=str2bool, help='set to true to perturb focal length (y/n)',
                        default='y')
    parser.add_argument('--perturb_principle_point', '-c', metavar='', type=str2bool, help='set to true to perturb principle point (y/n)',
                        default='y')
    parser.add_argument('--perturb_distortion', '-d', metavar='', type=str2bool, help='set to true to perturb distortion parameters (y/n)',
                        default='n')

    # Retrieve arguments
    args = parser.parse_args()
    return args

def main(args):
    """
    Script to evaluate performance of visual odometry algorithms against miscalibrations.
    The script does the following:
    1. Perturb the known camera parameters
    2. Generate the configuration file which is used to run OrbSLAM
    3. Perform VO-evaluation and plots results

    All outputs are stored in the "eval" directory.
    """

    ################################################
    ################## PARAMETERS ##################
    ################################################

    """
    Define parameters to run the OrbSLAM executable.
    1. Clone the repository: https://github.com/Mayankm96/ORB_SLAM2/tree/feature/projects 
    2. Build it by following the instructions present in the README 
    """
    # specify the root path to the OrbSLAM directory
    ORBSLAM_ROOT = args.orbslam_root
    # path to the vocabulary used in OrbSLAM
    vocab_path = os.path.join(ORBSLAM_ROOT, 'Vocabulary/ORBvoc.txt')
    # path to the executable file which is used to run the experiments
    orbslam_applet = os.path.join(ORBSLAM_ROOT, 'Projects/PLR_Evaluation/mono_plr')
    # sample ORBSLAM settings file which is used as a template to auto-generate settings for the expriments
    orbslam_config_file = 'config/ORBSLAM_TUM1.yaml'

    """
    We use KITTI dataset in the experiment.
    1. Download the sequence: http://www.cvlibs.net/datasets/kitti/eval_odometry.php
    2. Paste the following files into "KITTI_parameters/<seq-name>" directory 
        a. Paste the calibration file in  as 'camera.yaml'
        b. 'gt_kitti.txt' and 'times.txt' files corresponding to the sequence
    """
    # specify the name of the KITTI sequence
    KITTI_sequence = args.sequence_name
    # specify the directory containing the raw data
    dataset_folder = args.dataset_folder
    # path to camera calibration file
    camera_config_file = 'KITTI_parameters/' + KITTI_sequence + '/camera.yaml'
    # path to times.txt file
    timestamps_file = 'KITTI_parameters/' + KITTI_sequence + '/times.txt'
    # path to groundtruth file
    traj_gt_file = 'KITTI_parameters/' + KITTI_sequence + '/gt_kitti.txt'

    """
    The perturbation settings for camera calibration
    """
    # specify path to dataset generation configuration
    sampling_config_file = args.sampling_settings
    # specify the number of runs to perform
    num_of_runs = args.num_of_runs
    # specify the experiment type: 'random_perturbations' or 'range_perturbations'
    assert args.experiment_type in ['random', 'range']
    experiment_type = args.experiment_type + '_perturbations'
    # sepcify which parameters to perturb: 'f' for focal length, 'c' for principle point, 'd' for distortion parameters
    perturb_parameters = []
    if args.perturb_focal:
        perturb_parameters = perturb_parameters + ['f']
    if args.perturb_principle_point:
        perturb_parameters = perturb_parameters + ['c']
    if args.perturb_distortion:
        perturb_parameters = perturb_parameters + ['d']

    """
    Output directory to save all data related to the experiment
    """
    # path to the output directory
    output_folder = args.output_dir

    # trajectory evaluation
    boxplot_distances = [100, 200, 300, 400, 500]  # preset relative error trajectory distances in meters
    align_type = 'sim3'     # choose from: ['posyaw', 'sim3', 'se3', 'none']
    align_num_frames = -1  # all (note: this is only used for calculating absolute trajectory errors)

    ################################################
    ##################### DISP #####################
    ################################################

    print('-- Dataset Folder: ', dataset_folder)
    print('-- Number of Runs: ', num_of_runs)
    print('-- Perturbing Parameters: ', perturb_parameters)
    print('-- Experiment Type: ', experiment_type)
    print('-- Output Folder: ', output_folder)
    print('----------------------------------------')

    ################################################
    ##################### MAIN #####################
    ################################################

    # convert boxplot distances list to float
    boxplot_distances = [float(i) for i in boxplot_distances]
    ## construct output directory name
    experiment_name = experiment_type + '_'
    for param in perturb_parameters:
        experiment_name = experiment_name + param
    output_folder = os.path.join(output_folder, KITTI_sequence, experiment_name)

    ## combine KITTI poses and timestamps files to a TUM trajectory file
    # path to stamped trajectory groundtruth file
    stamped_traj_gt_file = 'KITTI_parameters/' + KITTI_sequence + '/stamped_groundtruth.txt'
    # create stamped trajectory groundtruth file if it does not exist
    if not os.path.exists(stamped_traj_gt_file):
        # create concatenated matrix with timestamps and pose details
        trajectory = kitti_poses_and_timestamps_to_trajectory(traj_gt_file, timestamps_file)
        # write the trajectory into output file
        file_interface.write_tum_trajectory_file(stamped_traj_gt_file, trajectory)

    ## read data from the configuration files
    with open(camera_config_file, 'r') as stream:
        camera = yaml.safe_load(stream)

    with open(orbslam_config_file, 'r') as stream:
        orbslam_settings = yaml.safe_load(stream)

    with open(sampling_config_file, 'r') as stream:
        sampling_settings = yaml.safe_load(stream)

    ## create pandas dataframe to save all the results
    dataframe = create_summary_dataframe(num_of_runs, boxplot_distances)

    ## perform runs for the given experiment type
    for i in range(num_of_runs):

        # create directory to save data from the run
        run_name = 'run_%d' % i
        run_output_folder = os.path.join(output_folder, run_name)
        os.makedirs(run_output_folder, exist_ok=True)

        # read the first raw image in the sequence
        image_dir = os.path.join(dataset_folder, 'image_00', 'data')
        im_file = sorted(os.listdir(image_dir))[0]
        input_im_file = os.path.join(image_dir, im_file)
        im = cv2.imread(input_im_file)

        """
        The following information is stored from each run:
        1. an example raw image from the dataset
        2. rectified image from the perturbed calibration parameters
        3. diagnostoc image with useful information about the perturbation
        4. configuration file for orbslam with the perturbed calibration settings
        5. keyframe trajectory outputted from running OrbSLAM on that sequence
        6. evaluation statistics on the estimated trajectory
        """
        raw_im_file = os.path.join(run_output_folder, 'raw_' + im_file)
        perturbed_im_file = os.path.join(run_output_folder, 'perturbed_' + im_file)
        diagostic_im_file = os.path.join(run_output_folder, 'diagnostic_' + im_file)
        output_config_file = os.path.join(run_output_folder, 'config.yaml')
        traj_output_file = os.path.join(run_output_folder, 'stamped_traj_estimate.txt')

        # generate map for correctly rectified image
        K, D = get_default_camera_matrices(camera)
        ir_correct, _ = process_correct_image(im, K, D)

        if experiment_type == 'random_perturbations':
            # perturbs parameters by uniformly sampling frame a range
            # first run is done with correct calibration parameters
            if i != 0:
                K, D = get_random_perturbed_camera_matrices(camera, sampling_settings, perturb_parameters)
        elif experiment_type == 'range_perturbations':
            # perturbs parameters in a range
            K, D = get_range_perturbed_camera_matrices(camera, sampling_settings, perturb_parameters, num_of_runs, i)

        print("Running number: ", i)
        print("Intrinsic Matrix: ", K)
        print("Distortion Coefficients: ", D)

        # store one sample of rectification output with generated calibration settings
        cv2.imwrite(raw_im_file, im)
        map_diff_stats = process_perturbed_image(im, K, D, sampling_settings, ir_correct, perturbed_im_file,
                                                 diagostic_im_file)

        # update dataframe with generated statistics on rectification
        dataframe = update_dataframe_with_rectification_stats(dataframe, run_name, K, D, map_diff_stats)

        # generate configuration file for the perturbed camera settings
        orbslam_settings = modify_orbslam_settings(orbslam_settings, K, D)
        with open(output_config_file, 'w') as outfile:
            outfile.write('%YAML:1.0\n')
            yaml.dump(orbslam_settings, outfile)

        # run OrbSLAM
        if not os.path.exists(traj_output_file):
            run_orbslam(orbslam_applet=orbslam_applet, vocab_path=vocab_path,
                        orbslam_settings_path=output_config_file,
                        output_file=traj_output_file,
                        dataset_folder=dataset_folder)

        # evaluate run
        error_stats_dict = perform_traj_evaluation(os.path.join(run_output_folder, 'evaluation'), run_name,
                                                   stamped_traj_gt_file, traj_output_file, align_type=align_type,
                                                   align_num_frames=align_num_frames,
                                                   boxplot_distances=boxplot_distances)
        for key, value in error_stats_dict.items():
            dataframe.at[run_name, key] = value

        print('-----------------------------------')

        # save the created data frame after every finite number of runs (checkpoints)
        if i % 5 == 0:
            dataframe.to_csv(os.path.join(output_folder, 'summary.csv'))

    # save the created dataframe
    dataframe.to_csv(os.path.join(output_folder, 'summary.csv'))


if __name__ == '__main__':
    ARGS = parse_args()
    main(ARGS)
