#!/usr/bin/env python

import os
import ruamel
from ruamel import yaml
import subprocess
import numpy as np
import cv2
import argparse

# Evo's Trajectory Toolbox
from evo.tools import file_interface

# Aleks's image rectification toolbox
import carnivalmirror as cm

# customized library toolboxes
from library.trajectory import Trajectory
from library.rangesampler import RangeSampler
from library.helper import str2bool, create_summary_dataframe, update_dataframe_with_rectification_stats, \
    kitti_poses_and_timestamps_to_trajectory, get_range_perturbed_camera_parameters, modify_orbslam_settings


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


def parse_args():
    """
    Parses CLI arguments applicable for this helper script
    """
    # Create parser instance
    parser = argparse.ArgumentParser(description='Evaluate ORBSLAM with different camera parameters')
    # Define arguments
    parser.add_argument('--orbslam_root', '-b', metavar='', dest='orbslam_root', help="path to ORBSLAM2 directory")
    parser.add_argument('--dataset_folder', '-k', metavar='', dest='dataset_folder',
                        help='path to a directory containing KITTI sequence dataset')
    parser.add_argument('--sequence_name', '-e', metavar='', dest='sequence_name', help='name of the KITTI sequence')
    parser.add_argument('--sampling_settings', '-s', metavar='', dest='sampling_settings',
                        help='path to a file with the sampling settings',
                        default="config/sampling_settings.yaml")
    parser.add_argument('--num_of_runs', '-n', metavar='', type=int, dest='num_of_runs',
                        help='number of runs to perform', default=6)
    parser.add_argument('--output_dir', '-o', metavar='', dest='output_dir',
                        help='path of the directory to save experiment resutls in',
                        default='eval')
    parser.add_argument('--experiment_type', '-t', metavar='', dest='experiment_type',
                        help='type of experiment to perform: random/range', default='range')
    parser.add_argument('--perturb_focal', '-f', metavar='', type=str2bool,
                        help='set to true to perturb focal length (y/n)',
                        default='y')
    parser.add_argument('--perturb_principle_point', '-c', metavar='', type=str2bool,
                        help='set to true to perturb principle point (y/n)',
                        default='y')
    parser.add_argument('--perturb_distortion', '-d', metavar='', type=str2bool,
                        help='set to true to perturb distortion parameters (y/n)',
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
    # path to the directory with the images
    image_dir = os.path.join(dataset_folder, 'image_00', 'data')
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
    align_type = 'sim3'  # choose from: ['posyaw', 'sim3', 'se3', 'none']
    align_num_frames = -1  # all (note: this is only used for calculating absolute trajectory errors)

    ################################################
    ##################### DISP #####################
    ################################################

    print('-- Dataset Folder: ', dataset_folder)
    print('-- Output Folder: ', output_folder)
    print('-- Number of Runs: ', num_of_runs)
    print('-- Experiment | Type: ', experiment_type)
    print('-- Experiment | Perturbing Parameters: ', perturb_parameters)
    print('-- Trajectory Evaluation | Align Type: ', align_type)
    print('-- Trajectory Evaluation | Align Number of Frames: ', align_num_frames)
    print('----------------------------------------')

    ##########################################################
    ##################### INITIALIZATION #####################
    ##########################################################

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

    # retrieve the correct camera calibration parameters
    K = np.array(camera["K"]).reshape((3, 3))
    D = np.array(camera["D"])

    # define the sampling ranges
    sampling_ranges = get_range_perturbed_camera_parameters(K, D, sampling_settings, perturb_parameters)

    # read the first raw image in the sequence (to get the shape)
    seq_first_im_file = sorted(os.listdir(image_dir))[0]
    seq_first_image = cv2.imread(os.path.join(image_dir, seq_first_im_file))
    # extract the size of the image
    height, width = seq_first_image.shape[:2]

    # Create a Calibration object for the correct calibration. That will be used
    # as a reference calibration when calculating APPD values
    cam_reference = cm.Calibration(K=K, D=D, width=width, height=height)

    # Create a Sampler to sample parameters from it
    if experiment_type == 'random_perturbations':
        sampler = cm.UniformAPPDSampler(ranges=sampling_ranges, cal_height=height, cal_width=width, reference=cam_reference, width=width, height=height)
    else:
        sampler = RangeSampler(ranges=sampling_ranges, cal_height=height, cal_width=width, num_of_runs=num_of_runs)

    ################################################
    ##################### MAIN #####################
    ################################################

    ## perform runs for the given experiment type
    for i in range(num_of_runs):

        # create directory to save data from the run
        run_name = 'run_%d' % i
        run_output_folder = os.path.join(output_folder, run_name)
        os.makedirs(run_output_folder, exist_ok=True)

        """
        The following information is stored from each run:
        1. an example raw image from the dataset
        2. rectified image from the perturbed calibration parameters
        3. diagnostoc image with useful information about the perturbation
        4. configuration file for orbslam with the perturbed calibration settings
        5. keyframe trajectory outputted from running OrbSLAM on that sequence
        6. evaluation statistics on the estimated trajectory
        """
        raw_im_file = os.path.join(run_output_folder, 'raw_' + seq_first_im_file)
        perturbed_im_file = os.path.join(run_output_folder, 'perturbed_' + seq_first_im_file)
        diagostic_im_file = os.path.join(run_output_folder, 'diagnostic_' + seq_first_im_file)
        output_config_file = os.path.join(run_output_folder, 'config.yaml')
        traj_output_file = os.path.join(run_output_folder, 'stamped_traj_estimate.txt')

        # first run is done with correct calibration parameters for
        if i == 0:
            cam_run = cam_reference
        else:
            # get parameters from the sampler
            cam_run = sampler.next()

        print("Running number: ", i + 1)
        print("Intrinsic Matrix: ", cam_run.get_K(height))
        print("Distortion Coefficients: ", cam_run.get_D())

        K = cam_run.get_K(height)
        D = cam_run.get_D()
        # Calculate the APPD value for these parameters
        while True:
            try:
                appd, diff_map = cam_run.appd(reference=cam_reference, width=width, height=height,
                                              return_diff_map=True, normalized=True)
                break
            except RuntimeError as e:
                continue

        # (Mis)rectify the image
        misrect_image = cam_run.rectify(seq_first_image, result_width=width, result_height=height, mode='preserving')

        # Save reference image ang with the rectified image
        cv2.imwrite(raw_im_file, seq_first_image)
        cv2.imwrite(perturbed_im_file, misrect_image)
        cv2.imwrite(diagostic_im_file, diff_map)

        # update dataframe with generated statistics on rectification
        dataframe = update_dataframe_with_rectification_stats(dataframe, run_name, K, D, appd)

        # generate configuration file for the perturbed camera settings
        orbslam_settings = modify_orbslam_settings(orbslam_settings, K, D)
        with open(output_config_file, 'w') as outfile:
            outfile.write('%YAML:1.0\n')
            yaml.dump(orbslam_settings, outfile, Dumper=ruamel.yaml.RoundTripDumper)

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
    # Read CLI Arguments
    ARGS = parse_args()
    # Run main()
    main(ARGS)
