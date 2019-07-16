from __future__ import print_function
import os
import time
import glob
import sys
import threading
from copy import deepcopy

import numpy as np
import pandas as pd
import cv2
from sklearn.preprocessing import StandardScaler

import carnivalmirror as cm


class Dataset(object):
    def __init__(self, index_csv, selector='',  num_of_samples=-1,
                 internal_shuffle=False, n_jobs=1, verbose=0):
        """
        NOTE:  - For larger datasets only process the paths and load the data later in the get_outputs() function.
               - selector should be formatted like: "image03+2011_09_26,image03+2011_09_28" to use all the folders with
                    tags image03 and 2011_09_26, and with tags image03 and 2011_09_28 (no spaces!)
        """

        self.use_scaler = False
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.resolution_reduction_factor = 4
        self.label_scale_factor = 100

        self._lock_appd = threading.Lock()

        folders = pd.read_csv(index_csv)
        # assumption: the index csv file is inside the folder where the folders with images are
        dir_path = os.path.dirname(os.path.abspath(index_csv))
        # apply the selector
        folders['selector'] = folders['tags'].apply(lambda tags: self.apply_selector(tags, selector))
        folders = folders[folders['selector']==True]


        # group the different calibrations (later we need one sampler per group)
        folders['cal_group'] = folders.groupby(['width', 'height', 'fx', 'cx', 'fy', 'cy', 'k1', 'k2',
                                                'p1', 'p2', 'k3']).ngroup()
        self.cal_groups = dict()
        for cal in folders['cal_group'].unique():
            self.cal_groups[cal] = folders.loc[[folders['cal_group'].eq(cal).idxmax()],
                                               ['width', 'height', 'fx', 'cx', 'fy', 'cy', 'k1', 'k2', 'p1', 'p2', 'k3']]

        # load the images from every folder
        data = pd.DataFrame(columns=['image_name', 'cal_group'])
        for folder_idx, folder in folders.iterrows():
            image_names = glob.glob(os.path.join(dir_path, folder['path_to_dir'])+'/*')
            n = len(image_names)
            new_df = pd.DataFrame({'image_name': image_names, 'cal_group': [folder['cal_group']]*n})
            data = pd.concat([data, new_df], sort=False)

        if self.verbose > 0:
            print("%d images found in %d folders grouped in %d groups from index file %s, when applying selector '%s'." %
                  (data.shape[0], folders.shape[0], len(folders['cal_group'].unique()), index_csv, selector))

        # Initialize the samplers:
        samplers_init_start = time.time()
        self.initialize_samplers()
        if self.verbose > 0:
            print("The samplers were initialized in %.02f sec." % (time.time()-samplers_init_start))

        # internal shuffle if true
        if internal_shuffle:
            data = data.sample(frac=1).reset_index(drop=True)
        # retrieve only the amount of samples as requested
        if num_of_samples <= 0:
            self.n_samples = data.shape[0]
        else:
            self.n_samples = num_of_samples
        data = data.head(self.n_samples)
        # read data from the data-frame
        self.image_paths = data['image_name'].tolist()
        self.cal_group_assignment = data['cal_group'].tolist()

        # Get final output shape.
        output, _ = self.get_outputs([0])
        self.shape = output[0].shape

    def train_scaler(self, remove_mean=False, remove_std=False,
                     scaler_batch_size=32, scaler_epochs=256):
        assert(remove_mean or remove_std)

        # Mean and std scaling.
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler(with_mean=remove_mean, with_std=remove_std)

        if self.verbose > 0:
            print_width = 10
            sys.stdout.write("Training scaler." + ' ' * print_width)

        # Arbitrary sampling of data to determine mean.
        # Hacky, but much faster than iterating over the whole data.
        for epoch in range(scaler_epochs):
            ids_batch = np.random.choice(self.n_samples, scaler_batch_size)
            images_batch, _ = self.get_outputs(ids_batch)
            images_batch = np.reshape(images_batch, (images_batch.shape[0], -1))
            self.scaler.partial_fit(images_batch)

            if self.verbose > 0:
                sys.stdout.write('\b' * print_width)
                sys.stdout.write(' %4d/%4d' % (epoch+1, scaler_epochs))
                sys.stdout.flush()

        if self.verbose > 0:
            print()

        self.use_scaler = True

    def set_scaler(self, scaler):
        self.use_scaler = True
        self.scaler = scaler

    def get_scaler(self):
        return deepcopy(self.scaler)

    def get_outputs(self, ids):
        image_outputs = []
        label_outputs = []

        # First load images
        images = []
        cal_infos = []
        for id in ids:
            # Load the image.
            image = cv2.imread(self.image_paths[id], cv2.IMREAD_COLOR)
            assert(image.shape[-1] == 3)
            images.append(image)

            # Load calibration info
            cal_group = self.cal_group_assignment[id]
            cal_width = self.cal_groups[cal_group]['width'].values[0]
            cal_height = self.cal_groups[cal_group]['height'].values[0]
            target_width = cal_width / self.resolution_reduction_factor
            target_height = cal_height / self.resolution_reduction_factor
            cal_infos.append((cal_group, cal_width, cal_height, target_width, target_height))

        miscals = []
        self._lock_appd.acquire()
        for cal_info in cal_infos:
            cal_group, cal_width, cal_height, target_width, target_height = cal_info

            # Sample a miscalibration, apply it, and calculate the respective APPD
            miscal = self.samplers[cal_group].next()
            appd = miscal.appd(reference=self.samplers[cal_group].reference,
                               width=target_width, height=target_width, normalized=True,
                               map_width=target_width, map_height=target_height)
            miscals.append(miscal)

            label = appd * self.label_scale_factor
            label_outputs.append(label)
        self._lock_appd.release()

        # Form image batch from raw data.
        for cal_info, miscal in zip(cal_infos, miscals):
            _, _, _, target_width, target_height = cal_info
            image = miscal.rectify(image,
                                   result_width=target_width,
                                   result_height=target_height,
                                   mode='preserving')

            # Collect batch data.
            image_outputs.append(image)

        # Convert to numpy array.
        image_outputs = np.array(image_outputs).astype(np.float)
        label_outputs = np.array(label_outputs).astype(np.float)

        # Pass batch thorough scaler to remove mean and/or std.
        if self.use_scaler:
            shape = image_outputs.shape
            image_outputs = np.reshape(image_outputs, (image_outputs.shape[0], -1))
            image_outputs = self.scaler.transform(image_outputs)
            image_outputs = np.reshape(image_outputs, shape)

        return image_outputs, label_outputs

    def apply_selector(self, tags, selector):
        """Checks if all the tags in 'selector' (comma,no-space separated) are in 'tags'"""
        for selector_group in selector.split(','):
            found_matching_tag_group = True
            for tag in selector_group.split('+'):
                if tag not in tags.split(','):
                    found_matching_tag_group = False
            if found_matching_tag_group:
                return True
        if len(selector.split(','))==0:
            return True
        else:
            return False

    def initialize_samplers(self):
        """Initialize a separate sampler for each group of calibrations"""

        self.samplers = dict()

        n_jobs_per_group = max(1, int(self.n_jobs/len(self.cal_groups)))

        for cal_group in self.cal_groups:
            cg = self.cal_groups[cal_group]

            # Calculate the rescaled output resolution
            output_width = int(self.cal_groups[cal_group]['width'] / self.resolution_reduction_factor)
            output_height = int(self.cal_groups[cal_group]['height'] / self.resolution_reduction_factor)

            # Create the reference (correct calibration)
            reference = cm.Calibration(K=[cg['fx'].values[0], cg['fy'].values[0], cg['cx'].values[0], cg['cy'].values[0]],
                                      D=[cg['k1'].values[0], cg['k2'].values[0], cg['p1'].values[0], cg['p2'].values[0], cg['k3'].values[0]],
                                      width=cg['width'].values[0], height=cg['height'].values[0])


            # Establish the perturbation ranges
            # [!!!] This needs to be fine tuned depending on the dataset used
            ranges = {'fx': (0.95 * cg['fx'].values[0], 1.20 * cg['fx'].values[0]),
                      'fy': (0.95 * cg['fy'].values[0], 1.20 * cg['fy'].values[0]),
                      'cx': (0.95 * cg['cx'].values[0], 1.05 * cg['cx'].values[0]),
                      'cy': (0.95 * cg['cy'].values[0], 1.05 * cg['cy'].values[0]),
                      'k1': (0.85 * cg['k1'].values[0], 1.15 * cg['k1'].values[0]),
                      'k2': (0.85 * cg['k2'].values[0], 1.15 * cg['k2'].values[0]),
                      'p1': (0.85 * cg['p1'].values[0], 1.15 * cg['p1'].values[0]),
                      'p2': (0.85 * cg['p2'].values[0], 1.15 * cg['p2'].values[0]),
                      'k3': (0.85 * cg['k3'].values[0], 1.15 * cg['k3'].values[0])}

            # Initialize the sampler
            sampler = cm.UniformAPPDSampler(ranges=ranges, cal_width=cg['width'].values[0], cal_height=cg['height'].values[0],
                                            reference=reference, temperature=5, appd_range_dicovery_samples=2000,
                                            appd_range_bins=20, init_jobs=self.n_jobs,
                                            width=output_width, height=output_height,
                                            min_cropped_size=(int(output_width / 1.5), int(output_height / 1.5)))
            #sampler = cm.ParallelBufferedSampler(sampler=sampler, buffer_size=self.n_jobs*64, n_jobs=n_jobs_per_group)
            self.samplers[cal_group] = sampler

    def stop(self):
        """This should be ran when we want to stop generating data samples and before exiting the script.
           It shuts off the background threads gracefully"""

        for cal_group in self.samplers:
            try:
                self.samplers[cal_group].stop()
            except:
                pass
