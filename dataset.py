import numpy as np
import pandas as pd
import time
import os
import cv2


class Dataset(object):
    def __init__(self, path, remove_mean=False, remove_std=False,
                 scaler_batch_size=32, num_of_samples=-1, internal_shuffle=False, verbose=0):
        """
        NOTE: For larger datasets only process the paths and load the data later in the get_outputs() function.
        """

        data = pd.read_csv(path)
        # assumption: csv file is present inside the folder where the images are
        dir_path = os.path.dirname(os.path.abspath(path))
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
        self.image_paths = data['image_name'].apply(lambda filename: os.path.join(dir_path, filename) + '.png').tolist()
        self.labels = data['d_avg'].to_numpy().astype(np.float)
        self.use_scaler = False
        self.verbose = verbose

        # Mean and std scaling.
        if remove_mean or remove_std:
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler(with_mean=remove_mean, with_std=remove_std)

            # Arbitrary sampling of data to determine mean.
            # Hacky, but much faster than iterating over the whole data.
            for epoch in range(128):
                ids_batch = np.random.choice(self.n_samples, scaler_batch_size)
                images_batch, _ = self.get_outputs(ids_batch)
                images_batch = np.reshape(images_batch, (images_batch.shape[0], -1))
                self.scaler.partial_fit(images_batch)
            self.use_scaler = True

        # Get final output shape.
        output, _ = self.get_outputs([0])
        self.shape = output[0].shape

    def get_outputs(self, ids):
        image_outputs = []
        label_outputs = []

        preprocess_start = time.time()

        # Form image batch from raw data.
        for id in ids:
            # Load data.
            image = cv2.imread(self.image_paths[id], cv2.IMREAD_COLOR)
            label = self.labels[id]

            # Convert image from BGRA/BGR to RGB
            if image.shape[-1] == 4:
                cv2.cvtColor(image, cv2.COLOR_BGRA2RGB, image)
            else:
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB, image)
            # Convert image to float matrix
            image = image.astype(np.float)

            # Image augmentation.
            # Cropping to remove car from AppoloScape dataset
            image = image[0:770, :, :]

            # Collect batch data.
            image_outputs.append(image)
            label_outputs.append(label)

        # Convert to numpy array.
        image_outputs = np.array(image_outputs).astype(np.float)
        label_outputs = np.array(label_outputs).astype(np.float)

        # Pass batch thorugh scaler to remove mean and/or std.
        if self.use_scaler:
            shape = image_outputs.shape
            image_outputs = np.reshape(image_outputs, (image_outputs.shape[0], -1))
            image_outputs = self.scaler.transform(image_outputs)
            image_outputs = np.reshape(image_outputs, shape)

        self.time_preprocess = time.time() - preprocess_start

        return image_outputs, label_outputs
