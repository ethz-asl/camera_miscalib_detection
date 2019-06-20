import numpy as np
import pandas as pd
import time


class Dataset(object):
    def __init__(self, path, remove_mean=False, remove_std=False,
                 scaler_batch_size=32, verbose=0):
        # Dataset options. For larger datasets only process the paths and load
        # the data later in the get_outputs() function
        data = pd.read_csv(path).to_numpy()
        self.images = data[:, 1:].astype(np.float).reshape((-1, 28, 28))
        self.labels = data[:, 0].astype(np.float)
        self.n_samples = self.labels.shape[0]
        self.use_scaler = False
        self.verbose = verbose

        # Mean and std scaling.
        if remove_mean or remove_std:
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler(with_mean=remove_mean,
                                         with_std=remove_std)

            # Arbitrary sampling of data to determine mean.
            # Hacky, but much faster than iterating over the whole data.
            for epoch in range(128):
                ids_batch = np.random.choice(self.n_samples, scaler_batch_size)
                images_batch, _ = self.get_outputs(ids_batch)
                images_batch = np.reshape(images_batch,
                                          (images_batch.shape[0], -1))
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
            image = self.images[id].copy()
            label = self.labels[id]

            # Image augmentation.
            # ...

            # Collect batch data.
            image_outputs.append(image)
            label_outputs.append(label)

        # Convert to numpy array.
        image_outputs = np.array(image_outputs)
        label_outputs = np.array(label_outputs).astype(np.int)

        # Pass batch thorugh scaler to remove mean and/or std.
        if self.use_scaler:
            shape = image_outputs.shape
            image_outputs = np.reshape(image_outputs,
                                       (image_outputs.shape[0], -1))
            image_outputs = self.scaler.transform(image_outputs)
            image_outputs = np.reshape(image_outputs, shape)

        self.time_preprocess = time.time() - preprocess_start

        return image_outputs, label_outputs
