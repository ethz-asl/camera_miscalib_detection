import numpy as np

def to_onehot(y, n_classes):
    y_onehot = np.zeros((len(y), n_classes))
    for i, cls in enumerate(y):
        y_onehot[i, cls] = 1

    return y_onehot

class Generator(object):
    def __init__(self, dataset, ids, n_classes, batch_size=16, shuffle=False,
            verbose=0):
        self.dataset = dataset
        self.ids = np.array(ids)
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.verbose = verbose

        self.n_samples = self.ids.size
        self.n_batches = int(np.ceil(float(self.n_samples)/self.batch_size))

        self._i = 0

    def __iter__(self):
        return self

    def next(self):
        if self.shuffle and self._i == 0:
            np.random.shuffle(self.ids)

        batch_ids = self.ids[self._i:self._i+self.batch_size]

        self._i = self._i + self.batch_size
        if self._i >= self.n_samples:
            self._i = 0

        images_batch, labels_batch = self.dataset.get_outputs(batch_ids)

        # For greyscale images add one more channel.
        images_batch = images_batch[:, :, :, None]

        # Transform classes to onehot encoding.
        labels_batch = to_onehot(labels_batch, self.n_classes)

        return images_batch, labels_batch
