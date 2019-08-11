from __future__ import print_function
import numpy as np
import threading
try:
    import Queue as queue
except:
    import queue
import sys
import time

class Generator(object):
    def __init__(self, dataset, ids, batch_size=16, shuffle=False,
                 buffer_size=32, verbose=0):
        self.dataset = dataset
        self.ids = np.array(ids)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.verbose = verbose

        self.n_samples = self.ids.size
        self.n_batches = int(np.ceil(float(self.n_samples) / self.batch_size))
        self.buffer_size = buffer_size

        self._i = 0
        self._buffer = queue.Queue()

        procs = []
        for i in range(self.buffer_size):
            procs.append(self._buffer_next())

        if self.verbose > 0:
            sys.stdout.write("Filling generator buffer. ")
            sys.stdout.flush()

        for proc in procs:
            proc.join()

        if self.verbose > 0:
            print("Done")

    def _buffer_next(self):
        if self.shuffle and self._i == 0:
            np.random.shuffle(self.ids)

        batch_ids = self.ids[self._i:self._i + self.batch_size]

        self._i = self._i + self.batch_size
        if self._i >= self.n_samples:
            self._i = 0

        proc = threading.Thread(target=self._buffer_next_worker,
                args=(batch_ids,))
        proc.start()

        return proc

    def _buffer_next_worker(self, batch_ids):
        images_batch, labels_batch = self.dataset.get_outputs(batch_ids)
        self._buffer.put([images_batch, labels_batch])

    def __iter__(self):
        return self

    def next(self):
        t=time.time()
        images_batch, labels_batch = self._buffer.get(block=True)
        #if time.time()-t > 0.05:
        #    print("BUFFER TIME: ", time.time()-t)
        self._buffer_next()
        return images_batch, labels_batch
