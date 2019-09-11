from __future__ import print_function
import numpy as np
import os
import sys
import time
import pickle

# Parse command line arguments.
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('index')
parser.add_argument('test_selector')
parser.add_argument('-n_test_samples', type=int, default=-1)
parser.add_argument('-batch_size', type=int, default=32)
parser.add_argument('-buffer_size', type=int, default=32)
parser.add_argument('-model_path', default='models/test_model/')
parser.add_argument('-model_name', default='model')
parser.add_argument('-v', type=int, default=0)
parser.add_argument('-njobs', type=int, default=8)

args = parser.parse_args()

# Load the dataset.
from dataset import Dataset

dataset_test = Dataset(args.index, selector=args.test_selector, internal_shuffle=True,
                       num_of_samples=args.n_test_samples, n_jobs=args.njobs, verbose=args.v, start=-1)

# Load previous scaler
scaler_path = os.path.join(args.model_path, 'scaler.p')
scaler = pickle.load(open(scaler_path, 'rb'))
dataset_test.set_scaler(scaler)

print('Test with %d images' % (dataset_test.n_samples))

ids_test = np.arange(dataset_test.n_samples)

# Create batch generators for the test sets.
from generator import Generator

gen_test = Generator(dataset_test, ids_test, batch_size=args.batch_size, shuffle=True,
                     buffer_size=args.buffer_size, verbose=args.v)

# Define tf model.
import tensorflow as tf
tf.reset_default_graph()

# Set tensorflow to only log errors
if args.v == 0:
    tf.logging.set_verbosity(tf.logging.ERROR)

# Load previous metegraph
meta_file = os.path.join(args.model_path, args.model_name + '.meta')
print("Loading model metafile: ", meta_file)
saver = tf.train.import_meta_graph(meta_file)
graph = tf.get_default_graph()

# Check input shape
print("Input shape: ", dataset_test.shape)

# Inputs.
input_image_tf = graph.get_tensor_by_name('input_image:0')
y_true_tf = graph.get_tensor_by_name('y_true:0')

training_tf = graph.get_tensor_by_name('training:0')

loss_tf = graph.get_tensor_by_name('loss_mse:0')
error_tf = graph.get_tensor_by_name('error_mae:0')
y_pred_tf = graph.get_tensor_by_name('Squeeze:0')

# Global step for logging.
global_step = 0

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    # Initialize tf variables.
    tf.global_variables_initializer().run()

    model_file = os.path.join(args.model_path, args.model_name)
    saver.restore(sess, model_file)

    # Open csv to log results
    fp = open('appd_result.csv', 'w')

    # Sequence of train and validation batches.
    test_loss = 0
    test_error = 0
    test_step = 0

    console_output_size = 0
    for b in range(gen_test.n_batches):
        images_batch, labels_batch = gen_test.next()

        # Calculate validation loss.
        batch_loss, batch_error, y_pred = sess.run(
            [loss_tf, error_tf, y_pred_tf],
            feed_dict={input_image_tf: images_batch,
                       y_true_tf: labels_batch})

        test_loss += batch_loss
        test_error += batch_error
        test_step += 1

        # Log results
        for label,pred in zip(labels_batch, y_pred):
            fp.write('%f,%f\n' % (label, pred))

        # Print results.
        sys.stdout.write('\b' * console_output_size)

        console_output = 'step %5d ' % test_step
        console_output += 'Test: loss_mse: %.4f err_mae %.4f' % (
            test_loss / test_step, test_error / test_step)

        console_output_size = len(console_output)

        sys.stdout.write(console_output)
        sys.stdout.flush()

    print()

dataset_test.stop()
