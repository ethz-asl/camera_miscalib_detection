import numpy as np
import os
import sys
import time

# Parse command line arguments.
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('index')
parser.add_argument('train_selector')
parser.add_argument('valid_selector')
parser.add_argument('-n_train_samples', type=int, default=-1)
parser.add_argument('-n_val_samples', type=int, default=-1)
parser.add_argument('-batch_size', type=int, default=64)
parser.add_argument('-epochs', type=int, default=100)
parser.add_argument('-model_path', default='models/test_model/')
parser.add_argument('-log_path', default='tensorboard/')
parser.add_argument('-log_name', default=None)
parser.add_argument('-checkpoints', default=5)
parser.add_argument('-v', type=int, default=0)
parser.add_argument('-njobs', type=int, default=os.cpu_count())

args = parser.parse_args()

# Load the dataset.
from dataset import Dataset

dataset_train = Dataset(args.index_csv, selector=args.train_selector, remove_mean=False, remove_std=False,
                        internal_shuffle=True, num_of_samples=args.n_train_samples, verbose=args.v, n_jobs=args.njobs)
dataset_valid = Dataset(args.index_csv, selector=args.valid_selector, remove_mean=False, remove_std=False,
                        internal_shuffle=True, num_of_samples=args.n_valid_samples, verbose=args.v, n_jobs=args.njobs)

print('Train with %d images' % (dataset_train.n_samples))
print('Valid with %d images' % (dataset_valid.n_samples))

ids_train = np.arange(dataset_train.n_samples)
ids_valid = np.arange(dataset_valid.n_samples)

# Create batch generators for the train and validation sets.
from generator import Generator

gen_train = Generator(dataset_train, ids_train, batch_size=args.batch_size, shuffle=False, verbose=args.v)
gen_valid = Generator(dataset_valid, ids_valid, batch_size=args.batch_size, shuffle=True, verbose=args.v)

# Create model directory if it doesn't exist.
if not os.path.exists(args.model_path):
    os.makedirs(args.model_path)

# Define tf model.
import tensorflow as tf

tf.reset_default_graph()

# Set tensorflow to only log errors
if args.v == 0:
    tf.logging.set_verbosity(tf.logging.ERROR)

from model import init_model

print("Input shape: ", dataset_train.shape)
init_model(dataset_train.shape)

graph = tf.get_default_graph()

# Inputs.
input_image_tf = graph.get_tensor_by_name('input_image:0')
y_true_tf = graph.get_tensor_by_name('y_true:0')

training_tf = graph.get_tensor_by_name('training:0')
time_preprocess_tf = graph.get_tensor_by_name('time_preprocess:0')
time_train_tf = graph.get_tensor_by_name('time_train:0')

loss_tf = graph.get_tensor_by_name('loss_mse:0')
error_tf = graph.get_tensor_by_name('error_mae:0')
train_op_tf = graph.get_operation_by_name('train_op')

# Summaries.
summary_tf = tf.summary.merge_all('summary')
summary_time_tf = tf.summary.merge_all('summary_time')

# Model save.
model_saver = tf.train.Saver(max_to_keep=args.checkpoints)

# Global step for logging.
global_step = 0

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    # Tensorboard statistics.
    if args.log_name:
        train_log = os.path.join(args.log_path, args.log_name, 'train')
        train_writer = tf.summary.FileWriter(train_log, sess.graph)

        valid_log = os.path.join(args.log_path, args.log_name, 'valid')
        valid_writer = tf.summary.FileWriter(valid_log)

    # Initialize tf variables.
    tf.global_variables_initializer().run()

    # Iterate network epochs.
    for epoch in range(0, args.epochs):
        train_loss = 0
        train_error = 0
        train_step = 0

        valid_loss = 0
        valid_error = 0
        valid_step = 0

        # Sequence of train and validation batches.
        batches = np.array([1] * gen_train.n_batches + [0] * gen_valid.n_batches)
        np.random.shuffle(batches)

        console_output_size = 0
        for train in batches:
            if train:
                images_batch, labels_batch = gen_train.next()

                train_start = time.time()

                # Run optimizer and calculate loss.
                batch_summary, batch_loss, batch_error, _ = sess.run(
                    [summary_tf, loss_tf, error_tf, train_op_tf],
                    feed_dict={input_image_tf: images_batch,
                               y_true_tf: labels_batch, training_tf: True})

                time_train = time.time() - train_start
                batch_summary_time = sess.run(
                    summary_time_tf,
                    feed_dict={time_preprocess_tf: dataset_train.time_preprocess,
                               time_train_tf: train_start})

                if args.log_name:
                    train_writer.add_summary(batch_summary, global_step)
                    train_writer.add_summary(batch_summary_time, global_step)

                train_loss += batch_loss
                train_error += batch_error
                train_step += 1
            else:
                images_batch, labels_batch = gen_valid.next()

                # Calculate validation loss.
                batch_summary, batch_loss, batch_error = sess.run(
                    [summary_tf, loss_tf, error_tf],
                    feed_dict={input_image_tf: images_batch,
                               y_true_tf: labels_batch})

                if args.log_name:
                    valid_writer.add_summary(batch_summary, global_step)

                valid_loss += batch_loss
                valid_error += batch_error
                valid_step += 1

            # Print results.
            sys.stdout.write('\b' * console_output_size)

            console_output = 'epoch %2d ' % epoch

            if train_step:
                console_output += 'loss %.6f err %.2f | ' % (
                    train_loss / train_step,
                    train_error / train_step * 100)

            if valid_step:
                console_output += 'val_loss: %.6f val_err %.2f' % (
                    valid_loss / valid_step,
                    valid_error / valid_step * 100)

            console_output_size = len(console_output)

            sys.stdout.write(console_output)
            sys.stdout.flush()

            global_step += 1

        print()

        # Only save the model if it's better.
        final_valid_loss = (valid_loss / valid_step)
        if epoch == 0 or final_valid_loss < best_loss:
            model_name = os.path.join(args.model_path, 'model-%d' % epoch)
            model_saver.save(sess, model_name, global_step=global_step)

            # Update best loss.
            best_loss = final_valid_loss
