from __future__ import print_function
import tensorflow as tf
import numpy as np

def add_dense_block(input, scope, training, growth_rate=12, bottleneck_features = 48, kernel_size=(3,3),
                    add_bottleneck=False, n_composite_functions=6, transition_compression_factor=0.5):

    with tf.variable_scope(scope):
        concatenated = input
        for i in range(n_composite_functions):
            with tf.variable_scope('compfun%d' % (i + 1)):
                # Bottleneck 1 x 1
                if add_bottleneck:
                    x_new = tf.keras.layers.BatchNormalization(axis=-1, name='btlnk_bn')(concatenated, training=training)

                    x_new = tf.keras.layers.ReLU(name='btlnk_relu')(x_new)

                    x_new = tf.keras.layers.Conv2D(filters=bottleneck_features, kernel_size=1,
                                                   padding='same', activation=None, use_bias=True,
                                                   kernel_initializer='glorot_uniform', name='btlnk_conv')(x_new)
                else:
                    x_new = concatenated

                # Actual convolution n x n
                batch_norm = tf.keras.layers.BatchNormalization(axis=-1, name='bn')(x_new, training=training)

                activation = tf.keras.layers.ReLU(name='relu')(batch_norm)

                convolution = tf.keras.layers.Conv2D(filters=growth_rate, kernel_size=kernel_size,
                                                     padding='same', activation=None, use_bias=True,
                                                     kernel_initializer='glorot_uniform', name='conv')(activation)

                concatenated = tf.concat([input, convolution], axis=-1)

        with tf.variable_scope('transition'):

            output = tf.keras.layers.BatchNormalization(axis=-1, name='bn')(concatenated, training=training)

            output_channels = int(np.floor(transition_compression_factor * output.get_shape().as_list()[1]))

            output = tf.keras.layers.Conv2D(filters=output_channels, kernel_size=1,
                                            padding='same', activation=None, use_bias=True,
                                            kernel_initializer='glorot_uniform', name='conv')(output)

            output = tf.keras.layers.AveragePooling2D(pool_size=2, padding='same')(output)

    return output


def add_conv_pool_layer(input, scope, filters=1, kernel_size=(3,3)):
    with tf.name_scope(scope):
        conv1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,
                padding="same", activation=tf.nn.relu, use_bias=True,
                kernel_initializer='glorot_uniform', name="conv1")(input)

        conv2 = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,
                padding="same", activation=tf.nn.relu, use_bias=True,
                kernel_initializer='glorot_uniform', name="conv2")(conv1)

        pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2),
                name="pool1")(conv2)

    return pool1

def init_model(input_shape):
    # Input layer block.
    input_image = tf.placeholder(dtype=tf.float32, shape=(None,) + input_shape, name='input_image')

    y_true = tf.placeholder(dtype=tf.float32, shape=(None), name="y_true")

    training = tf.placeholder_with_default(tf.constant(False, dtype=tf.bool), shape=(), name="training")

    # Densenet blocks 
    dense_block_1 = add_dense_block(input_image, 'dense_block_1', training=training, growth_rate=12,
                                    bottleneck_features = 4*12, kernel_size=(3,3),
                                    add_bottleneck=True, n_composite_functions=12, transition_compression_factor=0.5)

    dense_block_2 = add_dense_block(dense_block_1, 'dense_block_2', training=training, growth_rate=12,
                                    bottleneck_features = 4*12, kernel_size=(3,3),
                                    add_bottleneck=True, n_composite_functions=18, transition_compression_factor=0.5)

    dense_block_3 = add_dense_block(dense_block_2, 'dense_block_3', training=training, growth_rate=12,
                                    bottleneck_features = 4*12, kernel_size=(3,3),
                                    add_bottleneck=True, n_composite_functions=18, transition_compression_factor=0.5)

    dense_block_4 = add_dense_block(dense_block_3, 'dense_block_4', training=training, growth_rate=12,
                                    bottleneck_features = 4*12, kernel_size=(3,3),
                                    add_bottleneck=True, n_composite_functions=18, transition_compression_factor=0.5)

    # Flatten
    flatten = tf.keras.layers.Flatten()(dense_block_3)

    # Dense layers.
    dense1 = tf.keras.layers.Dense(units=512, activation=tf.nn.relu,
                                   kernel_initializer='glorot_uniform',
                                   use_bias=True, name="dense1")(flatten)

    dense1_drop = tf.keras.layers.Dropout(rate=0.5, name='dense1_drop')(dense1, training=training)

    dense2 = tf.keras.layers.Dense(units=256, activation=tf.nn.relu,
                                   kernel_initializer='glorot_uniform',
                                   use_bias=True, name="dense2")(dense1_drop)

    dense2_drop = tf.keras.layers.Dropout(rate=0.5, name='dense2_drop')(dense2, training=training)

    dense3 = tf.keras.layers.Dense(units=128, activation=tf.nn.relu,
                                   kernel_initializer='glorot_uniform',
                                   use_bias=True, name="dense3")(dense2_drop)

    dense3_drop = tf.keras.layers.Dropout(rate=0.5, name='dense3_drop')(dense3, training=training)

    dense4 = tf.keras.layers.Dense(units=64, activation=tf.nn.relu,
                                   kernel_initializer='glorot_uniform',
                                   use_bias=True, name="dense4")(dense3_drop)

    dense4_drop = tf.keras.layers.Dropout(rate=0.5, name='dense4_drop')(dense4, training=training)

    y_pred = tf.keras.layers.Dense(units=1, activation=None,
                                   kernel_initializer='glorot_uniform',
                                   use_bias=True, name="output")(dense4_drop)

    # Loss and optimizer.
    loss = tf.reduce_mean(tf.square(y_pred - y_true), name='loss_mse')

    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
    train_op = optimizer.minimize(loss, name='train_op')

    # Statistics.
    error = tf.reduce_mean(tf.abs(y_pred - y_true), name='error_mae')

    time_data = tf.placeholder(dtype=tf.float32, shape=(), name='time_data')
    time_train = tf.placeholder(dtype=tf.float32, shape=(), name='time_train')

    with tf.name_scope('Summary'):
        tf.summary.scalar('loss_mse', loss, collections=['summary'])
        tf.summary.scalar("error_mae", error, collections=['summary'])
        tf.summary.histogram('appd', y_true, collections=['summary'])

    with tf.name_scope('Timings'):
        tf.summary.scalar('data', time_data, collections=['summary_time'])
        tf.summary.scalar('train', time_train, collections=['summary_time'])
    
    with tf.name_scope('Images'):
        tf.summary.image('input_images', input_image, max_outputs=3, collections=['summary'])
