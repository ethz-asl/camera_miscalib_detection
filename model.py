from __future__ import print_function
import tensorflow as tf

def add_conv_pool_layer(input, scope, filters=1, kernel_size=(3,3), strides=(1,1)):
    with tf.name_scope(scope):
        conv1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,
                strides=strides, padding="same", activation=tf.nn.relu,
                use_bias=True, kernel_initializer='glorot_uniform',
                name="conv1")(input)

        conv2 = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,
                strides=strides, padding="same", activation=tf.nn.relu,
                use_bias=True, kernel_initializer='glorot_uniform',
                name="conv2")(conv1)

        pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2),
                name="pool1")(conv2)

    return pool1

def init_model(input_shape):
    # Input layer block.
    input_image = tf.placeholder(dtype=tf.float32, shape=(None,) + input_shape, name='input_image')

    y_true = tf.placeholder(dtype=tf.float32, shape=(None,), name="y_true")

    training = tf.placeholder_with_default(tf.constant(False, dtype=tf.bool), shape=(), name="training")

    # Convolutional layer blocks.
    conv_block_1 = add_conv_pool_layer(input_image, 'conv_block_1', filters=32)
    conv_block_2 = add_conv_pool_layer(conv_block_1, 'conv_block_2', filters=32)
    conv_block_3 = add_conv_pool_layer(conv_block_2, 'conv_block_3', filters=32)
    conv_block_4 = add_conv_pool_layer(conv_block_3, 'conv_block_4', filters=64)
    conv_block_5 = add_conv_pool_layer(conv_block_4, 'conv_block_5', filters=128)
    conv_block_6 = add_conv_pool_layer(conv_block_5, 'conv_block_6', filters=128)
    conv_block_7 = add_conv_pool_layer(conv_block_6, 'conv_block_7', filters=128)

    # Flatten
    flatten = tf.keras.layers.Flatten()(conv_block_7)

    # Dense layers.
    dense1 = tf.keras.layers.Dense(units=512, activation=tf.nn.relu,
                                   kernel_initializer='glorot_uniform',
                                   use_bias=True, name="dense1")(flatten)

    dense1_drop = tf.keras.layers.Dropout(rate=0.5, name='dense1_drop')(dense1, training=training)

    dense2 = tf.keras.layers.Dense(units=128, activation=tf.nn.relu,
                                   kernel_initializer='glorot_uniform',
                                   use_bias=True, name="dense2")(dense1_drop)

    dense2_drop = tf.keras.layers.Dropout(rate=0.0, name='dense2_drop')(dense2, training=training)

    y_pred = tf.keras.layers.Dense(units=1, kernel_initializer='glorot_uniform',
                                   use_bias=False, name="output")(dense2_drop)

    # Squeeze predictions for comparison with labels which are a simple 1D vector
    y_pred = tf.squeeze(y_pred)

    # Loss and optimizer.
    loss = tf.reduce_mean(tf.square(y_pred - y_true), name='loss_mse')

    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
    #optimizer = tf.train.MomentumOptimizer(1e-4, 0.1)
    #optimizer = tf.train.GradientDescentOptimizer(1e-4)

    train_op = optimizer.minimize(loss, name='train_op')

    # Statistics.
    error = tf.reduce_mean(tf.abs(y_pred - y_true), name='error_mae')

    time_data = tf.placeholder(dtype=tf.float32, shape=(), name='time_data')
    time_train = tf.placeholder(dtype=tf.float32, shape=(), name='time_train')

    with tf.name_scope('Summary'):
        tf.summary.scalar('loss_mse', loss, collections=['summary'])
        tf.summary.scalar("error_mae", error, collections=['summary'])
        tf.summary.histogram('appd_true', y_true, collections=['summary'])
        tf.summary.histogram('appd_pred', y_pred, collections=['summary'])
        tf.summary.histogram('appd_error', tf.abs(y_pred - y_true), collections=['summary'])

    with tf.name_scope('Timings'):
        tf.summary.scalar('data', time_data, collections=['summary_time'])
        tf.summary.scalar('train', time_train, collections=['summary_time'])

    #with tf.name_scope('Images'):
    #    tf.summary.image('input_images', input_image, max_outputs=3, collections=['summary'])
