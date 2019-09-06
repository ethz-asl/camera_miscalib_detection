from __future__ import print_function
import tensorflow as tf

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

    with tf.name_scope('model'):
        # Input layer block.
        input_image = tf.placeholder(dtype=tf.float32, shape=(None,) + input_shape, name='input_image')

        y_true = tf.placeholder(dtype=tf.float32, shape=(None, 2), name="y_true")

        training = tf.placeholder_with_default(tf.constant(False, dtype=tf.bool), shape=(), name="training")

        # Add VGG-16 Head
        VGG16_MODEL = tf.keras.applications.VGG16(input_shape=input_shape,
                                                  include_top=False,
                                                  weights='imagenet')
        VGG16_MODEL.trainable = False

        # Flatten using GAP
        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()(VGG16_MODEL)

        # Dense layers.
        dense1 = tf.keras.layers.Dense(units=512, activation=tf.nn.relu,
                                       kernel_initializer='glorot_uniform',
                                       use_bias=True, name="dense1")(global_average_layer)

        dense1_drop = tf.keras.layers.Dropout(rate=0.5, name='dense1_drop')(dense1, training=training)

        dense2 = tf.keras.layers.Dense(units=256, activation=tf.nn.relu,
                                       kernel_initializer='glorot_uniform',
                                       use_bias=True, name="dense2")(dense1_drop)

        dense2_drop = tf.keras.layers.Dropout(rate=0.5, name='dense2_drop')(dense2, training=training)

        y_logits = tf.keras.layers.Dense(units=2, activation='none',
                                       kernel_initializer='glorot_uniform',
                                       use_bias=True, name="output")(dense2_drop)
        y_pred = tf.nn.softmax(y_logits)

    # Loss and optimizer.
    with tf.name_scope('cross_entropy'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=y_logits), name='loss')

    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
        train_op = optimizer.minimize(loss, name='train_op')

    # Statistics.
    with tf.name_scope('test'):
        correct_pred = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name="accuracy")

    time_data = tf.placeholder(dtype=tf.float32, shape=(), name='time_data')
    time_train = tf.placeholder(dtype=tf.float32, shape=(), name='time_train')

    with tf.name_scope('Summary'):
        tf.summary.scalar('loss_categorical', loss, collections=['summary'])
        tf.summary.scalar("accuracy", accuracy, collections=['summary'])

    with tf.name_scope('Timings'):
        tf.summary.scalar('data', time_data, collections=['summary_time'])
        tf.summary.scalar('train', time_train, collections=['summary_time'])
    
    with tf.name_scope('Images'):
        tf.summary.image('input_images', input_image, max_outputs=3, collections=['summary'])
