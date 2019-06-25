"""
This script can be used to get a summary of the model created (during prototyping).
@note It works only with Keras API.
"""

import tensorflow as tf


def create_model(input_shape, training=False):
    """
    Creates a simple CNN model
    :param input_shape:
    :param output_size:
    :return:
    """
    # check if dimensions are a correct type
    assert isinstance(input_shape, tuple)

    # define input placeholder
    input_image = tf.keras.Input(shape=input_shape, name='input')

    ########################
    ### Paste model here ###
    ########################

    with tf.name_scope("conv_block_1"):
        conv1 = tf.keras.layers.Conv2D(filters=8, kernel_size=(3, 3),
                                       padding="same", activation=tf.nn.relu, use_bias=True,
                                       kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                       name="conv1")(input_image)

        conv2 = tf.keras.layers.Conv2D(filters=8, kernel_size=(3, 3),
                                       padding="same", activation=tf.nn.relu, use_bias=True,
                                       kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                       name="conv2")(conv1)

        pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2),
                                             name="pool1")(conv2)

    with tf.name_scope("conv_block_2"):
        conv3 = tf.keras.layers.Conv2D(filters=8, kernel_size=(3, 3),
                                       padding="same", activation=tf.nn.relu, use_bias=True,
                                       kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                       name="conv3")(pool1)

        conv4 = tf.keras.layers.Conv2D(filters=8, kernel_size=(3, 3),
                                       padding="same", activation=tf.nn.relu, use_bias=True,
                                       kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                       name="conv4")(conv3)

        pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2),
                                             name="pool2")(conv4)

    with tf.name_scope("conv_block_3"):
        conv5 = tf.keras.layers.Conv2D(filters=8, kernel_size=(3, 3),
                                       padding="same", activation=tf.nn.relu, use_bias=True,
                                       kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                       name="conv5")(pool2)

        conv6 = tf.keras.layers.Conv2D(filters=8, kernel_size=(3, 3),
                                       padding="same", activation=tf.nn.relu, use_bias=True,
                                       kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                       name="conv6")(conv5)

        pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2),
                                             name="pool3")(conv6)

    with tf.name_scope("conv_block_4"):
        conv7 = tf.keras.layers.Conv2D(filters=8, kernel_size=(3, 3),
                                       padding="same", activation=tf.nn.relu, use_bias=True,
                                       kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                       name="conv7")(pool3)

        conv8 = tf.keras.layers.Conv2D(filters=8, kernel_size=(3, 3),
                                       padding="same", activation=tf.nn.relu, use_bias=True,
                                       kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                       name="conv8")(conv7)

        pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2),
                                             name="pool4")(conv8)

    # Flatten
    flatten = tf.keras.layers.Flatten()(pool4)

    # Dense layers.
    dense1 = tf.keras.layers.Dense(units=256, activation=tf.nn.relu,
                                   kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                   use_bias=True, name="dense1")(flatten)

    dense1_drop = tf.keras.layers.Dropout(rate=0.5, name='dense1_drop')(dense1, training=training)

    dense2 = tf.keras.layers.Dense(units=64, activation=tf.nn.relu,
                                   kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                   use_bias=True, name="dense2")(dense1_drop)

    dense2_drop = tf.keras.layers.Dropout(rate=0.5, name='dense2_drop')(dense2, training=training)

    y_pred = tf.keras.layers.Dense(units=1, activation=None,
                                   kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                   use_bias=True, name="output")(dense2_drop)

    ####################
    ### Create model ###
    ####################

    # instantiate the model given inputs and outputs.
    model = tf.keras.Model(inputs=input_image, outputs=y_pred)

    return model


if __name__ == '__main__':
    """
    Example Usage
    """
    # retrieve the cnn architecture
    model = create_model((770//4, 1128//4, 3), True)

    # print model summary
    model.summary()

# EOF
