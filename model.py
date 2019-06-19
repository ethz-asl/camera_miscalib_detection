import tensorflow as tf

def init_model(input_shape, n_classes):
    # Input layer block.
    input_image = tf.placeholder(dtype=tf.float32, shape=(None,) +
        input_shape + (1,), name='input_image')

    y_true = tf.placeholder(dtype=tf.float32, shape=(None, n_classes),
        name="y_true")

    training = tf.placeholder_with_default(tf.constant(False, dtype=tf.bool),
        shape=(), name="training")

    # Convolutional layer block.
    conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3),
            padding="same", activation=tf.nn.relu, use_bias=True,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            name="conv1")(input_image)

    conv2 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3),
            padding="same", activation=tf.nn.relu, use_bias=True,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            name="conv2")(conv1)

    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2),
            name="pool1")(conv2)

    conv3 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3),
            padding="same", activation=tf.nn.relu, use_bias=True,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            name="conv3")(pool1)

    conv4 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3),
            padding="same", activation=tf.nn.relu, use_bias=True,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            name="conv4")(conv3)

    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2),
            name="pool2")(conv4)

    flatten = tf.keras.layers.Flatten()(pool2)

    # Dense layers.
    dense1 = tf.keras.layers.Dense(units=256, activation=tf.nn.relu,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
             use_bias=True, name="dense1")(flatten)

    dense1_drop = tf.keras.layers.Dropout(rate=0.5, name='dense1_drop')(
            dense1, training=training)

    dense2 = tf.keras.layers.Dense(units=64, activation=tf.nn.relu,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
             use_bias=True, name="dense2")(dense1_drop)

    dense2_drop = tf.keras.layers.Dropout(rate=0.5, name='dense2_drop')(
            dense2, training=training)

    y_pred = tf.keras.layers.Dense( units=n_classes, activation=None,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            use_bias=True, name="classes")(dense2_drop)

    # Loss and optimizer.
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=y_pred, labels=y_true), name="loss")

    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
    train_op = optimizer.minimize(loss, name='train_op')

    # Statistics.
    correct_pred = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32),
            name="accuracy")

    time_preprocess = tf.placeholder(dtype=tf.float32, shape=(),
            name='time_preprocess')
    time_train = tf.placeholder(dtype=tf.float32, shape=(), name='time_train')

    with tf.name_scope('summary'):
        tf.summary.scalar('loss', loss, collections=['summary'])
        tf.summary.scalar("accuracy", accuracy, collections=["summary"])

        with tf.name_scope('timings'):
            tf.summary.scalar('preprocess', time_preprocess,
                    collections=['summary_time'])
            tf.summary.scalar('train', time_train, collections=['summary_time'])
