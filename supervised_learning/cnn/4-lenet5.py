#!/usr/bin/env python3
"""LeNet-5 (Tensorflow 1) Module"""
import tensorflow.compat.v1 as tf


def lenet5(x, y):
    """Function that builds a modified version of the LeNet-5 architecture
    using tensorflow:

    x = a tf.placeholder of shape (m, 28, 28, 1) containing the input images
    for the network
    m = the number of images
    y = a tf.placeholder of shape (m, 10) containing the one-hot labels for
    the network
    -> The model should consist of the following layers in order:
    -> Convolutional layer with 6 kernels of shape 5x5 with same padding
    -> Max pooling layer with kernels of shape 2x2 with 2x2 strides
    -> Convolutional layer with 16 kernels of shape 5x5 with valid padding
    -> Max pooling layer with kernels of shape 2x2 with 2x2 strides
    -> Fully connected layer with 120 nodes
    -> Fully connected layer with 84 nodes
    -> Fully connected softmax output layer with 10 nodes
    All layers requiring initialization should initialize their kernels with
    the_normal initialization method: tf.keras.initializers.VarianceScaling(
    scale=2.0)
    All hidden layers requiring activation should use the relu activation
    function
    you may import tensorflow.compat.v1 as tf
    you may NOT use tf.keras only for the he_normal method.
    Returns:
    -> a tensor for the softmax activated output
    -> a training operation that utilizes Adam optimization (with default
    hyperparameters)
    -> a tensor for the loss of the network
    -> a tensor for the accuracy of the network

    """
    # init kernel weights for training stability
    init = tf.keras.initializers.VarianceScaling(scale=2.0)

    # convolutional and pooling layers
    conv_layer1 = tf.layers.conv2d(
        inputs=x,
        filters=6,
        kernel_size=(5, 5),
        padding='same',
        activation=tf.nn.relu,
        kernel_initializer=init
    )
    pool_layer1 = tf.layers.max_pooling2d(
        inputs=conv_layer1,
        pool_size=(2, 2),
        strides=(2, 2)
    )
    conv_layer2 = tf.layers.conv2d(
        inputs=pool_layer1,
        filters=16,
        kernel_size=(5, 5),
        padding='valid',
        activation=tf.nn.relu,
        kernel_initializer=init
    )
    pool_layer2 = tf.layers.max_pooling2d(
        inputs=conv_layer2,
        pool_size=(2, 2),
        strides=(2, 2)
    )

    # creating the flatten layer, dense 1 & 2, output layer
    flat_layer = tf.layers.flatten(pool_layer2)
    dense_layer1 = tf.layers.dense(
        inputs=flat_layer, units=120,
        activation=tf.nn.relu,
        kernel_initializer=init
    )
    dense_layer2 = tf.layers.dense(
        inputs=dense_layer1, units=84,
        activation=tf.nn.relu,
        kernel_initializer=init
    )
    output_layer = tf.layers.dense(
        inputs=dense_layer2, units=10,
        kernel_initializer=init
    )

    # applies softmax activation, calculate loss, set up optimizer
    softmax_output = tf.nn.softmax(output_layer)
    t_loss = tf.losses.softmax_cross_entropy(
        onehot_labels=y,
        logits=output_layer
    )
    adam_op = tf.train.AdamOptimizer().minimize(t_loss)

    # computes accuracy
    t_accuracy = tf.reduce_mean(
        tf.cast(
            tf.equal(tf.argmax(y, axis=1),
                     tf.argmax(output_layer, axis=1)),
                     tf.float32
        )
    )

    # returns the output, training op utilizing adam op, and loss and accuracy
    return (softmax_output, adam_op, t_loss, t_accuracy)
