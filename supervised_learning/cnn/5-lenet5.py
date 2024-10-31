#!/usr/bin/env python3
"""LeNet-5 (Keras) Module"""
from tensorflow import keras as K


def lenet5(X):
    """Function that builds a modified version of the LeNet-5 architecture
    using keras:

    X - a K.Input of shape (m, 28, 28, 1) containing the input images for
    the network
    m - the number of images
    The model should consist of the following layers in order:
    -> Convolutional layer with 6 kernels of shape 5x5 with same padding
    -> Max pooling layer with kernels of shape 2x2 with 2x2 strides
    -> Convolutional layer with 16 kernels of shape 5x5 with valid padding
    -> Max pooling layer with kernels of shape 2x2 with 2x2 strides
    -> Fully connected layer with 120 nodes
    -> Fully connected layer with 84 nodes
    -> Fully connected softmax output layer with 10 nodes
    All layers requiring initialization should initialize their kernels
    with the he_normal initialization method
    The seed for the he_normal initializer should be set to zero for each
    layer to ensure reproducibility.
    All hidden layers requiring activation should use the relu activation
    function
    you may from tensorflow import keras as K
    Returns: a K.Model compiled to use Adam optimization (with default
    hyperparameters) and accuracy metrics

    """
    # init weights for relu
    init = K.initializers.HeNormal(seed=0)

    # creating convolutional and pooling Layers
    pool_layer2 = K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(
        K.layers.Conv2D(filters=16, kernel_size=(5, 5),
                        padding='valid', activation='relu',
                        kernel_initializer=init)(
            K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(
                K.layers.Conv2D(filters=6, kernel_size=(5, 5),
                                padding='same', activation='relu',
                                kernel_initializer=init)(X)
            )
        )
    )

    # creating layers
    output_layer = K.layers.Dense(units=10,
                                  activation='softmax',
                                  kernel_initializer=init)(
        K.layers.Dense(units=84, activation='relu', kernel_initializer=init)(
            K.layers.Dense(units=120, activation='relu',
                           kernel_initializer=init)(
                K.layers.Flatten()(pool_layer2)
            )
        )
    )

    # creating and compiling the model
    model = K.Model(inputs=X, outputs=output_layer)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # returns the K model
    return (model)
