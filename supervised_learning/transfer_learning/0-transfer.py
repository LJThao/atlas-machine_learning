#!/usr/bin/env python3
"""Transfer Knowledge Module - trains a convolutional neural network
to classify the CIFAR 10 dataset - Keras Applications used for the
model = MobileNetV2"""
from tensorflow import keras as K


def preprocess_data(X, Y):
    """Function that pre-processes the data for your model:

    X = a numpy.ndarray of shape (m, 32, 32, 3) containing the CIFAR 10 data,
    where m is the number of data points
    Y = a numpy.ndarray of shape (m,) containing the CIFAR 10 labels for X
    Returns: X_p, Y_p
    X_p = a numpy.ndarray containing the preprocessed X
    Y_p = a numpy.ndarray containing the preprocessed Y

    """
    # preprocess input images for MobileNetV2 and one-hot encode labels
    X_p = K.applications.mobilenet_v2.preprocess_input(X)
    Y_p = K.utils.to_categorical(Y, 10)
    return X_p, Y_p


if __name__ == "__main__":
    # load and preprocess the CIFAR-10 data
    (X_train, Y_train), (X_test, Y_test) = K.datasets.cifar10.load_data()
    X_train, Y_train = preprocess_data(X_train, Y_train)
    X_test, Y_test = preprocess_data(X_test, Y_test)

    # define data augmentation layers
    data_aug = K.Sequential([
        K.layers.RandomFlip("horizontal"),
        K.layers.RandomRotation(0.1),
        K.layers.RandomZoom(0.1),
        K.layers.RandomTranslation(0.1, 0.1)
    ])

    # build the MobileNetV2 model, apply augmentation, resize shape
    inputs = K.Input(shape=(32, 32, 3))
    augmented = data_aug(inputs)
    resized = K.layers.Resizing(160, 160)(augmented)
    base_model = K.applications.MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(160, 160, 3),
        pooling="avg"
    )
    # freeze the base layers of the model
    base_model.trainable = False

    # build the head with dense layers, normalization, dropout, output
    x = base_model(resized)
    x = K.layers.Dense(512, activation="relu")(x)
    x = K.layers.BatchNormalization()(x)
    x = K.layers.Dropout(0.5)(x)
    x = K.layers.Dense(256, activation="relu")(x)
    x = K.layers.BatchNormalization()(x)
    x = K.layers.Dropout(0.5)(x)
    outputs = K.layers.Dense(10, activation="softmax")(x)

    # creating the full model
    model = K.Model(inputs, outputs)

    # compile the model with adam opt and categorical crossentropy loss
    model.compile(
        optimizer=K.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["acc"],
    )

    # define the learning rate scheduler
    lr_scheduler = K.callbacks.ReduceLROnPlateau(
        monitor="val_acc",
        factor=0.1,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )

    # train the frozen model
    history = model.fit(
        X_train, Y_train,
        validation_data=(X_test, Y_test),
        batch_size=64,
        epochs=10,
        callbacks=[lr_scheduler]
    )

    # fine-tuning the base model, unfreezing the base model
    base_model.trainable = True
    # freeze all of the layers except for the last 30
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    # recompile the model with a lower learning rate
    model.compile(
        optimizer=K.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["acc"],
    )

    # train the fine-tuned model
    fine_tune_history = model.fit(
        X_train, Y_train,
        validation_data=(X_test, Y_test),
        batch_size=64,
        # add more epochs for fine tuning to raise accuracy
        epochs=5,
        callbacks=[lr_scheduler]
    )

    # save the model
    model.save("cifar10.h5")
