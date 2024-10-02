#!/usr/bin/env python3
"""Train Function"""
import tensorflow.compat.v1 as tf
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations, alpha, iterations, save_path="/tmp/model.ckpt"):
    """builds, trains, and saves a neural network classifier:

    X_train = numpy.ndarry containing the training input data
    Y_train = numpy.ndarray containing the training labels
    X_valid = numpy.ndarray containing the validation input data
    Y_valid = numpy.ndarray containing the validation labels
    layer_sizes = list containing the number of nodes in each layer of the network
    activations = list containing the activation functions for each layer of the network
    alpha = the learning rate
    iterations = the number of iterations to train over
    save_path = designates where to save the model

    """
    # creating placeholders x and y that will train data(x) and labels(y)
    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    # using forward prop function to get the predicted output
    y_pred = forward_prop(x, layer_sizes, activations)
    # using calculate loss function to compare y and y_pred, then calc loss
    loss = calculate_loss(y, y_pred)
    # computing the accuracy of y to see if it matches the true labels
    accuracy = calculate_accuracy(y, y_pred)
    # using gradient descent to update the network's weights
    train_op = create_train_op(loss, alpha)
    # initializing TF session and start training
    with tf.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        for i in range(iterations + 1):
            # 0th to 1000th iterations, then training and accuracy are printed
            if i % 100 == 0 or i == iterations:
                t_cost, t_accuracy = sess.run(
                    [loss, accuracy], feed_dict={x: X_train, y: Y_train}
                )
                v_cost, v_accuracy = sess.run(
                    [loss, accuracy], feed_dict={x: X_valid, y: Y_valid}
                )
                print(f"After {i} iterations:")
                print(f"\tTraining Cost: {t_cost}")
                print(f"\tTraining Accuracy: {t_accuracy}")
                print(f"\tValidation Cost: {v_cost}")
                print(f"\tValidation Accuracy: {v_accuracy}")
            # training steps for adjusting the weight/loss function to predict
            sess.run(train_op, feed_dict={x: X_train, y: Y_train})
        # initializing the model to save
        model = tf.train.Saver()
        # the training model is saved to the path
        save_path = model.save(sess, save_path)
    # returns the path to the saved model
    return save_path
