#!/usr/bin/env python3
"""Class Neuron that defines a single
neuron performing binary classification based on
5-neuron.py"""
import numpy as np
import matplotlib.pyplot as plt


class Neuron():
    """Class Neuron"""
    def __init__(self, nx):
        """Class Constructor"""
        # checks if nx is an int, if not raise
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        # checks is nx is less than 1, if it is raise
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # setting up the private instance attributes __W, __b, __A
        self.__W = np.random.normal(0, 1, size=(1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """getter method to get W"""
        return self.__W

    @property
    def b(self):
        """getter method to get b"""
        return self.__b

    @property
    def A(self):
        """getter method to get A"""
        return self.__A

    def forward_prop(self, X):
        """public method to calculate the forward propagation
        of the neuron.

        Formula for z = W * X + b

        """
        W = self.W
        b = self.b

        # calculating z by performing matrix multiplication
        z = np.dot(W, X) + b

        # applies sigmoid activation function
        self.__A = 1 / (1 + np.exp(-z))

        # returns the activated output of the neuron
        return self.__A

    def cost(self, Y, A):
        """calculates the cost of the model using a logistic
        regression

        Y = correct labels for input data
        A = the activated output of a neuron for each example

        """
        log_A = np.log(A)
        log_1_minus_A = np.log(1.0000001 - A)
        loss = Y * log_A + (1 - Y) * log_1_minus_A

        # calculate avg cost
        return -np.sum(loss) / A.shape[1]

    def evaluate(self, X, Y):
        """evaluates the neuron's predictions

        X = input data
        Y = correct labels for the input data

        then returns the prediction and cost

        """
        A = self.forward_prop(X)
        prediction = (A >= 0.5).astype(int)
        cost = self.cost(Y, A)
        return prediction, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """calculates one pass of gradient descent on the neuron

        X = input data
        Y = correct labels for the input data
        A = activated output of the neuron for each example
        alpha = the learning rate

        """
        m = X.shape[1]
        dZ = A - Y
        self.__W -= alpha * np.dot(dZ, X.T) / m
        self.__b -= alpha * np.sum(dZ) / m

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        """trains the neuron and returns the evaluation of the training
        data"""
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if graph or verbose:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step < 1 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        # storing costs and iterations to a list for the graph
        if graph:
            g_costs = []
            g_iters = []

        # using forward propagation and gradient descent
        for iteration in range(iterations):
            A = self.forward_prop(X)
            self.gradient_descent(X, Y, A, alpha)

            # calculates the current cost and print at the set interval
            if verbose and iteration % step == 0:
                cost = self.cost(Y, A)
                print(f"Cost after {iteration} iterations: {cost}")

            # appending cost and iteration to the list
            if graph and iteration % step == 0:
                cost = self.cost(Y, A)
                g_costs.append(cost)
                g_iters.append(iteration)

        # plotting the graph and then displaying
        if graph:
            plt.plot(g_costs, g_iters, label='Cost', color='blue')
            plt.title('Training Cost')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.show()

        # returning the evaluation of the training data
        return self.evaluate(X, Y)
