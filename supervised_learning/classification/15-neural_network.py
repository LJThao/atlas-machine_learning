#!/usr/bin/env python3
"""Class NeuralNetwork that defines a neural network with a hidden
layer performing binary classification based on 14-neural_network.py"""
import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork():
    """Class NeuralNetwork"""
    def __init__(self, nx, nodes):
        """Class constructor"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        # setting private instance attributes
        self.__W1 = np.random.normal(size=(nodes, nx))
        self.__b1 = np.zeros(shape=(nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.normal(size=(1, nodes))
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """use getter method for W1"""
        return self.__W1

    @property
    def b1(self):
        """use getter method for b1"""
        return self.__b1

    @property
    def A1(self):
        """use getter method for a1"""
        return self.__A1

    @property
    def W2(self):
        """use getter method for w2"""
        return self.__W2

    @property
    def b2(self):
        """use getter method for b2"""
        return self.__b2

    @property
    def A2(self):
        """use getter method for a2"""
        return self.__A2

    def forward_prop(self, X):
        """calculating the forward propagation of the neural network:

        X = input data, nx = number of input features of a neuron,
        m = number of examples

        """
        self.__A1 = 1 / (1 + np.exp(-(np.dot(self.W1, X) + self.b1)))
        self.__A2 = 1 / (1 + np.exp(-(np.dot(self.W2, self.A1) + self.b2)))
        return self.A1, self.A2

    def cost(self, Y, A):
        """calculates the cost of the model using logistic regression"""
        log_A = np.log(A)
        log_1_minus_A = np.log(1.0000001 - A)
        loss = Y * log_A + (1 - Y) * log_1_minus_A

        # calculate the average cost
        return -np.sum(loss) / A.shape[1]

    def evaluate(self, X, Y):
        """evaluates the neural networks's predictions and returns its
        predictions"""
        A = self.forward_prop(X)[1]
        prediction = (A >= 0.5).astype(int)
        cost = self.cost(Y, A)
        return prediction, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """calculates one pass of gradient descent on the neural network"""
        m = Y.shape[1]
        dz2 = A2 - Y
        dw2 = np.dot(dz2, A1.T) / m
        db2 = np.sum(dz2, axis=1, keepdims=True) / m
        dz1 = np.dot(self.__W2.T, dz2) * (A1 * (1 - A1))
        dw1 = np.dot(dz1, X.T) / m
        db1 = np.sum(dz1, axis=1, keepdims=True) / m

        self.__W2 -= alpha * dw2
        self.__b2 -= alpha * db2
        self.__W1 -= alpha * dw1
        self.__b1 -= alpha * db1

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        """trains the neural network and returning the evaluation of
        the training data after iterations of training have occurred"""
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
        for iteration in range(iterations + 1):
            A1, A2 = self.forward_prop(X)
            self.gradient_descent(X, Y, A1, A2, alpha)

            # calculates the cost in each step
            cost = self.cost(Y, A2)

            # calculates the current cost and print at the set interval
            if verbose and iteration % step == 0:
                print(f"Cost after {iteration} iterations: {cost}")

            # appending cost and iteration to the list
            if graph and iteration % step == 0:
                g_costs.append(cost)
                g_iters.append(iteration)

        # plotting the graph and then displaying
        if graph:
            plt.plot(g_iters, g_costs, label='Cost', color='blue')
            plt.title('Training Cost')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.show()

        # returning the evaluation of the training data
        return self.evaluate(X, Y)
