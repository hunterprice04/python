# this is not my code i am using to better learn how a neural network really works
# I used this resource for this code:
# http://neuralnetworksanddeeplearning.com/chap1.html

import random
import numpy as np


class Network(object):

    # this initializes all needed values of the network
    # sizes is a list containing the sizes of each layer of the network
    def __init__(self, sizes):
        # the number of layers is equal to the size of the sizes list
        self.num_layers = len(sizes)

        self.sizes = sizes

        # generates random values as starting values for the biases and weights
        # the program assumes that the first layer is the input layer so
        # we do not generate the biases for the first layer
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    # Returns the output of the network given the input a
    # a is the vector of activations
    # w is the weight vector for the current layer
    # b is the bias vector for the current layer
    def feedforward(self, a):
        # iterates through the iterator of tuples of biases and weights
        # and applies the sigmoid function to each item
        for b, w in zip(self.biases, self.weights):
            # passes in the dot product of the vector of weights and the vector of activations
            # and adds the biases
            a = sigmoid(np.dot(w, a) + b)
        return a

    # training data is a list of tuples (x, y) that represent the training inputs
    # and corresponding desired outputs respectively
    # epochs is the number of epochs to train for
    # mini_batch_size is the size of the mini_batches to use when sampling
    # eta is the learning rate
    # test data if provided will evaluate the network after each epoch training and print the partial progress
    def stochastic_gradient_descent(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        # if test_data is provided print out assign a value to n_test
        if test_data:
            n_test = len(test_data)

        # n is the number of training data cases
        n = len(training_data)

        # iterate epoch times
        for j in range(epochs):
            # shuffle the training data in order to better train the network
            random.shuffle(training_data)

            # this separates the training data into multiple mini_batches that
            # have the size of mini_batch_size
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in range(0, n, mini_batch_size)]

            # for each mini_batch we will apply a single step of gradient descent
            # updates the networks weights and biases according to a single iteration of gradient descent
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {}: {} / {}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {} complete".format(j))

    # mini_batch is a small batch of the training set in the format of an (x, y) tuple
    # eta is the learning rate
    def update_mini_batch(self, mini_batch, eta):
        # create a list of numpy arrays in the same size as biases and weights all filled with 0's
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # iterate through each test case in the mini batch
        for x, y in mini_batch:
            # calculate the gradient of the cost function using the backpropagation algorithm
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)

            # adjust the nabla_b and nabla_w matrices with the gradient
            # adds the gradient of each to its corresponding matrix
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        # update the weights and biases using the calculated nabla_w and nabla_b
        # from the mini_batch
        self.weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]

    # currently working on understanding this function
    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return nabla_b, nabla_w

    # currently working on understanding this function
    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    # currently working on understanding this function
    def cost_derivative(self, output_activations, y):
        return (output_activations - y)


#### Helper Functions

# this function uses the sigmoid and is used to bound the data between 0 and 1
# the input z is a vector or Numpy array. it performs the operation to each element.
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# this function simply returns the derivative of the sigmoid function
def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

