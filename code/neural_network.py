import math
import numpy as np
import random

def sigmoid(x):
    return 1 / (1 + math.exp(-x))


class neuralNode:
    """Base class for node in neural network"""
    def __init__(self, weights):
        if isinstance(weights, np.ndarray):
            if len(weights.shape) == 1:
                self.weights = weights
        else:
            raise ValueError('Argument is not a one dimensional ndarray!')

    def output(self, inputs):
        return sigmoid((inputs * self.weights).sum())

class neuralLayer:
    """Class for layer in network"""
    def __init__(self, weight_matrix, thresholded = False):
        try:
            self.thresholded = thresholded
            self.weight_matrix = weight_matrix
            self.last_weight_update = np.zeros(weight_matrix.shape)
        except:
            raise ValueError("Unexpected arguments")

    def output(self, inputs):
        output_values = []
        if self.thresholded:
            inputs = np.insert(inputs,0,1)
        for row in self.weight_matrix.T:
            output_values.append(neuralNode(row).output(inputs))
        output_vector = np.array(output_values)
        # Keep track of last output and input
        self.last_output = np.copy(output_vector)
        self.last_input = np.copy(inputs)
        return output_vector

    def emit_deltas(self, next_deltas):
        deltas = np.multiply(self.weight_matrix, next_deltas)
        if self.thresholded:
            deltas = np.delete(deltas,0,0)
        return deltas.sum(axis = 1)

class neuralNet:
    """Base class for neural net"""

    def __init__(self, n_layers, layer_list):
        if not len(layer_list) == n_layers:
            raise ValueError('Number of layers and number of matrices supplied do not match')
        try:
            self.n_layers = n_layers
            self.layers = []
            for layer in layer_list:
                self.layers.append(layer)
        except:
            raise ValueError('DEBUG_MESSAGE_HERE')

    def propagate(self, inputs):
        for layer in self.layers:
            inputs = layer.output(inputs)
        return inputs
#         for row in self.layers[current_layer].T:
            # output_values.append(neuralNode(row).output(inputs))
        # output_vector = np.array(output_values)
        # if current_layer == self.n_layers - 1 or forward_one:
            # return output_vector
        # else:
#             return self.propagate(output_vector, current_layer + 1)
