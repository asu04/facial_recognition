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


class neuralNet:
    """Base class for neural net"""

    def __init__(self, n_layers, matrix_list):
        if not len(matrix_list) == n_layers:
            raise ValueError('Number of layers and number of matrices supplied do not match')
        try:
            self.n_layers = n_layers
            self.layers = []
            for layer in matrix_list:
                self.layers.append(layer)

        except:
            raise ValueError('DEBUG_MESSAGE_HERE')

    def propagate(self, inputs, current_layer = 0, forward_one = False):
        output_values = []
        for row in self.layers[current_layer].T:
            output_values.append(neuralNode(row).output(inputs))
        output_vector = np.array(output_values)
        if current_layer == self.n_layers - 1 or forward_one:
            return output_vector
        else:
            return self.propagate(output_vector, current_layer + 1)
