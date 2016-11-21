import numpy as np
import neural_network as ann
from scipy.misc import imread

DATA_PATH = "/home/asu/Projects/facial_recognition/code"
RANDOM_HIGH = 0.5
RANDOM_LOW = -0.5

#Initialize weight matrices with small random weights by layer

layer1 = ann.neuralLayer(np.zeros((960,3)))
layer2 = ann.neuralLayer(((RANDOM_HIGH - RANDOM_LOW) * np.random.rand(4,4) + RANDOM_LOW),thresholded=True)

layer_list = [layer1, layer2]

#Initialize network

faceNet = ann.neuralNet(2, layer_list)

#Backpropagation

def backpropagate(training_data, rate, momentum):
    total_weight_1 = np.zeros(faceNet.layers[1].weight_matrix.shape)
    total_weight_2 = np.zeros(faceNet.layers[0].weight_matrix.shape)
    for data in training_data:
        target = data[1]
        features = data[0]
        final_output = faceNet.propagate(features)
        print final_output
        error = target - final_output
        delta = error*(final_output)*(1-final_output)
        # Now backpropgate all the output deltas to find the rest of the deltas
        hidden_deltas = faceNet.layers[1].emit_deltas(final_delta)
        # Now compute all necessary weight updates and store back in weight
        # matrix
        weight_1 = np.tile(faceNet.layers[1].last_input, (faceNet.layers[1].weight_matrix.shape[1],1)).T
        weight_2 = np.tile(faceNet.layers[0].last_input, (faceNet.layers[0].weight_matrix.shape[1],1)).T
        weight_1 *= (final_delta * rate)
        weight_2 *= (hidden_deltas * rate)
        total_weight_1 += weight_1
        total_weight_2 += weight_2

    total_weight_1 += (momentum * faceNet.layers[1].last_weight_update)
    total_weight_2 += (momentum * faceNet.layers[0].last_weight_update)
    faceNet.layers[0].last_weight_update = np.copy(total_weight_2)
    faceNet.layers[1].last_weight_update = np.copy(total_weight_1)
    faceNet.layers[0].weight_matrix += total_weight_2
    faceNet.layers[1].weight_matrix += total_weight_1

# def backpropagate(training_data, rate, momentum, current_layer):
    # final_delta = np.zeros(faceNet.layers[current_layer].shape[1])
    # for data in training_data:
        # target = data[1]
        # features = data[0]
        # final_output = faceNet.output(features)
        # error = target - final_output
        # delta = error*(final_output)*(1-final_output)
        # final_delta += delta
    # for i in range(faceNet.layers[current_layer].shape[0]):
        # faceNet.layers[current_layer].T[i] += rate * hidden_output * final_delta


