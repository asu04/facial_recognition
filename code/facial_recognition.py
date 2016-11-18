import numpy as np
import neural_network as ann

DATA_PATH = "/home/asu/Projects/facial_recognition/code"
RANDOM_HIGH = 0.5
RANDOM_LOW = -0.5

#Initialize weight matrices with small random weights by layer

layer1 = np.zeros((960,3))
layer2 = (RANDOM_HIGH - RANDOM_LOW) * np.random.rand(4,4) + RANDOM_LOW

matrix_list = [layer1, layer2]

#Initialize network

faceNet = ann.neuralNet(2, matrix_list)

#Backpropagation
#Testcomment

def backpropagate(training_data, rate, momentum, current_layer):
    final_delta = np.zeros(faceNet.layers[current_layer].shape[1])
    for data in training_data:
        target = data[1]
        features = data[0]
        # Output units are thresholded, so need to add a 1 in the output of the
        # hidden layer here
        hidden_output = np.insert(faceNet.propagate(features, forward_one=True),0, 1.0)
        final_output = faceNet.propagate(hidden_output, current_layer = 1)
        error = target - final_output
        delta = error*(final_output)*(1-final_output)
        final_delta += delta
    for i in range(faceNet.layers[current_layer].shape[0]):
        faceNet.layers[current_layer].T[i] += rate * hidden_output * final_delta


