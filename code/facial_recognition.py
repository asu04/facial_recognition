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

def backpropagate(training_data, rate, momentum):
    for data in training_data:
        target = data[1]
        features = data[0]
        output = faceNet.propagate(features)

