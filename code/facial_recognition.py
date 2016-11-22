import numpy as np
import neural_network as ann
from scipy.misc import imread
from os import listdir
import random

DATA_PATH = "/home/asu/Projects/facial_recognition/data/quarter_size/"
RANDOM_HIGH = 0.5
RANDOM_LOW = -0.5
TARGET_VALUES = {'left': np.array([0.9,0.1,0.1,0.1]), 'right': np.array([0.1,0.9,0.1,0.1]), 'straight': np.array([0.1,0.1,0.9,0.1]), 'up': np.array([0.1,0.1,0.1,0.9])}
ORIENTATIONS = set(['left','right','straight','up'])
PREDICTIONS = {0: 'left', 1: 'right', 2: 'straight', 3: 'up'}

#Backpropagation (might want to implement more generally later)

def backpropagate(training_data, rate, momentum):
    total_weight_1 = np.zeros(faceNet.layers[1].weight_matrix.shape)
    total_weight_2 = np.zeros(faceNet.layers[0].weight_matrix.shape)
    for data in training_data:
        target = data[1]
        features = data[0]
        final_output = faceNet.propagate(features)
        error = target - final_output
        delta = error*(final_output)*(1-final_output)
        # Now backpropgate all the output deltas to find the rest of the deltas
        hidden_deltas = faceNet.layers[1].emit_deltas(delta)
        # Now compute all necessary weight updates and store back in weight
        # matrix
        weight_1 = np.tile(faceNet.layers[1].last_input, (faceNet.layers[1].weight_matrix.shape[1],1)).T
        weight_2 = np.tile(faceNet.layers[0].last_input, (faceNet.layers[0].weight_matrix.shape[1],1)).T
        weight_1 *= (delta * rate)
        weight_2 *= (hidden_deltas * rate)
        total_weight_1 += weight_1
        total_weight_2 += weight_2

    total_weight_1 += (momentum * faceNet.layers[1].last_weight_update)
    total_weight_2 += (momentum * faceNet.layers[0].last_weight_update)
    faceNet.layers[0].last_weight_update = np.copy(total_weight_2)
    faceNet.layers[1].last_weight_update = np.copy(total_weight_1)
    faceNet.layers[0].weight_matrix += total_weight_2
    faceNet.layers[1].weight_matrix += total_weight_1

#Initialize weight matrices with small random weights by layer

layer1 = ann.neuralLayer(np.zeros((960,3)))
layer2 = ann.neuralLayer(((RANDOM_HIGH - RANDOM_LOW) * np.random.rand(4,4) + RANDOM_LOW),thresholded=True)

layer_list = [layer1, layer2]

#Initialize network

faceNet = ann.neuralNet(2, layer_list)

# Construct input data

names = set()
full_data = set(listdir(DATA_PATH))

for filename in full_data:
    names.add(filename.split("_")[0])

# Probably want a good sample from each name, say 13 each with a good mix of
# orientations.

training = []

for name in names:
    for direction in ORIENTATIONS:
        subset = [x for x in full_data if (x.split("_")[0] == name and x.split("_")[1] == direction)]
        training += random.sample(subset, 4)

training = set(training)
validation = full_data.difference(training)

training_data = []

for filename in training:
    features = imread(DATA_PATH + filename).flatten()/255.0
    target = TARGET_VALUES[filename.split("_")[1]]
    training_data.append((features, target))

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

candidate_networks = []

for i in range(500):
    backpropagate(training_data, 0.3, 0.3)
    if (i+1)%50 == 0:
        weight_matrix_0 = np.copy(faceNet.layers[0].weight_matrix)
        weight_matrix_1 = np.copy(faceNet.layers[1].weight_matrix)
        candidate_networks.append([weight_matrix_0, weight_matrix_1])

for network in candidate_networks:
    correct = 0
    test_network = ann.neuralNet(2, [ann.neuralLayer(network[0]), ann.neuralLayer(network[1], thresholded=True)])
    for filename in validation:
        features = imread(DATA_PATH + filename).flatten()/255.0
        target = filename.split("_")[1]
        output = test_network.propagate(features)
        prediction = PREDICTIONS[np.argmax(output)]
        if prediction == target:
            correct += 1
    network.append(correct)



