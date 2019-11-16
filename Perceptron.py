#!/usr/bin/env python3


import argparse

import numpy as np
import pandas as pd


'''
An implementation of a multi-layer perceptron.

Author: Cody Lewis
Date: 2019-09-21
'''


class Neuron:
    '''A neuron of the perceptron'''
    def __init__(self, input_layer=False):
        self.activation_strength = 0
        self.connections = []
        self.is_input_layer = input_layer

    def connect(self, other):
        '''Connect a neuron to this'''
        self.connections.append(Connection(self, other))

    def input(self, value):
        '''Input a value into this neuron'''
        self.activation_strength = value

    def output(self):
        '''Recieve a final output from this neuron with softmax activation'''
        val = np.exp(self.activation_strength)
        self.activation_strength = 0
        return val

    def mutate(self, step_size):
        '''Mutate the weights of the connections on this neuron'''
        for connection in self.connections:
            connection.mutate_weight(step_size)

    def revert(self):
        '''Reset the weights of the connections on this neuron'''
        for connection in self.connections:
            connection.revert_weight()

    def activate(self):
        '''Activate this neuron'''
        for connection in self.connections:
            if self.is_input_layer:
                connection.activate(self.activation_strength)
            else:
                connection.activate(activate(self.activation_strength))
        self.activation_strength = 0

    def activated(self, value):
        '''Get activated with the value'''
        self.activation_strength += value


class Connection:
    '''A connection between 2 neurons'''
    def __init__(self, from_node, to_node):
        self.from_node = from_node
        self.to_node = to_node
        self.weight = np.random.normal()
        self.cached_weight = 0

    def activate(self, value):
        '''Activate the neuron that this connects to with the value'''
        self.to_node.activated(value * self.weight)

    def mutate_weight(self, step_size):
        '''Mutate the weight of this'''
        self.cached_weight = self.weight
        self.weight += step_size * np.random.normal()

    def revert_weight(self):
        '''Reset the weight mutation'''
        self.weight = self.cached_weight


class Perceptron:
    '''A multi layer perceptron implementation'''
    def __init__(self, structure):
        self.layers = [
            [Neuron(i == 0) for _ in range(layer_size)]
            for i, layer_size in enumerate(structure)
        ]

        for i in range(len(self.layers) - 1):
            for neuron in self.layers[i]:
                for next_neuron in self.layers[i + 1]:
                    neuron.connect(next_neuron)

    def summary(self):
        print(f"{'-' * 5} [Perceptron Summary] {'-' * 5}")
        num_layers = len(self.layers)
        for i, layer in enumerate(self.layers):
            if i == 0:
                layer_type = "Input Layer"
            elif i == num_layers - 1:
                layer_type = "Output Layer"
            else:
                layer_type = f"Hidden Layer {i}"
            print(f"{layer_type}:\t{len(layer)}")
        print("-" * 32)

    def predict(self, inputs):
        '''Predict a value given the inputs'''
        for i, inp in enumerate(inputs):
            self.layers[0][i + 1].input(inp)
        # Set up bias
        self.layers[0][0].input(1)

        for layer in self.layers[:-1]:
            for neuron in layer:
                neuron.activate()

        outputs = []
        for neuron in self.layers[-1]:
            outputs.append(neuron.output())
        denominator = np.sum(outputs)
        activations = []
        for output in outputs:
            activations.append(output / denominator)

        return activations

    def classify(self, inputs):
        '''
        Find the classification given a prediction for the prediction of the
        inputs
        '''
        return np.argmax(self.predict(inputs))

    def learn(self, error_goal, inputs, target_responses, verbose=False):
        '''Learn the weights for the network using evolutionary hill descent'''
        counter = 0
        n_epochs = 10_000
        error_champ = find_error(inputs, target_responses, self)
        errors = [error_champ]
        while(error_goal < error_champ) and (counter < n_epochs):
            if verbose:
                print(f"\rEpoch {counter + 1}, error: {error_champ}", end="")
            step_size = 0.02 * np.random.normal()
            for layer in self.layers[:-1]:
                for neuron in layer:
                    neuron.mutate(step_size)

            error_mutant = find_error(inputs, target_responses, self)

            if error_mutant < error_champ:
                error_champ = error_mutant
            else:
                for layer in self.layers[:-1]:
                    for neuron in layer:
                        neuron.revert()
            counter += 1
            errors.append(error_champ)
        if verbose:
            print()
        return error_champ, errors


def activate(x_value, coefficient=1, constant=0):
    '''
    Perform the sigmoid function to determine whether a neuron is activated.
    '''
    return 1 / (1 + np.exp(-coefficient * x_value - constant))


def find_error(inputs, target_responses, perceptron):
    '''
    Find the error of the perceptron.
    '''
    error = 0
    for i, target_response in zip(inputs, target_responses):
        predictions = perceptron.predict(i)
        error += log_loss(1, predictions[target_response])
    return -error / len(inputs)


def log_loss(label, prediction, eps=1e-15):
    '''Find the log loss'''
    # clip performs a minmax
    np.clip(prediction, eps, 1 - eps)
    return label * np.log2(prediction)


def eval_predictions(model, data, labels):
    '''Evaluate the accuracy of predictions on the data made by the model'''
    print("Input data\tPrediction\tTrue Label")
    accuracy = 0
    for row, label in zip(data, labels):
        prediction = model.classify(row)
        print(f"{row}\t\t{prediction}\t\t{label}")
        accuracy += prediction == label
    print(f"Accuracy: {100 * accuracy / len(labels)}%")
    print()


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="A multi-layer perceptron")
    PARSER.add_argument("-f", "--file", dest="file", type=str, action="store",
                        default="XOR2.csv",
                        help="csv file cotaining the training data and labels")
    PARSER.add_argument("-l", "--layers", dest="layers", type=int,
                        action="store", default=[4], nargs="+",
                        help="Number of neurons in the hidden layers")
    ARGS = PARSER.parse_args()
    DF = pd.read_csv(ARGS.file, header=None)
    DATA = DF[DF.columns[:-1]].to_numpy()
    LABELS = DF[DF.columns[-1]].to_numpy()
    MODEL = Perceptron(
        [np.shape(DATA)[1] + 1] + ARGS.layers + [len(np.unique(LABELS))]
    )
    print(f"Constructed Perceptron:")
    MODEL.summary()
    print()
    eval_predictions(MODEL, DATA, LABELS)
    MODEL.learn(0.01, DATA, LABELS, True)
    print()
    eval_predictions(MODEL, DATA, LABELS)
