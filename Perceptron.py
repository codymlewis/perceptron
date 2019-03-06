#!/usr/bin/env python3

'''
An implementation of a simple perceptron.

Author: Cody Lewis
Date: 2019-03-06
'''

import numpy as np
import matplotlib.pyplot as plt

def activate(x, a=1, b=0):
    return 1 / (1 + np.exp(-a * x - b))

def activation_strength(weight, node_state):
    '''
    Find the activation strength of a neuron.
    '''
    strength = 0

    for i in zip(weight, node_state):
        strength += i[0] * i[1]

    return strength

def predict(inputs, connected_matrix, output_neuron, weights):
    '''
    Predict from the perceptron for a given input.
    '''
    node_states = np.array([0 for _ in range(len(inputs))])

    for i in range(len(inputs)):
        node_states[i] = inputs[i] * connected_matrix[output_neuron][i]

    return activate(activation_strength(weights, node_states))

def get_response(inputs, connected_matrix, output_neuron, weights):
    '''
    Get a response from the perceptron for a given input.
    '''
    return int(np.round(predict(inputs, connected_matrix, output_neuron, weights)))

def find_error(inputs, target_responses, connected_matrix, output_neuron, weights):
    '''
    Find the error of the perceptron.
    '''
    error = 0

    for i in range(len(target_responses)):
        prediction = predict(inputs[i], connected_matrix, output_neuron, weights)
        error += np.power((target_responses[i] - prediction), 2)

    return np.sqrt(error)

def hill_climb(error_goal, inputs, target_responses, weights, connected_matrix, output_neuron):
    '''
    Evolutionary algorithm to find the optimal weights for the perceptron.
    '''
    counter = 0
    n_epochs = 10_000
    error_champ = find_error(inputs, target_responses, connected_matrix, output_neuron, weights)
    errors = []

    while(error_goal < error_champ) and (counter < n_epochs):
        step_size = 0.02 * np.random.normal()
        mutant_weights = weights.copy()
        for i in range(len(mutant_weights)):
            mutant_weights[i] += step_size * np.random.normal()

        error_mutant = find_error(inputs, target_responses,
                                  connected_matrix, output_neuron,
                                  mutant_weights)

        if error_mutant < error_champ:
            weights = mutant_weights
            error_champ = error_mutant
        errors.append(error_champ)
        counter += 1

    return weights, error_champ, errors


if __name__ == '__main__':
    ERROR_GOAL = 0.1
    INPUTS = np.array([
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1]
    ])
    TARGET_RESPONSES = np.array([0, 1, 1, 1])
    CONNECTED_MATRIX = np.array([
        [0, 0, 0, 1],
        [0, 0, 0, 1],
        [0, 0, 0, 1],
        [1, 1, 1, 0]
    ])
    OUTPUT_NEURON = 3
    WEIGHTS = np.array([np.random.normal() for _ in range(len(CONNECTED_MATRIX) - 1)])
    print(WEIGHTS)
    WEIGHTS, ERROR_CHAMP, ERRORS = hill_climb(ERROR_GOAL, INPUTS, TARGET_RESPONSES, WEIGHTS, CONNECTED_MATRIX, OUTPUT_NEURON)
    CORRECT_RESPONSES = 0
    for i in zip(INPUTS, TARGET_RESPONSES):
        OUTPUT = get_response(i[0], CONNECTED_MATRIX, OUTPUT_NEURON, WEIGHTS)
        if i[1] == OUTPUT:
            CORRECT_RESPONSES += 1
        print(f"Input: {i[0]}\tOutput: {OUTPUT}\tShould be: {i[1]}")
    print(f"Percentage of correct responses: {CORRECT_RESPONSES / len(INPUTS) * 100}%")
    print(f"Final Error: {ERROR_CHAMP}")
    print(f"Final weights: {WEIGHTS}")

    plt.plot(range(len(ERRORS)), ERRORS)
    plt.xlabel("Iteration")
    plt.ylabel("Average Error")
    plt.show()
