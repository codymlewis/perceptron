#!/usr/bin/env python3

'''
An implementation of a simple perceptron.

Author: Cody Lewis
Date: 2019-03-06
'''

import sys
import numpy as np
import matplotlib.pyplot as plt

def read_csv(filename):
    '''
    Read a csv detailing the inputs and outputs and return them in a np array format.
    '''
    inputs = []
    responses = []

    with open(filename) as csv:
        for line in csv:
            input_line = []
            while line.find(",") > -1:
                input_line.append(int(line[:line.find(",")]))
                line = line[line.find(",") + 1:]
            inputs.append(input_line)
            responses.append(int(line))

    return np.array(inputs), np.array(responses)

def create_connected_matrix(input_len):
    '''
    Create the connectedness matrix for a single layer neural network perceptron.
    '''
    connected_matrix = []

    for _ in range(input_len):
        connected_row = [0 for _ in range(input_len)]
        connected_row.append(1)
        connected_matrix.append(connected_row)
    final_row = [1 for _ in range(input_len)]
    final_row.append(0)
    connected_matrix.append(final_row)

    return np.array(connected_matrix)

def activate(x_value, coefficient=1, constant=0):
    '''
    Perform the sigmoid function to determine whether a neuron is activated.
    '''
    return 1 / (1 + np.exp(-coefficient * x_value - constant))

def activation_strength(weight, node_state):
    '''
    Find the activation strength of a neuron.
    '''
    strength = 0

    for weight_node_state in zip(weight, node_state):
        strength += weight_node_state[0] * weight_node_state[1]

    return strength

def predict(inputs, connected_matrix, output_neuron, weights):
    '''
    Predict from the perceptron for a given input.
    '''
    node_states = np.array([0 for _ in range(len(inputs))])

    for index_input in enumerate(inputs):
        node_states[index_input[0]] = index_input[1] * \
            connected_matrix[output_neuron][index_input[0]]

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

    for index_target_responses in enumerate(target_responses):
        prediction = predict(inputs[index_target_responses[0]],
                             connected_matrix, output_neuron, weights)
        error += np.power((index_target_responses[1] - prediction), 2)

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
        for index_mutant_weights in enumerate(mutant_weights):
            mutant_weights[index_mutant_weights[0]] += step_size * \
                np.random.normal()

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
    ARGS = sys.argv[1:]
    INPUTS, TARGET_RESPONSES = read_csv(ARGS[0] if ARGS else "AND2.csv")
    CONNECTED_MATRIX = create_connected_matrix(len(INPUTS[0]))
    OUTPUT_NEURON = len(CONNECTED_MATRIX) - 1
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
