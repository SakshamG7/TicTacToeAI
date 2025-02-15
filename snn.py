"""
Saksham's Custom Neural Network Implementation, will be used for creating the Tic Tac Toe AI
The AIs will play against each other and learn from their mistakes, no data collection, just play!
not sure if this will work, but its worth a try!

Author: Saksham Goel
Date: Feb 15, 2025
Version: 1.0

Github: @SakshamG7
Organization: AceIQ
Website: https://aceiq.ca
Contact: mail@aceiq.ca
Location: Canada, ON, Oakville
"""

# Required Libraries
from __future__ import annotations # Required for type hinting the class itself
import gpy
import math
import random

# Activation Function, Just personal preference
def LeakyReLU(x: float) -> float:
    return x if x > 0 else 0.01 * x

# Derivative of the Activation Function
def dLeakyReLU(x: float) -> float:
    return 1 if x > 0 else 0.01

# Output Activation Function, useful for probabilities distribution
def softmax(x: gpy.matrix) -> gpy.matrix:
    return x.apply(math.exp) / x.sum().apply(math.exp)

# Neural Network Class
class SimpleNeuralNetwork(object):
    # __init__: constructor for the neural network, initializes the input and output size
    # input_size -> int: the size of the input layer
    # hidden_layers -> int: the number of hidden layers
    # hidden_size -> list -> int: the size of each hidden layer
    # output_size -> int: the size of the output layer
    def __init__(self, input_size: int = 1, hidden_layers: int = 1, hidden_size: list = [1], output_size: int = 1) -> None:
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers
        self.hidden_size = hidden_size
        self.fitness = 0
        # Initialize the weights and biases with random values
        self.weights = []
        self.biases = []

        for i in range(hidden_layers + 1):
            if i == 0:
                self.weights.append(gpy.matrix([[random.uniform(-1, 1) for j in range(hidden_size[i])] for k in range(input_size)]))
                self.biases.append(gpy.matrix([[random.uniform(-1, 1)] for j in range(hidden_size[i])]))
            elif i == hidden_layers:
                self.weights.append(gpy.matrix([[random.uniform(-1, 1) for j in range(output_size)] for k in range(hidden_size[i - 1])]))
                self.biases.append(gpy.matrix([[random.uniform(-1, 1)] for j in range(output_size)]))
            else:
                self.weights.append(gpy.matrix([[random.uniform(-1, 1) for j in range(hidden_size[i])] for k in range(hidden_size[i - 1])]))
                self.biases.append(gpy.matrix([[random.uniform(-1, 1)] for j in range(hidden_size[i])]))

    # Set the fitness of the neural network
    def set_fitness(self, fitness: float) -> None:
        self.fitness = fitness

    # get_fitness: returns the fitness of the neural network
    # returns -> float: the fitness of the neural network
    def get_fitness(self) -> float:
        return self.fitness

    # forward: forward pass of the neural network
    # x -> list: the input list
    # returns -> gpy.matrix: the output of the neural network
    def forward(self, x: list) -> gpy.matrix:
        inputs = gpy.matrix(data=x)
        final_output = gpy.matrix(rows=1, cols=self.output_size)

        for i in range(self.hidden_layers + 1):
            if i == 0:
                final_output = gpy.dot_product(inputs, self.weights[i]) + self.biases[i]
                # Update the final_output with the activation function
                final_output = final_output.apply(LeakyReLU)
            elif i == self.hidden_layers:
                # Final Layer
                final_output = gpy.dot_product(final_output, self.weights[i]) + self.biases[i]
            else:
                # Intermediate Hidden Layer
                final_output = gpy.dot_product(final_output, self.weights[i]) + self.biases[i]
                final_output = final_output.apply(LeakyReLU)

        return softmax(final_output) # Apply the softmax function to get the probabilities

    # copy: returns a copy of the neural network
    # returns -> SimpleNeuralNetwork: the copy of the neural network
    def copy(self) -> SimpleNeuralNetwork:
        new_nn = SimpleNeuralNetwork(self.input_size, self.hidden_layers, self.hidden_size, self.output_size)
        new_nn.weights = [x.copy() for x in self.weights]
        new_nn.biases = [x.copy() for x in self.biases]
        return new_nn

    # crossover: crossover function for the neural network
    # partner -> SimpleNeuralNetwork: the partner neural network
    # returns -> SimpleNeuralNetwork: the child neural network
    def crossover(self, partner: SimpleNeuralNetwork) -> SimpleNeuralNetwork:
        # Very simple crossover, just a 50-50 split, and we know the structure of the neural network is the same, so no Augmenting Topologies sadly
        child = self.copy()
        # Crossover the weights
        for i in range(len(self.weights)):
            for j in range(self.weights[i].rows):
                for k in range(self.weights[i].cols):
                    if random.uniform(-1, 1) < 0.5:
                        child.weights[i].data[j][k] = partner.weights[i].data[j][k]

        # Crossover the biases
        for i in range(len(self.biases)):
            for j in range(self.biases[i].rows):
                    if random.uniform(-1, 1) < 0.5:
                        child.biases[i].data[j][0] = partner.biases[i].data[j][0] # shape is (rows, 1)

        return child
    
    # mutate: mutation function for the neural network
    # mutation_rate -> float: the rate of mutation
    def mutate(self, mutation_rate: float) -> None:
        # Mutate the weights
        for i in range(len(self.weights)):
            for j in range(self.weights[i].rows):
                for k in range(self.weights[i].cols):
                    if random.uniform(-1, 1) < mutation_rate:
                        self.weights[i].data[j][k] += random.uniform(-1, 1)

        for i in range(len(self.biases)):
            for j in range(self.biases[i].rows):
                    if random.uniform(-1, 1) < mutation_rate:
                        self.biases[i].data[j][0] += random.uniform(-1, 1)