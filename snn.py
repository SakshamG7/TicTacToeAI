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
        # Initialize the weights and biases with random values

    # forward: forward pass of the neural network
    # x -> gpy.matrix: the input matrix
    # returns -> gpy.matrix: the output matrix
    def forward(self, x: gpy.matrix) -> gpy.matrix:
        return x