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
        self.weights = []
        self.biases = []

        for i in range(hidden_layers + 1):
            if i == 0:
                self.weights.append(gpy.matrix([[random.random() for j in range(hidden_size[i])] for k in range(input_size)]))
                self.biases.append(gpy.matrix([[random.random()] for j in range(hidden_size[i])]))
            elif i == hidden_layers:
                self.weights.append(gpy.matrix([[random.random() for j in range(output_size)] for k in range(hidden_size[i - 1])]))
                self.biases.append(gpy.matrix([[random.random()] for j in range(output_size)]))
            else:
                self.weights.append(gpy.matrix([[random.random() for j in range(hidden_size[i])] for k in range(hidden_size[i - 1])]))
                self.biases.append(gpy.matrix([[random.random()] for j in range(hidden_size[i])]))

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
    
    # mse_loss: the mean squared error loss function, very simple and easy to implement
    # x -> list: the input list
    # y -> list: the output list
    # returns -> float: the loss value
    def mse_loss(self, x: list, y: list) -> float:
        y_hat = self.forward(x) # Get the output of the neural network
        loss = 0
        for i in range(len(y)):
            loss += (y_hat.data[0][i] - y[i]) ** 2
        return loss / len(y)

    # backward: backward pass of the neural network, used for training
    # x -> list: the input list
    # y -> list: the output list
    # learning_rate -> float: the learning rate
    
