"""
Saksham's Self Learning Neural Network
This Neural Network will play tic-tac-toe and maybe chess, to learn from its mistakes and improve its gameplay

Author: Saksham Goel
Date: Feb 15, 2025
Version: 1.0

Github: @SakshamG7
Organization: AceIQ
Website: https://aceiq.ca
Contact: mail@aceiq.ca
Location: Canada, ON, Oakville
"""

# Importing Libraries
from tictactoe import TicTacToe
import math
import random

# Activation Function
# Have great luck with this one, just works with simple tasks fast
def SakshamsLinearCutOff(x: float) -> float:
    if x > 1:
        return x * 0.01 + 0.99
    elif x < -1:
        return x * 0.01 - 0.99
    return x

def softmax(x):
    e_x = [math.exp(i) for i in x]
    return [i / sum(e_x) for i in e_x]

# Neural Network Class
class SelfLearningNeuralNetwork(object):
    # Initialize the Neural Network
    def __init__(self):
        self.input_size = 0
        self.output_size = 0
        self.neurons = {}
        self.connections = {}
        self.input_ids = []
        self.output_ids = []
        
        # Add Input and Output Neurons to neurons
        for i in range(input_size):
            self.neurons[f'input_{i}'] = [random.random(), 0, 0]
        for i in range(output_size):
            self.neurons[f'output_{i}'] = [random.random(), 0, 0]

    # Add Neuron to the Neural Network
    # neuron_id: Neuron ID
    # bias: Bias of the Neuron
    def add_neuron(self, neuron_id: int, bias):
        # The Neuron is a list with 2 elements
        # 1. Bias
        # 2. Usage
        # 3. Value
        self.neurons[neuron_id] = [bias, 0, 0]
    
    # Add Connection to the Neural Network
    # connection_id: Connection ID
    # source_neuron_id: Source Neuron ID
    # target_neuron_id: Target Neuron ID
    # weight: Weight of the Connection
    def add_connection(self, connection_id, source_neuron_id, target_neuron_id, weight):
        self.connections[connection_id] = (source_neuron_id, target_neuron_id, weight)
    
    # Forward Propagation
    # inputs: Inputs to the Neural Network
    # expected_outputs: Expected Number Outputs
    def forward_propagation(self, inputs, expected_outputs):
        if len(inputs) < self.input_size:
            # Grow more input neurons
            for i in range(self.input_size - len(inputs)):
                new_id = max(self.neurons.keys()) + 1
                self.add_neuron(new_id, random.random())
                self.input_ids.append(new_id)
                self.input_size += 1
        elif len(inputs) > self.input_size:
            # Padding, fill with 0s
            for i in range(len(inputs) - self.input_size):
                inputs.append(0)
        if len(expected_outputs) < self.output_size:
            # Grow more output neurons
            for i in range(self.output_size - len(expected_outputs)):
                new_id = max(self.neurons.keys()) + 1
                self.add_neuron(new_id, random.random())
                self.output_ids.append(new_id)
                self.output_size += 1
        elif len(expected_outputs) > self.output_size:
            # Padding, fill with 0s
            for i in range(len(expected_outputs) - self.output_size):
                expected_outputs.append(0)
        self.neurons['input'] = inputs
        for connection_id in self.connections:
            source_neuron_id, target_neuron_id, weight = self.connections[connection_id]
            self.neurons[target_neuron_id][2] += self.neurons[source_neuron_id][2] * weight + self.neurons[source_neuron_id][0]
            # Activation Function
            self.neurons[target_neuron_id][2] = SakshamsLinearCutOff(self.neurons[target_neuron_id][2])
            self.neurons[target_neuron_id][1] += 1
            self.neurons[source_neuron_id][1] += 1
        return self.neurons['output']

    # is_stuck: Check if the Neural Network is stuck
