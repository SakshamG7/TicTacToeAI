#!/usr/bin/env python3
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

import math
import random
import json  # For saving and loading the Neural Network
from tictactoe import TicTacToe

# Activation Function
def SakshamsLinearCutOff(x: float) -> float:
    if x > 1:
        return x * 0.01 + 0.99
    elif x < -1:
        return x * 0.01 - 0.99
    return x

def softmax(x):
    e_x = [math.exp(i) for i in x]
    return [i / sum(e_x) for i in e_x]

class SelfLearningNeuralNetwork(object):
    def __init__(self):
        self.input_size = 0
        self.output_size = 0
        self.neurons = {}       # Dictionary: neuron_id -> [bias, usage, value]
        self.connections = {}   # Dictionary: connection_id -> [source_neuron_id, target_neuron_id, weight, usage]
        self.input_ids = []     # List of input neuron IDs
        self.output_ids = []    # List of output neuron IDs

    def add_neuron(self, neuron_id: int, bias):
        self.neurons[neuron_id] = [bias, 0, 0]

    def add_connection(self, connection_id, source_neuron_id, target_neuron_id, weight):
        self.connections[connection_id] = [source_neuron_id, target_neuron_id, weight, 0]

    def forward_propagation(self, inputs: list, expected_outputs: int, treshold: float = 0.75, mutation_rate: float = 0.1):
        # Grow output neurons if needed
        if expected_outputs > self.output_size:
            for i in range(expected_outputs - self.output_size):
                new_id = max(self.neurons.keys()) + 1 if self.neurons else 0
                self.add_neuron(new_id, random.random())
                self.output_ids.append(new_id)
                self.output_size += 1

        # Grow input neurons if needed
        if len(inputs) > self.input_size:
            for i in range(len(inputs) - self.input_size):
                new_id = max(self.neurons.keys()) + 1 if self.neurons else 0
                self.add_neuron(new_id, random.random())
                self.input_ids.append(new_id)
                self.input_size += 1
                # Connect new input to a random output neuron
                n2 = random.choice(self.output_ids)
                new_connection_id = max(self.connections.keys()) + 1 if self.connections else 0
                self.add_connection(new_connection_id, new_id, n2, random.uniform(-1, 1))
        elif len(inputs) < self.input_size:
            # Padding with 0s
            for i in range(self.input_size - len(inputs)):
                inputs.append(0)

        # Propagate values via connections (iterate over a copy of the keys)
        for connection_id in list(self.connections.keys()):
            self.connections[connection_id][3] += 1
            source_neuron_id, target_neuron_id, weight, usage = self.connections[connection_id]
            # Remove connection if source or target is missing
            if source_neuron_id not in self.neurons or target_neuron_id not in self.neurons:
                del self.connections[connection_id]
                continue
            self.neurons[target_neuron_id][2] += self.neurons[source_neuron_id][2] * weight + self.neurons[source_neuron_id][0]
            self.neurons[target_neuron_id][2] = SakshamsLinearCutOff(self.neurons[target_neuron_id][2])
            self.neurons[target_neuron_id][1] += 1
            self.neurons[source_neuron_id][1] += 1

        # Collect outputs from designated output neurons
        outputs = []
        for output_id in self.output_ids:
            outputs.append(self.neurons[output_id][2])
        outputs = softmax(outputs)

        # If confidence is low, allow mutations/adaptations
        if max(outputs) < treshold:
            # Add new connections to random input-output pairs
            for i in range(random.randint(1, 1 + len(self.neurons) // 2)):
                source_neuron_id = random.choice(self.input_ids)
                target_neuron_id = random.choice(self.output_ids)
                weight = random.uniform(-1, 1)
                connection_id = max(self.connections.keys()) + 1 if self.connections else 0
                self.add_connection(connection_id, source_neuron_id, target_neuron_id, weight)

                # 50% chance: add a new hidden neuron between two random neurons
                if random.random() < 0.5:
                    new_id = max(self.neurons.keys()) + 1
                    n1 = random.choice(list(self.neurons.keys()))
                    n2 = random.choice(list(self.neurons.keys()))
                    while n1 == n2:
                        n2 = random.choice(list(self.neurons.keys()))
                    self.add_neuron(new_id, random.random())
                    self.add_connection(max(self.connections.keys()) + 1, n1, new_id, random.uniform(-1, 1))
                    self.add_connection(max(self.connections.keys()) + 1, n2, new_id, random.uniform(-1, 1))

                # 33% chance: add a new connection between any two neurons
                if random.random() < 0.33:
                    source_neuron_id = random.choice(list(self.neurons.keys()))
                    target_neuron_id = random.choice(list(self.neurons.keys()))
                    weight = random.uniform(-1, 1)
                    connection_id = max(self.connections.keys()) + 1 if self.connections else 0
                    self.add_connection(connection_id, source_neuron_id, target_neuron_id, weight)

                # 25% chance: remove a connection (and cleanup hidden neurons only)
                if random.random() < 0.25 and self.connections:
                    connection_id = random.choice(list(self.connections.keys()))
                    if self.connections[connection_id][3] < 10:
                        # Cleanup only hidden neurons (inputs/outputs are preserved)
                        for neuron_id in list(self.neurons.keys()):
                            if neuron_id in self.input_ids or neuron_id in self.output_ids:
                                continue
                            if neuron_id not in [self.connections[connection_id][0], self.connections[connection_id][1]]:
                                del self.neurons[neuron_id]
                        del self.connections[connection_id]

                # 10% chance: remove a hidden neuron that is not used often
                if random.random() < 0.1 and self.neurons:
                    neuron_id = random.choice(list(self.neurons.keys()))
                    # Only consider deletion if it's a hidden neuron
                    if neuron_id in self.input_ids or neuron_id in self.output_ids:
                        continue
                    if self.neurons[neuron_id][1] < 10:
                        del self.neurons[neuron_id]
                        # Cleanup: remove any connection that refers to a missing neuron
                        for connection_id in list(self.connections.keys()):
                            src, tgt, _, _ = self.connections[connection_id]
                            if src not in self.neurons or tgt not in self.neurons:
                                del self.connections[connection_id]

                # Mutate weights with given mutation_rate
                if random.random() < mutation_rate and self.connections:
                    connection_id = random.choice(list(self.connections.keys()))
                    self.connections[connection_id][2] += random.uniform(-0.1, 0.1)

                # Mutate biases with given mutation_rate
                if random.random() < mutation_rate and self.neurons:
                    neuron_id = random.choice(list(self.neurons.keys()))
                    self.neurons[neuron_id][0] += random.uniform(-0.1, 0.1)

        final_outputs = []
        for i in range(expected_outputs):
            final_outputs.append(outputs[i])
        return softmax(final_outputs)

    def save(self, filename: str):
        with open(filename, 'w') as file:
            json.dump({
                'input_size': self.input_size,
                'output_size': self.output_size,
                'neurons': self.neurons,
                'connections': self.connections,
                'input_ids': self.input_ids,
                'output_ids': self.output_ids
            }, file)

    def load(self, filename: str):
        with open(filename, 'r') as file:
            data = json.load(file)
            self.input_size = data['input_size']
            self.output_size = data['output_size']
            self.neurons = data['neurons']
            self.connections = data['connections']
            self.input_ids = data['input_ids']
            self.output_ids = data['output_ids']

def main():
    SSNN = SelfLearningNeuralNetwork()
    game = TicTacToe()
    TURNS = 1000

    for i in range(TURNS):
        print(f'Turn: {i + 1}')
        game.reset()
        while not game.is_over():
            state = game.board
            move = SSNN.forward_propagation(state, 9, 0.9)
            best_move = None
            best_value = -1
            for i in range(len(move)):
                if game.is_valid_move(i) and move[i] > best_value:
                    best_move = i
                    best_value = move[i]
            game.play(best_move)
    SSNN.save('ssnn.json')

def play(filename):
    SSNN = SelfLearningNeuralNetwork()
    SSNN.load(filename)
    game = TicTacToe()
    user_first = input('Do you want to play first? (y/n): ')
    user_turn = user_first.lower() == 'y'

    while not game.is_over():
        if user_turn:
            game.print_board()
            move = int(input('Enter your move (1-9): '))
            game.play(move - 1)
        else:
            state = game.board
            moves = SSNN.forward_propagation(state, 9, 0.9)
            best_move = None
            best_value = -1
            for i in range(len(moves)):
                if game.is_valid_move(i) and moves[i] > best_value:
                    best_move = i
                    best_value = moves[i]
            game.play(best_move)
        user_turn = not user_turn

    game.print_board()
    if game.winner == 0:
        print('It is a draw!')
    elif game.winner == 1:
        print('You win!' if user_turn else 'Neural Network wins!')
    else:
        print('Neural Network wins!' if user_turn else 'You win!')

if __name__ == '__main__':
    # Uncomment one of the following lines to run training or play mode:
    # main()
    play('ssnn.json')
