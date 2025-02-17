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
    diff = 0.01
    cut_off = 1
    if x > cut_off:
        return x * diff + (cut_off - diff)
    elif x < -cut_off:
        return x * diff - (cut_off - diff)
    return x

def softmax(x):
    e_x = [math.exp(i) for i in x]
    return [i / sum(e_x) for i in e_x]

class SelfLearningNeuralNetwork(object):
    def __init__(self, input_size: int = 9, output_size: int = 9):
        self.input_size = input_size
        self.output_size = output_size
        self.neurons = {}       # Dictionary: neuron_id -> [bias, usage, value]
        self.connections = {}   # Dictionary: connection_id -> [source_neuron_id, target_neuron_id, weight, usage]
        self.input_ids = []     # List of input neuron IDs
        self.output_ids = []    # List of output neuron IDs
        self.fitness = 0        # Fitness of the Neural Network, useful for NEAT

        # Create input neurons
        for i in range(input_size):
            self.add_neuron(i, 0)
            self.input_ids.append(i)

        # Create output neurons
        for i in range(input_size, input_size + output_size):
            self.add_neuron(i, 0)
            self.output_ids.append(i)


    def add_neuron(self, neuron_id: int, bias):
        self.neurons[neuron_id] = [bias, 0, 0]


    def add_connection(self, connection_id, source_neuron_id, target_neuron_id, weight):
        self.connections[connection_id] = [source_neuron_id, target_neuron_id, weight, 0]


    def forward(self, inputs: list):
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
        return outputs
    

    # Crossover the Neural Network with another Neural Network
    # parent: the other Neural Network to crossover with
    def crossover(self, parent):
        child = SelfLearningNeuralNetwork(self.input_size, self.output_size)
        # Cross over shared neurons
        for neuron_id in self.neurons:
            if neuron_id in parent.neurons:
                if random.random() < 0.5:
                    child.neurons[neuron_id] = self.neurons[neuron_id].copy()
                else:
                    child.neurons[neuron_id] = parent.neurons[neuron_id].copy()
        # Cross over shared connections
        for connection_id in self.connections:
            if connection_id in parent.connections:
                if random.random() < 0.5:
                    child.connections[connection_id] = self.connections[connection_id].copy()
                else:
                    child.connections[connection_id] = parent.connections[connection_id].copy()
        # Take the remaining neurons from the fitter parent
        if self.fitness > parent.fitness:
            for neuron_id in self.neurons:
                if neuron_id not in child.neurons:
                    child.neurons[neuron_id] = self.neurons[neuron_id].copy()
        else:
            for neuron_id in parent.neurons:
                if neuron_id not in child.neurons:
                    child.neurons[neuron_id] = parent.neurons[neuron_id].copy()
        
        # Take the remaining connections from the fitter parent, also check if they are valid
        if self.fitness > parent.fitness:
            for connection_id in self.connections:
                if connection_id not in child.connections:
                    src, tgt, _, _ = self.connections[connection_id]
                    if src in child.neurons and tgt in child.neurons:
                        child.connections[connection_id] = self.connections[connection_id].copy()
        else:
            for connection_id in parent.connections:
                if connection_id not in child.connections:
                    src, tgt, _, _ = parent.connections[connection_id]
                    if src in child.neurons and tgt in child.neurons:
                        child.connections[connection_id] = parent.connections[connection_id].copy()

        # Update input and output neuron IDs
        child.input_ids = self.input_ids.copy()
        child.output_ids = self.output_ids.copy()

        return child


    # Mutate the Neural Network
    # Add new connections, neurons, remove connections, neurons, mutate weights and biases
    def mutate(self, mutation_rate: float = 0.1):
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
    
    # Copy the Neural Network
    def copy(self):
        copy = SelfLearningNeuralNetwork(self.input_size, self.output_size)
        copy.neurons = self.neurons.copy()
        copy.connections = self.connections.copy()
        copy.input_ids = self.input_ids.copy()
        copy.output_ids = self.output_ids.copy()
        return copy


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
            move = SSNN.forward(state, 9, 0.9)
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
            moves = SSNN.forward(state, 9, 0.9)
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
