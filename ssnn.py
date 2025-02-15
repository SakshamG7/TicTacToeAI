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
import math
import random
import json # For saving and loading the Neural Network
from tictactoe import TicTacToe

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
        self.connections[connection_id] = [source_neuron_id, target_neuron_id, weight, 0]
    
    # Node: This is my messiest work I have ever done, I am sorry
    # Forward Propagation
    # inputs: Inputs to the Neural Network
    # expected_outputs: Expected Number Outputs
    # treshold: Treshold for confidence, if the confidence is less than the treshold, the Neural Network must adapt/change because it is not confident and stuck
    # mutation_rate: Mutation Rate, the rate at which the Neural Network will
    def forward_propagation(self, inputs: list, expected_outputs: int, treshold: float =0.75, mutation_rate: float =0.1):
        if expected_outputs > self.output_size:
            # Grow more output neurons
            for i in range(expected_outputs - self.output_size):
                if len(self.neurons.keys()) != 0:
                    new_id = max(self.neurons.keys()) + 1
                else:
                    new_id = 0
                self.add_neuron(new_id, random.random())
                self.output_ids.append(new_id)
                self.output_size += 1
        if len(inputs) > self.input_size:
            # Grow more input neurons
            for i in range(len(inputs) - self.input_size):
                if len(self.neurons.keys()) != 0:
                    new_id = max(self.neurons.keys()) + 1
                else:
                    new_id = 0
                self.add_neuron(new_id, random.random())
                self.input_ids.append(new_id)
                self.input_size += 1
                # Connect this input to a random output neuron
                n2 = random.choice(self.output_ids)
                if len(self.connections.keys()) != 0:
                    new_connection_id = max(self.connections.keys()) + 1
                else:
                    new_connection_id = 0
                self.add_connection(new_connection_id, new_id, n2, random.uniform(-1, 1))
        elif len(inputs) < self.input_size:
            # Padding, fill with 0s
            for i in range(self.input_size - len(inputs)):
                inputs.append(0)

        for connection_id in list(self.connections):
            # Update usage of the connection
            self.connections[connection_id][3] += 1
            source_neuron_id, target_neuron_id, weight, usage = self.connections[connection_id]
            # Check if the target neuron even exists, if not, remove the connection and continue
            if target_neuron_id not in self.neurons:
                del self.connections[connection_id]
                continue
            self.neurons[target_neuron_id][2] += self.neurons[source_neuron_id][2] * weight + self.neurons[source_neuron_id][0]
            # Activation Function
            self.neurons[target_neuron_id][2] = SakshamsLinearCutOff(self.neurons[target_neuron_id][2])
            self.neurons[target_neuron_id][1] += 1
            self.neurons[source_neuron_id][1] += 1
        
        # Return the output neurons
        outputs = []
        for output_id in self.output_ids:
            outputs.append(self.neurons[output_id][2])
        outputs = softmax(outputs)

        if max(outputs) < treshold:
            # The Neural Network is not confident, it must adapt
            # Add new connections to unused neurons
            for i in range(random.randint(1, 1 + len(self.neurons) // 2)):
                source_neuron_id = random.choice(self.input_ids)
                target_neuron_id = random.choice(self.output_ids)
                weight = random.uniform(-1, 1)
                connection_id = max(self.connections.keys()) + 1
                self.add_connection(connection_id, source_neuron_id, target_neuron_id, weight)
            
                # Add new neurons, 50 percent chance of adding a new neuron
                if random.random() < 0.5:
                    new_id = max(self.neurons.keys()) + 1
                    # Connect the new neuron to 2 random neurons
                    n1 = random.choice(list(self.neurons.keys()))
                    n2 = random.choice(list(self.neurons.keys()))
                    while n1 == n2:
                        n2 = random.choice(list(self.neurons.keys()))
                    self.add_neuron(new_id, random.random())
                    self.add_connection(max(self.connections.keys()) + 1, n1, new_id, random.uniform(-1, 1))
                    self.add_connection(max(self.connections.keys()) + 1, n2, new_id, random.uniform(-1, 1))
                
                # Add new connections to existing neurons, 33 percent chance of adding a new connection
                if random.random() < 0.33:
                    source_neuron_id = random.choice(list(self.neurons.keys()))
                    target_neuron_id = random.choice(list(self.neurons.keys()))
                    weight = random.uniform(-1, 1)
                    connection_id = max(self.connections.keys()) + 1
                    self.add_connection(connection_id, source_neuron_id, target_neuron_id, weight)
                
                # Remove connections, 25 percent chance of removing a connection that is not used often
                if random.random() < 0.25:
                    connection_id = random.choice(list(self.connections.keys()))
                    if self.connections[connection_id][3] < 10:
                        # Clean up unconected neurons
                        for neuron_id in list(self.neurons.keys()):
                            if neuron_id not in self.input_ids and neuron_id not in self.output_ids:
                                if neuron_id not in [self.connections[connection_id][0], self.connections[connection_id][1]]:
                                    del self.neurons[neuron_id]
                        del self.connections[connection_id]
                
                # Remove neurons, 10 percent chance of removing a neuron that is not used often
                if random.random() < 0.1:
                    neuron_id = random.choice(list(self.neurons.keys()))
                    if self.neurons[neuron_id][1] < 10 and neuron_id not in self.input_ids and neuron_id not in self.output_ids:
                        del self.neurons[neuron_id]
                        # Clean up unconected neurons and connections
                        for neuron_id in list(self.neurons.keys()):
                            if neuron_id not in self.input_ids and neuron_id not in self.output_ids:
                                if neuron_id not in [self.connections[connection_id][0], self.connections[connection_id][1]]:
                                    del self.neurons[neuron_id]
                        for connection_id in list(self.connections.keys()):
                            if self.connections[connection_id][0] not in self.neurons or self.connections[connection_id][1] not in self.neurons:
                                # Clean up unconected neurons
                                for neuron_id in list(self.neurons.keys()):
                                    if neuron_id not in self.input_ids and neuron_id not in self.output_ids:
                                        if neuron_id not in [self.connections[connection_id][0], self.connections[connection_id][1]]:
                                            del self.neurons[neuron_id]
                                del self.connections[connection_id]
                
                # Mutate weights, mutation rate chance of mutating a weight
                if random.random() < mutation_rate:
                    connection_id = random.choice(list(self.connections.keys()))
                    self.connections[connection_id][2] += random.uniform(-0.1, 0.1)
                
                # Mutate biases, mutation rate chance of mutating a bias
                if random.random() < mutation_rate:
                    neuron_id = random.choice(list(self.neurons.keys()))
                    self.neurons[neuron_id][0] += random.uniform(-0.1, 0.1)
        final_outputs = []
        for i in range(expected_outputs):
            final_outputs.append(outputs[i])
        return softmax(final_outputs) # Return the outputs in the range of expected outputs
    
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

# Main Function
def main():
    # Self train the Neural Network to play tic-tac-toe, it will play against itself.
    # The Neural Network will learn from its mistakes and improve its gameplay
    
    # Initialize the Neural Network
    SSNN = SelfLearningNeuralNetwork()

    game = TicTacToe()

    TURNS = 1000

    for i in range(TURNS):
        # Play the game
        while not game.is_over():
            # Get the current state of the game
            state = game.board
            # Get the Neural Network's move
            move = SSNN.forward_propagation(state, 9, 0.9)
            
            # Find the best legal move
            best_move = None
            best_value = -1
            for i in range(len(move)):
                if game.is_valid_move(i):
                    if move[i] > best_value:
                        best_move = i
                        best_value = move[i]

            # Make the move
            game.play(best_move)
            # Print the game
            game.print_board()
    
    # Save the Neural Network
    SSNN.save('ssnn.json')


def play(filename):
    # Player with the human against the Neural Network
    SSNN = SelfLearningNeuralNetwork()
    SSNN.load(filename)

    game = TicTacToe()

    # Ask the user if they want to play first
    user_first = input('Do you want to play first? (y/n): ')
    if user_first.lower() == 'y':
        user_turn = True
    else:
        user_turn = False
    
    while not game.is_over():
        if user_turn:
            game.print_board()
            move = int(input('Enter your move (1-9): '))
            game.play(move - 1)
        else:
            state = game.board
            move = SSNN.forward_propagation(state, 9, 0.9)
            best_move = None
            best_value = -1
            for i in range(len(move)):
                if game.is_valid_move(i):
                    if move[i] > best_value:
                        best_move = i
                        best_value = move[i]
            game.play(best_move)
        user_turn = not user_turn

    game.print_board()
    if game.winner == 0:
        print('It is a draw!')
    elif game.winner == 1:
        if user_turn:
            print('You win!')
        else:
            print('Neural Network wins!')
    else:
        if user_turn:
            print('Neural Network wins!')
        else:
            print('You win!')

if __name__ == '__main__':
    main()
    # play('ssnn.json')