#!/usr/bin/env python3
"""
Saksham's Self Learning Neural Network
This Neural Network will play tic-tac-toe and maybe chess, to learn from its mistakes and improve its gameplay

Author: Saksham Goel
Date: Feb 17, 2025
Version: 2.0

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
        self.wins = 0           # Number of wins
        self.losses = 0         # Number of losses
        self.draws = 0          # Number of draws
        self.legal_count = 0    # Number of legal moves made by the Neural Network

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
        if len(inputs) != self.input_size:
            raise ValueError('Invalid input size')
        # Take in the inputs and set the values of the input neurons
        c = 0
        for i in self.input_ids:
            self.neurons[i][2] = inputs[c]
            c += 1
        # Propagate values via connections (iterate over a copy of the keys), I just relized this is a completly wrong but new implementation for a forward pass, lets see how it goes!
        for connection_id in list(self.connections.keys()):
            self.connections[connection_id][3] += 1
            source_neuron_id, target_neuron_id, weight, usage = self.connections[connection_id]
            # Remove connection if source or target is missing
            if source_neuron_id not in self.neurons or target_neuron_id not in self.neurons:
                del self.connections[connection_id]
                continue
            self.neurons[target_neuron_id][2] += self.neurons[source_neuron_id][2] * weight + self.neurons[source_neuron_id][0] # Update target's value
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
            data = {
                'neurons': self.neurons,
                'connections': self.connections,
                'input_ids': self.input_ids,
                'output_ids': self.output_ids,
                'fitness': self.fitness,
                'wins': self.wins,
                'losses': self.losses,
                'draws': self.draws,
                'legal_count': self.legal_count
            }
            json.dump(data, file)
    
    def load(self, filename: str):
        with open(filename, 'r') as file:
            data = json.load(file)
            self.neurons = {int(k): v for k, v in data['neurons'].items()}
            self.connections = {int(k): [data['connections'][k][0], data['connections'][k][1], data['connections'][k][2], data['connections'][k][3]] for k in data['connections']}
            self.input_ids = data['input_ids']
            self.output_ids = data['output_ids']
            self.fitness = data['fitness']
            self.wins = data['wins']
            self.losses = data['losses']
            self.draws = data['draws']
            self.legal_count = data['legal_count']



def get_top_move(moves: list, game: TicTacToe):
    best_move = None
    best_value = -1
    for i in range(len(moves)):
        if game.is_valid_move(i) and moves[i] > best_value:
            best_move = i
            best_value = moves[i]
    return best_move

def train():
    game = TicTacToe()
    # Parameters
    POPULATION_SIZE = 20
    ELITE_SIZE = 4
    GENERATIONS = 100
    MUTATION_RATE = 0.1

    population = []

    # Setup the initial population, mutate it too to add some diversity
    for _ in range(POPULATION_SIZE):
        SSNN = SelfLearningNeuralNetwork(9, 9)
        SSNN.mutate(0.25)
        population.append(SSNN)

    # Training loop, find the best Neural Network
    for generation in range(GENERATIONS):
        for NN in population:
            # Complete with the rest of the population
            for opp in range(POPULATION_SIZE):
                game.reset()
                user_turn = True
                # Get the other Neural Network
                opponent = population[opp]
                if opponent == NN:
                    continue
                while not game.is_over():
                    if user_turn:
                        state = game.board
                        moves = NN.forward(state)
                        top_move = moves.index(max(moves))
                        best_move = get_top_move(moves, game)
                        if top_move == best_move:
                            NN.fitness += 100 # Reward for making the best move
                            NN.legal_count += 1
                        else:
                            NN.fitness -= 100 # Greatly penalize for making an illegal move
                        game.play(best_move)
                    else:
                        state = game.board
                        moves = opponent.forward(state)
                        top_move = moves.index(max(moves))
                        best_move = get_top_move(moves, game)
                        if top_move == best_move:
                            opponent.fitness += 100 # Reward for making the best move
                            opponent.legal_count += 1
                        else:
                            opponent.fitness -= 100 # Greatly penalize for an illegal move
                        game.play(best_move)
                    user_turn = not user_turn
                # Reward the winner, penalize the loser, and partially reward a draw
                # game.winner, if X wins, returns 1, if O wins, returns -1, if draw, returns 0
                if game.winner == 0:
                    NN.draws += 1
                    opponent.draws += 1
                    NN.fitness += 5
                    opponent.fitness += 5
                elif game.winner == 1:
                    NN.wins += 1
                    opponent.losses += 1
                    NN.fitness += 10
                    opponent.fitness -= 10
                else:
                    NN.losses += 1
                    opponent.wins += 1
                    NN.fitness -= 10
                    opponent.fitness += 10

        population.sort(key=lambda x: x.fitness, reverse=True)
        
        elite_population = population[:ELITE_SIZE]

        # Save the best Neural Network with its normalized fitness and generation
        elite_population[0].save(f'models/ssnn_gen_{generation + 1}_fit_{elite_population[0].fitness / POPULATION_SIZE}.json')

        # Print the best Neural Network's fitness
        print(f'Generation {generation + 1}, Fitness: {elite_population[0].fitness / POPULATION_SIZE}')
        # Print the best Neural Network's wins, losses, and draws
        print(f'Wins: {elite_population[0].wins}, Losses: {elite_population[0].losses}, Draws: {elite_population[0].draws}, Legal Moves: {elite_population[0].legal_count}')

        # Crossover the elite population to create the next generation and keep the elite population
        new_population = elite_population.copy()
        for _ in range(POPULATION_SIZE - ELITE_SIZE):
            parent1 = random.choice(elite_population)
            elite_population.remove(parent1) # Prevents the same parent from being selected twice
            parent2 = random.choice(elite_population)
            elite_population.append(parent1) # Add back the removed parent
            child = parent1.crossover(parent2)
            child.mutate(MUTATION_RATE)
            new_population.append(child)
        
        # Reset the stats of the new population
        for NN in new_population:
            NN.fitness = 0
            NN.wins = 0
            NN.losses = 0
            NN.draws = 0
            NN.legal_count = 0
        
        population = new_population.copy()

    # Save the best Neural Network
    population[0].save('ssnn.json')


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
            if not game.is_valid_move(move - 1):
                print('Invalid move!')
                continue
            game.play(move - 1)
        else:
            state = game.board
            moves = SSNN.forward(state)
            best_move = get_top_move(moves, game)
            game.play(best_move)
        user_turn = not user_turn

    game.print_board()
    if game.winner == 0:
        print('It is a draw!')
    elif game.winner == 1:
        if user_first == 'y':
            print('You win!')
        else:
            print('You lose!')
    else:
        if user_first == 'n':
            print('You win!')
        else:
            print('You lose!')
if __name__ == '__main__':
    # Uncomment one of the following lines to run training or play mode:
    # train()
    play('ssnn.json')
