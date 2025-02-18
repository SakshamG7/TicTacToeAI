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
    diff = 0.5
    # 0.4 seems to work the best for tuning Confidence and 0.5 works best with wins/loss/draw ratio, found with even further testing
    # 0.45 looks promising
    # 0.5 is getting better, just takes some time
    # 0.8 is also promising and might be better than 0.5
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
        self.neurons = {}       # neuron_id -> [bias, usage, value]
        self.connections = {}   # connection_id -> [source_neuron_id, target_neuron_id, weight, usage]
        self.input_ids = []     # List of input neuron IDs
        self.output_ids = []    # List of output neuron IDs
        self.fitness = 0
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.legal_count = 0
        self.total_moves = 0
        # Cached topological order for efficient forward passes.
        self.cached_topological_order = None

        # Create input neurons.
        for i in range(input_size):
            self.add_neuron(i, 0)
            self.input_ids.append(i)

        # Create output neurons.
        for i in range(input_size, input_size + output_size):
            self.add_neuron(i, 0)
            self.output_ids.append(i)

    def invalidate_cache(self):
        """Invalidates the cached topological order."""
        self.cached_topological_order = None

    def add_neuron(self, neuron_id: int, bias):
        self.neurons[neuron_id] = [bias, 0, 0]
        self.invalidate_cache()

    def add_connection(self, connection_id, source_neuron_id, target_neuron_id, weight):
        self.connections[connection_id] = [source_neuron_id, target_neuron_id, weight, 0]
        self.invalidate_cache()

    def creates_cycle(self, source: int, target: int) -> bool:
        """
        Checks whether adding a connection from source to target would create a cycle.
        A depth-first search (DFS) is performed starting from the target to see if we can reach the source.
        """
        visited = set()

        def dfs(node):
            if node == source:
                return True
            visited.add(node)
            for conn in self.connections.values():
                if conn[0] == node:
                    next_node = conn[1]
                    if next_node not in visited and dfs(next_node):
                        return True
            return False

        return dfs(target)

    def topological_sort(self):
        """Returns a cached topological sort if available; otherwise computes and caches it."""
        if self.cached_topological_order is not None:
            return self.cached_topological_order

        in_degree = {neuron: 0 for neuron in self.neurons}
        for conn in self.connections.values():
            target = conn[1]
            if target in in_degree:
                in_degree[target] += 1

        queue = [n for n in in_degree if in_degree[n] == 0]
        order = []
        while queue:
            current = queue.pop(0)
            order.append(current)
            for conn in self.connections.values():
                if conn[0] == current:
                    target = conn[1]
                    if target not in in_degree:
                        continue
                    in_degree[target] -= 1
                    if in_degree[target] == 0:
                        queue.append(target)
        self.cached_topological_order = order if len(order) == len(self.neurons) else None
        return self.cached_topological_order

    def forward(self, inputs: list):
        """
        Executes a forward pass. If the network is acyclic, neurons are processed in topological order.
        Otherwise, iterative updates are performed until convergence.
        """
        if len(inputs) != self.input_size:
            raise ValueError('Invalid input size')

        # Reset neuron values.
        for neuron_id in self.neurons:
            self.neurons[neuron_id][2] = 0

        # Set input neuron values.
        for idx, neuron_id in enumerate(self.input_ids):
            self.neurons[neuron_id][2] = inputs[idx]

        order = self.topological_sort()
        if order is not None:
            # Propagate in topological order.
            for neuron_id in order:
                if neuron_id in self.input_ids:
                    continue
                total_input = self.neurons[neuron_id][0]  # Start with bias.
                for conn in self.connections.values():
                    if conn[1] == neuron_id:
                        source_id = conn[0]
                        if source_id not in self.neurons:
                            continue
                        total_input += self.neurons[source_id][2] * conn[2]
                self.neurons[neuron_id][2] = SakshamsLinearCutOff(total_input)
        else:
            # Cycle detected: use iterative updates.
            max_iterations = 10
            epsilon = 1e-3
            for _ in range(max_iterations):
                delta = 0
                updated_values = {}
                for neuron_id in self.neurons:
                    if neuron_id in self.input_ids:
                        updated_values[neuron_id] = self.neurons[neuron_id][2]
                    else:
                        total_input = self.neurons[neuron_id][0]
                        for conn in self.connections.values():
                            if conn[1] == neuron_id:
                                source_id = conn[0]
                                if source_id not in self.neurons:
                                    continue
                                total_input += self.neurons[source_id][2] * conn[2]
                        new_value = SakshamsLinearCutOff(total_input)
                        updated_values[neuron_id] = new_value
                        delta = max(delta, abs(new_value - self.neurons[neuron_id][2]))
                for neuron_id in self.neurons:
                    self.neurons[neuron_id][2] = updated_values[neuron_id]
                if delta < epsilon:
                    break

        outputs = [self.neurons[n][2] for n in self.output_ids]
        return softmax(outputs)

    def crossover(self, parent):
        """
        Performs NEAT-style crossover with another network by aligning genes (neurons and connections)
        based on their (source, target) pairs. Matching genes are randomly chosen, while disjoint and excess genes 
        are inherited from the fitter parent. In case of equal fitness, genes are inherited from both parents.
        """
        # Determine the fitter parent; if fitnesses are equal, set fitter to None.
        if self.fitness > parent.fitness:
            fitter = self
        elif self.fitness < parent.fitness:
            fitter = parent
        else:
            fitter = None  # Equal fitness

        child = SelfLearningNeuralNetwork(self.input_size, self.output_size)

        # Neuron crossover: take the union of neuron IDs from both parents.
        for neuron_id in set(self.neurons.keys()).union(parent.neurons.keys()):
            if neuron_id in self.neurons and neuron_id in parent.neurons:
                # Matching neuron: choose randomly from either parent.
                child.neurons[neuron_id] = (
                    self.neurons[neuron_id].copy() if random.random() < 0.5
                    else parent.neurons[neuron_id].copy()
                )
            elif neuron_id in self.neurons:
                # Disjoint/excess gene from self.
                if fitter is None or fitter is self:
                    child.neurons[neuron_id] = self.neurons[neuron_id].copy()
            elif neuron_id in parent.neurons:
                # Disjoint/excess gene from parent.
                if fitter is None or fitter is parent:
                    child.neurons[neuron_id] = parent.neurons[neuron_id].copy()

        # Build connection dictionaries keyed by (source, target).
        conn_self = {(data[0], data[1]): data for data in self.connections.values()}
        conn_parent = {(data[0], data[1]): data for data in parent.connections.values()}

        # Connection crossover: align genes by their (source, target) pairs.
        for key in set(conn_self.keys()).union(conn_parent.keys()):
            if key in conn_self and key in conn_parent:
                # Matching gene: select randomly.
                gene = random.choice([conn_self[key], conn_parent[key]])
            elif key in conn_self:
                if fitter is None or fitter is self:
                    gene = conn_self[key]
                else:
                    continue
            elif key in conn_parent:
                if fitter is None or fitter is parent:
                    gene = conn_parent[key]
                else:
                    continue

            source, target, weight, usage = gene
            # Add gene only if both neurons exist and adding it does not create a cycle.
            if source in child.neurons and target in child.neurons and not child.creates_cycle(source, target):
                new_id = max(child.connections.keys()) + 1 if child.connections else 0
                child.connections[new_id] = gene.copy()

        child.input_ids = self.input_ids.copy()
        child.output_ids = self.output_ids.copy()
        child.invalidate_cache()
        child.topological_sort()
        return child


    def mutate(self, mutation_rate: float = 0.1, MAX_MUT: int = 500, MIN_MUT: int = 1):
        """
        Mutates the network by adding new connections/neurons or modifying existing ones.
        New connections are added only if they do not create a cycle.
        """
        self.invalidate_cache()
        # Add new connections to random input-output pairs.
        for i in range(max(MIN_MUT, min(MAX_MUT, random.randint(0, 1 + len(self.neurons) // 2)))):
            # Remove a connection (and clean up hidden neurons).
            if random.random() < mutation_rate / 4 and self.connections:
                connection_id = random.choice(list(self.connections.keys()))
                if self.connections[connection_id][3] < 10:
                    for neuron_id in list(self.neurons.keys()):
                        if neuron_id in self.input_ids or neuron_id in self.output_ids:
                            continue
                        if neuron_id not in [self.connections[connection_id][0], self.connections[connection_id][1]]:
                            del self.neurons[neuron_id]
                    del self.connections[connection_id]

            # 1% chance: remove a hidden neuron that is not used often.
            if random.random() < mutation_rate / 8 and self.neurons:
                neuron_id = random.choice(list(self.neurons.keys()))
                if neuron_id not in self.input_ids and neuron_id not in self.output_ids:
                    if self.neurons[neuron_id][1] < 10:
                        del self.neurons[neuron_id]
                        for connection_id in list(self.connections.keys()):
                            src, tgt, _, _ = self.connections[connection_id]
                            if src not in self.neurons or tgt not in self.neurons:
                                del self.connections[connection_id]

            # Add a new hidden neuron between two random neurons.
            if random.random() < mutation_rate / 4:
                new_id = max(self.neurons.keys()) + 1
                n1 = random.choice(list(self.neurons.keys()))
                n2 = random.choice(list(self.neurons.keys()))
                while n1 == n2:
                    n2 = random.choice(list(self.neurons.keys()))
                self.add_neuron(new_id, random.random())
                if not self.creates_cycle(n1, new_id):
                    connection_id = max(self.connections.keys()) + 1 if self.connections else 0
                    self.add_connection(connection_id, n1, new_id, random.uniform(-1, 1))
                if not self.creates_cycle(n2, new_id):
                    connection_id = max(self.connections.keys()) + 1 if self.connections else 0
                    self.add_connection(connection_id, n2, new_id, random.uniform(-1, 1))

            # Add a new connection between any two neurons.
            if random.random() < mutation_rate / 2:
                source_neuron_id = random.choice(list(self.neurons.keys()))
                target_neuron_id = random.choice(list(self.neurons.keys()))
                if source_neuron_id != target_neuron_id and not self.creates_cycle(source_neuron_id, target_neuron_id):
                    weight = random.uniform(-1, 1)
                    connection_id = max(self.connections.keys()) + 1 if self.connections else 0
                    self.add_connection(connection_id, source_neuron_id, target_neuron_id, weight)

            # Mutate weights.
            if random.random() < mutation_rate and self.connections:
                connection_id = random.choice(list(self.connections.keys()))
                self.connections[connection_id][2] += random.uniform(-1, 1)

            # Mutate biases.
            if random.random() < mutation_rate / 2 and self.neurons:
                neuron_id = random.choice(list(self.neurons.keys()))
                self.neurons[neuron_id][0] += random.uniform(-1, 1)


    def copy(self):
        new_network = SelfLearningNeuralNetwork(self.input_size, self.output_size)
        new_network.neurons = {k: v.copy() for k, v in self.neurons.items()}
        new_network.connections = {k: v.copy() for k, v in self.connections.items()}
        new_network.input_ids = self.input_ids.copy()
        new_network.output_ids = self.output_ids.copy()
        new_network.fitness = self.fitness
        new_network.wins = self.wins
        new_network.losses = self.losses
        new_network.draws = self.draws
        new_network.legal_count = self.legal_count
        new_network.total_moves = self.total_moves
        new_network.invalidate_cache()
        new_network.topological_sort()
        return new_network

    def save(self, filename: str):
        with open(filename, 'w') as file:
            data = {
                'neurons': self.neurons,
                'connections': self.connections,
                'input_ids': self.input_ids,
                'output_ids': self.output_ids
            }
            json.dump(data, file)

    def load(self, filename: str):
        with open(filename, 'r') as file:
            data = json.load(file)
            self.neurons = {int(k): v for k, v in data['neurons'].items()}
            self.connections = {
                int(k): [data['connections'][k][0], data['connections'][k][1],
                         data['connections'][k][2], data['connections'][k][3]]
                for k in data['connections']
            }
            self.input_ids = data['input_ids']
            self.output_ids = data['output_ids']


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
    POPULATION_SIZE = 100
    ELITE_SIZE = 10
    GENERATIONS = 10000
    MUTATION_RATE = 0.1
    RANDO_TURNS = POPULATION_SIZE * 4 # The number of times that the AI plays with a player that makes random moves, this allows the AI to explore more and learn more

    population = []

    # Setup the initial population, mutate it too to add some diversity
    for _ in range(POPULATION_SIZE):
        SSNN = SelfLearningNeuralNetwork(9, 9)
        SSNN.mutate(0.25)
        population.append(SSNN)

    # Training loop, find the best Neural Network
    for generation in range(GENERATIONS):
        # random.seed(generation) might be useful to keep the randomness more consistent and make it less dependant on luck
        for NN in population:
            # Complete with the rest of the population
            for opponent in population:
                game.reset()
                user_turn = True
                while not game.is_over():
                    if user_turn:
                        state = game.board
                        moves = NN.forward(state)
                        top_move = moves.index(max(moves))
                        best_move = get_top_move(moves, game)
                        if top_move == best_move:
                            NN.legal_count += 1
                        NN.total_moves += 1
                        game.play(best_move)
                    else:
                        state = game.board
                        moves = opponent.forward(state)
                        top_move = moves.index(max(moves))
                        best_move = get_top_move(moves, game)
                        if top_move == best_move:
                            opponent.legal_count += 1
                        opponent.total_moves += 1
                        game.play(best_move)
                    user_turn = not user_turn
                # Reward the winner, penalize the loser, and partially reward a draw
                # game.winner, if X wins, returns 1, if O wins, returns -1, if draw, returns 0
                if game.winner == 0:
                    NN.draws += 1
                    opponent.draws += 1
                elif game.winner == 1:
                    NN.wins += 1
                    opponent.losses += 1
                else:
                    NN.losses += 1
                    opponent.wins += 1
            
            # Play against a random player to explore more
            for turn in range(RANDO_TURNS // 2):
                # random.seed(generation + turn) # This is to make the randomness more consistent and less dependant on luck
                game.reset()
                user_turn = True
                while not game.is_over():
                    state = game.board
                    if user_turn:
                        moves = NN.forward(state)
                        top_move = moves.index(max(moves))
                        best_move = get_top_move(moves, game)
                        if top_move == best_move:
                            NN.legal_count += 1
                        NN.total_moves += 1
                        game.play(best_move)
                    else:
                        moves = [0 for _ in range(9)]
                        # Delete non valid moves
                        for move in moves:
                            if not game.is_valid_move(move):
                                moves.remove(move)
                        best_move = random.choice(moves)
                        game.play(best_move)
                    user_turn = not user_turn
                if game.winner == 0:
                    NN.draws += 1
                elif game.winner == 1:
                    NN.wins += 1
                else:
                    NN.losses += 10 # Really shouldnt lose to a player that makes random moves
            # Now the random player plays first
            for _ in range(RANDO_TURNS // 2):
                # random.seed(generation + turn) # Lowers the chance of the AI winning by luck
                game.reset()
                user_turn = False
                while not game.is_over():
                    state = game.board
                    if user_turn:
                        moves = NN.forward(state)
                        top_move = moves.index(max(moves))
                        best_move = get_top_move(moves, game)
                        if top_move == best_move:
                            NN.legal_count += 1
                        NN.total_moves += 1
                        game.play(best_move)
                    else:
                        moves = [0 for _ in range(9)]
                        # Delete non valid moves
                        for move in moves:
                            if not game.is_valid_move(move):
                                moves.remove(move)
                        best_move = random.choice(moves)
                        game.play(best_move)
                    user_turn = not user_turn
                if game.winner == 0:
                    NN.draws += 1
                elif game.winner == 1:
                    NN.wins += 1
                else:
                    NN.losses += 10 # Really shouldnt lose to a player that makes random moves

        # Calculate the fitness of each Neural Network
        for NN in population:
            confidence = 100 * NN.legal_count / NN.total_moves
            #T1
            NN.fitness = ((NN.wins - NN.losses + NN.draws / 2) * confidence) / (POPULATION_SIZE + RANDO_TURNS) # Normalize the fitness
            # T2
            # NN.fitness = (NN.wins + NN.draws / 2) * confidence ** 2 / 100 / (POPULATION_SIZE + RANDO_TURNS) # Normalize the fitness
            # NN.fitness -= NN.losses ** 2 / (POPULATION_SIZE) # Penalize the Neural Network for losing

        population.sort(key=lambda x: x.fitness, reverse=True)
        
        elite_population = population[:ELITE_SIZE]

        # Save the best Neural Network with its normalized fitness and generation
        elite_population[0].save(f'models/ssnn_gen_{generation + 1}_fit_{elite_population[0].fitness}.json')

        # Print the best Neural Network's fitness
        print(f'Generation {generation + 1}, Fitness: {elite_population[0].fitness}')
        # Print the best Neural Network's wins, losses, and draws
        print(f'Wins: {elite_population[0].wins}, Losses: {elite_population[0].losses}, Draws: {elite_population[0].draws}, Confidence: {round(100 * elite_population[0].legal_count / elite_population[0].total_moves, 2)}%')

        # Crossover the elite population to create the next generation and keep the elite population
        new_population = elite_population.copy()
        for _ in range(POPULATION_SIZE - ELITE_SIZE):
            parent1 = random.choice(elite_population)
            parent1 = parent1.copy() # Prevents the parent from being modified
            parent2 = random.choice(elite_population)
            parent2 = parent2.copy() # Prevents the parent from being modified
            child = parent1.crossover(parent2)
            child.mutate(MUTATION_RATE)
            new_population.append(child)
        
        population = []
        # Reset the stats of the new population
        for NN in new_population:
            NN.fitness = 0
            NN.wins = 0
            NN.losses = 0
            NN.draws = 0
            NN.legal_count = 0
            NN.total_moves = 0
            NN.invalidate_cache()
            population.append(NN)
        POPULATION_SIZE = len(new_population) # Update the population size

    # Save the best Neural Network
    population[0].save('ssnn_v2.json')


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
    train()
    # play('models/ssnn.json')
