"""
Saksham's Custom Training Implementation, will be used for training the Tic Tac Toe AI.
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
import snn
import random

# Tic Tac Toe Game
class TicTacToe(object):
    # __init__: constructor for the Tic Tac Toe game
    def __init__(self) -> None:
        self.board = [0] * 9
        self.turn = 1
        self.winner = 0
        self.over = False
    
    def reset(self) -> None:
        self.board = [0] * 9
        self.turn = 1
        self.winner = 0
        self.over = False

    # play: plays a move on the board
    # move -> int: the move to play
    def play(self, move: int) -> None:
        if self.over:
            return
        
        if self.board[move] == 0:
            self.board[move] = self.turn
            self.turn = -self.turn
        else:
            return

        self.check_winner()
        self.check_draw()

    def print_board(self) -> None:
        for i in range(3):
            print(self.board[3 * i:3 * i + 3])

    # check_winner: checks if there is a winner
    def check_winner(self) -> bool:
        for i in range(3):
            if self.board[i] == self.board[i + 3] == self.board[i + 6] != 0:
                self.winner = self.board[i]
                self.over = True
                return True
            if self.board[3 * i] == self.board[3 * i + 1] == self.board[3 * i + 2] != 0:
                self.winner = self.board[3 * i]
                self.over = True
                return True
        if self.board[0] == self.board[4] == self.board[8] != 0:
            self.winner = self.board[0]
            self.over = True
            return True
        if self.board[2] == self.board[4] == self.board[6] != 0:
            self.winner = self.board[2]
            self.over = True
            return True
        if all([x != 0 for x in self.board]):
            self.over = True
            return True
        return False

    # check_draw: checks if the game is a draw
    def check_draw(self) -> bool:
        if all([x != 0 for x in self.board]):
            self.over = True
            return True
        return False    

# Training the AI

# Hyperparameters, most numbers are arbitrary
population_size = 100
generations = 1000
mutation_rate = 0.05
elite_percentage = 0.1 # The top 10% of the population will be carried over to the next generation, ensures that the best AI is not lost
model_dir = "models" # Directory to save the models

elite_cutoff = max(2, int(elite_percentage * population_size)) # Ensure that at least two AIs is carried over

# Parameters
input_size = 9
hidden_layers = 3 # Some arbitrary number I chose
hidden_size = [9, 3, 9] # Some arbitrary numbers I chose, resembles a bottleneck architecture
output_size = 9

# Initialize the population
population = []

# Tracker for the best AI
best_fitness = float("-inf") # Litterally anything is better than this

for i in range(population_size):
    population.append(snn.SimpleNeuralNetwork(input_size, hidden_layers, hidden_size, output_size))

# Play the games
for generation in range(generations):
    for i in range(population_size):
        for j in range(population_size):
            if i == j:
                continue
            ai1 = population[i]
            ai2 = population[j]
            game = TicTacToe()
            while not game.over:
                if game.turn == 1:
                    move = ai1.forward(game.board).data.index(max(ai1.forward(game.board).data))
                else:
                    move = ai2.forward(game.board).data.index(max(ai2.forward(game.board).data))
                game.play(move)
            if game.winner == 1: # AI 1 wins
                # Reward the winning AI
                ai1.update_fitness(1)
                # Punish the losing AI
                ai2.update_fitness(-0.5)
            elif game.winner == -1: # AI 2 wins
                # Reward the winning
                ai2.update_fitness(1)
                # Punish the losing AI
                ai1.update_fitness(-0.5)
            elif game.winner == 0: # Draw, better than losing so give both AIs half a point
                ai1.update_fitness(0.5)
                ai2.update_fitness(0.5)
            game.reset()

    # Sort the population based on the fitness
    population.sort(key=lambda x: x.get_fitness(), reverse=True)

    # Check if the best AI in this generation is better than the best AI in the previous generation
    if population[0].get_fitness() > best_fitness:
        best_fitness = population[0].get_fitness()
        population[0].save(f"{model_dir}/best_ai_gen_{generation}_fitness_{best_fitness}.json")

    # Print the best AI in the generation
    print(f"Generation {generation + 1}: Best AI Fitness: {population[0].get_fitness()} | Best Overall Fitness: {best_fitness}")

    # Carry over the top 10% of the population
    elite = population[:elite_cutoff] # Ensure that at least one AI is carried over

    # Create the next generation via crossover and mutation
    new_population = elite.copy()
    for i in range(population_size - len(elite)):
        # pick two random parents, ensure they are not the same
        parent1 = random.choice(elite)
        elite.remove(parent1)
        parent2 = random.choice(elite)
        elite.append(parent1)
        child = parent1.crossover(parent2)
        child.mutate(mutation_rate)
        new_population.append(child)

    # Reset the fitness of the population
    for ai in new_population:
        ai.reset_fitness()

    # Update the population
    population = new_population

# Save the best AI
population[0].save(f"{model_dir}/best_ai_gen_{generation}_fitness_{best_fitness}_final.json")

# End of Training
print("Training Complete!")