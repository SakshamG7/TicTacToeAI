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
import ssnn
import random
from tictactoe import TicTacToe
import math

# Training the AI
# Hyperparameters, most numbers are arbitrary
population_size = 20
generations = 1000000
mutation_rate = 0.01
elite_percentage = 0.3 # The top 10% of the population will be carried over to the next generation, ensures that the best AI is not lost
model_dir = "models" # Directory to save the models

elite_cutoff = max(2, int(elite_percentage * population_size)) # Ensure that at least two AIs is carried over

# Parameters
input_size = 9
hidden_size = [64, 32, 16] # Some arbitrary numbers I chose
hidden_layers = len(hidden_size) # Some arbitrary number I chose
output_size = 9

print("Training the AI...")

# Initialize the population
population = []

best_fitness = float("-inf")

for i in range(population_size):
    population.append(ssnn.SimpleNeuralNetwork(input_size, hidden_layers, hidden_size, output_size))

# First encourage playing legal moves, then encourage winning, then encourage not losing

# Play the games
for generation in range(generations):
    for i in range(population_size):
        for j in range(population_size):
            if i == j:
                continue
            invalid_move = False
            ai1 = population[i]
            ai2 = population[j]
            game = TicTacToe()
            while not game.over:
                input_data = game.board
                if game.turn == 1:
                    moves = ai1.forward(input_data).data[0]
                else:
                    moves = ai2.forward(input_data).data[0]
                # Choose the highest probability move, if it is not valid, choose the next highest probability move, and so on, if the first move is not valid, the AI only gets partial credit for the move.
                # Check if the highest probability move is valid
                first_move = moves.index(max(moves))
                move = first_move
                thought_away_from_first_move = 0
                while not game.is_valid_move(move):
                    moves[move] = float("-1")
                    move = moves.index(max(moves))
                    thought_away_from_first_move += 1
                if not game.is_valid_move(first_move):
                    invalid_move = True
                    # Punish the AI for making an invalid move and by how far it was from the first move
                    if game.turn == 1:
                        ai1.update_fitness(-1 * thought_away_from_first_move**2)
                    else:
                        ai2.update_fitness(-1 * thought_away_from_first_move**2)
                else:
                    # Reward the AI for making a valid move
                    if game.turn == 1:
                        ai1.update_fitness(moves[first_move])
                    else:
                        ai2.update_fitness(moves[first_move])
                game.play(move)
            if invalid_move:
                # Punish the AI for making an invalid move, reset all their progress
                if game.turn == 1:
                    ai1.wins = 0
                    ai1.losses = 0
                    ai1.draws = 0
                else:
                    ai2.wins = 0
                    ai2.losses = 0
                    ai2.draws = 0
            elif game.winner == 1: # AI 1 wins
                ai1.wins += 1
                ai2.losses += 1
            elif game.winner == -1: # AI 2 wins                
                ai2.wins += 1
                ai1.losses += 1
            elif game.winner == 0: # Draw, better than losing so give both AIs half a point
                ai1.draws += 1
                ai2.draws += 1
    
    # Calculate the fitness of the AIs
    for ai in population:
        ai.update_fitness((ai.wins + ai.draws // 2) - ai.losses ** 4)
        # Normalize the fitness
        ai.update_fitness(ai.get_fitness() / population_size)

        

    # Sort the population based on the fitness
    population.sort(key=lambda x: x.get_fitness(), reverse=True)

    # Check if the best AI in this generation is better than the best AI in the previous generation
    # Fitness will change every generation, so we can only trust that the latest AI is the best, especially since the previous elite models are carried over, fitness will lower as there will be more draws
    if population[0].get_fitness() > best_fitness:
        population[0].save(f"{model_dir}/best_ai_gen_{generation}_fitness_{population[0].get_fitness()}.json")
        best_fitness = population[0].get_fitness()

    # Print the best AI in the generation
    print(f"Generation {generation + 1}: Best AI Fitness: {population[0].get_fitness()} | Best Overall Fitness: {best_fitness}")
    print(f"Wins: {population[0].wins}, Losses: {population[0].losses}, Draws: {population[0].draws}")

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
        ai.set_fitness(0)
        ai.wins = 0
        ai.losses = 0
        ai.draws = 0

    # Update the population
    population = new_population

population[0].save(f"{model_dir}/best_ai_gen_final.json")

# End of Training
print("Training Complete!")
