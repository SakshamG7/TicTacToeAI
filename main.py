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

# Parameters
input_size = 9
hidden_layers = 3 # Some arbitrary number I chose
hidden_size = [9, 3, 9] # Some arbitrary numbers I chose, resembles a bottleneck architecture
output_size = 9