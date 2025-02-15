import ssnn
from tictactoe import TicTacToe

# Play the game against the AI
print("Playing against the AI...")
# Load best ai from the models directory
ai = ssnn.load("models/best_ai_gen_2029_fitness_68.02656886710022.json")
game = TicTacToe()
while not game.over:
    if game.turn == 1:
        game.print_board()
        move = int(input("Enter your move: "))
    else:
        # Plays the move with the highest probability, if the move is not valid, it plays the next highest probability move, and so on
        moves = ai.forward(game.board).data[0]
        first_move = moves.index(max(moves))
        move = first_move

        while not game.is_valid_move(move):
            moves[move] = float("-1")
            move = moves.index(max(moves))
    game.play(move)
game.print_board()
if game.winner == 1:
    print("You Win!")
elif game.winner == -1:
    print("AI Wins!")
else:
    print("Draw!")
