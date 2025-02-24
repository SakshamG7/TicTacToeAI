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
        
        if self.is_valid_move(move):
            self.board[move] = self.turn
            self.turn = -self.turn
        else:
            return

        self.check_winner()
        self.check_draw()

    def is_valid_move(self, move: int) -> bool:
        return self.board[move] == 0

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
    
    def is_over(self) -> bool:
        return self.over
    
    # Check if a player has a move that will win the game
    def check_win_move(self, player: int) -> int:
        for i in range(9):
            if self.board[i] == 0:
                self.board[i] = player
                if self.check_winner():
                    self.board[i] = 0
                    return i
                self.board[i] = 0
        return -1