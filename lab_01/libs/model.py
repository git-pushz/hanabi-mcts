import numpy as np

NUM_COLUMNS = 7
COLUMN_HEIGHT = 6
WINDOW_LENGTH = 4
FOUR = 4

PLAYER1 = 1
PLAYER2 = -1

class Model:
    def __init__(self, board):
        self.board = board
    
    def valid_moves(self):
        """Returns columns where a disc may be played"""
        return [n for n in range(NUM_COLUMNS) if self.board[n, COLUMN_HEIGHT - 1] == 0]

    def make_move(self, move):
        """Updates `board` as `player` drops a disc in `column`"""
        (index,) = next((i for i, v in np.ndenumerate(self.board[move.position]) if v == 0))
        self.board[move.position, index] = move.player

    def make_random_move(self, player):
        legal_moves = self.valid_moves()
        if (len(legal_moves)==0):
            return False
        random_move = np.random.choice(legal_moves)
        self.make_move(GameMove(player, random_move))
        return True
    
    def check_win(self):
        if (self.four_in_a_row(PLAYER1)):
            return PLAYER1
        if (self.four_in_a_row(PLAYER2)):
            return PLAYER2
        else:
            return 0

    def four_in_a_row(self, player):
        """Checks if `player` has a 4-piece line"""
        return (
            any(
                all(self.board[c, r] == player)
                for c in range(NUM_COLUMNS)
                for r in (list(range(n, n + FOUR)) for n in range(COLUMN_HEIGHT - FOUR + 1))
            )
            or any(
                all(self.board[c, r] == player)
                for r in range(COLUMN_HEIGHT)
                for c in (list(range(n, n + FOUR)) for n in range(NUM_COLUMNS - FOUR + 1))
            )
            or any(
                np.all(self.board[diag] == player)
                for diag in (
                    (range(ro, ro + FOUR), range(co, co + FOUR))
                    for ro in range(0, NUM_COLUMNS - FOUR + 1)
                    for co in range(0, COLUMN_HEIGHT - FOUR + 1)
                )
            )
            or any(
                np.all(self.board[diag] == player)
                for diag in (
                    (range(ro, ro + FOUR), range(co + FOUR - 1, co - 1, -1))
                    for ro in range(0, NUM_COLUMNS - FOUR + 1)
                    for co in range(0, COLUMN_HEIGHT - FOUR + 1)
                )
            )
        )
    
    def copy(self):
        model = Model(np.copy(self.board))
        return model


class GameMove:
    def __init__(self, player, position):
        self.player = player
        self.position = position

