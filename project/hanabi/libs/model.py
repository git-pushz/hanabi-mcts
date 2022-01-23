import numpy as np
import copy
from ..agent import Agent

NUM_COLUMNS = 7
COLUMN_HEIGHT = 6
WINDOW_LENGTH = 4
FOUR = 4

PLAYER1 = 1
PLAYER2 = -1

class GameMove:
    '''
    sender: string, name of the sender
    destination: string, name of the destination player
    action_type: can be "color" or "value"
    hint_value: can be the color or the value of the card
    hand_card_ordered: the card in hand that has been played/ discarded

    action_type == "play" use action_type, sender, hand_card_ordered
    action_type == "discard" use action_type, sender, hand_card_ordered
    action_type == "hint" use action_type, sender, destination, hint_value
    '''
    # def __init__(self, player, position):
    def __init__(self, player: int, action_type: str, sender: str, destination: str = None, hand_card_ordered: int = None, hint_value = None):
        ## removed
        # self.player = player
        # self.position = position
        ##
        ## added
        self.player = player
        self.action_type = action_type
        self.sender = sender
        self.destination = destination
        self.hand_card_ordered = hand_card_ordered
        self.hint_value = hint_value
        ##

class Model:
    # def __init__(self, board):
    def __init__(self, agent: Agent):
        ## removed
        # self.board = board
        ##
        ## added
        # in agent it is stored the state of the game
            # players: list of players in turn order (Player: name, ready, hand (list of cards in hand order))
            # hands
            # knowledge
            # trash
            # board
            # hints
            # errors
        self.agent = copy.deepcopy(agent)
        ##
    
    def valid_moves(self):
        '''
        Returns all possible moves available at the current state 
        (that correspond to a certain tree level
        that corresponds to a certain player)
        '''
        ## TODO
        return
        ## removed
        # """Returns columns where a disc may be played"""
        # return [n for n in range(NUM_COLUMNS) if self.board[n, COLUMN_HEIGHT - 1] == 0]

    def make_move(self, move: GameMove):
        '''
        Makes a move and updates the agent state accordingly
        '''
        ## TODO
        return
        ## removed
        # """Updates `board` as `player` drops a disc in `column`"""
        # (index,) = next((i for i, v in np.ndenumerate(self.board[move.position]) if v == 0))
        # self.board[move.position, index] = move.player

    # the name should be changed to something like make_intentional_move, because it shouldn't be random
    def make_random_move(self, player: int):
        ## TODO
        return
        # legal_moves = self.valid_moves()
        # if (len(legal_moves)==0):
        #     return False
        # random_move = np.random.choice(legal_moves)
        # self.make_move(GameMove(player, random_move))
        # return True
    
    # the name should be changed to something like check_if_ended, because there is no winner
    def check_win(self):
        ## TODO
        return
        # if (self.four_in_a_row(PLAYER1)):
        #     return PLAYER1
        # if (self.four_in_a_row(PLAYER2)):
        #     return PLAYER2
        # else:
        #     return 0

    def copy(self):
        # model = Model(np.copy(self.board))
        model = Model(copy.deepcopy(self.agent))
        return model



