import GameData
import numpy as np
from enum import Enum

Colors = Enum("GREEN", "YELLOW", "BLUE", "RED", "WHITE")

class MentalState():
    '''
    Mental state representation for a single card in a player's hand
    '''
    def __init__(self):
        col = np.array([3, 2, 2, 2, 1])
        col = col.reshape(col.size, 1)
        self.table = np.tile(col, 5)

    def rank_hint_received(self, rank: int):
        '''
        Update mental state for a card when received an hint based on card rank
        '''
        i = [j for j in range(5) if j != rank]
        self.table[i, :] = 0

    def color_hint_received(self, color: Colors):
        '''
        Update mental state for a card when received an hint based on color
        '''
        i = [j for j in range(5) if j != color]
        self.table[:, i] = 0

    def specific_hint_received(self, rank: int, color: Colors):
        '''
        Update mental state for a card when a card is drawn from deck
        '''
        # self.table[rank, color] must be > 0
        assert(self.table[rank, color] > 0)
        self.table[rank, color] -= 1
        
class MentalStateGlobal():
    '''
    Class for holding the mental state of each card in the hand of each player
    '''
    def __init__(self, players_number: int, hands: dict):
        self.matrix = np.array([np.array([MentalState() for _ in range(5)], dtype=MentalState) for _ in range(players_number)])

class Agent():
    def __init__(self, name: str, data: GameData.ServerGameStateData, players_names: list):
        self.name = name
        # name of the current player
        self.currentPlayer = data.currentPlayer
        # list of players in turn order
        self.players = players_names
        # position of the player in the round
        self.nr = self.players.index(self.name)
        # hands[player_name] = list of cards
        self.hands = {player[name]: player[hand] for player in data.players}
        self.hands[self.name] = []
        # knowledge[player_number][card position][card_color][card_rank]
        self.knowledge = np.empty((len(players), 5, 5, ))
        self.trash = []
        self.played = []
        # list of cards that are currently on the top of each stack on the board
        self.board = [0] * 5
        self.hints = data.usedNoteTokens
        self.errors = data.usedStormTokens

    ## TODO
    # funzione chiamata alla ricezione di un ready
    # funzione inizializzazione variabili di stato ad inizio partita (quando tutti ready)
    # funzione per aggiornare variabili di stato
    # funzione per decidere la mossa da fare

    # def update_state():

    # def make_move():
 