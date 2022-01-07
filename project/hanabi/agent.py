import GameData
import numpy as np
from game import Card

colors = ["green", "yellow", "blue", "red", "white"]
    
HAND_SIZE = 5

class MentalState():
    '''
    Mental state representation for a single card in a player's hand
    '''
    def __init__(self):
        col = np.array([3, 2, 2, 2, 1])
        col = col.reshape(col.size, 1)
        self.table = np.tile(col, len(colors))

    def rank_hint_received(self, rank: int):
        '''
        Update mental state for a card when received an hint based on card rank
        '''
        rank -= 1
        i = [j for j in range(HAND_SIZE) if j != rank]
        self.table[i, :] = 0

    def color_hint_received(self, color: int):
        '''
        Update mental state for a card when received an hint based on color
        '''
        i = [j for j in range(HAND_SIZE) if j != color]
        self.table[:, i] = 0

    def specific_hint_received(self, rank: int, color: int):
        '''
        Update mental state for a card when a card is drawn from deck
        '''
        rank -= 1
        # self.table[rank, color] must be > 0
        assert(self.table[rank, color] > 0)
        self.table[rank, color] -= 1
        
    def to_string(self) -> str:
        s = '\n'.join([''.join(['{:4}'.format(item) for item in row]) for row in self.table])
        s += '\n'
        return s
        

class PlayerMentalState():
    '''
    Class for holding the mental state of the hand of one player
    '''
    def __init__(self):
        self.ms_hand = np.array([MentalState() for _ in range(HAND_SIZE)], dtype=MentalState)

    def update_whole_hand(self, rank: int, color: int):
        for c in self.ms_hand:
            c.specific_hint_received(rank, color)
    
    def update_card(self, card_index: int, rank: int = None, color: int = None):
        assert((rank is None) != (color is None))
        if rank is None:
            self.ms_hand[card_index].color_hint_received(color)
        else:
            self.ms_hand[card_index].rank_hint_received(rank)

    def to_string(self) -> str:
        s = ''
        for i in range(HAND_SIZE):
            s += self.ms_hand[i].to_string()
        return s


class MentalStateGlobal():
    '''
    Class for holding the mental state of each card in the hand of each player
    '''
    def __init__(self, hands: dict):
        self.matrix = {k: PlayerMentalState() for k in hands.keys()}

        for name, hand in hands.items():
            for card in hand:
                for n in hands.keys():
                    if n != name:
                        self.matrix[n].update_whole_hand(card.value, colors.index(card.color))
    
    def card_discovered(self, hands:dict, last_player: str, old_card: Card, new_card: Card):
        '''
        Update mental state of each player when a card is played and a new card is drawn
        '''
        for name in hands.keys():
            if name != last_player:
                self.matrix[name].update_whole_hand(new_card.value, colors.index(new_card.color))
        self.matrix[last_player].update_whole_hand(old_card.value, old_card.color)

    def to_string(self) -> str:
        s = ''
        for k, v in self.matrix.items():
            s += f"Player {k}:\n"
            s += v.to_string()
        return s


class LastAction():
    '''
    Mini-class to store info about the last action performed by other players
    '''
    def __init__(self):
        self.last_player = None
        self.card_index = None

    def update_last_action(self, last_player: str, card_index: int):
        self.last_player = last_player
        self.card_index = card_index

class Agent():
    '''
    Our AI Hanabi agent player
    '''
    def __init__(self, name: str, data: GameData.ServerGameStateData, players_names: list):
        self.name = name
        # name of the current player
        self.currentPlayer = data.currentPlayer
        # list of players in turn order
        self.players = players_names
        # position of the player in the round
        self.nr = self.players.index(self.name)
        # hands[player_name] = list of cards
        self.hands = {player.name: player.hand for player in data.players}
        self.hands[self.name] = []
        # knowledge['player_name'][card position][card_color][card_rank]
        self.knowledge = MentalStateGlobal(self.hands)
        self.trash = []
        self.played = []
        # list of cards that are currently on the top of each stack on the board
        self.board = [0] * 5
        self.hints = data.usedNoteTokens
        self.errors = data.usedStormTokens
        self.last_action = LastAction()


    def update_last_action(self, data):
        if type(data) is GameData.ServerActionValid:
            self.last_action.update_last_action(data.lastPlayer, data.cardHandIndex)
            self.trash.append(data.card)
        if type(data) is GameData.ServerPlayerMoveOk:
            self.last_action.update_last_action(data.lastPlayer, data.cardHandIndex)
            self.played.append(data.card)
            self.board[colors.index(data.card.color)] += 1
        if type(data) is GameData.ServerPlayerThunderStrike:
            self.last_action.update_last_action(data.lastPlayer, data.cardHandIndex)
            self.errors += 1
        if type(data) is GameData.ServerHintData:
            self.last_action.update_last_action(data.lastPlayer, data.cardHandIndex)
            self.hints += 1

    def update_knowledge(self, players: list):
        new_card = players[self.last_action.last_player].hand[-1]
        self.knowledge.card_discovered(self.hands, self.hands[self.last_action.last_player], new_card)
        self.hands[self.last_action.last_player] = players[self.last_action.last_player].hand
        
    def update_knowledge_on_hint_received(self, type, value, positions):
        for pos in positions:
            if type == 'color':
                self.knowledge[self.name].update_card(pos, color=colors.index(value))
            if type == 'value':
                self.knowledge[self.name].update_card(pos, rank=value)

    ## TODO
    # funzione chiamata alla ricezione di un ready
    # funzione inizializzazione variabili di stato ad inizio partita (quando tutti ready)
    # funzione per aggiornare variabili di stato
    # funzione per decidere la mossa da fare

    # def update_state():

    # def make_move():
 