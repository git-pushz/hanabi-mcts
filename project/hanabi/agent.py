import GameData
import numpy as np
from game import Card

colors = ["green", "yellow", "blue", "red", "white"]
card_states = ["none",  # default state for each card
               "playable", # card is playable (i.e. it's the next useful rank for its color on the table stack)
               ## - I know exactly the rank and the color of the card AND the pile of that color has exactly rank-1 on top
               # - All the stacks have the same top AND I know the rank of the card, which is exactly TOS+1
               "expendable", # card can be possibly discarded (it's not the only one in the deck)
               ## - I know exactly the rank and the color of the card AND the trash doesn't contain all of the remaining of the same color and rank
               "useless", # card can be surely discarded (it has already been played)
               ## - I know exactly the rank and the color of the card AND the pile of that color has rank >= of this one
               # - I know the rank AND all piles have TOS >= of this one
               # - I know the color AND the pile of that color is filled (TOS == 5 OR check the trash)
               "risky"] # card is the only one in the deck (opposite of expendable)
               # - I know it's a 5 
               ## - I know exactly the rank and the color of the card AND the trash contains all of the remaining of the same color and rank
               #   (it's the last one in game)

HAND_SIZE = 5
CARD_QUANTITIES = [3, 2, 2, 2, 1]

class Agent():
    '''
    Our AI Hanabi agent player
    '''

    class MentalState():
        '''
        Mental state representation for a single card in a player's hand
        '''
        def __init__(self, agent):
            col = np.array(CARD_QUANTITIES)
            col = col.reshape(col.size, 1)
            self.table = np.tile(col, len(colors))
            self.state = card_states[0]
            self.agent = agent

        def rank_hint_received(self, rank: int):
            '''
            Update mental state for a card when received an hint based on card rank
            '''
            rank -= 1
            i = [j for j in range(HAND_SIZE) if j != rank]
            self.table[i, :] = 0
            self.update_card_state()

        def color_hint_received(self, color: int):
            '''
            Update mental state for a card when received an hint based on color
            '''
            i = [j for j in range(HAND_SIZE) if j != color]
            self.table[:, i] = 0
            self.update_card_state()

        def card_drawn(self, rank: int, color: int, is_template=False):
            '''
            Update mental state for a card when a card is drawn from deck
            '''
            rank -= 1
            # self.table[rank, color] must be > 0
            # this assertion is technically wrong, if someone received an hint on a 4, all rows a part from the 4th
            # will become 0, but this doesn't mean that there are no more 3s around
            # assert(self.table[rank, color] > 0)
            if (self.table[rank, color] == 0):
                return
            self.table[rank, color] -= 1
            if (is_template):
                return
            self.update_card_state()
            
        def update_card_state(self):
            r, c = np.nonzero(self.table)
            if (len(r) == 1 and len(c) == 1): # if the card is fully determined
                print("card is fully determined")
                r = r[0]+1  # rank of the card
                c = c[0]    # color of the card
                if self.agent.board[c]+1 == r:
                    print("card playable")
                    self.state = card_states[1] # playable
                    return
                elif self.agent.board[c] >= r:
                    print("card useless")
                    self.state = card_states[3] # useless
                    return
                counter = 0
                for t in self.agent.trash:
                    if t.value == r and t.color == colors[c]:
                        counter += 1
                if CARD_QUANTITIES[r] - counter == 1:
                    print("card risky")
                    self.state = card_states[4] # risky
                else:
                    print("card expendable")
                    self.state = card_states[2] # expendable

            # values = set(board)
            # if len(values) == 1:
            #   value = values.pop()
            #   mask = numpy.full(5, True, dtype=bool)
            #   mask[value - 1] = False
            #   if np.sum(self.table[mask, :]) == 0: self.state = card_states[1]


        def to_string(self) -> str:
            s = '\n'.join([''.join(['{:4}'.format(item) for item in row]) for row in self.table])
            s += '\n'
            return s
            

    class PlayerMentalState():
        '''
        Class for holding the mental state of the hand of one player
        '''
        def __init__(self, agent):
            self.ms_hand = [Agent.MentalState(agent) for _ in range(HAND_SIZE)]
            self.agent = agent

        def update_whole_hand(self, rank: int, color: int):
            for c in self.ms_hand:
                c.card_drawn(rank, color)

        def update_card(self, card_index: int, rank: int = None, color: int = None):
            assert((rank is None) != (color is None))
            if rank is None:
                self.ms_hand[card_index].color_hint_received(color)
            else:
                self.ms_hand[card_index].rank_hint_received(rank)

        def get_cards_from_state(self, state: int):
            '''
            Return a list of MentalStates for cards whose state is card_states[state]
            '''
            return [idx[0] for idx, card in np.ndenumerate(self.ms_hand) if card.state == card_states[state]]

        def reset_card_mental_state(self, card_index: int, player_ms_template):
            '''
            Reset the specified card mental state with the template mental state of the player
            '''
            print("removing card at index from mental state", card_index)
            self.ms_hand.pop(card_index)
            print("appending the mental state template")
            print(player_ms_template.to_string())
            self.ms_hand.append(player_ms_template)

        def to_string(self) -> str:
            s = ''
            for i in range(HAND_SIZE):
                s += self.ms_hand[i].to_string()
                s += '\n'
            return s


    class MentalStateGlobal():
        '''
        Class for holding the mental state of each card in the hand of each player
        '''
        def __init__(self, hands: dict, agent):
            self.matrix = {k: Agent.PlayerMentalState(agent) for k in hands.keys()}
            # it is used to store information of the past knowledge of the match
            # used as the starting mental state for newly drawn cards by the agent
            # it is updated in 3 cases:
            # at MentalStateGlobal initialitiation X
            # when a card is drawn X
            # when a card of the agent hand is fully determined
            self.templates_ms = {k: Agent.MentalState(agent) for k in hands.keys()}
            self.agent = agent

            for name, hand in hands.items():
                for card in hand:
                    for n in hands.keys():
                        if n != name:
                            self.templates_ms[n].card_drawn(card.value, colors.index(card.color), True)
                            self.matrix[n].update_whole_hand(card.value, colors.index(card.color))
        
        def update_templates_ms(self):
            '''
            Update mental state template of each player
            '''
            # it actually re-compute it
            self.templates_ms = {k: Agent.MentalState(self.agent) for k in self.agent.hands.keys()}
            for name, hand in self.agent.hands.items():
                for card in hand:
                    for n in self.agent.hands.keys():
                        if n != name:
                            self.templates_ms[n].card_drawn(card.value, colors.index(card.color), True)
                for card in self.agent.trash:
                    self.templates_ms[name].card_drawn(card.value, colors.index(card.color), True)
                for i, pile in enumerate(self.agent.board):
                    for rank in range(pile):
                        self.templates_ms[name].card_drawn(rank+1, i, True)

        def card_discovered(self, hands:dict, last_player: str, old_card: Card, new_card: Card=None):
            '''
            Update mental state of each player when a card is played and a new card is drawn
            '''
            self.matrix[last_player].update_whole_hand(old_card.value, colors.index(old_card.color))
            print("discarded/ played card", old_card)
            # new card is optional because if the player who played/ discarded the card is the agent, there is
            # no way to know which card it draw
            if (new_card == None):
                return
            for name in hands.keys():
                if name != last_player:
                    self.matrix[name].update_whole_hand(new_card.value, colors.index(new_card.color))
            print("drawn card", new_card)

        def player_mental_state(self, player_name):
            return self.matrix[player_name]

        def player_template_ms(self, player_name):
            return self.templates_ms[player_name]

        def to_string(self) -> str:
            s = ''
            for k, v in self.matrix.items():
                s += f"Player {k}:\n"
                s += v.to_string()
                s += '-'*30
                s += '\n'
            return s


    class LastAction():
        '''
        Mini-class to store info about the last action performed by other players
        '''
        def __init__(self):
            self.last_player = None
            self.card_index = None

        def update_last_action(self, last_player: str, card_index: int, card):
            self.last_player = last_player
            self.card_index = card_index
            self.card = card
    
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
        self.knowledge = Agent.MentalStateGlobal(self.hands, self)
        self.trash = []
        self.played = []
        # list of cards that are currently on the top of each stack on the board
        self.board = [0] * 5
        self.hints = data.usedNoteTokens
        self.errors = data.usedStormTokens
        self.last_action = Agent.LastAction()

    def make_move(self):
        # checks that hint token is available
        #1. playable?
        cards = self.knowledge.player_mental_state(self.name).get_cards_from_state(1)
        print("cards_from_state", cards)
        if(len(cards) > 0):
            print("played a playable card")
            return GameData.ClientPlayerPlayCardRequest(self.name, cards[0])
        cards = self.knowledge.player_mental_state(self.name).get_cards_from_state(3)
        if(len(cards) > 0):
            print("discarded a useless card")
            return GameData.ClientPlayerDiscardCardRequest(self.name, cards[0])
        # TODO hint
        print("discarded a random card")
        return GameData.ClientPlayerDiscardCardRequest(self.name, 0)

    def update_last_action(self, data):
        if type(data) is GameData.ServerActionValid:
            self.last_action.update_last_action(data.lastPlayer, data.cardHandIndex, data.card)
            self.trash.append(data.card)
        if type(data) is GameData.ServerPlayerMoveOk:
            self.last_action.update_last_action(data.lastPlayer, data.cardHandIndex, data.card)
            self.played.append(data.card)
            self.board[colors.index(data.card.color)] += 1
        if type(data) is GameData.ServerPlayerThunderStrike:
            self.last_action.update_last_action(data.lastPlayer, data.cardHandIndex, data.card)
            self.errors += 1
            self.trash.append(data.card)
        # if type(data) is GameData.ServerHintData:
        #     self.last_action.update_last_action(data.lastPlayer, data.cardHandIndex)
        #     self.hints += 1

    def update_knowledge(self, players: list):
        print("updating knowledge..")
        # if the agent is the one drawing a card, we have no information on the card
        if (self.last_action.last_player == self.name):
            new_card = None
            self.knowledge.card_discovered(self.hands, self.last_action.last_player, self.last_action.card, new_card)
        else:
            new_card = players[self.players.index(self.last_action.last_player)].hand[-1]
            self.knowledge.card_discovered(self.hands, self.last_action.last_player, self.last_action.card, new_card)
            # update player hand with new card
            self.hands[self.last_action.last_player] = players[self.players.index(self.last_action.last_player)].hand
        # update mental state templates for all players
        self.knowledge.update_templates_ms()
        # update mental state of "card_index"th card of the player who drawn a new card
        self.knowledge.player_mental_state(self.last_action.last_player).reset_card_mental_state(self.last_action.card_index, self.knowledge.player_template_ms(self.last_action.last_player))
        
    def update_knowledge_on_hint_received(self, data: GameData.ServerHintData):
        # data.type, data.value, data.positions, data.destination
        for pos in data.positions:
            if data.type == 'color':
                self.knowledge.player_mental_state(data.destination).update_card(pos, color=colors.index(data.value))
            if data.type == 'value':
                self.knowledge.player_mental_state(data.destination).update_card(pos, rank=data.value)

    ## TODO
    # 1 quando l'agent conosce, grazie ad un hint, colore e valore di una o più carte, va aggiornato il mental state degli altri giocatori
    # 2 Quando l'agent gioca una carta, va fatto un update sul mental state di tutti i giocatori (visto che l'agent scopre la carta che ha giocato)
     # in più bisogna aggiornare il mental state dell'agent per fare in modo che venga scartata la carta giocata e pescata una di cui non sa niente a parte per 
     # quello che c'è in campo, in mano agli altri e nel trash
    # 3 Documentazione
    # 4 Implementare gestione dei rimanenti card_states

    