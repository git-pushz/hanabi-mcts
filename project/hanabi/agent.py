import sys

import GameData
import numpy as np

import game
from game import Card

colors = ["red", "yellow", "green", "blue", "white"]
card_states = ["none",  # default state for each card
               "playable",  # card is playable (i.e. it's the next useful rank for its color on the table stack)
               ## - I know exactly the rank and the color of the card AND the pile of that color has exactly rank-1 on top
               # - All the stacks have the same top AND I know the rank of the card, which is exactly TOS+1
               "expendable",  # card can be possibly discarded (it's not the only one in the deck)
               ## - I know exactly the rank and the color of the card AND the trash doesn't contain all of the remaining of the same color and rank
               "useless",  # card can be surely discarded (it has already been played)
               ## - I know exactly the rank and the color of the card AND the pile of that color has rank >= of this one
               # - I know the rank AND all piles have TOS >= of this one
               # - I know the color AND the pile of that color is filled (TOS == 5 OR check the trash)
               "risky"]  # card is the only one in the deck (opposite of expendable)
# - I know it's a 5
## - I know exactly the rank and the color of the card AND the trash contains all of the remaining of the same color and rank
#   (it's the last one in game)

HAND_SIZE = 5
CARD_QUANTITIES = [3, 2, 2, 2, 1]


class Agent:
    '''
    Our AI Hanabi agent player
    
    Attributes:
        name: The name of the agent
        currentPlayer: name of the current player in the game
        players: A list of the players' names in turn order
        nr: The position of the (agent) player in the round
        hands: A dictionary where the key is the name of a player and the value is the corresponding hand (list of cards)
        knowledge: Global MentaState of the agent (knowledge['player_name'][card position][card_color][card_rank])
        trash: A card list representing the trash
        played: A card list representing the played cards
        board: A list representing the fireworks on the table (board[color_idx] is the value of the highest card for the color)
        hints: The number of used note tokens
        errors: The number of used storm tokens
        last_action: The last performed action (card played/discarded) known to the agent
    '''

    class MentalState:
        '''
        Mental state representation for a single card in a player's hand.
        
        Attributes:
            table:  A 2D numpy array representing the possible values of the card.
                    table[r][c] contains the number of cards with rank 'r' and color 'c' that
                    have not been discovered yet and whose value may be the one of the card.
                    (i.e. (table[r][c] / np.sum(table)) is the probability that the card's
                    value is table[r][c])
            state:  The state of the card, according to the Agent's knowledge.
                    (the possible values are the ones contained in the 'card_states' list)
            agent:  The agent that "owns" this mental state (i.e. card)
        '''

        def __init__(self, agent):
            col = np.array(CARD_QUANTITIES)
            col = col.reshape(col.size, 1)
            self.table = np.tile(col, len(colors))
            self.fully_determined = False
            self.fully_determined_now = False
            self.state = card_states[0]
            self.agent = agent

        def rank_hint_received(self, rank: int):
            '''
            Update the mental state when a rank hint is received for this card.
            Calling this functions will fully determine the card's rank (setting all the other
            ranks in self.table to 0) and update the card's state according to the new information

            Args:
                rank: the rank of the card
            '''
            rank -= 1
            i = [j for j in range(HAND_SIZE) if j != rank]
            self.table[i, :] = 0
            self.update_card_state()

        def color_hint_received(self, color: int):
            '''
            Update the mental state when a color hint is received for this card.
            Calling this functions will fully determine the card's color (setting all the other
            colors in self.table to 0) and update the card's state according to the new information

            Args:
                color: the index of the color in the list 'colors'
            '''
            i = [j for j in range(HAND_SIZE) if j != color]
            self.table[:, i] = 0
            self.update_card_state()

        def card_drawn(self, rank: int, color: int, is_template=False):
            '''
            Remove a new discovered card from the "possibilities" of this card
            (i.e. self.table[rank][color] will be decremented by 1)

            Args:
                rank:   the rank of the new discovered card
                color:  the index of new discovered card's color in the list 'colors'
                is_template:
            '''
            rank -= 1
            # self.table[rank, color] must be > 0
            # this assertion is technically wrong, if someone received an hint on a 4, all rows a part from the 4th
            # will become 0, but this doesn't mean that there are no more 3s around
            # assert(self.table[rank, color] > 0)
            if self.table[rank, color] == 0:
                return
            self.table[rank, color] -= 1
            if is_template:
                return
            self.update_card_state()

        def update_card_state(self):
            '''
            Update the card's state according to the currently known informations
            (stored in self.table)
            '''
            if self.state == "useless":
                return
            r, c = np.nonzero(self.table)
            # r is a numpy array of the row (=rank) indices of nonzero entries of self.table
            # c is a numpy array of the column (=color) indices of nonzero entries of self.table
            if len(r) == 1 and len(c) == 1:  # I know both
                r, c = r[0] + 1, c[0]
                self.fully_determined = True
                self.fully_determined_now = True
                if self.agent.board[c] == self.agent.maximums[c]:
                    self.state = "useless"
                    return
                elif self.agent.board[c] == r - 1:
                    self.state = "playable"
                    return
                elif self.agent.board[c] >= r:
                    self.state = "useless"
                    return
                elif self.agent.maximums[c] < r - 1:
                    self.state = "useless"
                    return
                elif self.table[r, c] == 1:
                    self.state = "risky"
                    return
                elif self.table[r, c] > 1:
                    self.state = "expendable"
                    return
                else:
                    print("Should not be here")
                    print(f"rank: {r}, color: {colors[c]} (index {c})")
                    print(f"Mental State of current card: {self.table}")
            elif len(np.unique(r)) == 1:  # I know only the rank
                r = r[0] + 1
                if max(self.agent.board) == min(self.agent.board) == r - 1:
                    self.state = "playable"
                    return
                elif min(self.agent.board) >= r:
                    self.state = "useless"
                    return
                elif max(self.agent.maximums) < r - 1:
                    self.state = "useless"
                    return
                elif max(self.table[r - 1, :]) == 1:
                    self.state = "risky"
                    return
                elif min(self.table[r - 1, :]) > 1:
                    self.state = "expendable"
                    return
                else:
                    print("Should not be here")
                    print(f"rank: {r}")
                    print(f"Mental State of current card: {self.table}")
            elif len(np.unique(c)) == 1:  # I know only the color
                c = c[0]
                if self.agent.board[c] == self.agent.maximums[c]:
                    self.state = "useless"
                    return
                elif max(self.table[:, c]) == 1:
                    self.state = "risky"
                    return
                elif min(self.table[:, c]) > 1:
                    self.state = "expendable"
                    return
                else:
                    print("Should not be here")
                    print(f"color: {colors[c]} (index {c})")
                    print(f"Mental State of current card: {self.table}")
            else:  # I know nothing
                if max(self.table == 1):
                    self.state = "risky"
                    return
                elif min(self.table) > 1:
                    self.state = "expendable"
                    return
                else:
                    print("Should not be here")

        def get_table(self):
            '''
            Return the numpy array which is the mental state rappresentation of the card
            '''
            return self.table

        def is_fully_determined(self):
            if self.fully_determined:
                r, c = np.nonzero(self.table)
                assert len(r) == len(c) == 1, "card isn't actually fully determined"
                return True, r[0]+1, c[0]
            else:
                return False, None, None

        def to_string(self) -> str:
            s = '\n'.join([''.join(['{:4}'.format(item) for item in row]) for row in self.table])
            s += '\n'
            return s

    class PlayerMentalState:
        '''
        The mental states of all the cards in a player's hand

        Attributes:
            ms_hand: A numpy array of mental states, one for each card in the player's hand
            agent: The agent this instance is referring to
        '''

        def __init__(self, agent):
            self.ms_hand = [Agent.MentalState(agent) for _ in range(HAND_SIZE)]
            self.agent = agent

        def update_whole_hand(self, rank: int, color: int, fully_determined=None):
            '''
            Update all the mental states of the agent's hand when a new card is discovered

            Args:
                rank:   the rank of the new discovered card
                color:  the index of new discovered card's color in the list 'colors'
                fully_determined: whether the card was fully determined or not
            '''
            if fully_determined is None:
                for c in self.ms_hand:
                    c.card_drawn(rank, color)
            else:
                for c in self.ms_hand:
                    # decrease value of other cards in agent's hand if a recent fully determined card is detected
                    # (the recent FD cards must not be decreased)
                    if not c.fully_determined_now:
                        c.get_table()[rank, color] -= 1

        def reset_recent_fully_determined_cards(self):
            """
            This function reset to False the fully_determined_now attribute of each card of a Player's hand
            """
            for card in self.ms_hand:
                if card.fully_determined_now:
                    card.fully_determined_now = False

        def update_card(self, card_index: int, rank: int = None, color: int = None):
            '''
            Update the mental state of a card in the player's hand when a hint for it is received.
            (WARNING: only one between 'rank' and 'color' must be set)

            Args:
                card_index: the index of the card in the hand
                rank:       the rank of the card
                color:      the index of card's color in the list 'colors'
            '''
            assert ((rank is None) != (color is None))
            if rank is None:
                self.ms_hand[card_index].color_hint_received(color)
            else:
                self.ms_hand[card_index].rank_hint_received(rank)

        def get_cards_from_state(self, state: int):
            '''
            Get the hand's cards that have a certain state 

            Args:
                state: the index of the state in the 'card_states' list

            Returns:
                A list of indices of the cards in the specified state inside the hand
            '''
            return [idx[0] for idx, card in np.ndenumerate(self.ms_hand) if card.state == card_states[state]]

        def get_new_fully_determined_cards(self):
            '''
            Return the index of all RECENT Fully Determined cards in a hand/PlayerMentalState, specifically a list of MentalStates 
            '''
            return [idx[0] for idx, card in np.ndenumerate(self.ms_hand) if
                    (card.fully_determined and card.fully_determined_now)]

        def get_card_from_index(self, index: int):
            '''
            Return a mental state of a card given the index of it in a hand/PlayerMentalState
            '''
            return self.ms_hand[index]

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

    class MentalStateGlobal:
        '''
        The PlayerMentalStates for all the players in the game,
        according to the knowledge of the agent

        Attributes:
            matrix: A dictionary where the key is a player's name and
                    the value the corresponding PlayerMentalState
            agent:  The agent this instance is referring to
        '''

        def __init__(self, hands: dict, agent):
            self.matrix = {k: Agent.PlayerMentalState(agent) for k in hands.keys()}
            # it is used to store information of the past knowledge of the match
            # used as the starting mental state for newly drawn cards by the agent
            # it is updated in 3 cases:
            # at MentalStateGlobal initialization X
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
            # it actually re-computes it
            # TODO: VERIFY
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
                        self.templates_ms[name].card_drawn(rank + 1, i, True)

        def card_discovered(self, hands: dict, last_player: str, old_card: Card, new_card: Card = None):
            '''
            Update the PlayerMentalState of all the players when a card is played/discarded
            and a new one is taken from the deck.
            For the player who just performed the action the discovered card will be the played/discarded one,
            while for all the other players it will be the drawn one.

            Args:
                hands:          A dictionary where the key is the name of a player ad the value
                                is the corresponding hand (list of cards)
                last_player:    The name of the player who performed the last action
                old_card:       The played/discarded card
                new_card:       The drawn card
            '''
            self.matrix[last_player].update_whole_hand(old_card.value, colors.index(old_card.color))
            print("discarded/ played card", old_card)
            # new card is optional because if the player who played/ discarded the card is the agent, there is
            # no way to know which card it draw
            if new_card is None:
                return
            for name in hands.keys():
                if name != last_player:
                    self.matrix[name].update_whole_hand(new_card.value, colors.index(new_card.color))
            print("drawn card", new_card)

        def player_mental_state(self, player_name):
            '''
            Get the PlayerMentalState of a certain player

            Args:
                player_name: The name of the player whose PlayerMentalState must be retrieved

            Returns:
                The PlayerMentalState of the desired player
            '''
            return self.matrix[player_name]

        def player_template_ms(self, player_name):
            return self.templates_ms[player_name]

        def to_string(self) -> str:
            s = ''
            for k, v in self.matrix.items():
                s += f"Player {k}:\n"
                s += v.to_string()
                s += '-' * 30
                s += '\n'
            return s

    class LastAction:
        '''
        Some informations about the last action (card played/discarded)
        performed in the game by a player

        Attributes:
            last_player:    The name of the player who performed the last action
            card_index:     The index of the played/discarded card in the last_player's hand 
        '''

        def __init__(self):
            self.last_player = None
            self.card_index = None

        def update_last_action(self, last_player: str, card_index: int, card):
            self.last_player = last_player
            self.card_index = card_index
            self.card = card

    def __init__(self, name: str, data: GameData.ServerGameStateData, players_names: list):
        '''
        Create a new Agent

        Args:
            name: The name of the agent
            data: The game state to use to initialize the Agent
            players_names: The list of players names in turn order
        '''
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
        self.maximums = [5] * 5

    def make_move(self):
        '''
        Perform the best possible move according to the Agent's knowledge

        Returns:
            A GameData object representing the chosen move
        '''
        cards = self.knowledge.player_mental_state(self.name).get_cards_from_state(1)
        print("cards_from_state", cards)
        if (len(cards) > 0):
            print("played a playable card")
            return GameData.ClientPlayerPlayCardRequest(self.name, cards[0])
        cards = self.knowledge.player_mental_state(self.name).get_cards_from_state(3)
        if (len(cards) > 0):
            print("discarded a useless card")
            return GameData.ClientPlayerDiscardCardRequest(self.name, cards[0])
        # TODO hint
        print("discarded a random card")
        return GameData.ClientPlayerDiscardCardRequest(self.name, 0)

    def update_last_action(self, data):
        '''
        Update the last action (card played/discarded) known to the agent

        Args:
            data: The GameData object containing the action to register
        '''
        # the action was "discarding a card"
        if type(data) is GameData.ServerActionValid:
            self.last_action.update_last_action(data.lastPlayer, data.cardHandIndex, data.card)
            self.trash.append(data.card)
            self.maximums = self.board_maximums()
        # the action was "successfully playing a card"
        if type(data) is GameData.ServerPlayerMoveOk:
            self.last_action.update_last_action(data.lastPlayer, data.cardHandIndex, data.card)
            self.played.append(data.card)
            self.board[colors.index(data.card.color)] += 1
        # the action was "unsuccessfully playing a card"
        if type(data) is GameData.ServerPlayerThunderStrike:
            self.last_action.update_last_action(data.lastPlayer, data.cardHandIndex, data.card)
            self.trash.append(data.card)
            self.maximums = self.board_maximums()
            self.errors += 1
            self.trash.append(data.card)
        # if type(data) is GameData.ServerHintData:
        #     self.last_action.update_last_action(data.lastPlayer, data.cardHandIndex)
        #     self.hints += 1

    def update_knowledge(self, players: list):
        '''
        Update the agent's knowledge (GlobalMentalState) according to the last performed action
        (card played/discarded)

        Args:
            players: The list of the players objects in turn order (must be consistent with self.players)
        '''
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
        self.knowledge.player_mental_state(self.last_action.last_player).reset_card_mental_state(
            self.last_action.card_index, self.knowledge.player_template_ms(self.last_action.last_player))

    def update_knowledge_on_hint_received(self, data: GameData.ServerHintData):
        '''
        Update the agent's knowledge (GlobalMentalState) when an hint is sent from a player to another.
        (note that the destination of the hint doesn't have to be the agent itself)

        Args:
            data: the object describing the hint
        '''
        self.hints += 1

        for pos in data.positions:
            if data.type == 'color':
                self.knowledge.player_mental_state(data.destination).update_card(pos, color=colors.index(data.value))
            if data.type == 'value':
                self.knowledge.player_mental_state(data.destination).update_card(pos, rank=data.value)

    def board_maximums(self):
        trash_colors = dict.fromkeys(colors, [])
        for card in self.trash:
            trash_colors[card.color].append(card.value)

        maximums = [5] * 5
        for color in colors:
            for i in range(0, 5):
                if trash_colors[color].count(i + 1) == CARD_QUANTITIES[i]:
                    maximums[colors.index(color)] = i
                    break

        return maximums

    def discover_card(self, card: Card, card_index: int, action_type: str):
        """
        Called whenever the agent plays or discards a card: if it wasn't fully determined, update the structures

        Args:
            card: the played/discarded card
            card_index: the index of card in agent's hand
            action_type: it's one of ['play', 'mistake', 'discard'] FOR DEBUG ONLY
        """
        ms = self.knowledge.player_mental_state(self.name)
        card_ms: Agent.MentalState = ms[card_index]
        if not card_ms.fully_determined:
            # TODO: update all remaining MS in my hand
            # TODO: update MS of all other players
            # TODO: update the template
            # TODO: draw card -> init its mental state
            pass
        else:
            if action_type == 'mistake':
                print("Should not be here: made a mistake with a fully determined card.", file=sys.stderr)

    def update_board(self, card: Card):
        self.played.append(card)
        self.board[colors.index(card.color)] += 1
        if self.board[colors.index(card.color)] == 5:
            self.hint_gained()

    def update_trash(self, card: Card):
        self.trash.append(card)

    def hint_consumed(self):
        self.hints = min(self.hints + 1, 8)

    def hint_gained(self):
        self.hints = max(0, self.hints - 1)

    def mistake_made(self):
        self.errors += 1
        assert self.errors < 3

    def assert_aligned_with_server(self, hints_used: int, mistakes_made: int, board: list, trash: list, players: list):
        assert self.hints == hints_used, "wrong count of hints"
        assert self.errors == mistakes_made, "wrong count of errors"
        assert self.board == board, "wrong board"
        assert self.trash == trash, " wrong trash"
        for player in players:
            assert player.hand == self.hands[player.name], f"player {player.name} wrong hand"

    def track_played_card(self, player_name: str, card_index: int):
        del self.hands[player_name][card_index]

    def track_drawn_card(self, players: list):
        different_hands = 0
        new_card = None
        player = None
        for p in players:
            if len(p.hand) != len(self.hands[p.name]):
                different_hands += 1
                # NB: newly drawn cards are appended to the right
                new_card = p.hand[-1]
                player = p.name
        assert new_card is not None, "new card not found"
        assert different_hands == 1, "too many different cards"

        self.hands[player].append(new_card)
        for p in self.players:
            if p != player:
                self.knowledge.player_mental_state(p).update_whole_hand(new_card.rank, new_card.color)

    def update_knowledge_on_hint(self, hint_type: str, value, positions: list, destination: str):
        if destination == self.name:
            if hint_type == 'rank':
                for index in positions:
                    self.knowledge.player_mental_state(destination).update_card(index, rank=value)
                    fully_determined, rank, color = self.knowledge.player_mental_state(destination).get_card_from_index(index).is_fully_determined()
                    if fully_determined:
                        for player in self.players:
                            if player != self.name:
                                self.knowledge.player_mental_state(player).card_drawn(rank, color)

            else:
                for index in positions:
                    self.knowledge.player_mental_state(destination).update_card(index, color=value)
                    fully_determined, rank, color = self.knowledge.player_mental_state(destination).get_card_from_index(index).is_fully_determined()
                    if fully_determined:
                        for player in self.players:
                            if player != self.name:
                                self.knowledge.player_mental_state(player).card_drawn(rank, color)
        else:
            if hint_type == 'rank':
                for index in positions:
                    self.knowledge.player_mental_state(destination).update_card(index, rank=value)
            else:
                for index in positions:
                    self.knowledge.player_mental_state(destination).update_card(index, color=value)
