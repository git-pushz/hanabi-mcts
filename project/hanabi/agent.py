from turtle import color
import GameData
import numpy as np
import copy
import sys
from game import Card
from collections import deque

DEBUG = False
HINT_DEBUG = True
VERBOSE = True

colors = ["red", "yellow", "green", "blue", "white"]
card_states = ["none",  # default state for each card
               "playable",  # card is playable (i.e. it's the next useful rank for its color on the table stack)
               "expendable",  # card can be possibly discarded (it's not the only one in the deck)
               "useless",  # card can be surely discarded (it has already been played)
               "risky"]  # card is the only one in the deck (opposite of expendable)

HAND_SIZE = 5
CARD_QUANTITIES = [3, 2, 2, 2, 1]


class Agent:
    """
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
    """

    # __COMPARE_BOARD = np.array([[3, -1, 0, 0, 0],
    #                             [-1, 2, 1, 0, 0],
    #                             [-1, 1, 1, 0, 0],
    #                             [-1, -1, -1, 1, 1],
    #                             [-1, -1, 0, 1, 0]])
    __COMPARE_BOARD = np.array([[6, 0, 0, 0, 0],
                                [0, 4, 0, 0, 0],
                                [0, 0, 2, 0, 0],
                                [0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0]])

    __GOALS_LABELS = ['play', 'discard', 'maydiscard', 'protect', 'keep']

    class MentalState:
        """
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
        """

        def __init__(self, agent):
            col = np.array(CARD_QUANTITIES)
            col = col.reshape(col.size, 1)
            self.table = np.tile(col, len(colors))
            self.fully_determined = False
            self.fully_determined_now = False
            self.state = card_states[0]
            self.agent = agent

        def rank_hint_received(self, rank: int):
            """
            Update the mental state when a rank hint is received for this card.
            Calling this functions will fully determine the card's rank (setting all the other
            ranks in self.table to 0) and update the card's state according to the new information

            Args:
                rank: the rank of the card
            """
            rank -= 1
            i = [j for j in range(HAND_SIZE) if j != rank]
            self.table[i, :] = 0
            self.update_card_state()

        def color_hint_received(self, color: int):
            """
            Update the mental state when a color hint is received for this card.
            Calling this functions will fully determine the card's color (setting all the other
            colors in self.table to 0) and update the card's state according to the new information

            Args:
                color: the index of the color in the list 'colors'
            """
            i = [j for j in range(HAND_SIZE) if j != color]
            self.table[:, i] = 0
            self.update_card_state()

        def card_drawn(self, rank: int, color: int, is_template=False):
            """
            Remove a new discovered card from the "possibilities" of this card
            (i.e. self.table[rank][color] will be decremented by 1)

            Args:
                rank:   the rank of the new discovered card
                color:  the index of new discovered card's color in the list 'colors'
                is_template:
            """
            rank -= 1
            # self.table[rank, color] must be > 0
            # this assertion is technically wrong, if someone received an hint on a 4, all rows a part from the 4th
            # will become 0, but this doesn't mean that there are no more 3s around
            # assert(self.table[rank, color] > 0)
            if self.table[rank, color] == 0:
                return
            ## TODO
            self.table[rank, color] -= 1
            if is_template:
                return
            self.update_card_state()

        def update_card_state(self):
            """
            Update the card's state according to the currently known information
            (stored in self.table)
            """
            if self.state == "useless":
                return
            r, c = np.nonzero(self.table)
            # r is a numpy array of the row (=rank) indices of nonzero entries of self.table
            # r is a numpy array of the row (=rank) indices of nonzero entries of self.table
            # c is a numpy array of the column (=color) indices of nonzero entries of self.table

            # UPDATE: FROM PLAYABLE, A CARD CAN ONLY BECOME USELESS OR RISKY -> COVER THE CASE OF EXPENDABLE

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
                elif self.table[r - 1, c] == 1:
                    self.state = "risky"
                    return
                elif self.table[r - 1, c] > 1 and self.state != 'playable':
                    self.state = "expendable"
                    return
                else:
                    if DEBUG:
                        print("Should not be here")
                        print(f"rank: {r}, color: {colors[c]} (index {c})")
                        print(f"Mental State of current card:\n {self.table}")
            elif len(np.unique(r)) == 1:  # I know only the rank
                r = r[0] + 1
                if np.max(self.agent.board) == np.min(self.agent.board) == r - 1:
                    self.state = "playable"
                    return
                elif np.min(self.agent.board) >= r:
                    self.state = "useless"
                    return
                elif np.max(self.agent.maximums) < r - 1:
                    self.state = "useless"
                    return
                elif np.max(self.table[r - 1, :]) == 1:
                    self.state = "risky"
                    return
                elif np.min(self.table[r - 1, :]) > 1 and self.state != 'playable':
                    self.state = "expendable"
                    return
                # the state remains "none"
                # else:
                # print("Should not be here")
                # print(f"rank: {r}")
                # print(f"Mental State of current card:\n {self.table}")
            elif len(np.unique(c)) == 1:  # I know only the color
                c = c[0]
                if self.agent.board[c] == self.agent.maximums[c]:
                    self.state = "useless"
                    return
                elif np.max(self.table[:, c]) == 1:
                    self.state = "risky"
                    return
                elif np.min(self.table[:, c]) > 1 and self.state != 'playable':
                    self.state = "expendable"
                    return
                # the state remains "none"
                # else:
                #     if self.fully_determined:
                #         print("Should not be here")
                #         print(f"color: {colors[c]} (index {c})")
                #         print(f"Mental State of current card:\n {self.table}")
            else:  # I know nothing
                if np.max(self.table) == 1:
                    self.state = "risky"
                    return
                elif np.min(self.table) > 1 and self.state != 'playable':
                    self.state = "expendable"
                    return
                else:
                    if self.state != "none":
                        print("Should not be here")

        def get_table(self):
            """
            Return the mental state representation of the card (2D numpy array)
            """
            return self.table

        def is_fully_determined_now(self):
            """
            Return whether the current MS is fully determined now, fully determined and possibly its color and rank
            """
            if self.fully_determined_now:
                r, c = np.nonzero(self.table)
                assert len(r) == len(c) == 1, "card isn't actually fully determined"
                return True, self.fully_determined, r[0] + 1, c[0]
            else:
                return False, self.fully_determined, None, None

        def reset_fully_determined_now(self):
            """
            Reset to False the fully_determined_now field
            """
            self.fully_determined_now = False

        def to_string(self) -> str:
            s = "   r   y   g   b   w\n"
            s += '\n'.join([''.join(['{:4}'.format(item) for item in row]) for row in self.table])
            s += '\n'
            if VERBOSE:
                s += '\t' + 'state: ' + self.state + '\n'
            return s

    class PlayerMentalState:
        """
        The mental states of all the cards in a player's hand

        Attributes:
            ms_hand: A list of mental states, one for each card in the player's hand
            agent: The agent this instance is referring to
        """

        def __init__(self, agent):
            self.ms_hand = [Agent.MentalState(agent) for _ in range(HAND_SIZE)]
            self.agent = agent

        def update_whole_hand(self, rank: int, color: int, fully_determined=None):
            """
            Update all the mental states of the agent's hand when a new card is discovered

            Args:
                rank:   the rank of the new discovered card
                color:  the index of new discovered card's color in the list 'colors'
                fully_determined: whether the card was fully determined or not
            """
            # rank -= 1
            if fully_determined is None:
                for c in self.ms_hand:
                    c.card_drawn(rank, color)
            else:
                for c in self.ms_hand:
                    # decrease value of other cards in agent's hand if a recent fully determined card is detected
                    # (the recent FD cards must not be decreased)
                    if not c.fully_determined_now:
                        ## TODO
                        c.get_table()[rank, color] -= 1

        # def reset_recent_fully_determined_cards(self):
        #     """
        #     This function reset to False the fully_determined_now attribute of each card of a Player's hand
        #     """
        #     for card in self.ms_hand:
        #         if card.fully_determined_now:
        #             card.fully_determined_now = False

        def update_card(self, card_index: int, rank: int = None, color: int = None):
            """
            Update the mental state of a card in the player's hand when a hint for it is received.
            (WARNING: only one between 'rank' and 'color' must be set)

            Args:
                card_index: the index of the card in the hand
                rank:       the rank of the card
                color:      the index of card's color in the list 'colors'
            """
            assert ((rank is None) != (color is None))
            if rank is None:
                self.ms_hand[card_index].color_hint_received(color)
            else:
                self.ms_hand[card_index].rank_hint_received(rank)

        def get_cards_from_state(self, state: str):
            """
            Get the hand's cards that have a certain state

            Args:
                state: the index of the state in the 'card_states' list

            Returns:
                A list of indices of the cards in the specified state inside the hand
            """
            return [idx[0] for idx, card in np.ndenumerate(self.ms_hand) if card.state == state]

        def get_new_fully_determined_cards(self):
            """
            Return the index of all RECENT Fully Determined cards in a hand/PlayerMentalState, specifically a list of MentalStates
            """
            return [idx[0] for idx, card in np.ndenumerate(self.ms_hand) if
                    (card.fully_determined and card.fully_determined_now)]

        def get_card_from_index(self, index: int):
            """
            Return a mental state of a card given the index of it in a hand/PlayerMentalState
            """
            return self.ms_hand[index]

        def reset_card_mental_state(self, card_index: int, player_ms_template):
            """
            Reset the specified card mental state with the template mental state of the player
            """
            if DEBUG:
                print("removing card at index from mental state", card_index)
            self.ms_hand.pop(card_index)
            if DEBUG:
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
        """
        The PlayerMentalStates for all the players in the game,
        according to the knowledge of the agent

        Attributes:
            matrix: A dictionary where the key is a player's name and
                    the value the corresponding PlayerMentalState
            agent:  The agent this instance is referring to
        """

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
            """
            Update mental state template of each player
            """
            # it actually re-computes it
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
            """
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
            """
            self.matrix[last_player].update_whole_hand(old_card.value, colors.index(old_card.color))
            if DEBUG:
                print("discarded/ played card", old_card)
            # new card is optional because if the player who played/ discarded the card is the agent, there is
            # no way to know which card it draw
            if new_card is None:
                return
            for name in hands.keys():
                if name != last_player:
                    self.matrix[name].update_whole_hand(new_card.value, colors.index(new_card.color))
            if DEBUG:
                print("drawn card", new_card)

        def player_mental_state(self, player_name):
            """
            Get the PlayerMentalState of a certain player

            Args:
                player_name: The name of the player whose PlayerMentalState must be retrieved

            Returns:
                The PlayerMentalState of the desired player
            """
            return self.matrix[player_name]

        def player_template_ms(self, player_name):
            """
            Return the template for a new card of player_name's hand

            Args:
                player_name: the name of the player
            """
            return self.templates_ms[player_name]

        def to_string(self, print_templates=False) -> str:
            s = ''
            for k, v in self.matrix.items():
                s += f"Player {k}:\n"
                s += v.to_string()
                s += '-' * 30
                s += '\n'

            if print_templates:
                for k, v in self.templates_ms.items():
                    s += f"template player {k}:\n{v.to_string()}\n"

            return s

    class LastAction:
        """
        Some informations about the last action (card played/discarded)
        performed in the game by a player

        Attributes:
            last_player:    The name of the player who performed the last action
            card_index:     The index of the played/discarded card in the last_player's hand
        """

        def __init__(self):
            self.last_player = None
            self.card_index = None

        def update_last_action(self, last_player: str, card_index: int, card):
            self.last_player = last_player
            self.card_index = card_index
            self.card = card

    # AGENT INIT
    def __init__(self, name: str, data: GameData.ServerGameStateData, players_names: list):
        global HAND_SIZE
        """
        Create a new Agent

        Args:
            name: The name of the agent
            data: The game state to use to initialize the Agent
            players_names: The list of players names in turn order
        """
        self.name = name
        # name of the current player
        self.currentPlayer = data.currentPlayer
        # list of players in turn order
        self.players = players_names
        if len(players_names) >= 4:
            HAND_SIZE = 4
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
        self.__hint_history = deque(maxlen=4)

    def make_move(self):
        """
        Perform the best possible move according to the Agent's knowledge

        Returns:
            A GameData object representing the chosen move
        """
        if VERBOSE:
            print(f"Player {self.name}:")
            print(f"\tBoard: {self.board}")
            print(f"\tTrash: {self.trash}")
            print(f"\tHands: {self.hands}")
            print(f"\tHints used: {self.hints}")
            print(f"\tErrors made: {self.errors}")

        cards = self.knowledge.player_mental_state(self.name).get_cards_from_state("playable")
        if DEBUG:
            print("cards_from_state", cards)
        if len(cards) > 0:
            if DEBUG:
                print("played a playable card")
            return GameData.ClientPlayerPlayCardRequest(self.name, cards[0])
        cards = self.knowledge.player_mental_state(self.name).get_cards_from_state("useless")
        if len(cards) > 0:
            if DEBUG:
                print("discarded a useless card")
            return GameData.ClientPlayerDiscardCardRequest(self.name, cards[0])
        if self.hints < 8:
            best_hint = self.decide_hint()
        else:
            best_hint = None
        if best_hint is not None:
            if DEBUG:
                print("giving an hint to the player: ", best_hint[2])
            return GameData.ClientHintData(self.name, best_hint[2], best_hint[0], best_hint[1])
        if DEBUG:
            print("no good hint found..")

        return self.discard()

    def discard(self):
        # Reminder -> card_states = ["none", "playable","expendable","useless","risky"]

        # check if there are some useless cards in agent's hand...
        cards = self.knowledge.player_mental_state(self.name).get_cards_from_state("useless")
        if len(cards) > 0:
            if DEBUG:
                print("discarded a useless card")
            return GameData.ClientPlayerDiscardCardRequest(self.name, np.random.choice(cards))

        # ...if not check if there are some expendable cards in agent's hand
        cards = self.knowledge.player_mental_state(self.name).get_cards_from_state("expendable")
        if len(cards) == 1:
            if DEBUG or VERBOSE:
                print("discarded an expendable card")
            return GameData.ClientPlayerDiscardCardRequest(self.name, cards[0])
        if len(cards) > 0:
            # if agent has more than one expendable card in hand, discard the one with the highest rank
            ranks_list = []
            for card_index in cards:
                # get ranks of expendable cards and add them to a list
                rank, color = np.nonzero(
                    self.knowledge.player_mental_state(self.name).get_card_from_index(card_index).get_table())
                ranks_list.append((rank[0] + 1))
            # get the index of the max rank in the list, which is the same index of the corresponding card index in exp_cards
            to_discard = ranks_list.index(max(ranks_list))
            if DEBUG or VERBOSE:
                print("discarded an expendable card")
            return GameData.ClientPlayerDiscardCardRequest(self.name, cards[to_discard])

        # if here better discard a none card than a risky one, specifically the oldest one
        cards = self.knowledge.player_mental_state(self.name).get_cards_from_state("none")
        if len(cards) > 0:
            if DEBUG:
                print("discarded a none card")
            return GameData.ClientPlayerDiscardCardRequest(self.name, cards[0])

        # last option, discard a risky one
        cards = self.knowledge.player_mental_state(self.name).get_cards_from_state("risky")
        if len(cards) > 0:
            if DEBUG:
                print("discarded a risky card")
            return GameData.ClientPlayerDiscardCardRequest(self.name, np.random.choice(cards))

        print("HERE", '+'*50)

    def update_last_action(self, data):
        """
        Update the last action (card played/discarded) known to the agent

        Args:
            data: The GameData object containing the action to register
        """
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
        """
        Update the agent's knowledge (GlobalMentalState) according to the last performed action
        (card played/discarded)

        Args:
            players: The list of the players objects in turn order (must be consistent with self.players)
        """
        if DEBUG:
            print("updating knowledge..")
        # if the agent is the one drawing a card, we have no information on the card
        if self.last_action.last_player == self.name:
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
        """
        Update the agent's knowledge (GlobalMentalState) when an hint is sent from a player to another.
        (note that the destination of the hint doesn't have to be the agent itself)

        Args:
            data: the object describing the hint
        """
        self.hints += 1

        for pos in data.positions:
            if data.type == 'color':
                self.knowledge.player_mental_state(data.destination).update_card(pos, color=colors.index(data.value))
            if data.type == 'value':
                self.knowledge.player_mental_state(data.destination).update_card(pos, rank=data.value)

    def decide_hint(self):
        """
        Return an action, a tuple with action[0] = "value" or "color" and action[1] the rank or the color value
        and action[2] the destination player
        the action represents the best hint calculated by the agent (== None if no good hint has been found)
        """
        if DEBUG:
            print("deciding if there is any good hint to give...")
        best_score = -1
        best_action = None
        if DEBUG:
            print(self.hands)
        for name, hand in self.hands.items():
            maxscore = -1
            best_player_score = -1
            action = None
            best_player_action = None
            if name != self.name:
                # goals = ["play", "discard", "maydiscard", "protect", "keep"]
                goals = self.calculate_goals(hand)
                if DEBUG or HINT_DEBUG:
                    print(" ")
                    print("for player ", name, "calculated the goals:")
                    print(goals)
                for c in colors:
                    move = self.predict(self.knowledge.player_mental_state(name), ("color", c), hand)
                    if DEBUG or HINT_DEBUG:
                        print(" for color", c, "predicted the moves:")
                        print(" ", move)
                    score = self.compare(move, goals)
                    if DEBUG or HINT_DEBUG:
                        print("  score of the hint:", score, "maxscore: ", maxscore)
                        print(" ")
                    if score > maxscore:
                        maxscore = score
                        action = ("color", c, name)
                for rank in range(1, 6):
                    move = self.predict(self.knowledge.player_mental_state(name), ("value", rank), hand)
                    if DEBUG or HINT_DEBUG:
                        print(" for rank", rank, "predicted the moves:")
                        print(" ", move)
                    score = self.compare(move, goals)
                    if DEBUG or HINT_DEBUG:
                        print("  score of the hint:", score, "maxscore: ", maxscore)
                        print(" ")
                    if score > maxscore:
                        maxscore = score
                        action = ("value", rank, name)
                if maxscore > 0:
                    best_player_action = action
                    best_player_score = maxscore
            if best_player_score > best_score and best_player_action not in self.__hint_history:
                best_score = best_player_score
                best_action = best_player_action
            if DEBUG or HINT_DEBUG:
                print("final score:", best_score)
                print("best action: ", best_action)

        if best_action is None:
            print(f"History: {self.__hint_history}")
            players = copy.deepcopy(list(self.hands.keys()))
            players.remove(self.name)
            action = np.random.choice(["value", "color"])
            value = None
            if action == "value":
                value = np.random.choice(range(5))
            else:
                value = np.random.choice(colors)
            best_action = (action, value, np.random.choice(players))
            print("BEST ACTION: ", best_action)

        self.__hint_history.appendleft(best_action)
        return best_action

    def calculate_goals(self, hand):
        """
        Return goals, a list that map for each card of the player hand a goal based on the full knowledge the agent
        has about the player hand
        """
        goals = []
        for card in hand:
            if self.compute_card_state(card) == "playable":
                goals.append("play")
            elif self.compute_card_state(card) == "useless":
                goals.append("discard")
            elif self.compute_card_state(card) == "expendable":
                goals.append("maydiscard")
            elif self.compute_card_state(card) == "risky":
                goals.append("protect")
            else:
                goals.append("keep")
        return goals

    # action object examples: ("value", 2) or ("color", "red")
    def predict(self, player_ms: PlayerMentalState, action, hand):
        """
        Returns predictions, a mapping for every card in player hand to a predicted action (the action
        the agent expect the player to play if given the hint represented by the input object action
        """
        # TODO: add the check that an hint that doesn't target a color or rank in the player hand can't be given
        # and that an hint that give no new information (i.e. M' = M) can't be given
        predictions = []
        ms = copy.deepcopy(player_ms)
        if action[0] == "value":
            for i, card in enumerate(hand):
                if card.value == action[1]:
                    ms.update_card(i, action[1], None)
        else:
            for i, card in enumerate(hand):
                if card.color == action[1]:
                    ms.update_card(i, None, action[1])
        for card_ms in ms.ms_hand:
            card_ms.update_card_state()
            if card_ms.state == "playable":
                predictions.append("play")
            elif card_ms.state == "useless":
                predictions.append("discard")
            elif card_ms.state == "expendable":
                predictions.append("maydiscard")
            elif card_ms.state == "risky":
                predictions.append("protect")
            else:
                predictions.append("keep")
        return predictions

    def compare(self, move, goals):
        """
        Return a score based on a comparison between the goals calculated by the agent and the move the agent predicted
        the player would play (for each card) given a particular hint
        """
        # goals: what I would like the other player to do
        # move: what the other player would do if he received the hint I'm considering in this iteration
        score = 0
        # for i, goal in enumerate(goals):
        #     if goal == move[i]:
        #         if goal == "play":
        #             score += 3
        #         elif goal == "discard":
        #             score += 2
        #         elif goal == "maydiscard":
        #             score += 1
        #         elif goal == "protect":
        #             score += 1
        #     if goal != move[i]:
        #         if goal == "discard" or goal == "maydiscard":
        #             if move[i] == "discard" or move[i] == "maydiscard":
        #                 score += 1
        #         elif goal == "keep" or goal == "protect":
        #             if move[i] == "keep" or move[i] == "protect":
        #                 score += 1
        #         else:
        #             score -= 1
        #         # it means the player is playing a card they are not supposed to play
        #         # or discarding a card they are not supposed to discard
        #         # or keeping a card they are not supposed to keep
        #         # if goal == "play" and move[i] == "discard":
        #         #     return -1
        for goal, m in zip(goals, move):
            score += Agent.__COMPARE_BOARD[Agent.__GOALS_LABELS.index(goal), Agent.__GOALS_LABELS.index(m)]
        return score

    def board_maximums(self):
        """
        Compute and return the maximum that each pile on the board can reach in the current game
        """
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

    def compute_card_state(self, card):
        '''
        Update the card's state according to the board, trash and other's player hand
        it receive in input a card, not the mental state of the card
        '''
        if self.board[colors.index(card.color)] == self.maximums[colors.index(card.color)]:
            return "useless"
        elif self.board[colors.index(card.color)] == card.value - 1:
            return "playable"
        elif self.board[colors.index(card.color)] >= card.value:
            return "useless"
        elif self.maximums[colors.index(card.color)] < card.value - 1:
            return "useless"
        if card.value == 5:
            return "risky"
        expendable = True
        for t in self.trash:
            if t.value == card.value and t.color == card.color:
                expendable = False
        # card.value == 1 is either playable or useless
        if expendable:
            return "expendable"
        else:
            return "risky"

    def discover_card(self, card: Card, card_index: int, action_type: str):
        """
        Called whenever the agent plays or discards a card: if it wasn't fully determined, update the structures

        Args:
            card: the played/discarded card
            card_index: the index of card in agent's hand
            action_type: it's one of ['play', 'mistake', 'discard'] FOR DEBUG ONLY
        """
        ms = self.knowledge.player_mental_state(self.name)
        card_ms: Agent.MentalState = ms.get_card_from_index(card_index)
        if not card_ms.fully_determined:
            # card wasn't fully determined
            for index in range(5):
                if index != card_index:
                    # self.knowledge.player_mental_state(self.name).update_whole_hand(card.value,
                    #                                                                 colors.index(card.color),
                    #                                                                 fully_determined=False)
                    self.knowledge.player_mental_state(self.name).get_card_from_index(index).card_drawn(card.value,
                                                                                                        colors.index(
                                                                                                            card.color))
                    fully_determined_now, fully_determined, rank, color = self.knowledge.player_mental_state(self.name) \
                        .get_card_from_index(index).is_fully_determined_now()
                    if fully_determined_now:
                        for player in self.players:
                            if player != self.name:
                                self.knowledge.player_mental_state(player).update_whole_hand(rank, color,
                                                                                             fully_determined)

            for player in self.players:
                if player != self.name:
                    self.knowledge.player_mental_state(player).update_whole_hand(card.value, colors.index(card.color))

            self.knowledge.update_templates_ms()

        else:
            if action_type == 'mistake':
                if DEBUG:
                    print("Should not be here: made a mistake with a fully determined card.", file=sys.stderr)

        # TODO: update agent's hand (fully det)
        self.knowledge.player_mental_state(self.name).reset_card_mental_state(card_index,
                                                                              self.knowledge.player_template_ms(
                                                                                  self.name))
        if DEBUG:
            print(f"Hands: {self.hands}")
            print(f"Mental states: {self.knowledge.to_string(print_templates=True)}")

    def update_board(self, card: Card):
        """
        Update the representation of the board piles after a card is played correctly
        Args:
            card: the played card
        """
        self.played.append(card)
        self.board[colors.index(card.color)] += 1
        if self.board[colors.index(card.color)] == 5:
            self.hint_gained()

    def update_trash(self, card: Card):
        """
        Update the representation of the trash after a card is played by mistake or is discarded
        Args:
            card: the discarded card
        """
        self.trash.append(card)

    def hint_consumed(self):
        """
        Increment (if possible) the count of used hint tokens
        """
        self.hints = min(self.hints + 1, 8)

    def hint_gained(self):
        """
        Decrement (if possible) the count of used hint tokens
        """
        self.hints = max(0, self.hints - 1)

    def mistake_made(self):
        """
        Increment (if possible) the count of used error tokens
        """
        self.errors += 1
        assert self.errors < 3

    def assert_aligned_with_server(self, hints_used: int, mistakes_made: int, board, trash: list, players: list):
        """
        FOR DEBUG ONLY: assert that every internal structure is consistent with the server knowledge
        Args:
            hints_used: the number of hint tokens used in the actual game
            mistakes_made: the number of mistakes made in the actual game
            board: the actual current board game
            trash: the list of actually discarded cards
            players: the list of objects of class Player - they also store their hands
        """
        assert self.hints == hints_used, "wrong count of hints"
        assert self.errors == mistakes_made, "wrong count of errors"
        # b = [max(v) if len(v) > 0 else 0 for _, v in board.items()]
        # assert self.board == b, f"wrong board: self.board: {self.board}, board: {board}"
        assert self.trash == trash, " wrong trash"
        for player in players:
            assert player.hand == self.hands[player.name], f"player {player.name} wrong hand"

    def track_played_card(self, player_name: str, card_index: int):
        """
        Remove the played card from another player's hand
        Args:
            player_name: the player who played the card
            card_index: the index of the played card in player_name's hand
        """
        del self.hands[player_name][card_index]
        # TODO: update agent's hand (fully det)
        self.knowledge.player_mental_state(player_name).reset_card_mental_state(card_index,
                                                                                self.knowledge.player_template_ms(
                                                                                    player_name))
        if DEBUG:
            print(f"Hands: {self.hands}")
            print(f"Mental states: {self.knowledge.to_string(print_templates=True)}")

    def track_drawn_card(self, players: list):
        """
        Only called if the player who played/discarded a card also drew a new one (i.e. if the has more than 0 cards)
        Args:
            players: the list of objects of class Player - they also store their hands
        """
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
                self.knowledge.player_mental_state(p).update_whole_hand(new_card.value, colors.index(new_card.color))

    def update_knowledge_on_hint(self, hint_type: str, value, positions: list, destination: str):
        """
        Update the knowledge of the hinted player. If it's the agent, also update the knowledge of other players
        Args:
            hint_type: 'value' or 'color'
            value: the rank or the color, depending on hint_type
            positions: the list of the positions of the cards which match the hint
            destination: the name of the hinted player
        """
        if destination == self.name:
            if hint_type == 'value':
                for index in positions:
                    self.knowledge.player_mental_state(destination).update_card(index, rank=value)
                    fully_determined_now, fully_determined, rank, color = self.knowledge.player_mental_state(
                        destination).get_card_from_index(
                        index).is_fully_determined_now()
                    if fully_determined_now:
                        for player in self.players:
                            if player != self.name:
                                self.knowledge.player_mental_state(player).update_whole_hand(rank, color,
                                                                                             fully_determined)
                        self.knowledge.player_mental_state(destination).get_card_from_index(
                            index).reset_fully_determined_now()

            else:
                for index in positions:
                    self.knowledge.player_mental_state(destination).update_card(index, color=colors.index(value))
                    fully_determined_now, fully_determined, rank, color = self.knowledge.player_mental_state(
                        destination).get_card_from_index(
                        index).is_fully_determined_now()
                    if fully_determined_now:
                        for player in self.players:
                            if player != self.name:
                                self.knowledge.player_mental_state(player).update_whole_hand(rank, color,
                                                                                             fully_determined)
                        self.knowledge.player_mental_state(destination).get_card_from_index(
                            index).reset_fully_determined_now()
        else:
            if hint_type == 'value':
                for index in positions:
                    self.knowledge.player_mental_state(destination).update_card(index, rank=value)
            else:
                for index in positions:
                    self.knowledge.player_mental_state(destination).update_card(index, color=colors.index(value))

        if DEBUG:
            print(f"Hands: {self.hands}")
            print(f"Mental states: {self.knowledge.to_string()}")

# TODO
## when a card of the agent hand is fully determined add it to hands list
## when a card is fully determined a update_template needs to be triggered
## nella logica degli hint vengono dati punti anche alle carte che il giocatore sa già essere playable
