import copy
from enum import IntEnum
import numpy as np
import random
from itertools import product
import GameData
from constants import SEED
import game

colors = ["red", "yellow", "green", "blue", "white"]


class Color(IntEnum):
    RED = 0
    YELLOW = 1
    GREEN = 2
    BLUE = 3
    WHITE = 4


color_str2enum = {
    "red": Color.RED,
    "yellow": Color.YELLOW,
    "green": Color.GREEN,
    "blue": Color.BLUE,
    "white": Color.WHITE,
}

color_enum2str = {
    Color.RED: "red",
    Color.YELLOW: "yellow",
    Color.GREEN: "green",
    Color.BLUE: "blue",
    Color.WHITE: "white",
}

HAND_SIZE = 5
CARD_QUANTITIES = [3, 2, 2, 2, 1]
MAX_HINTS = 8
MAX_ERRORS = 3


### WARNING ###
# When a player will compute the rules the decide the next move, he will have to
# add his hand back into the deck before performing any inference.
# When the computation is done, he will have to remove the cards from the deck again


class Card:
    def __init__(self, rank: int, color: Color) -> None:
        # id ?
        self.rank = rank
        self.color = color
        self.rank_known = False
        self.color_known = False
        rank is not None and color is not None

    def __eq__(self, other):
        if type(other) is not Card and type(other) is not game.Card:
            raise TypeError(f"Cannot compare type card with {type(other)}")
        if hasattr(other, "rank"):
            return self.rank == other.rank and self.color == color_str2enum[other.color]
        elif hasattr(other, "value"):
            return (
                self.rank == other.value and self.color == color_str2enum[other.color]
            )
        else:
            raise AttributeError(
                f"Object {other} doesn't have attribute rank nor value."
            )

    def __ne__(self, other):
        return not self.__eq__(other)

    def reveal_rank(self, rank=None):
        if rank is not None:
            # assert self.rank is None
            self.rank = rank
        self.rank_known = True

    def reveal_color(self, color=None):
        if color is not None:
            # assert self.color is None
            self.color = color
        self.color_known = True

    def is_fully_determined(self):
        return self.rank_known and self.color_known


class Deck:
    def __init__(self) -> None:
        col = np.array(CARD_QUANTITIES)
        col = col.reshape(col.size, 1)
        self._table = np.tile(col, len(colors))
        self._reserved_ranks = np.zeros(len(CARD_QUANTITIES), dtype=np.int8)
        self._reserved_colors = np.zeros(len(Color), dtype=np.int8)

    def __deepcopy__(self, memo={}):
        cls = self.__class__
        result = cls.__new__(cls)
        result._table = np.copy(self._table)
        result._reserved_ranks = np.copy(self._reserved_ranks)
        result._reserved_colors = np.copy(self._reserved_colors)
        return result

    def __len__(self):
        """
        Return the number of cards still available in the deck
        """
        return np.sum(self._table)

    def __getitem__(self, item):
        if type(item) is tuple:
            return self._table[item[0], item[1]]
        else:
            raise IndexError

    def _decrement(self, rank: int, color: Color) -> None:
        assert (
            self._table[rank - 1][color] > 0
        ), "trying to decrement zero value from Deck"
        self._table[rank - 1][color] -= 1

    def _increment(self, rank: int, color: Color) -> None:
        assert (
            self._table[rank - 1][color] < CARD_QUANTITIES[rank - 1]
        ), "trying to increment maximum value from Deck"
        self._table[rank - 1][color] += 1

    def remove_cards(self, cards: list[Card]) -> None:
        for card in cards:
            self._decrement(card.rank, card.color)

    def add_cards(self, cards: list[Card], redeterminizing=False) -> None:
        # reset reservations
        if redeterminizing:
            assert np.all(
                self._reserved_colors == 0
            ), "Color reservation not reset correctly"
            assert np.all(
                self._reserved_ranks == 0
            ), "Rank reservation not reset correctly"
        for card in cards:
            if not redeterminizing or not card.is_fully_determined():
                self._increment(card.rank, card.color)
            # redeterminizing and not fully determined determined card
            if redeterminizing and not card.is_fully_determined():
                if card.rank_known:
                    self._reserved_ranks[card.rank - 1] += 1
                elif card.color_known:
                    self._reserved_colors[card.color] += 1

    def draw(self, rank: int = None, color: Color = None) -> Card:
        if rank is None and color is None:
            # rows, columns = np.nonzero(self._table)
            # pos = np.random.choice(rows.size)
            # rank = rows[pos] + 1
            # color = columns[pos]
            possibilities = [
                (r, c)
                for r in range(len(CARD_QUANTITIES))
                for c in range(len(Color))
                for _ in range(self._table[r][c])
            ]
            rank, color = random.choice(possibilities)
            rank += 1
        elif rank is not None:
            # rows = np.nonzero(self._table[:, color])[0]
            # rank = np.random.choice(rows) + 1
            possibilities = [
                c for c in range(len(Color)) for _ in range(self._table[rank - 1][c])
            ]
            color = random.choice(possibilities)
            assert color is not None
        elif color is not None:
            # columns = np.nonzero(self._table[rank - 1, :])[0]
            # color = np.random.choice(columns)
            possibilities = [
                r
                for r in range(len(CARD_QUANTITIES))
                for _ in range(self._table[r][color])
            ]
            rank = random.choice(possibilities) + 1
            assert rank is not None
        self._decrement(rank, color)
        return Card(rank, color)

    def draw2(self, rank: int = None, color: Color = None) -> Card:
        # OBS: if rank or color are not None, for sure we are redeterminizing

        # not fully determined
        if rank is None or color is None:

            table = np.copy(self._table)

            update_table = True
            iteration = 0
            max_iterations = 100

            while update_table:
                update_table = False
                iteration += 1
                if iteration > max_iterations:
                    raise RuntimeError("Stuck in draw2")
                if rank is None:
                    # if no rank is specified, do not pick any rank-reserved card
                    r_idx = np.sum(table, axis=1) <= self._reserved_ranks
                    table[r_idx, :] = 0
                    update_table = np.any(r_idx)

                if color is None:
                    # if no color is specified, do not pick any rank-reserved card
                    c_idx = np.sum(table, axis=0) <= self._reserved_colors
                    table[:, c_idx] = 0
                    update_table = update_table or np.any(c_idx)

            # completely unknown
            if rank is None and color is None:
                possibilities = [
                    coordinates
                    for coordinates, occurrencies in np.ndenumerate(table)
                    for _ in range(occurrencies)
                ]

                rank, color = random.choice(possibilities)
                rank += 1

            # known rank
            elif rank is not None:
                assert (
                    self._reserved_ranks[rank - 1] > 0
                ), f"No card with rank{rank} was previously reserved"
                self._reserved_ranks[rank - 1] -= 1
                possibilities = [
                    c for c in range(table.shape[1]) for _ in range(table[rank - 1][c])
                ]
                color = random.choice(possibilities)

            # known color
            elif color is not None:
                assert (
                    self._reserved_colors[color] > 0
                ), f"No card with color {color} was previously reserved"
                self._reserved_colors[color] -= 1
                possibilities = [
                    r for r in range(table.shape[0]) for _ in range(table[r][color])
                ]
                rank = random.choice(possibilities) + 1

            self._decrement(rank, color)

        assert rank is not None and color is not None
        return Card(rank, color)

    def is_empty(self) -> bool:
        return not np.any(self._table != 0)


class Trash:
    def __init__(self) -> None:
        self.list = []
        self.maxima = [5] * 5
        col = np.array(CARD_QUANTITIES)
        col = col.reshape(col.size, 1)
        self._table = np.tile(col, len(colors))

    def __deepcopy__(self, memo={}):
        cls = self.__class__
        result = cls.__new__(cls)
        result._table = np.copy(self._table)
        result.maxima = copy.copy(self.maxima)
        result.list = copy.deepcopy(self.list)
        return result

    def _decrement(self, rank: int, color: Color) -> None:
        assert (
            self._table[rank - 1][color] > 0
        ), "trying to decrement zero value from Trash"
        self._table[rank - 1][color] -= 1
        if self._table[rank - 1][color] == 0:
            self.maxima[color] = min(rank - 1, self.maxima[color])

    def append(self, card: Card) -> None:
        self.list.append(card)
        self._decrement(card.rank, card.color)

    def get_table(self):
        return self._table


class GameState:
    """
    Attributes:
        players:            list of player names in turn order
        root_player:   name of the root player (agent)
        hands:              dictionary with player names as keys and hands (list of cards) as values
        board:             successfully played cards (currently in the table)
        trash:              list of discarded cards
        hints:              the number of used note tokens
        errors:             the number of used storm tokens
        deck:               the cards currently in the deck
    """

    def __init__(
        self,
        players_names: list[str],
        root_player: str,
        data: GameData.ServerGameStateData = None,
    ) -> None:
        """
        Create a new GameState

        Args:
            players_names: the list of the player names in turn order
            root_player: the name of the root player (agent)
            data: the server game state to use to initialize the client game state
        """
        global HAND_SIZE
        if len(players_names) >= 4:
            HAND_SIZE = 4
        self.players = copy.deepcopy(players_names)
        self.root_player = root_player
        if data is not None:
            self.board = [0] * 5
            self.deck = Deck()
            self.trash = Trash()
            self.hints = data.usedNoteTokens
            self.errors = data.usedStormTokens
            self.hands = {
                player.name: GameState.server_to_client_hand(player.hand)
                for player in data.players
            }
            self.hands[root_player] = [Card(None, None) for _ in range(HAND_SIZE)]
            for player, hand in self.hands.items():
                if player != self.root_player:
                    self.deck.remove_cards(hand)

    def __deepcopy__(self, memo={}):
        cls = self.__class__
        result = cls.__new__(cls)
        result.players = copy.deepcopy(self.players)
        result.root_player = copy.copy(self.root_player)
        result.hands = copy.deepcopy(self.hands)
        result.board = self.board[:]
        result.trash = copy.deepcopy(self.trash)
        result.deck = copy.deepcopy(self.deck)
        result.hints = self.hints
        result.errors = self.errors
        return result

    @staticmethod
    def server_to_client_hand(server_hand: list) -> list[Card]:
        """
        Generate a client-hand (list of cards) given a server-hand

        Args:
            server_hand: list of server cards

        Returns:
            a list of client cards
        """
        hand = []
        for card in server_hand:
            hand.append(Card(rank=card.value, color=color_str2enum[card.color]))
        return hand

    def get_prev_player_name(self, current_player: str) -> str:
        current_player_idx = self.players.index(current_player)
        prev_player_idx = current_player_idx - 1
        if prev_player_idx < 0:
            prev_player_idx = len(self.players) - 1
        return self.players[prev_player_idx]

    def get_next_player_name(self, current_player: str) -> str:
        current_player_idx = self.players.index(current_player)
        next_player_idx = (current_player_idx + 1) % len(self.players)
        return self.players[next_player_idx]

    def remove_card_from_hand(self, player: str, card_idx: int) -> None:
        """ """
        del self.hands[player][card_idx]

    def append_card_to_player_hand(self, player: str, card: Card):
        """ """
        self.hands[player].append(card)
        if player != self.root_player:
            self.deck.remove_cards([card])

    def give_hint(
        self, cards_idx: list[int], destination: str, hint_type: str, hint_value: int
    ) -> None:
        """ """
        hand = self.hands[destination]
        for idx in cards_idx:
            if hint_type == "value":
                hand[idx].reveal_rank(hint_value)
            elif hint_type == "color":
                hand[idx].reveal_color(hint_value)
        self.hints += 1

    def discover_card_root(self, rank: int, color: Color, card_idx: int) -> None:
        card = self.hands[self.root_player][card_idx]
        if not card.is_fully_determined():
            card.reveal_rank(rank)
            card.reveal_color(color)
            self.deck.remove_cards([card])

    def update_trash(self, card: Card) -> None:
        """ """
        self.trash.append(card)

    def gain_hint(self) -> None:
        """ """
        if self.hints == 0:
            raise RuntimeError(f"Trying to gain more than {MAX_HINTS} hint tokens.")
        self.hints -= 1

    def use_hint(self) -> None:
        """ """
        if self.hints == MAX_HINTS:
            raise RuntimeError("Trying to use more token hints than allowed")
        self.hints += 1

    def mistake_made(self) -> None:
        """ """
        if self.errors >= MAX_ERRORS:
            raise RuntimeError("Too many error tokens")
        self.errors += 1

    def card_correctly_played(self, color: Color) -> None:
        """ """
        if self.board[color] >= self.trash.maxima[color]:
            raise RuntimeError("Trying to play a card that doesn't exists")
        self.board[color] += 1
        self.hints = max(0, self.hints - 1)  # gain an hint if possible

    def game_ended(self) -> tuple[bool, int]:
        """
        Checks if the game is ended for some reason. If it's ended, it returns True and the score of the game.
        If the game isn't ended, it returns False, None
        """
        if self.errors == MAX_ERRORS:
            return True, 0
        if self.board == self.trash.maxima:
            return True, sum(self.board)
        if self.deck.is_empty():
            return True, sum(self.board)
        return False, None


class MCTSState(GameState):
    """ """

    def __init__(self, initial_state: GameState) -> None:
        super().__init__(
            copy.deepcopy(initial_state.players),
            copy.copy(initial_state.root_player),
        )
        self.hands = copy.deepcopy(initial_state.hands)
        self.board = initial_state.board[:]
        self.deck = copy.deepcopy(initial_state.deck)
        self.trash = copy.deepcopy(initial_state.trash)
        self.hints = initial_state.hints
        self.errors = initial_state.errors
        # determinize root's hand
        root_hand = self.hands[self.root_player]
        for idx, card in enumerate(root_hand):
            if not card.is_fully_determined():
                assert card.rank is None or card.color is None
                new_card = self.deck.draw(rank=card.rank, color=card.color)
                assert new_card.rank is not None and new_card.color is not None
                new_card.rank_known = card.rank_known
                new_card.color_known = card.color_known
                root_hand[idx] = new_card

    # MCTS
    def play_card(self, player: str, card_idx: int) -> None:
        """
        Track a card played by "player". The played card will be removed from the player's hand
        and added to either the board or the trash depending on its value
        (the number of tokens will be adjusted accordingly)
        A new card drawn from the deck will be appended in the last position of the player's hand

        Args:
            player: the name of the player
            card_idx: the index of the card in the player's hand
        """
        card = self.hands[player].pop(card_idx)
        self.hands[player].append(self.deck.draw())
        # if player == self.root_player and not card.is_fully_determined():
        #     self.deck.remove_cards([card])
        if self.board[card.color] == card.rank - 1:
            self.board[card.color] += 1
            if card.rank == 5:
                self.hints = max(self.hints - 1, 0)
        else:
            self.trash.append(card)
            self.errors += 1

    def discard_card(self, player: str, card_idx: int) -> None:
        """
        Track a card discarded by "player". The discarded card will be removed from the player's hand
        and added to the trash (the number of tokens will be adjusted accordingly)
        A new card drawn from the deck will be appended in the last position of the player's hand

        Args:
            player: the name of the player
            card_idx: the index of the card in the player's hand
        """
        card = self.hands[player].pop(card_idx)
        self.trash.append(card)
        self.hands[player].append(self.deck.draw())
        # if player == self.root_player and not card.is_fully_determined():
        #     self.deck.remove_cards([card])
        self.hints = max(self.hints - 1, 0)

    def give_hint(self, destination: str, hint_type: str, hint_value: int) -> None:
        """
        This works asssuming that all the cards in all the players' hands have a defined rank and color
        (either known or not)
        """
        hand = self.hands[destination]
        for card in hand:
            if hint_type == "value" and card.rank == hint_value:
                card.reveal_rank()
            elif hint_type == "color" and card.color == hint_value:
                card.reveal_color()

    # MCTS
    def hints_available(self) -> int:
        return MAX_HINTS - self.hints

    def redeterminize_hand(self, player_name: str) -> None:
        hand = self.hands[player_name]
        if player_name == self.root_player:
            raise RuntimeError("Cannot re-determinize root player's hand")
        self.deck.add_cards(hand, redeterminizing=True)

        # for idx, card in enumerate(hand):
        #     rank = card.rank if card.rank_known else None
        #     color = card.color if card.color_known else None
        #     new_card = self.deck.draw(rank=rank, color=color)
        #     new_card.rank_known = card.rank_known
        #     new_card.color_known = card.color_known
        #     hand[idx] = new_card

        for idx, card in enumerate(hand):
            rank = card.rank if card.rank_known else None
            color = card.color if card.color_known else None
            new_card = self.deck.draw2(rank=rank, color=color)
            new_card.rank_known = card.rank_known
            new_card.color_known = card.color_known
            hand[idx] = new_card

    # MCTS
    def restore_hand(self, player_name: str, saved_hand: list[Card]) -> None:
        """
        Restore the specified hand for the specified player, removing all the "illegal" cards
        and re-determinizing their slots

        Args:
            player_name: the name of the player
            saved_hand: the hand to restore
        """
        self.deck.add_cards(self.hands[player_name])  # put cards back in deck
        self.hands[player_name] = []
        self._remove_illegal_cards(saved_hand)  # remove inconsistencies
        self.deck.remove_cards(filter(lambda card: card is not None, saved_hand))  # pick cards from deck
        self._determinize_empty_slots(saved_hand)
        self.hands[player_name] = saved_hand

    # MCTS
    def _determinize_empty_slots(self, hand: list[Card]) -> None:
        """
        Determinize the empty slots of the hand (where card = None)

        Args:
            hand: the hand to adjust
        """
        for idx in range(len(hand)):
            if hand[idx] is None:
                hand[idx] = self.deck.draw()

    # MCTS
    def _remove_illegal_cards(self, cards: list[Card]) -> None:
        """
        Remove the illegal cards from the list (considering all the cards in the trash,
        in the player's hands and on the table)

        Args:
            cards: the list of cards to modify
        """
        locations = self.trash.list
        for p in self.players:
            locations += self.hands[p]

        for idx, card in enumerate(cards):
            # if card is None:
            #     continue
            quantity = 1 + self.deck[card.rank, card.color]  # 1 is for "card" itself
            for c in locations:
                if c.rank == card.rank and c.color == card.color:
                    quantity += 1
            if self.board[card.color] >= card.rank:
                quantity += 1
            if quantity > CARD_QUANTITIES[card.rank - 1]:
                # TODO: idx not ok for index
                cards[idx] = None

    def assert_consistency(self):
        col = np.array(CARD_QUANTITIES)
        col = col.reshape(col.size, 1)
        full_table = np.tile(col, len(colors))
        table = np.copy(self.deck[:, :])

        # trash
        trash_table = full_table - self.trash.get_table()
        table += trash_table

        # hands
        for player in self.players:
            for card in self.hands[player]:
                table[card.rank - 1][card.color] += 1

        # board
        for c_idx, tos in enumerate(self.board):
            for i in range(tos):
                table[i][c_idx] += 1

        assert np.all(table == full_table), "Consistency failed"


### TODO
# * Gestire ultimo giro di giocate dopo che il mazzo e' finito
# * to_string per la classe Tree
