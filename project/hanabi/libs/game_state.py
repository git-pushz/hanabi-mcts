import copy
import typing
from enum import Enum

import numpy as np
from typing import List, Tuple

from .. import GameData


colors = ["red", "yellow", "green", "blue", "white"]


class Color(Enum):
    RED = 0
    YELLOW = 1
    GREEN = 2
    BLUE = 3
    WHITE = 4


color2enum = {
    "red": Color.RED,
    "yellow": Color.YELLOW,
    "green": Color.GREEN,
    "blue": Color.BLUE,
    "white": Color.WHITE,
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

    def reveal_rank(self):
        self.rank_known = True

    def reveal_color(self):
        self.color_known = True

    def is_fully_determined(self):
        return self.rank_known and self.color_known


class Deck:
    def __init__(self) -> None:
        col = np.array(CARD_QUANTITIES)
        col = col.reshape(col.size, 1)
        self._table = np.tile(col, len(colors))

    def _decrement(self, rank: int, color: Color) -> None:
        assert (
            self._table[rank - 1][color] > 0,
            "trying to decrement zero value from Deck",
        )
        self._table[rank - 1][color] -= 1

    def _increment(self, rank: int, color: Color) -> None:
        assert (
            self._table[rank - 1][color] < CARD_QUANTITIES[rank - 1],
            "trying to increment maximum value from Deck",
        )
        self._table[rank - 1][color] += 1

    def remove_cards(self, cards: List[Card]) -> None:
        for card in cards:
            self._decrement(card.rank, card.color)

    def add_cards(self, cards: List[Card]) -> None:
        for card in cards:
            self._increment(card.rank, card.color)

    def draw(self, rank: int = None, color: Color = None) -> Card:
        rows, columns = np.nonzero(self._table)
        if rank is None:
            rank = np.random.choice(rows)
        if color is None:
            color = np.random.choice(columns)
        self._decrement(rank, color)
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

    def _decrement(self, rank: int, color: Color) -> None:
        assert (
            self._table[rank][color] > 0,
            "trying to decrement zero value from Deck",
        )
        self._table[rank - 1][color] -= 1
        if self._table[rank - 1][color] == 0:
            self.maxima[color] = min(rank - 1, self.maxima[color])

    def append(self, card: Card) -> None:
        self.list.append(card)
        self._decrement(card.rank, card.color)


class GameState:
    """
    Atttributes:
        players:            list of player names in turn order
        root_player_name:   name of the root player (agent)
        hands:              dictionary with player names as keys and hands (list of cards) as values
        boards:             successfully played cards (currently in the table)
        maximums:           the maximum values that can be reached by each firework
        trash:              list of discarded cards
        hints:              the number of used note tokens
        errors:             the number of used storm tokens
        deck:               the cards currently in the deck
    """

    def __init__(
        self,
        players_names: List[str],
        root_player_name: str,
        data: GameData.ServerGameStateData,
    ) -> None:
        """
        Create a new GameState

        Args:
            player_names: the list of the player names in turn order
            root_player_name: the name of the root player (agent)
            data: the server game state to use to initialize the client game state
        """
        global HAND_SIZE
        self.players = copy.deepcopy(players_names)
        if len(players_names) >= 4:
            HAND_SIZE = 4
        self.root_player_name = root_player_name
        self.hands = {
            player.name: GameState.server_to_client_hand(player.hand)
            for player in data.players
        }
        self.hands[root_player_name] = []
        self.board = [0] * 5
        self.trash = Trash()
        self.hints = data.usedNoteTokens
        self.errors = data.usedStormTokens
        self.deck = Deck()
        for hand in self.hands.values():
            self.deck.remove_cards(hand)

    def __deepcopy__(self, memo={}):
        cls = self.__class__
        result = cls.__new__(cls)
        result.players = copy.deepcopy(self.players)
        result.root_player_name = copy.copy(self.root_player_name)
        result.hands = copy.deepcopy(self.hands)
        result.board = self.board[:]
        result.trash = copy.deepcopy(self.trash)
        result.hints = self.hints
        result.errors = self.errors
        return result

    @staticmethod
    def server_to_client_hand(server_hand: list) -> List[Card]:
        """
        Generate a client-hand (list of cards) given a server-hand

        Args:
            server_hand: list of server cards

        Returns:
            a list of client cards
        """
        hand = []
        for card in server_hand:
            hand.append(Card(rank=card.rank, color=card.color))
        return hand

    def remove_card_from_hand(self, player: str, card_idx: int) -> None:
        """
        """
        del self.hands[player][card_idx]

    def append_card_to_player_hand(self, player: str, card: Card):
        """
        """
        self.hands[player].append(card)
        if player != self.root_player_name:
            self.deck.draw(rank=card.rank, color=card.color)

    def give_hint(self, destination: str, hint_type: str, hint_value: int) -> None:
        """
        """
        hand = self.hands[destination]
        for card in hand:
            if hint_type == "rank" and card.rank == hint_value:
                card.reveal_rank()
            if hint_type == "color" and card.color == hint_value:
                card.reveal_color()
        self.hints += 1

    def update_trash(self, card: Card) -> None:
        """
        """
        self.trash.append(card)

    def gain_hint(self) -> None:
        """
        """
        if self.hints == 0:
            raise RuntimeError(f"Trying to gain more than {MAX_HINTS} hint tokens.")
        self.hints -= 1

    def use_hint(self) -> None:
        """
        """
        if self.hints == MAX_HINTS:
            raise RuntimeError("Trying to use more token hints than allowed")
        self.hints += 1

    def mistake_made(self) -> None:
        """
        """
        if self.errors >= MAX_ERROR:
            raise RuntimeError("Too many error tokens")
        self.errors += 1

    def card_correctly_played(self, color: Color) -> None:
        """
        """
        if self.board[color] >= self.trash.maxima[color]:
            raise RuntimeError("Trying to play a card that doesn't exists")
        self.board[color] += 1
        self.hints = max(0, self.hints - 1)  # gain an hint if possible

    def game_ended(self) -> Tuple[bool, typing.Any]:
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

    def __init__(self, initial_state: GameState, player_names: List[str], root_player_name: str,
                 data: GameData.ServerGameStateData) -> None:
        super().__init__(player_names, root_player_name, data)
        self.players = copy.deepcopy(initial_state.players)
        self.root_player_name = copy.copy(initial_state.root_player_name)
        self.hands = copy.deepcopy(initial_state.hands)
        self.board = initial_state.board[:]
        self.trash = copy.deepcopy(initial_state.trash)
        self.hints = initial_state.hints
        self.errors = initial_state.errors

    # MCTS
    def play_card(self, player: str, card_idx: int) -> None:
        card = self.hands[player].pop(card_idx)
        self.hands[player].append(self.deck.draw())
        if player == self.root_player_name and not card.is_fully_determined():
            self.deck.remove_cards([card])
        if self.board[card.color] == card.rank - 1:
            self.board[card.color] += 1
            if card.rank == 5:
                self.hints = min(self.hints + 1, MAX_HINTS)
        else:
            self.trash.append(card)
            self.errors += 1

    def discard_card(self, player: str, card_idx: int) -> None:
        card = self.hands[player].pop(card_idx)
        self.hands[player].append(self.deck.draw())
        if player == self.root_player_name and not card.is_fully_determined():
            self.deck.remove_cards([card])
        self.trash.append(card)
        self.hints = max(self.hints - 1, 0)

    # MCTS
    def hints_available(self) -> int:
        return MAX_HINTS - self.hints

    # MCTS
    def get_prev_player_name(self, current_player: str) -> str:
        current_player_idx = self.players.index(current_player)
        prev_player_idx = current_player_idx - 1
        if prev_player_idx < 0:
            prev_player_idx = len(self.players) - 1
        return self.players[prev_player_idx]

    # MCTS
    def get_next_player_name(self, current_player: str) -> str:
        current_player_idx = self.players.index(current_player)
        next_player_idx = (current_player_idx + 1) % len(self.players)
        return self.players[next_player_idx]

    # MCTS
    def restore_hand(self, player_name: str, saved_hand: List[Card]) -> None:
        """
        Restore the specified hand for the specified player, removing all the "illegal" cards
        and re-determinizing their slots

        Args:
            player_name: the name of the player
            saved_hand: the hand to restore
        """
        self.deck.add_cards(self.hands[player_name])  # put cards back in deck
        self._remove_illegal_cards(saved_hand)  # remove inconsistencies
        self.deck.remove_cards(saved_hand)  # pick cards from deck
        self._determinize_empty_slots(saved_hand)
        self.hands[player_name] = saved_hand

    # MCTS
    def _determinize_empty_slots(self, hand: List[Card]) -> None:
        """
        Determinize the empty slots of the hand (where card = None)

        Args:
            hand: the hand to adjust
        """
        for idx in range(len(hand)):
            if hand[idx] is None:
                hand[idx] = self.deck.draw()

    # MCTS
    def _remove_illegal_cards(self, cards: List[Card]) -> None:
        """
        Remove the illegal cards from the list (considering all the cards in the trash,
        in the player's hands and on the table)

        Args:
            cards: the list of cards to modify
        """
        locations = self.trash.list
        for p in self.players:
            locations += p.hand

        for idx in range(len(cards)):
            card = cards[idx]
            if card is None:
                continue
            quantity = 1 + self.deck[card.rank][card.color]  # 1 is for "card" itself
            for c in locations:
                if c.rank == card.rank and c.color == card.color:
                    quantity += 1
            if self.board[card.color] >= card.rank:
                quantity += 1
            if quantity > CARD_QUANTITIES[card.rank - 1]:
                cards[idx] = None


### TODO
# * Gestire ultimo giro di giocate dopo che il mazzo e' finito
