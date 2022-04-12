import copy
import numpy as np
from typing import List, Tuple, Optional
import GameData
from hyperparameters import SCORE_3_ERRORS

from utils import (
    CARD_QUANTITIES,
    color_str2enum,
    Color,
    Card,
    Deck,
    Trash,
)

MAX_HINTS = 8
MAX_ERRORS = 3
HAND_SIZE = 5

### WARNING ###
# When a player will compute the rules the decide the next move, he will have to
# add his hand back into the deck before performing any inference.
# When the computation is done, he will have to remove the cards from the deck again


class GameState:
    """
    Stores all the state of the game.

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
        players_names: List[str],
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
            self.board = np.full(len(Color), 0, dtype=np.uint8)
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
        self.last_turn_played = None  # Only used in MCTSState

    def __deepcopy__(self, memo={}):
        cls = self.__class__
        result = cls.__new__(cls)
        result.players = copy.deepcopy(self.players)
        result.root_player = copy.copy(self.root_player)
        result.hands = copy.deepcopy(self.hands)
        result.board = np.copy(self.board)
        result.trash = copy.deepcopy(self.trash)
        result.deck = copy.deepcopy(self.deck)
        result.hints = self.hints
        result.errors = self.errors
        result.last_turn_played = copy.deepcopy(self.last_turn_played)
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
            hand.append(Card(rank=card.value, color=color_str2enum[card.color]))
        return hand

    def get_prev_player_name(self, current_player: str) -> str:
        """
        Returns the name of the player whose turn was before current_player

        Args:
            current_player: the successor of the returned player
        """
        current_player_idx = self.players.index(current_player)
        prev_player_idx = current_player_idx - 1
        if prev_player_idx < 0:
            prev_player_idx = len(self.players) - 1
        return self.players[prev_player_idx]

    def get_next_player_name(self, current_player: str) -> str:
        """
        Returns the name of the player whose turn is after current_player

        Args:
            current_player: the predecessor of the returned player
        """
        current_player_idx = self.players.index(current_player)
        next_player_idx = (current_player_idx + 1) % len(self.players)
        return self.players[next_player_idx]

    def root_card_discovered(self, card_idx: int, rank: int, color: Color) -> None:
        """
        Let the root player discover a card in his own hand
        """
        card = self.hands[self.root_player][card_idx]
        if not card.is_fully_determined():
            card.reveal_rank(rank)
            card.reveal_color(color)
            self.deck.remove_cards([card])

    def card_discarded(self, player: str, card_idx: int) -> None:
        """
        Remove a card from the player's hand and put it in the trash.
        The card must be fully specified, even for the root_player
        """
        card = self.hands[player].pop(card_idx)
        assert card.rank is not None and card.color is not None
        self.trash.append(card)
        assert self.hints > 0
        self.hints -= 1

    def card_played(self, player: str, card_idx: int, correctly: bool) -> None:
        """
        Remove a card from the player's hand and put it in the trash.
        The card must be fully specified, even for the root_player
        """
        card = self.hands[player].pop(card_idx)
        assert card.rank is not None and card.color is not None
        if correctly:
            assert self.board[card.color] < self.trash.maxima[card.color]
            assert card.rank == self.board[card.color] + 1
            self.board[card.color] += 1
            if card.rank == 5 and self.hints > 0:
                self.hints -= 1
        else:
            self.trash.append(card)
            assert self.errors < MAX_ERRORS
            self.errors += 1

    def card_drawn(self, player: str, card: Card) -> None:
        """
        Update the state when a new card is drawn by a player
        """
        if player == self.root_player:
            assert card.rank is None and card.color is None
        else:
            assert card.rank is not None and card.color is not None
            self.deck.remove_cards([card])
        self.hands[player].append(card)

    def hint_given(
        self, destination: str, cards_idx: List[int], hint_type: str, hint_value: int
    ) -> None:
        """
        Updates destination's knowledge based on the hint received

        Args:
            destination: the name of the destination of the hint
            cards_idx: the list of the positions of destination's cards which match the hint
            hint_type: whether the hint was a `value` or a `color` one
            hint_value: the rank or the color of the hint
        """
        assert self.hints < MAX_HINTS
        hand = self.hands[destination]
        for idx in cards_idx:
            card = hand[idx]
            if card.is_fully_determined():
                continue
            if hint_type == "value":
                card.reveal_rank(hint_value)
            elif hint_type == "color":
                card.reveal_color(hint_value)
            # the root player fully determined a card and now knows it's not in the deck
            if destination == self.root_player and card.is_fully_determined():
                self.deck.remove_cards([card])
        self.hints += 1


class MCTSState(GameState):
    """
    Subclass of GameState, holds the state of the game during the MCTS. It has the same attributes of its superclass
    but it has more methods.
    """

    def __init__(self, initial_state: GameState) -> None:
        super().__init__(
            copy.deepcopy(initial_state.players),
            copy.copy(initial_state.root_player),
        )
        self.hands = copy.deepcopy(initial_state.hands)
        self.board = np.copy(initial_state.board)
        self.deck = copy.deepcopy(initial_state.deck)
        self.trash = copy.deepcopy(initial_state.trash)
        self.hints = initial_state.hints
        self.errors = initial_state.errors
        # determinize root's hand
        root_hand = self.hands[self.root_player]
        self.deck.reserve_semi_determined_cards(root_hand)
        for idx, card in enumerate(root_hand):
            if not card.is_fully_determined():
                new_card = self.deck.draw(rank=card.rank, color=card.color)
                assert new_card.rank_known == card.rank_known
                assert new_card.color_known == card.color_known
                root_hand[idx] = new_card
        self.deck.assert_no_reserved_cards()
        self.last_turn_played = dict.fromkeys(self.hands.keys(), False)
        self.assert_consistency()

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
        if len(self.deck) > 0:
            self.deck.assert_no_reserved_cards()
            self.hands[player].append(self.deck.draw())
        if self.board[card.color] == card.rank - 1:
            self.board[card.color] += 1
            if card.rank == 5 and self.hints > 0:
                self.hints -= 1
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
        # if self.hints == 0:
        #     raise RuntimeError("No used hint tokens")
        card = self.hands[player].pop(card_idx)
        self.trash.append(card)
        if len(self.deck) > 0:
            self.deck.assert_no_reserved_cards()
            self.hands[player].append(self.deck.draw())
        self.hints = max(self.hints - 1, 0)

    def give_hint(self, destination: str, hint_type: str, hint_value: int) -> None:
        """
        This works asssuming that all the cards in all the players' hands have a defined rank and color
        (either known or not)
        """
        # if self.hints == MAX_HINTS:
        #     raise RuntimeError("Maximum number of hints already reached")
        hand = self.hands[destination]
        for card in hand:
            if hint_type == "value" and card.rank == hint_value:
                card.reveal_rank()
            elif hint_type == "color" and card.color == hint_value:
                card.reveal_color()
        self.hints = min(self.hints + 1, MAX_HINTS)

    # MCTS
    def available_hints(self) -> int:
        """
        Returns the count of available hints.
        """
        return MAX_HINTS - self.hints

    # MCTS
    def used_hints(self) -> int:
        """
        Returns the count of used hints
        """
        return self.hints

    def redeterminize_hand(self, player_name: str) -> None:
        """
        Redeterminizes player_name's hand before exiting the node associated with their move.

        Args:
            player_name: the name of the player
        """
        hand = self.hands[player_name]
        if player_name == self.root_player:
            raise RuntimeError("Cannot re-determinize root player's hand")

        self.deck.add_cards(hand, ignore_fd=True)
        success = False
        while not success:
            success = True
            self.deck.reserve_semi_determined_cards(hand)

            new_hand = []
            for idx, card in enumerate(hand):
                if card.is_fully_determined():
                    new_hand.append(card)
                else:
                    rank = card.rank if card.rank_known else None
                    color = card.color if card.color_known else None
                    new_card = self.deck.draw(rank=rank, color=color)
                    if new_card is None:
                        self.deck.reset_reservations()
                        self.deck.add_cards(new_hand, ignore_fd=True)
                        success = False
                        break
                    assert new_card.rank_known == card.rank_known
                    assert new_card.color_known == card.color_known
                    new_hand.append(new_card)

            self.deck.assert_no_reserved_cards()
            self.hands[player_name] = new_hand

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
        if len(self.hands[player_name]) > len(saved_hand):
            self.deck.assert_no_reserved_cards()
            saved_hand.append(self.deck.draw())
            assert len(self.hands[player_name]) == len(saved_hand)  # at most 1 card
        self.hands[player_name] = saved_hand

    # MCTS
    def _remove_illegal_cards(self, cards: List[Card]) -> None:
        """
        Remove the illegal cards from the list (considering all the cards in the trash,
        in the player's hands and on the table)

        Args:
            cards: the list of cards to modify
        """

        for idx in range(len(cards)):
            card = cards[idx]
            if self.deck[card.rank, card.color] > 0:
                self.deck.remove_cards([card])
            else:
                cards[idx] = self.deck.draw()

    def assert_consistency(self) -> None:
        """
        Utility function, asserts that all its knowledge is consistent (i.e. the cards from the trash + the cards from
        the board + the cards from the hands + the cards remaining in the deck should always be equal to the cards from
        a full deck)s
        """
        col = np.array(CARD_QUANTITIES)
        col = col.reshape(col.size, 1)
        full_table = np.tile(col, len(Color))
        table = np.copy(self.deck[slice(5), slice(5)])

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

    def game_ended(self) -> Tuple[bool, Optional[float]]:
        """
        Checks if the game is ended for some reason. If it's ended, it returns True and the score of the game.
        If the game isn't ended, it returns False, None
        """
        if self.errors == MAX_ERRORS:
            return True, sum(self.board) * SCORE_3_ERRORS
        # if self.board == self.trash.maxima:
        if np.all(self.board == 5):
            return True, sum(self.board)
        if all(self.last_turn_played.values()):
            return True, sum(self.board)
        return False, None
