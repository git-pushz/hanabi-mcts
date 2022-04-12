from typing import List, Any
import random
import numpy as np
from constants import SEED
from game_state import GameState
from utils import Card, Color, color_enum2str, color_str2enum
from mcts import MCTS
import GameData
from hyperparameters import MCTS_ITERATIONS, MCTS_TIME_BUDGET

DEBUG = False
VERBOSE = True
LOG = False


class Agent:
    def __init__(
        self, name: str, data: GameData.ServerGameStateData, players_names: list
    ) -> None:
        self.name = name
        self._game_state = GameState(players_names, name, data)
        self.turn = 0
        self.hand_size = 5 if len(players_names) < 4 else 4
        if SEED is not None:
            np.random.seed(SEED)
            random.seed(SEED)

    def make_move(self) -> GameData.ClientToServerData:
        """
        Runs the MCTS and returns the GameData.ClientToServerData object corresponding to the action chosen.
        """
        self.turn += 1
        mcts = MCTS(self._game_state, self.name)
        move = mcts.run_search(time_budget=MCTS_TIME_BUDGET, iterations=MCTS_ITERATIONS)
        if move.action_type == "hint":
            hint_value = (
                move.hint_value
                if move.hint_type == "value"
                else color_enum2str[move.hint_value]
            )
            return GameData.ClientHintData(
                self.name, move.destination, move.hint_type, hint_value
            )
        elif move.action_type == "play":
            return GameData.ClientPlayerPlayCardRequest(self.name, move.card_idx)
        elif move.action_type == "discard":
            return GameData.ClientPlayerDiscardCardRequest(self.name, move.card_idx)
        else:
            raise RuntimeError(f"Unknown action type received: {move.action_type}")

    def discover_own_card(self, card, card_idx: int) -> None:
        """
        Called whenever the agent plays or discards a card, this function update the deck knowledge if the discovered card is NOT fully determined.

        Args:
            card: the played/discarded card
            card_idx: the index of card in agent's hand
        """
        self._game_state.root_card_discovered(
            card_idx, card.value, color_str2enum[card.color]
        )

    def track_discarded_card(self, player: str, card_idx: int) -> None:
        """
        Calls the GameState function to keep track of a discarded card
        """
        self._game_state.card_discarded(player, card_idx)

    def track_played_card(self, player: str, card_idx: int, correctly: bool) -> None:
        self._game_state.card_played(player, card_idx, correctly)

    def track_drawn_card(self, players: list) -> None:
        """
        Track a card drawn by another player form the deck
        """
        different_hands = 0
        new_card = None
        player = None
        for p in players:
            if p.name != self.name:
                if len(p.hand) != len(self._game_state.hands[p.name]):
                    different_hands += 1
                    # NB: newly drawn cards are appended to the right
                    new_card = p.hand[-1]
                    player = p.name
        assert new_card is not None, "new card not found"
        assert different_hands == 1, "too many different cards"
        assert player != self.name, "Cannot discover my cards"
        self._game_state.card_drawn(player, Card.from_server(new_card))

    def draw_card(self) -> None:
        """
        Draw a card from the deck. This will append a new unknwon
        card (rank = None, color = None) to the agent's hand
        """
        self._game_state.card_drawn(self.name, Card(None, None))

    def track_hint(
        self, destination: str, cards_idx: List[int], hint_type: str, hint_value: int
    ) -> None:
        """
        Update the agent's knowledge based on the hint that was just given
        """
        value = hint_value if hint_type == "value" else color_str2enum[hint_value]
        self._game_state.hint_given(destination, cards_idx, hint_type, value)

    def assert_aligned_with_server(
        self,
        hints_used: int,
        mistakes_made: int,
        board: Any,
        trash: list,
        players: list,
    ) -> None:
        """
        FOR DEBUG ONLY: assert that every internal structure is consistent with the server knowledge
        Args:
            hints_used: the number of hint tokens used in the actual game
            mistakes_made: the number of mistakes made in the actual game
            board: the actual current board game
            trash: the list of actually discarded cards
            players: the list of objects of class Player - they also store their hands
        """
        assert self._game_state.hints == hints_used, "wrong count of hints"
        assert self._game_state.errors == mistakes_made, "wrong count of errors"

        # Board
        b = np.array([len(cards) for cards in board.values()])
        assert np.all(
            self._game_state.board == b
        ), f"wrong board: self.board: {self._game_state.board}, board: {b}"

        # Trash
        client_trash = self._game_state.trash.list
        assert len(client_trash) == len(trash)
        for idx in range(len(trash)):
            assert client_trash[idx] == Card.from_server(
                trash[idx]
            ), f"Mismatch between cards in trash at idx {idx}"

        # Hands
        for player in players:
            if player.name != self.name:
                client_hand = self._game_state.hands[player.name]
                assert len(player.hand) == len(client_hand)
                for idx in range(len(player.hand)):
                    assert client_hand[idx] == Card.from_server(
                        player.hand[idx]
                    ), f"player {player.name} wrong card in hand at idx {idx}"

    def known_status(self) -> str:
        """
        Utility function, returns a formatted string with all the knowledge up to this point.
        """
        s = f"At turn {self.turn} my knowledge is\n"
        s += "Colors: "
        for color in Color:
            s += color_enum2str[color] + " "
        s += "\n"
        s += f"Board: {self._game_state.board}: score = {np.sum(self._game_state.board)}\n"
        s += "Hands:"
        for p, h in self._game_state.hands.items():
            s += f"\tPlayer {p}: {h}\n"
        s += f"\nTrash: {self._game_state.trash}\n"
        s += f"Used hints: {self._game_state.hints}\n"
        s += f"Errors made: {self._game_state.errors}\n"
        s += f"Deck table: {str(self._game_state.deck)}\n"
        return s
