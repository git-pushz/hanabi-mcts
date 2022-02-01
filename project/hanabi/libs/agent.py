import numpy as np
from constants import SEED
from game_state import GameState, Card, color_enum2str, color_str2enum
from mcts import MCTS
import GameData

DEBUG = False
VERBOSE = True
LOG = False


class Agent:
    def __init__(
        self, name: str, data: GameData.ServerGameStateData, players_names: list
    ):
        self.name = name
        self._game_state = GameState(players_names, name, data)
        self.turn = 0
        np.random.seed(SEED)

    def make_move(self) -> GameData.ClientToServerData:
        """ """
        self.turn += 1
        mcts = MCTS(self._game_state, self.name)
        move = mcts.run_search(50)
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

    def track_drawn_card(self, players: list) -> None:
        """ """
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
        # TODO: for the agent, a card with (None, None) should be appended?
        self._game_state.append_card_to_player_hand(
            player, Card(new_card.value, color_str2enum[new_card.color])
        )

    def draw_unknown_card(self) -> None:
        self._game_state.append_card_to_player_hand(self.name, Card(None, None))

    def track_played_card(self, player: str, card_idx: int) -> None:
        """ """
        self._game_state.remove_card_from_hand(player, card_idx)

    def update_trash(self, card) -> None:
        """ """
        self._game_state.update_trash(
            Card(rank=card.value, color=color_str2enum[card.color])
        )

    def hint_gained(self) -> None:
        """ """
        self._game_state.gain_hint()

    def hint_consumed(self) -> None:
        """ """
        self._game_state.use_hint()

    def mistake_made(self) -> None:
        """ """
        self._game_state.mistake_made()

    def discover_card(self, card, card_idx: int, action_type: str = None) -> None:
        """
        Called whenever the agent plays or discards a card, this function update the deck knowledge if the discovered card is NOT fully determined.

        Args:
            card: the played/discarded card
            card_idx: the index of card in agent's hand
            action_type: it's one of ['play', 'mistake', 'discard'] FOR DEBUG ONLY
        """
        self._game_state.discover_card_root(
            card.value, color_str2enum[card.color], card_idx
        )

    def update_board(self, card) -> None:
        """ """
        self._game_state.card_correctly_played(color_str2enum[card.color])

    def update_knowledge_on_hint(
        self, hint_type: str, hint_value: int, cards_idx: list[int], destination: str
    ) -> None:
        """ """
        value = hint_value if hint_type == "value" else color_str2enum[hint_value]
        self._game_state.give_hint(cards_idx, destination, hint_type, value)

    def assert_aligned_with_server(
        self,
        hints_used: int,
        mistakes_made: int,
        board: list,
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
        # assert self._game_state.hints == hints_used, "wrong count of hints"
        assert self._game_state.errors == mistakes_made, "wrong count of errors"

        # TODO: remove
        self._game_state.hints = hints_used

        # Board
        b = [len(cards) for cards in board.values()]
        assert (
            self._game_state.board == b
        ), f"wrong board: self.board: {self._game_state.board}, board: {b}"

        # Trash
        client_trash = self._game_state.trash.list
        assert len(client_trash) == len(trash)
        for idx in range(len(trash)):
            assert (
                trash[idx].value == client_trash[idx].rank
            ), f"Mismatch between cards value in trash (idx={idx})"
            assert (
                color_str2enum[trash[idx].color] == client_trash[idx].color
            ), f"Mismatch between cards color in trash (idx={idx})"

        # Hands
        for player in players:
            if player.name != self.name:
                client_hand = self._game_state.hands[player.name]
                assert len(player.hand) == len(client_hand)
                for idx in range(len(player.hand)):
                    assert (
                        client_hand[idx] == player.hand[idx]
                    ), f"player {player.name} wrong card in hand at idx {idx}"
