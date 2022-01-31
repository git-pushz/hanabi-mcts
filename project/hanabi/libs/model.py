import numpy as np
import copy
from game_state import GameState, MCTSState
from constants import SEED

NUM_COLUMNS = 7
COLUMN_HEIGHT = 6
WINDOW_LENGTH = 4
FOUR = 4

PLAYER1 = 1
PLAYER2 = -1


class GameMoveOld:
    """
    Attribute:
        player: the (index of) player that made the move
        sender: string, name of the sender
        destination: string, name of the destination player
        action_type: can be "color" or "value"
        hint_value: can be the color or the value of the card (depending on "action_type")
        hand_card_ordered: the card in hand that has been played/ discarded

    action_type = "play" uses action_type, sender, hand_card_ordered
    action_type = "discard" uses action_type, sender, hand_card_ordered
    action_type = "hint" uses action_type, sender, destination, hint_value
    """

    def __init__(
        self,
        player: int,
        action_type: str,
        sender: str,
        destination: str = None,
        hand_card_ordered: int = None,
        hint_value: int = None,
    ) -> None:
        self.player = player
        self.action_type = action_type
        self.sender = sender
        self.destination = destination
        self.hand_card_ordered = hand_card_ordered
        self.hint_value = hint_value


class GameMove:
    """
    A move in the game

    Attributes:
        player: the name of the player that made the move
        action_type: a string among ["play", "discard", "hint"]
        card_idx: (non-hint) the index in the hand of the played/discarded card
        destination: (hint-only) string, name of the destination player
        hint_type: (hint-only) a string among ["value", "color"]
        hint_value: (hint-only) can be the color or the value of the card (depending on "action_type")
    """

    def __init__(
        self,
        player: str,
        action_type: str,
        card_idx: int = None,
        destination: str = None,
        hint_type: str = None,
        hint_value: int = None,
    ) -> None:
        self.player = player
        self.action_type = action_type
        self.card_idx = card_idx
        self.destination = destination
        self.hint_type = hint_type
        self.hint_value = hint_value

    def __eq__(self, other):
        equal = self.player == other.player and self.action_type == other.action_type
        if equal:
            if self.action_type == "hint":
                equal = (
                    equal
                    and self.destination == other.destination
                    and self.hint_type == other.hint_type
                    and self.hint_value == other.hint_value
                )
            else:
                equal = equal and self.card_idx == other.card_idx
        return equal

    def __ne__(self, other):
        return not self.__eq__(other)

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.player = self.player
        result.action_type = self.action_type
        result.card_idx = self.card_idx
        result.destination = self.destination
        result.hint_type = self.hint_type
        result.hint_value = self.hint_value
        return result


class Model:
    def __init__(self, mcts_state: MCTSState) -> None:
        np.random.seed(SEED)
        self.state = mcts_state
        self._saved_hand = None

    def __deepcopy__(self, memo={}):
        cls = self.__class__
        result = cls.__new__(cls)
        result.state = copy.deepcopy(self.state)
        result._saved_hand = copy.deepcopy(self._saved_hand)
        return result

    def enter_node(self, player: str) -> None:
        """
        Save the player's hand and re-determinize it

        Args:
            player: the name of the player
        """
        if player != self.state.root_player:
            # TODO deepcopy
            self._saved_hand = copy.copy(self.state.hands[player])
            self.state.redeterminize_hand(player)

    def exit_node(self, player: str) -> None:
        """
        Restore the player's hand with the previous saved one

        Args:
            player: the name of the player
        """
        if player != self.state.root_player and self._saved_hand is not None:
            self.state.restore_hand(player, self._saved_hand)
            self._saved_hand = None

    def valid_moves(self, this_player: str) -> list[GameMove]:
        """
        Returns all possible moves available at the current state
        (that correspond to a certain tree level
        that corresponds to a certain player)

        Args:
            this_player: the name of the playing player
        """
        moves = []

        hand = self.state.hands[this_player]
        for idx, card in enumerate(hand):
            actions = ["play"]
            if self.state.hints > 0:
                actions.append("discard")
            for action in actions:
                moves.append(GameMove(this_player, action, card_idx=idx))

        if self.state.hints_available() > 0:
            action_type = "hint"
            for player in self.state.players:
                if player == this_player:
                    continue
                hand = self.state.hands[player]
                for rank in range(1, 6):
                    if any(card.rank == rank for card in hand):
                        moves.append(
                            GameMove(
                                this_player,
                                action_type,
                                destination=player,
                                hint_type="value",
                                hint_value=rank,
                            )
                        )
                for color in range(5):
                    if any(card.color == color for card in hand):
                        moves.append(
                            GameMove(
                                this_player,
                                action_type,
                                destination=player,
                                hint_type="color",
                                hint_value=color,
                            )
                        )

        return moves

    def make_move(self, move: GameMove) -> None:
        """
        Makes a move and updates the game state accordingly

        Args:
            move: the move to perform
        """
        if move.action_type == "play":
            self.state.play_card(move.player, move.card_idx)
        elif move.action_type == "discard":
            self.state.discard_card(move.player, move.card_idx)
        elif move.action_type == "hint":
            self.state.give_hint(move.destination, move.hint_type, move.hint_value)
        else:
            raise RuntimeError(f"Unknown action type: {move.action_type}")

    # the name should be changed to something like make_intentional_move, because it shouldn't be random
    def make_random_move(self, player: str) -> bool:
        """
        Makes a random move for the simulation phase

        Args:
            player: the current player
        """
        legal_moves = self.valid_moves(player)
        if len(legal_moves) == 0:
            return False
        random_move = np.random.choice(legal_moves)
        self.make_move(random_move)
        return True

    def check_ended(self) -> tuple[bool, int]:
        """
        Returns True and the score of the game, if the game is ended. Returns False, None otherwise.
        """
        return self.state.game_ended()
