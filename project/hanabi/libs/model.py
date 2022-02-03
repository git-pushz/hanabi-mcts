import numpy as np
import copy
from typing import Tuple, List
from game_state import MCTSState
from game_move import GameMove
from rules import Rules


class Model:
    def __init__(self, mcts_state: MCTSState) -> None:
        self.state = mcts_state
        self._saved_hand = None
        self.state.assert_consistency()

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
        self.state.assert_consistency()

    def exit_node(self, player: str) -> None:
        """
        Restore the player's hand with the previous saved one

        Args:
            player: the name of the player
        """
        if player != self.state.root_player and self._saved_hand is not None:
            self.state.restore_hand(player, self._saved_hand)
            self._saved_hand = None
        self.state.assert_consistency()

    def valid_moves_old(self, this_player: str) -> List[GameMove]:
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
            actions = []
            # if card.rank_known or (card.color_known and self.state.board[card.color] == card.rank - 1):
            if card.rank_known:
                actions.append("play")
            if self.state.used_hints() > 0:
                actions.append("discard")
            for action in actions:
                moves.append(GameMove(this_player, action, card_idx=idx))

        if self.state.available_hints() > 0:
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

    def valid_moves(self, this_player: str) -> List[GameMove]:
        return Rules.get_rules_moves(self.state, this_player)

    def make_move(self, move: GameMove) -> None:
        """
        Makes a move and updates the game state accordingly

        Args:
            move: the move to perform
        """
        if self.state.last_turn_played[move.player]:
            raise RuntimeError(f"{move.player} already performed the last turn play")

        is_last_move = len(self.state.deck) == 0

        if move.action_type == "play":
            self.state.play_card(move.player, move.card_idx)
        elif move.action_type == "discard":
            assert self.state.used_hints() > 0
            self.state.discard_card(move.player, move.card_idx)
        elif move.action_type == "hint":
            assert self.state.available_hints() > 0
            self.state.give_hint(move.destination, move.hint_type, move.hint_value)
        else:
            raise RuntimeError(f"Unknown action type: {move.action_type}")

        assert not self.state.last_turn_played[move.player]

        if is_last_move:
            self.state.last_turn_played[move.player] = True

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

    def check_ended(self) -> Tuple[bool, int]:
        """
        Returns True and the score of the game, if the game is ended. Returns False, None otherwise.
        """
        return self.state.game_ended()
