from typing import List, Callable, Optional
import copy
from game_state import MCTSState
from game_move import GameMove
from utils import Card, Color, CARD_QUANTITIES
import numpy as np


class Rules:
    """
    Wrapper static class for the 9 'smart' rules.
    """

    @staticmethod
    def get_rules_moves(state: MCTSState, player: str) -> List[GameMove]:
        """
        The only method exposed. Returns a list of 'smart' moves based on the rules coded in this class.

        Args:
             state: the current game state
             player: the player of the current node
        """
        moves = []
        # RULE 1
        moves.append(Rules._tell_most_information(state, player))
        # RULE 2
        moves.append(Rules._tell_anyone_about_useful(state, player))
        # RULE 3
        moves.append(Rules._tell_dispensable(state, player))
        # RULE 4
        moves.append(Rules._complete_tell_playable_card(state, player))
        # RULE 5
        moves.append(Rules._complete_tell_dispensable_card(state, player))
        # RULE 6
        moves.append(Rules._complete_tell_currently_not_playable_card(state, player))
        # RULE 7
        moves.append(Rules._play_probably_safe(state, player, 0.7))
        # RULE 8
        moves.append(Rules._play_probably_safe_late(state, player, 0.4))
        # RULE 9
        moves.append(Rules._discard_probably_useless(state, player, 0.7))
        return [m for m in moves if m is not None]

    @staticmethod
    def _all_equal(iterator):
        iterator = iter(iterator)
        try:
            first = next(iterator)
        except StopIteration:
            return True
        return all(first == x for x in iterator)

    @staticmethod
    def _is_playable(card: Card, board) -> bool:
        return board[card.color] == card.rank - 1

    @staticmethod
    def _is_discardable(card: Card, board) -> bool:
        return board[card.color] >= card.rank

    @staticmethod
    def _get_probabilities(
        hand: List[Card],
        mental_state,
        fn_condition,
        board,
    ):
        probabilities = np.empty(len(hand), dtype=np.float)

        for idx, card in enumerate(hand):
            possibilities = np.copy(mental_state[:, :])
            if card.rank_known:
                mask = np.full(len(CARD_QUANTITIES), False)
                mask[card.rank - 1] = True
                possibilities[mask, :] = 0
            if card.color_known:
                mask = np.full(len(Color), False)
                mask[card.color] = True
                possibilities[:, mask] = 0

            number_of_determinizations = np.sum(possibilities)
            assert number_of_determinizations > 0
            number_of_playable = 0
            for r, c in zip(*np.nonzero(possibilities)):
                if fn_condition(Card(r + 1, c), board):
                    number_of_playable += possibilities[r][c]
            probabilities[idx] = number_of_playable / number_of_determinizations

        return probabilities

    # RULE 1
    @staticmethod
    def _tell_most_information(state: MCTSState, player: str) -> GameMove:
        if state.hints_available() == 0:
            return None

        best_move = None
        best_affected = -1
        new_information = True

        p = player
        while True:
            p = state.get_next_player_name(p)
            if p == player:
                break

            hand = state.hands[p]

            for r in range(1, 6):
                total_affected = 0
                for card in hand:
                    # TODO: JAVA: hand.hasCard
                    if not new_information or not card.rank_known:
                        if card.rank == r:
                            total_affected += 1

                if total_affected > best_affected:
                    new_option = GameMove(
                        player, "hint", destination=p, hint_type="value", hint_value=r
                    )
                    # TODO: CONVENTIONS?
                    best_affected = total_affected
                    best_move = new_option

            for c in range(len(Color)):
                total_affected = 0
                for card in hand:
                    # TODO: JAVA: hand.hasCard
                    if not new_information or not card.color_known:
                        if card.color == c:
                            total_affected += 1

                if total_affected > best_affected:
                    new_option = GameMove(
                        player, "hint", destination=p, hint_type="color", hint_value=c
                    )
                    # TODO: CONVENTIONS?
                    best_affected = total_affected
                    best_move = new_option

        return best_move

    # RULE 2
    @staticmethod
    def _tell_anyone_about_useful(state: MCTSState, player: str) -> Optional[GameMove]:
        action_type = "hint"
        if state.hints_available() == 0:
            return None
        destination = player
        while True:
            destination = state.get_next_player_name(destination)
            if destination == player:
                return None
            for idx, card in enumerate(state.hands[destination]):
                if (
                    Rules._is_playable(card, state.board)
                    and not card.is_fully_determined()
                ):
                    if not card.rank_known:
                        hint_type = "value"
                        hint_value = card.rank
                    else:
                        hint_type = "color"
                        hint_value = card.color
                    return GameMove(
                        player,
                        action_type,
                        destination=destination,
                        hint_type=hint_type,
                        hint_value=hint_value,
                    )

    # RULE 3
    @staticmethod
    def _tell_dispensable(state: MCTSState, player: str) -> Optional[GameMove]:
        action_type = "hint"
        if state.hints_available() == 0:
            return None
        destination = player
        while True:
            destination = state.get_next_player_name(destination)
            if destination == player:
                return None
            for idx, card in enumerate(state.hands[destination]):
                if (
                    Rules._is_discardable(card, state.board)
                    and not card.is_fully_determined()
                ):
                    if not card.rank_known:
                        hint_type = "value"
                        hint_value = card.rank
                    else:
                        hint_type = "color"
                        hint_value = card.color
                    return GameMove(
                        player,
                        action_type,
                        destination=destination,
                        hint_type=hint_type,
                        hint_value=hint_value,
                    )

    # RULE 4
    @staticmethod
    def _complete_tell_playable_card(
        state: MCTSState, player: str
    ) -> Optional[GameMove]:
        action_type = "hint"
        if state.hints_available() == 0:
            return None
        move = None
        destination = player
        while True:
            destination = state.get_next_player_name(destination)
            if destination == player:
                break
            for card in state.hands[destination]:
                if card.is_fully_determined:
                    continue
                if card.rank != state.board[card.color] + 1:
                    continue
                if not card.color_known and not card.rank_known:
                    continue
                elif card.color_known and not card.rank_known:
                    hint_type = "value"
                    hint_value = card.rank
                # elif not card.color_known and card.rank_known:
                else:
                    hint_type = "color"
                    hint_value = card.color
                move = GameMove(
                    player,
                    action_type,
                    destination=destination,
                    hint_type=hint_type,
                    hint_value=hint_value,
                )
                return move
        return move

    # RULE 5
    @staticmethod
    def _complete_tell_dispensable_card(
        state: MCTSState, player: str
    ) -> Optional[GameMove]:
        action_type = "hint"
        if state.hints_available() == 0:
            return None
        move = None
        destination = player
        while True:
            destination = state.get_next_player_name(destination)
            if destination == player:
                break
            for _, card in enumerate(state.hands[destination]):
                if card.is_fully_determined():
                    continue
                if card.rank > state.board[card.color]:
                    continue
                if not card.color_known and not card.rank_known:
                    continue
                elif not card.color_known and card.rank_known:
                    hint_type = "color"
                    hint_value = card.color
                # elif card.color_known and not card.rank_known:
                else:
                    hint_type = "value"
                    hint_value = card.rank
                move = GameMove(
                    player,
                    action_type,
                    destination=destination,
                    hint_type=hint_type,
                    hint_value=hint_value,
                )
                return move
        return move

    # RULE 6
    @staticmethod
    def _complete_tell_currently_not_playable_card(
        state: MCTSState, player: str
    ) -> Optional[GameMove]:
        action_type = "hint"
        if state.hints_available() == 0:
            return None
        move = None
        destination = player
        while True:
            destination = state.get_next_player_name(destination)
            if destination == player:
                break
            for _, card in enumerate(state.hands[destination]):
                if card.is_fully_determined or card.rank_known:
                    continue
                if card.rank <= state.board[card.color] + 1:
                    continue
                if not card.color_known and not card.rank_known:
                    continue
                elif not card.color_known and card.rank_known:
                    hint_type = "color"
                    hint_value = card.color
                # elif card.color_known and not card.rank_known:
                else:
                    hint_type = "value"
                    hint_value = card.rank
                move = GameMove(
                    player,
                    action_type,
                    destination=destination,
                    hint_type=hint_type,
                    hint_value=hint_value,
                )
                return move
        return move

    # RULE 7
    @staticmethod
    def _play_probably_safe(
        state: MCTSState, player: str, threshold: float = 0.7
    ) -> Optional[GameMove]:
        action_type = "play"
        hand = state.hands[player]
        mental_state = copy.deepcopy(state.deck)
        mental_state.add_cards(hand, ignore_fd=True)

        probabilities = Rules._get_probabilities(
            hand, mental_state[:, :], Rules._is_playable, state.board
        )

        if np.max(probabilities) >= threshold:
            best_idx = np.argmax(probabilities)
            return GameMove(player, action_type, card_idx=best_idx)
        else:
            return None

    # RULE 8
    @staticmethod
    def _play_probably_safe_late(
        state: MCTSState, player: str, threshold: float = 0.4
    ) -> GameMove:
        move = None
        if len(state.deck) <= 5:
            move = Rules._play_probably_safe(state, player, threshold)
        return move

    # RULE 9
    @staticmethod
    def _discard_probably_useless(
        state: MCTSState, player: str, threshold: float
    ) -> GameMove:
        action_type = "discard"
        if state.hints == 0:
            return None
        hand = state.hands[player]
        mental_state = copy.deepcopy(state.deck)
        mental_state.add_cards(hand, ignore_fd=True)

        probabilities = Rules._get_probabilities(
            hand, mental_state[:, :], Rules._is_discardable, state.board
        )

        if np.max(probabilities) >= threshold:
            best_idx = np.argmax(probabilities)
        else:
            best_idx = 0

        return GameMove(player, action_type, card_idx=best_idx)
