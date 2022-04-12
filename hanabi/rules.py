from typing import List, Callable, Optional
import copy
from game_state import MCTSState
from game_move import GameMove
from utils import Card, Color, CARD_QUANTITIES, Deck, Trash
import numpy as np
from hyperparameters import (
    PLAY_SAFE_PROBABILITY,
    PLAY_SAFE_LATE_PROBABILITY,
    DISCARD_PROBABILITY,
    RULE_9_MIN_HINTS,
    RULE_9_BEST_IDX_0,
    EXPEND_PROBABILITY,
    RULE_8_DECK_LENGTH,
)


class Rules:
    """
    Wrapper static class for the 10 'smart' rules.
    """

    _state: MCTSState = None
    _player: str = None
    _mental_state: Deck = None

    @staticmethod
    def get_rules_moves(state: MCTSState, player: str) -> List[GameMove]:
        """
        The only method exposed. Returns a list of 'smart' moves based on the rules coded in this class.

        Args:
             state: the current game state
             player: the player of the current node
        """

        Rules._state = state
        Rules._player = player
        Rules._mental_state = copy.deepcopy(state.deck)
        Rules._mental_state.add_cards(state.hands[player], ignore_fd=False)

        moves = []
        # RULE 1
        moves.append(Rules._tell_most_information())
        # RULE 2
        moves.append(Rules._tell_anyone(Rules._is_playable))
        # RULE 3
        moves.append(Rules._tell_anyone(Rules._is_discardable))
        # RULE 3b
        moves.append(Rules._tell_anyone(Rules._is_risky))
        # RULE 4
        moves.append(Rules._complete_tell_anyone(Rules._is_playable))
        # RULE 5
        moves.append(Rules._complete_tell_anyone(Rules._is_discardable))
        # RULE 6
        moves.append(Rules._complete_tell_anyone(Rules._is_unplayable))
        # RULE 7
        # # probability p ∈ [0.4, 0.8]
        # highest = 0.8
        # lowest = 0.4
        # # p = highest + len(deck)*(lowest-highest)/50  NB: 50 is the max length of deck
        # p = highest + len(state.deck)*(lowest-highest/50)
        moves.append(Rules._play_probably_safe(PLAY_SAFE_PROBABILITY))
        # RULE 8
        moves.append(Rules._play_probably_safe_late(PLAY_SAFE_LATE_PROBABILITY))
        # RULE 9
        moves.append(Rules._discard_probably_useless(DISCARD_PROBABILITY))
        # RULE 10
        # moves.append(Rules._discard_least_likely_to_be_necessary(EXPEND_PROBABILITY))
        return [m for m in moves if m is not None]

    @staticmethod
    def _all_equal(iterator) -> bool:
        iterator = iter(iterator)
        try:
            first = next(iterator)
        except StopIteration:
            return True
        return all(first == x for x in iterator)

    @staticmethod
    def _is_playable(card: Card, board: np.ndarray, trash: Trash) -> bool:
        """
        Determines whether the card is currently playable or not.

        Args:
            card: the card being evaluated
            board: the game board of played cards
            trash: unused argument, needed for the signature of the method
        """
        return board[card.color] == card.rank - 1

    @staticmethod
    def _is_discardable(card: Card, board: np.ndarray, trash: Trash) -> bool:
        """
        Determines whether the card is currently discardable or not.

        Args:
            card: the card being evaluated
            board: the game board of played cards
            trash: the trash of the game
        """
        return board[card.color] >= card.rank or trash.maxima[card.color] < card.rank

    @staticmethod
    def _is_unplayable(card: Card, board: np.ndarray, trash: Trash) -> bool:
        """
        Determines whether the card is currently unplayable or not.

        Args:
            card: the card being evaluated
            board: the game board of played cards
            trash: the trash of the game
        """
        return not (
            Rules._is_playable(card, board, trash)
            or Rules._is_discardable(card, board, trash)
        )

    @staticmethod
    def _is_expendable(card: Card, board: np.ndarray, trash: Trash) -> bool:
        """
        Determines whether the card is currently expendable (it isn't the last of its kind) or not.

        Args:
            card: the card being evaluated
            board: unused argument, needed for the signature of the method
            trash: the trash of the game
        """
        return trash[int(card.rank), card.color] > 1

    @staticmethod
    def _is_risky(card: Card, board: np.ndarray, trash: Trash) -> bool:
        """
        Determines whether the card is currently risky (it is the last of its kind) or not.

        Args:
            card: the card being evaluated
            board: unused argument, needed for the signature of the method
            trash: the trash of the game
        """
        return trash[int(card.rank), card.color] == 1

    @staticmethod
    def _get_probabilities(
        hand: List[Card],
        fn_condition: Callable[[Card, np.ndarray, Trash], bool],
        board: np.ndarray,
        trash: Trash,
    ) -> np.ndarray:
        """
        Returns the probability for each card of hand of matching the fn_condition

        Args:
            hand: the hand being currently evaluated
            fn_condition: the condition being tested
            board: the board of the game
            trash: the trash of the game
        """
        probabilities = np.empty(len(hand), dtype=np.float)

        for idx, card in enumerate(hand):
            possibilities = np.copy(Rules._mental_state[:, :])
            if card.rank_known:
                mask = np.full(len(CARD_QUANTITIES), True)
                mask[card.rank - 1] = False
                possibilities[mask, :] = 0
            if card.color_known:
                mask = np.full(len(Color), True)
                mask[card.color] = False
                possibilities[:, mask] = 0

            number_of_determinizations = np.sum(possibilities)
            assert number_of_determinizations > 0
            matching_count = 0
            for r, c in zip(*np.nonzero(possibilities)):
                if fn_condition(Card(r + 1, c), board, trash):
                    matching_count += possibilities[r][c]
            probabilities[idx] = matching_count / number_of_determinizations

        return probabilities

    # RULE 1
    @staticmethod
    def _tell_most_information() -> Optional[GameMove]:
        """
        Rule 1. It tries to give the hint that tells the most information.
        """
        if Rules._state.available_hints() == 0:
            return None

        action_type = "hint"

        best_move = None
        best_affected = -1
        new_information = True

        destination = Rules._player
        while True:
            destination = Rules._state.get_next_player_name(destination)
            if destination == Rules._player:
                break

            hand = Rules._state.hands[destination]

            for rank in range(1, 1 + len(CARD_QUANTITIES)):
                total_affected = 0
                for card in hand:
                    if not new_information or not card.rank_known:
                        if card.rank == rank:
                            total_affected += 1

                if total_affected > best_affected:
                    new_option = GameMove(
                        Rules._player,
                        action_type=action_type,
                        destination=destination,
                        hint_type="value",
                        hint_value=rank,
                    )
                    # TODO: CONVENTIONS?
                    best_affected = total_affected
                    best_move = new_option

            for color in range(len(Color)):
                total_affected = 0
                for card in hand:
                    if not new_information or not card.color_known:
                        if card.color == color:
                            total_affected += 1

                if total_affected > best_affected:
                    new_option = GameMove(
                        Rules._player,
                        action_type=action_type,
                        destination=destination,
                        hint_type="color",
                        hint_value=color,
                    )
                    # TODO: CONVENTIONS?
                    best_affected = total_affected
                    best_move = new_option

        assert best_move is not None
        return best_move

    # RULES 2 and 3
    @staticmethod
    def _tell_anyone(
        fn_condition: Callable[[Card, np.ndarray, Trash], bool],
    ) -> Optional[GameMove]:
        """
        Rules 2 and 3. Tries to hint about a playable or discardable card.

        Args:
            fn_condition: the condition being tested
        """
        if Rules._state.available_hints() == 0:
            return None

        action_type = "hint"

        destination = Rules._player
        while True:
            destination = Rules._state.get_next_player_name(destination)
            if destination == Rules._player:
                return None
            for idx, card in enumerate(Rules._state.hands[destination]):
                if (
                    fn_condition(card, Rules._state.board, Rules._state.trash)
                    and not card.is_fully_determined()
                ):
                    if not card.rank_known:
                        hint_type = "value"
                        hint_value = card.rank
                    else:
                        hint_type = "color"
                        hint_value = card.color
                    return GameMove(
                        Rules._player,
                        action_type,
                        destination=destination,
                        hint_type=hint_type,
                        hint_value=hint_value,
                    )

    # RULES 4, 5 and 6
    @staticmethod
    def _complete_tell_anyone(
        fn_condition: Callable[[Card, np.ndarray, Trash], bool],
    ) -> Optional[GameMove]:
        """
        Rules 4, 5 and 6. Tries to hint another player about an information that completes their knowledge on some card

        Args:
            fn_condition: the condition being tested
        """
        if Rules._state.available_hints() == 0:
            return None

        action_type = "hint"

        destination = Rules._player
        while True:
            destination = Rules._state.get_next_player_name(destination)
            if destination == Rules._player:
                return None
            for card in Rules._state.hands[destination]:
                if card.is_semi_determined() and fn_condition(
                    card, Rules._state.board, Rules._state.trash
                ):
                    if not card.rank_known:
                        hint_type = "value"
                        hint_value = card.rank
                    else:  # not card.color_known
                        hint_type = "color"
                        hint_value = card.color
                    return GameMove(
                        Rules._player,
                        action_type,
                        destination=destination,
                        hint_type=hint_type,
                        hint_value=hint_value,
                    )

    @staticmethod
    def _tell_risky_card() -> Optional[GameMove]:
        pass

    # RULE 7
    @staticmethod
    def _play_probably_safe(threshold: float = 0.7) -> Optional[GameMove]:
        """
        Rule 7. Tries to play the most probably safe card.

        Args:
            threshold: the threshold of probability above which a card is considered safe to play.
        """
        action_type = "play"

        probabilities = Rules._get_probabilities(
            Rules._state.hands[Rules._player],
            Rules._is_playable,
            Rules._state.board,
            Rules._state.trash,
        )

        if np.max(probabilities) >= threshold:
            return GameMove(
                Rules._player, action_type, card_idx=np.argmax(probabilities)
            )
        else:
            return None

    # RULE 8
    @staticmethod
    def _play_probably_safe_late(threshold: float = 0.4) -> Optional[GameMove]:
        """
        Rule 8. Tries to play the most probably safe card in the last rounds of the game. It's the same as rule 7,
        but its threshold is way lower.

        Args:
            threshold: the threshold of probability above which a card is considered safe to play.
        """

        move = None
        if len(Rules._state.deck) <= RULE_8_DECK_LENGTH:
            move = Rules._play_probably_safe(threshold)
        return move

    # RULE 9
    @staticmethod
    def _discard_probably_useless(threshold: float) -> Optional[GameMove]:
        """
        Rule 9. Tries to discard the most probably useless card.

        Args:
            threshold: the threshold of probability above which a card is considered safe to discard.
        """
        if Rules._state.used_hints() == 0:
            return None

        action_type = "discard"
        hand = Rules._state.hands[Rules._player]

        # TODO: improve
        # Search for duplicate fully determined cards in hand
        # for idx, card in enumerate(hand):
        #     if card.is_fully_determined():
        #         for c in hand:
        #             if c.is_fully_determined() and c == card:
        #                 return GameMove(player, action_type, card_idx=idx)

        probabilities = Rules._get_probabilities(
            hand, Rules._is_discardable, Rules._state.board, Rules._state.trash
        )

        move = Rules._discard_least_likely_to_be_necessary(EXPEND_PROBABILITY)

        if np.max(probabilities) >= threshold:
            best_idx = np.argmax(probabilities)
        elif move is not None:
            return move
        elif Rules._state.used_hints() >= RULE_9_MIN_HINTS:
            # Choose the oldest card whose rank is unknown (or 0 if all the ranks are known)
            if RULE_9_BEST_IDX_0:
                best_idx = 0
            else:
                best_idx = next(
                    (idx for idx, card in enumerate(hand) if not card.rank_known), 0
                )
        else:
            # if only 1 or none used hints, prefer a hint over a discard
            return None

        return GameMove(Rules._player, action_type, card_idx=best_idx)

    # RULE 10
    @staticmethod
    def _discard_least_likely_to_be_necessary(threshold: int) -> Optional[GameMove]:
        """
        Rule 10. Tries to discard the most probably expendable card.

        Args:
            threshold: the threshold of probability above which a card is considered safe to expend.
        """
        if Rules._state.used_hints() == 0:
            return None
        action_type = "discard"
        hand = Rules._state.hands[Rules._player]
        probabilities = Rules._get_probabilities(
            hand, Rules._is_expendable, Rules._state.board, Rules._state.trash
        )
        if np.max(probabilities) >= threshold:
            best_idx = np.argmax(probabilities)
            return GameMove(Rules._player, action_type, card_idx=best_idx)
        else:
            return None
