from typing import List, Callable
import copy
from game_state import MCTSState
from model import GameMove
from utils import Card, Color, CARD_QUANTITIES
import numpy as np


def _all_equal(iterator):
    iterator = iter(iterator)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return all(first == x for x in iterator)


def _is_playable(card: Card, board: np.typing.NDArray) -> bool:
    return board[card.color] == card.rank - 1


def _is_discardable(card: Card, board: np.typing.NDArray) -> bool:
    return board[card.color] >= card.rank


def _get_probabilities(
    hand: List[Card],
    mental_state,
    fn_condition: Callable[[Card, np.typing.NDArray], bool],
    board: np.typing.NDArray,
) -> List[float]:
    probabilities = np.empty(len(hand), dtype=np.float)

    for idx, card in enumerate(hand):
        possibilities = np.copy(mental_state._table)
        if card.rank_known:
            mask = np.full(len(CARD_QUANTITIES), False)
            mask[card.rank - 1] = True
            possibilities[mask, :] = 0
        if card.color_known:
            mask = np.full(len(Color), False)
            mask[card.color] = True
            possibilities[:, mask] = 0

        number_of_determinizations = np.sum(possibilities)
        number_of_playable = 0
        for r, c in zip(*np.nonzero(possibilities)):
            if fn_condition(Card(r + 1, c), board):
                number_of_playable += possibilities[r][c]
        probabilities[idx] = number_of_playable / number_of_determinizations

    return probabilities


# RULE 1
def tell_most_information(state: MCTSState, player: str) -> GameMove:
    best_move = None
    best_affected = -1
    new_information = True
    players = list(state.hands.keys())
    player_idx = players.index(player)
    for i in range(1, len(players)):
        # iterate over all players starting from next player
        p = players[(player_idx + i) % len(players)]
        if p == player:
            continue

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


def tell_anyone_about_useful(state: MCTSState, player: str) -> GameMove:
    action_type = "hint"
    while True:
        destination = state.get_next_player_name(player)
        if destination == player:
            return None
        for idx, card in enumerate(state.hands[destination]):
            if _is_playable(card, state.board) and not card.is_fully_determined():
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


def tell_dispensable(state: MCTSState, player: str) -> GameMove:
    action_type = "hint"
    while True:
        destination = state.get_next_player_name(player)
        if destination == player:
            return None
        for idx, card in enumerate(state.hands[destination]):
            if _is_discardable(card, state.board) and not card.is_fully_determined():
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


def complete_tell_playable_card(state: MCTSState, player: str) -> GameMove:
    action_type = "hint"
    move = None
    while True:
        destination = state.get_next_player_name(player)
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
            elif not card.color_known and card.rank_known:
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


def complete_tell_dispensable_card(state: MCTSState, player: str) -> GameMove:
    action_type = "hint"
    move = None
    while True:
        destination = state.get_next_player_name(player)
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
            elif card.color_known and not card.rank_known:
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


def complete_tell_currently_not_playable_card(
    state: MCTSState, player: str
) -> GameMove:
    action_type = "hint"
    move = None
    while True:
        destination = state.get_next_player_name(player)
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
            elif card.color_known and not card.rank_known:
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
def play_probably_safe(
    state: MCTSState, player: str, threshold: float = 0.7
) -> GameMove:
    action_type = "play"
    hand = state.hands[player]
    mental_state = copy.deepcopy(state.deck)
    mental_state.add_cards(hand, ignore_fd=True)

    probabilities = _get_probabilities(hand, mental_state._table, _is_playable)

    if np.max(probabilities) >= threshold:
        best_idx = np.argmax(probabilities)
        return GameMove(player, action_type, card_idx=best_idx)
    else:
        return None


# RULE 9
def discard_probably_useless(
    state: MCTSState, player: str, threshold: float
) -> GameMove:
    action_type = "discard"
    hand = state.hands[player]
    mental_state = copy.deepcopy(state.deck)
    mental_state.add_cards(hand, ignore_fd=True)

    probabilities = _get_probabilities(hand, mental_state._table, _is_discardable)

    if np.max(probabilities) >= threshold:
        best_idx = np.argmax(probabilities)
    else:
        best_idx = 0

    return GameMove(player, action_type, card_idx=best_idx)
