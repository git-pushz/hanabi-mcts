from game_state import MCTSState
from model import GameMove
from utils import Color


# RULE 1
def tell_most_information(state: MCTSState, player: str) -> GameMove:
    pass


def tell_anyone_about_useful(state: MCTSState, player: str) -> GameMove:
    best_move = None
    best_affected = -1
    new_information = True
    players = list(state.hands.keys())
    player_idx = players.index(player)
    for i in range(1, len(players)):
        # iterate over all players starting from next player
        p = players[(player_idx+i) % len(players)]
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
                new_option = GameMove(player, 'hint', destination=p, hint_type="value", hint_value=r)
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
                new_option = GameMove(player, 'hint', destination=p, hint_type="color", hint_value=c)
                # TODO: CONVENTIONS?
                best_affected = total_affected
                best_move = new_option

    return best_move
