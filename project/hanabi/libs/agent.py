from game_state import GameState, Card
from .. import GameData

class Agent:
    def __init__(self, name: str, data: GameData.ServerGameStateData, players_names: list):
        self.name = name
        self._game_state = GameState(player_names, name, data)

    def make_move(self) -> None:
        """
        """
        pass

    def track_drawn_card(self, players: list) -> None:
        """
        """
        different_hands = 0
        new_card = None
        player = None
        for p in players:
            if len(p.hand) != len(self.hands[p.name]):
                different_hands += 1
                # NB: newly drawn cards are appended to the right
                new_card = p.hand[-1]
                player = p.name
        assert new_card is not None, "new card not found"
        assert different_hands == 1, "too many different cards"

        assert player != self.name, "Cannot discover my cards"
        self._game_state.append_card_to_player_hand(player, Card(new_card.value, new_new_card.color))


    def track_played_card(self, player: str, card_idx: int) -> None:
        """
        """
        self._game_state.remove_card_from_hand(player, card_idx)

    def update_trash(self, card) -> None:
        """
        """
        self._game_state.update_trash(Card(card=card.value, color=card.color))

    def hint_gained(self) -> None:
        """
        """
        self._game_state.gain_hint()

    def hint_consumed(self) -> None:
        """
        """
        self._game_state.use_hint()

    def mistake_made(self) -> None:
        """
        """
        self._game_state.mistake_made()

    def discover_card(self, card, card_idx: int, action_type: str) -> None:
        """
        """
        pass

    def update_board(self, card) -> None:
        """
        """
        self._game_state.card_correctly_played(card.color)

    def assert_aligned_with_server(self) -> None:
        """
        """
        pass

    def update_knowledge_on_hint(self, hint_type: str, hint_value: int, cards_idx: list[int], destination: str) -> None:
        """
        """
        pass