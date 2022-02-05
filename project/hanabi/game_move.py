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
