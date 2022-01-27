import copy
import typing
import numpy as np
from .. import GameData


def determinize_root(agent, players):
    """
    Determinize the root's hand and adjust all the other players'
    mental states accordingly
    """
    redeterminize_hand(root_player)
    for p in players:
        player_ms = agent.knowledge.player_mental_state(p)  # maybe deepcopy?
        for ms in player_ms.ms_hand:
            for card in root_player.hand:
                ms[card.value][card.color] -= 1


def restore_root_hand(agent, players):  # only necessary if we don't make a deepcopy
    """
    Restore the root's hand as "unknown" and udjust all the other
    players' mental states accordingly
    """
    for p in players:
        player_ms = agent.knowledge.player_mental_state(p)
        for ms in player_ms.ms_hand:
            for card in root_player.hand:
                ms[card.value][card.color] += 1
    root_player.hand = []


def enter_node(player, root_player_name):
    saved_player_hand = None
    if player.name != root_player_name:
        saved_player_hand = copy.deepcopy(player.hand)
        redeterminize_hand(player)
    return saved_player_hand


def exit_node(player, saved_player_hand):
    if saved_player_hand is not None:
        player.hand = saved_player_hand
        remove_incompatible_cards(player.hand)
        determinize_empty_slots


def redeterminize_hand(player, player_ms):
    pass


##### OR #####

# Keep a single "mental-state-like" table for the cards still in the deck
# and apply masks "dinamically"

colors = ["red", "yellow", "green", "blue", "white"]


class Color(Enum):
    RED = 0
    YELLOW = 1
    GREEN = 2
    BLUE = 3
    WHITE = 4


color2enum = {
    "red": RED,
    "yellow": YELLOW,
    "green": GREEN,
    "blue": BLUE,
    "white": WHITE,
}

HAND_SIZE = 5
CARD_QUANTITIES = [3, 2, 2, 2, 1]
MAX_HINTS = 8
MAX_ERRORS = 3


### WARNING ###
# When a player will compute the rules the decide the next move, he will have to
# add his hand back into the deck before performing any inference.
# When the computation is done, he will have to remove the cards from the deck again


class Card:
    def __init__(self, rank: int, color: Color) -> None:
        # id ?
        self.rank = rank
        self.color = color
        self.rank_known = False
        self.color_known = False

    def reveal_rank(self):
        self.rank_known = True

    def reveal_color(self):
        self.color_known = True

    def is_fully_determined(self):
        return self.rank_known and self.color_known


class Deck:
    def __init__(self) -> None:
        col = np.array(CARD_QUANTITIES)
        col = col.reshape(col.size, 1)
        self._table = np.tile(col, len(colors))

    def _decrement(self, rank: int, color: Color) -> None:
        assert (self._table[rank - 1][color] > 0, "trying to decrement zero value from Deck")
        self._table[rank - 1][color] -= 1

    def _increment(self, rank: int, color: Color) -> None:
        assert (self._table[rank - 1][color] < CARD_QUANTITIES[rank - 1], "trying to increment maximum value from Deck")
        self._table[rank - 1][color] += 1

    def remove_cards(self, cards: list[Card]) -> None:
        for card in cards:
            self._decrement(card.rank, card.color)

    def add_cards(self, cards: list[Card]) -> None:
        for card in cards:
            self._increment(card.rank, card.color)

    def draw(self, rank: int = None, color: Color = None) -> Card:
        rows, columns = np.nonzero(self._table)
        if rank is None:
            rank = np.random.choice(rows)
        if color is None:
            color = np.random.choice(columns)
        self._decrement(rank, color)
        return Card(rank, color)

    def is_empty(self) -> bool:
        return not np.any(self._table != 0)


class Trash:
    def __init__(self) -> None:
        self.list = []
        self.maxima = [5] * 5
        col = np.array(CARD_QUANTITIES)
        col = col.reshape(col.size, 1)
        self._table = np.tile(col, len(colors))

    def _decrement(self, rank: int, color: Color) -> None:
        assert (self._table[rank][color] > 0, "trying to decrement zero value from Deck")
        self._table[rank - 1][color] -= 1
        if self._table[rank - 1][color] == 0:
            self.maxima[color] = min(rank - 1, self.maxima[color])

    def append(self, card: Card) -> None:
        self.list.append(card)
        self._decrement(card.rank, card.color)

        

class GameState:
    """
    Atttributes:
        players:            list of player names in turn order
        root_player_name:   name of the root player (agent)
        hands:              dictionary with player names as keys and hands (list of cards) as values
        boards:             successfully played cards (currently in the table)
        maximums:           the maximum values that can be reached by each firework
        trash:              list of discarded cards
        hints:              the number of used note tokens
        errors:             the number of used storm tokens
        deck:               the cards currently in the deck
    """

    def __init__(
        self,
        player_names: list[str],
        root_player_name: str,
        data: GameData.ServerGameStateData,
    ) -> None:
        """
        Create a new GameState

        Args:
            player_names: the list of the player names in turn order
            root_player_name: the name of the root player (agent)
            data: the server game state to use to initialize the client game state
        """
        global HAND_SIZE
        self.players = copy.deepcopy(player_names)
        if len(players_names) >= 4:
            HAND_SIZE = 4
        self.root_player_name = root_player_name
        self.hands = {
            player.name: GameState.generate_hand(player.hand) for player in data.players
        }
        self.hands[root_player_name] = []
        self.board = [0] * 5
        self.trash = Trash()
        self.hints = data.usedNoteTokens
        self.errors = data.usedStormTokens
        self.deck = Deck()
        for hand in self.hands.values():
            deck.remove_cards(hand)

    @staticmethod
    def generate_hand(server_hand: list) -> list[Card]:
        """
        Generate a client-hand (list of cards) given a server-hand

        Args:
            server_hand: list of server cards

        Returns:
            a list of client cards
        """
        hand = []
        for card in server_hand:
            hand.append(Card(rank=card.rank, color=card.color))
        return hand

    def give_hint(self, destination: str, hint_type: str, hint_value: int) -> None:
        hand = self.hands[destination]
        for card in hand:
            if hint_type == "rank" and card.rank == hint_value:
                card.reveal_rank()
            if hint_type == "color" and card.color == hint_value:
                card.reveal_color()
        self.hints += 1

    def play_card(self, player: str, card_idx: int) -> None:
        card = self.hands[player].pop(card_idx)
        self.hands[player].append(self.deck.draw())
        if (player == self.root_player_name and not card.is_fully_determined()):
            self.deck.remove_cards([card])
        if self.board[card.color] == card.rank - 1:
            self.board[card.color] += 1
            if card.rank == 5:
                self.hints = min(self.hints + 1, MAX_HINTS)
        else:
            self.trash.append(card)
            self.errors += 1
    
    def discard_card(self, player: str, card_idx: int) -> None:
        card = self.hands[player].pop(card_idx)
        self.hands[player].append(self.deck.draw())
        if (player == self.root_player_name and not card.is_fully_determined()):
            self.deck.remove_cards([card])
        self.trash.append(card)
        self.hints = max(self.hints - 1, 0)

    def game_ended(self) -> tuple[bool, int]:
        """
        Checks if the game is ended for some reason. If it's ended, it returns True and the score of the game.
        If the game isn't ended, it returns False, None
        """
        if self.errors == MAX_ERRORS: 
           return True, 0
        if self.board == self.trash.maxima:
            return True, sum(self.board)
        if self.deck.is_empty():
            return True, sum(self.board)
        return False, None

    def restore_hand(self, player_name: str, saved_hand: list[Card]) -> None:
        """
        Restore the specified hand for the specified player, removing all the "illegal" cards
        and re-determinizing their slots

        Args:
            player_name: the name of the player
            saved_hand: the hand to restore
        """
        self.deck.add_cards(self.hands[player_name])  # put cards back in deck
        self._remove_illegal_cards(saved_hand)  # remove inconsistencies
        self.deck.remove_cards(saved_hand)  # pick cards from deck
        self._determinize_empty_slots(saved_hand)
        self.hands[player_name] = saved_hand

    def _determinize_empty_slots(self, hand: list[Card]) -> None:
        """
        Determinize the empty slots of the hand (where card = None)

        Args:
            hand: the hand to adjust
        """
        for idx in range(len(hand)):
            if hand[idx] is None:
                hand[idx] = deck.draw()

    def _remove_illegal_cards(self, cards: list[Card]) -> None:
        """
        Remove the illegal cards from the list (considering all the cards in the trash,
        in the player's hands and on the table)

        Args:
            cards: the list of cards to modify
        """
        locations = self.trash.list
        for p in players:
            locations += p.hand

        for idx in range(len(cards)):
            card = cards[idx]
            if card is None:
                continue
            quantity = 1 + self.deck[card.rank][card.color]  # 1 is for "card" itself
            for c in locations:
                if c.rank == card.rank and c.color == card.color:
                    quantity += 1
            if board[card.color] >= card.rank:
                quantity += 1
            if quantity > CARD_QUANTITIES[card.rank - 1]:
                cards[idx] = None


### TODO
# * Gestire ultimo giro di giocate dopo che il mazzo e' finito