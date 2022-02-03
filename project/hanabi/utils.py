import copy
import numpy as np
from enum import IntEnum
import random
from typing import List


class Color(IntEnum):
    RED = 0
    YELLOW = 1
    GREEN = 2
    BLUE = 3
    WHITE = 4


color_str2enum = {
    "red": Color.RED,
    "yellow": Color.YELLOW,
    "green": Color.GREEN,
    "blue": Color.BLUE,
    "white": Color.WHITE,
}

color_enum2str = {
    Color.RED: "red",
    Color.YELLOW: "yellow",
    Color.GREEN: "green",
    Color.BLUE: "blue",
    Color.WHITE: "white",
}

CARD_QUANTITIES = [3, 2, 2, 2, 1]


class Card:
    def __init__(self, rank: int, color: Color) -> None:
        self.rank = rank
        self.color = color
        self.rank_known = False
        self.color_known = False

    @classmethod
    def from_server(cls, server_card):
        "Initialize Client crad from server card"
        return cls(server_card.value, color_str2enum[server_card.color])

    def __eq__(self, other):
        if type(other) is not Card:
            raise TypeError(f"Cannot compare type card with {type(other)}")
        return self.rank == other.rank and self.color == other.color

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        rank = str(self.rank) if self.rank is not None else ""
        color = color_enum2str[self.color] if self.color is not None else ""
        return f"Card ({rank},{color})"

    def reveal_rank(self, rank=None) -> None:
        if rank is None:
            assert self.rank is not None
        else:
            assert self.rank is None or self.rank == rank
            self.rank = rank
        self.rank_known = True

    def reveal_color(self, color=None) -> None:
        if color is None:
            assert self.color is not None
        else:
            assert self.color is None or self.color == color
            self.color = color
        self.color_known = True

    def is_fully_determined(self) -> bool:
        return self.rank_known and self.color_known

    def is_semi_determined(self) -> bool:
        return self.rank_known != self.color_known


class Deck:
    def __init__(self) -> None:
        col = np.array(CARD_QUANTITIES)
        col = col.reshape(col.size, 1)
        self._table = np.tile(col, len(Color))
        self._reserved_ranks = np.zeros(len(CARD_QUANTITIES), dtype=np.int8)
        self._reserved_colors = np.zeros(len(Color), dtype=np.int8)

    def __deepcopy__(self, memo={}):
        cls = self.__class__
        result = cls.__new__(cls)
        result._table = np.copy(self._table)
        result._reserved_ranks = np.copy(self._reserved_ranks)
        result._reserved_colors = np.copy(self._reserved_colors)
        return result

    def __len__(self):
        """
        Return the number of cards still available in the deck
        """
        return np.sum(self._table)

    def __getitem__(self, item):
        if type(item) is tuple:
            if type(item[0]) is int:
                return self._table[item[0] - 1, item[1]]
            elif type(item[0]) is slice:
                return self._table[item[0], item[1]]
            else:
                raise IndexError
        else:
            raise IndexError

    def __str__(self):
        s = "\nr y g b w\n"
        for r in range(self._table.shape[0]):
            for v in self._table[r, :]:
                s += f"{v} "
            s += "\n"
        return s

    def _decrement(self, rank: int, color: Color) -> None:
        assert (
            self._table[rank - 1][color] > 0
        ), "trying to decrement zero value from Deck"
        self._table[rank - 1][color] -= 1

    def _increment(self, rank: int, color: Color) -> None:
        assert (
            self._table[rank - 1][color] < CARD_QUANTITIES[rank - 1]
        ), "trying to increment maximum value from Deck"
        self._table[rank - 1][color] += 1

    def remove_cards(self, cards: List[Card]) -> None:
        for card in cards:
            self._decrement(card.rank, card.color)

    def add_cards(self, cards: List[Card], ignore_fd: bool = False) -> None:
        for card in cards:
            if not (ignore_fd and card.is_fully_determined()):
                self._increment(card.rank, card.color)

    def reserve_semi_determined_cards(self, cards: List[Card]) -> None:
        assert np.all(
            self._reserved_colors == 0
        ), "Color reservation not reset correctly"
        assert np.all(self._reserved_ranks == 0), "Rank reservation not reset correctly"
        for card in cards:
            if not card.is_fully_determined():
                if card.rank_known:
                    self._reserved_ranks[card.rank - 1] += 1
                elif card.color_known:
                    self._reserved_colors[card.color] += 1

    def draw(self, rank: int = None, color: Color = None) -> Card:
        if rank is None and color is None:
            possibilities = [
                (r, c)
                for r in range(len(CARD_QUANTITIES))
                for c in range(len(Color))
                for _ in range(self._table[r][c])
            ]
            rank, color = random.choice(possibilities)
            rank += 1
        elif rank is not None:
            possibilities = [
                c for c in range(len(Color)) for _ in range(self._table[rank - 1][c])
            ]
            color = random.choice(possibilities)
            assert color is not None
        elif color is not None:
            possibilities = [
                r
                for r in range(len(CARD_QUANTITIES))
                for _ in range(self._table[r][color])
            ]
            rank = random.choice(possibilities) + 1
            assert rank is not None
        self._decrement(rank, color)
        return Card(rank, color)

    def draw2(self, rank: int = None, color: Color = None) -> Card:
        # OBS: if rank or color are not None, for sure we are redeterminizing

        # fully determined
        if rank is not None and color is not None:
            raise RuntimeError("Cannot specify both rank and color when drawing")

        table = np.copy(self._table)

        update_table = True
        iterations = 0
        max_iterations = 100

        while update_table:
            update_table = False

            if rank is None:
                row_sums = np.sum(table, axis=1)
                # if no rank is specified, do not pick any rank-reserved card
                r_idx = np.logical_and(row_sums <= self._reserved_ranks, row_sums != 0)
                table[r_idx, :] = 0
                update_table = np.any(r_idx)

            if color is None:
                # if no color is specified, do not pick any rank-reserved card
                col_sums = np.sum(table, axis=0)
                c_idx = np.logical_and(col_sums <= self._reserved_colors, col_sums != 0)
                table[:, c_idx] = 0
                update_table = update_table or np.any(c_idx)

            iterations += 1
            if iterations > max_iterations:
                print(f"Rank: {rank}")
                print(f"Color: {color}")
                print(table)
                print(f"r_idx: {r_idx}")
                print(f"c_idx: {c_idx}")
                raise RuntimeError("Stuck in draw2")

        # completely unknown
        if rank is None and color is None:
            possibilities = [
                coordinates
                for coordinates, occurrencies in np.ndenumerate(table)
                for _ in range(occurrencies)
            ]
            rank, color = random.choice(possibilities)
            rank += 1

        # known rank
        elif rank is not None:
            assert (
                self._reserved_ranks[rank - 1] > 0
            ), f"No card with rank{rank} was previously reserved"
            self._reserved_ranks[rank - 1] -= 1
            possibilities = [
                c for c in range(table.shape[1]) for _ in range(table[rank - 1][c])
            ]
            color = random.choice(possibilities)

        # known color
        elif color is not None:
            assert (
                self._reserved_colors[color] > 0
            ), f"No card with color {color} was previously reserved"
            self._reserved_colors[color] -= 1
            possibilities = [
                r for r in range(table.shape[0]) for _ in range(table[r][color])
            ]
            rank = random.choice(possibilities) + 1

        assert rank is not None and color is not None
        self._decrement(rank, color)
        return Card(rank, color)


class Trash:
    def __init__(self) -> None:
        self.list = []
        self.maxima = [5] * 5
        col = np.array(CARD_QUANTITIES)
        col = col.reshape(col.size, 1)
        self._table = np.tile(col, len(Color))

    def __deepcopy__(self, memo={}):
        cls = self.__class__
        result = cls.__new__(cls)
        result.list = copy.deepcopy(self.list)
        result.maxima = copy.copy(self.maxima)
        result._table = np.copy(self._table)
        return result

    def __repr__(self):
        return str(self.list)

    def _decrement(self, rank: int, color: Color) -> None:
        assert (
            self._table[rank - 1][color] > 0
        ), "trying to decrement zero value from Trash"
        self._table[rank - 1][color] -= 1
        if self._table[rank - 1][color] == 0:
            self.maxima[color] = min(rank - 1, self.maxima[color])

    def append(self, card: Card) -> None:
        self.list.append(card)
        self._decrement(card.rank, card.color)

    def get_table(self) -> np.ndarray:
        return self._table
