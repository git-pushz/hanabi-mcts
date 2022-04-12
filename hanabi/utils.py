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
    def __init__(
        self,
        rank: int,
        color: Color,
        rank_known: bool = False,
        color_known: bool = False,
    ) -> None:
        self.rank = rank
        self.color = color
        self.rank_known = rank_known
        self.color_known = color_known

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
        if rank is not None:
            if self.rank is not None and self.rank != rank:
                raise RuntimeError(
                    "Cannot reveal card's rank: a different rank is already set"
                )
            self.rank = rank
        elif self.rank is None:
            raise RuntimeError("Cannot reveal card's rank: it's None")
        self.rank_known = True

    def reveal_color(self, color=None) -> None:
        if color is not None:
            if self.color is not None and self.color != color:
                raise RuntimeError(
                    "Cannot reveal card's color: a different color is already set"
                )
            self.color = color
        elif self.color is None:
            raise RuntimeError("Cannot reveal card's color: it's None")
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

    def assert_no_reserved_cards(self) -> None:
        assert not (
            np.any(self._reserved_ranks != 0) or np.any(self._reserved_colors != 0)
        )

    def reserve_semi_determined_cards(self, cards: List[Card]) -> None:
        self.assert_no_reserved_cards()
        for card in cards:
            if not card.is_fully_determined():
                if card.rank_known:
                    self._reserved_ranks[card.rank - 1] += 1
                elif card.color_known:
                    self._reserved_colors[card.color] += 1

    # def draw(self, rank: int = None, color: Color = None) -> Card:
    #     if rank is None and color is None:
    #         possibilities = [
    #             (r, c)
    #             for r in range(len(CARD_QUANTITIES))
    #             for c in range(len(Color))
    #             for _ in range(self._table[r][c])
    #         ]
    #         rank, color = random.choice(possibilities)
    #         rank += 1
    #     elif rank is not None:
    #         possibilities = [
    #             c for c in range(len(Color)) for _ in range(self._table[rank - 1][c])
    #         ]
    #         color = random.choice(possibilities)
    #         assert color is not None
    #     elif color is not None:
    #         possibilities = [
    #             r
    #             for r in range(len(CARD_QUANTITIES))
    #             for _ in range(self._table[r][color])
    #         ]
    #         rank = random.choice(possibilities) + 1
    #         assert rank is not None
    #     self._decrement(rank, color)
    #     return Card(rank, color)

    def draw(self, rank: int = None, color: Color = None) -> Card:
        rank_known = rank is not None
        color_known = color is not None

        # fully determined
        if rank_known and color_known:
            raise RuntimeError("Cannot specify both rank and color when drawing")

        table = np.copy(self._table)

        update_table = True
        iterations = 0
        max_iterations = 100

        while update_table:
            row_sums = np.sum(table, axis=1)
            r_idx = np.logical_and(row_sums <= self._reserved_ranks, row_sums != 0)
            if rank_known:
                r_idx[rank - 1] = False
            table[r_idx, :] = 0
            update_table = np.any(r_idx)

            col_sums = np.sum(table, axis=0)
            c_idx = np.logical_and(col_sums <= self._reserved_colors, col_sums != 0)
            if color_known:
                c_idx[color] = False
            table[:, c_idx] = 0
            update_table = update_table or np.any(c_idx)

            iterations += 1
            if iterations > max_iterations:
                raise RuntimeError("Stuck in draw2")

        # completely unknown
        if not rank_known and not color_known:
            possibilities = [
                coordinates
                for coordinates, occurrencies in np.ndenumerate(table)
                for _ in range(occurrencies)
            ]
            if len(possibilities) == 0:
                return None
            else:
                rank, color = random.choice(possibilities)
                rank += 1

        elif rank_known:
            if self._reserved_ranks[rank - 1] <= 0:
                raise RuntimeError(f"No card with rank {rank} was previously reserved")
            self._reserved_ranks[rank - 1] -= 1
            possibilities = [
                c for c in range(table.shape[1]) for _ in range(table[rank - 1][c])
            ]
            if len(possibilities) == 0:
                return None
            else:
                color = random.choice(possibilities)

        elif color_known:
            if self._reserved_colors[color] <= 0:
                raise RuntimeError(
                    f"No card with color {color} was previously reserved"
                )
            self._reserved_colors[color] -= 1
            possibilities = [
                r for r in range(table.shape[0]) for _ in range(table[r][color])
            ]
            if len(possibilities) == 0:
                return None
            else:
                rank = random.choice(possibilities) + 1

        assert rank is not None and color is not None
        self._decrement(rank, color)
        return Card(rank, color, rank_known=rank_known, color_known=color_known)

    def reset_reservations(self):
        self._reserved_ranks[:] = 0
        self._reserved_colors[:] = 0


class Trash:
    def __init__(self) -> None:
        self.list = []
        self.maxima = np.full(len(Color), 5, dtype=np.uint8)
        col = np.array(CARD_QUANTITIES)
        col = col.reshape(col.size, 1)
        self._table = np.tile(col, len(Color))

    def __deepcopy__(self, memo={}):
        cls = self.__class__
        result = cls.__new__(cls)
        result.list = copy.deepcopy(self.list)
        result.maxima = np.copy(self.maxima)
        result._table = np.copy(self._table)
        return result

    def __getitem__(self, item):
        if type(item) is tuple:
            if type(item[0]) is int:
                return self._table[item[0] - 1, item[1]]
            elif type(item[0]) is slice:
                return self._table[item[0], item[1]]
            else:
                raise IndexError(f"{item}, {type(item[0])}")
        else:
            raise IndexError

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
