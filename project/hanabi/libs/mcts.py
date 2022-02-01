import copy
from model import Model, GameMove
from game_state import GameState, MCTSState
from tree import Tree, Node, GameNode
from functools import reduce
import numpy as np
import random

DEBUG = False
SIMULATIONS_NUMBER = 50


def find(pred, iterable):
    for element in iterable:
        if pred(element):
            return element
    return None


class MCTS:
    def __init__(self, game_state: GameState, current_player: str) -> None:
        self.game_state = game_state
        prev_player = game_state.get_prev_player_name(current_player)
        root = Node(
            GameNode(GameMove(prev_player, action_type=None))
        )  # dummy game-move
        self.tree = Tree(root)

    def run_search(self, iterations: int = 50) -> GameMove:
        # each iteration represents the select, expand, simulate, backpropagate iteration
        for _ in range(iterations):
            self._run_search_iteration()

        # selecting from the direct children of the root the one containing the move with most number of simulations
        best_move_node = reduce(
            lambda a, b: a if a.data.simulations > b.data.simulations else b,
            self.tree.get_children(self.tree.get_root()),
        )
        return best_move_node.data.move

    def _run_search_iteration(self) -> None:
        select_leaf, select_model = self._select(Model(MCTSState(self.game_state)))

        # print('selected node ', select_leaf)
        expand_leaf, expand_model = self._expand(select_leaf, select_model)

        ## added
        simulation_score = 0
        for _ in range(SIMULATIONS_NUMBER):
            simulation_score += self._simulate(expand_leaf, copy.deepcopy(expand_model))
        simulation_score /= SIMULATIONS_NUMBER
        self._backpropagate(expand_leaf, simulation_score)
        if DEBUG:
            print(
                "children list of ",
                self.tree.get_root(),
                " simulations ",
                self.tree.get_root().data.simulations,
            )
            for child in self.tree.get_children(self.tree.get_root()):
                print(child)
                print("simulations ", child.data.simulations)
                print("value ", child.data.value)
                print("UCB1 ", self._UCB1(child, self.tree.get_root()))
                print("position", child.data.move)
                print("player", child.data.move.player)
                print(
                    "---------------------------------------------------------------------------------"
                )
            input("Enter...")

    def _select(self, model: Model) -> tuple[Node, Model]:
        node = self.tree.get_root()
        # model.state.redeterminize_hand(model.state.root_player)
        while not node.is_leaf() and self._is_fully_explored(node, model):
            node = self._get_best_child_UCB1(node)
            model.exit_node(node.data.move.player)  # restore hand
            model.make_move(node.data.move)
            model.enter_node(
                model.state.get_next_player_name(node.data.move.player)
            )  # re-determinize hand
        return node, model

    def _is_fully_explored(self, node: Node, model: Model) -> bool:
        """
        return True if there is no more moves playable at a certain level that has not been tried yet
        """
        return len(self._get_available_plays(node, model)) == 0

    def _get_available_plays(self, node: Node, model: Model) -> list[GameMove]:
        children = self.tree.get_children(node)
        player = model.state.get_next_player_name(node.data.move.player)
        # return only valid moves which haven't been already tried in children
        return list(
            filter(
                lambda move: not find(lambda child: child.data.move == move, children),
                model.valid_moves(player),
            )
        )

    def _expand(self, node: Node, model: Model) -> tuple[Node, Model]:
        expanded_node = None

        # model.check_win should check if the match is over, not if it is won (see simulation and backpropagation function)
        if not model.check_ended()[0]:
            legal_moves = self._get_available_plays(node, model)
            #####
            if len(legal_moves) == 0:
                player = model.state.get_next_player_name(node.data.move.player)
                hand = model.state.hands[player]
                for idx in range(len(hand)):
                    legal_moves.append(GameMove(player, "play", card_idx=idx))
            #####
            random_move = random.choice(legal_moves)
            model.make_move(random_move)
            expanded_node = Node(GameNode(random_move))
            self.tree.insert(expanded_node, node)
        else:
            expanded_node = node
            if DEBUG:
                print("winning node")
        if DEBUG:
            print("expanding..")
        return expanded_node, model

    def _simulate(self, node: Node, model: Model) -> int:
        current_player = node.data.move.player

        # here random moves are made until someone wins, then the winning player is passed to backpropagation function
        # the problem is that in hanabi there is no winner (and probably moves can't be random)
        # so this function need some changes (at the end it needs to return the score)

        while not model.check_ended()[0]:
            current_player = model.state.get_next_player_name(current_player)
            # if there are no more legal moves (=> draw)
            if not model.make_random_move(current_player):
                break
        score = model.check_ended()[1]
        assert score is not None

        return score

    # def backpropagate(self, node, winner: int):
    def _backpropagate(self, node: Node, score: int) -> None:
        # as the simulation function, this one needs to be changed
        # here nodes value is incremented if it leads to a winning game for the agent
        # but in our case need to be evaluated in proportion to the score
        # just to give and idea I implemented a simple version
        while not node.is_root():
            node.data.simulations += 1
            # it maps the score to [0, 1]
            node.data.value += score / 25
            node = self.tree.get_parent(node)
            # print('parent node ', node)
            # print('is ', node.data.move.position, ' root? ', node.is_root())
        node.data.simulations += 1

    def _UCB1(self, node: Node, parent: Node, c: float = 0.1) -> float:
        exploitation = node.data.value / node.data.simulations
        if parent.data.simulations == 0:
            exploration = 0
        else:
            exploration = np.sqrt(
                np.log(parent.data.simulations) / node.data.simulations
            )
        return exploitation + c * exploration

    def _get_best_child_UCB1(self, node: Node) -> Node:
        node_scores = map(
            lambda f: [f, self._UCB1(f, node)], self.tree.get_children(node)
        )
        return reduce(lambda a, b: a if a[1] > b[1] else b, list(node_scores))[0]
