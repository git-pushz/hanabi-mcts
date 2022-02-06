import copy
from typing import Tuple, List
import time
from model import Model, GameMove
from game_state import GameState, MCTSState
from tree import Tree, Node, GameNode
from functools import reduce
import numpy as np
import random
from hyperparameters import MCTS_SIMULATIONS

DEBUG = False


def find(pred, iterable):
    """
    Utility function.
    """
    for element in iterable:
        if pred(element):
            return element
    return None


class MCTS:
    """
    Wrapper class for the Monte Carlo Tree Search.

    Attributes:
        game_state: the GameState object corresponding to the current state of the "actual" game
        tree: the tree structure used for the search
    """
    def __init__(self, game_state: GameState, current_player: str) -> None:
        self.game_state = game_state
        prev_player = game_state.get_prev_player_name(current_player)
        root = Node(
            GameNode(GameMove(prev_player, action_type=None))
        )  # dummy game-move
        self.tree = Tree(root)

    def run_search(self, time_budget: int = None, iterations: int = None) -> GameMove:
        """
        Wrapper to the call of each iteration of the MCTS.

        Args:
            time_budget: the maximum amount of time for a set of iterations
            iterations: the maximum number of iterations
        """
        if (iterations is None) and (time_budget is None):
            raise RuntimeError(
                "At least one between iterations and time_budget must be specified"
            )

        # each iteration represents the select, expand, simulate, backpropagate iteration

        if time_budget is not None and iterations is not None:
            elapsed_time = 0
            start_time = time.time()
            n_iterations = 0
            while elapsed_time < time_budget or n_iterations < iterations:
                self._run_search_iteration()
                elapsed_time = time.time() - start_time
                n_iterations += 1
        elif time_budget is not None:
            elapsed_time = 0
            start_time = time.time()
            while elapsed_time < time_budget:
                self._run_search_iteration()
                elapsed_time = time.time() - start_time
        else:
            for _ in range(iterations):
                self._run_search_iteration()

        children = self.tree.get_children(self.tree.get_root())
        # selecting from the direct children of the root the one containing the move with most number of simulations
        best_move_node = reduce(
            lambda a, b: a if a.data.simulations > b.data.simulations else b,
            children,
        )
        return best_move_node.data.move

    def _run_search_iteration(self) -> None:
        """
        Performs a single iteration of the run_search.
        """
        select_leaf, select_model = self._select(Model(MCTSState(self.game_state)))

        # print('selected node ', select_leaf)
        expand_leaf, expand_model = self._expand(select_leaf, select_model)

        ## added
        simulation_score = 0
        for _ in range(MCTS_SIMULATIONS):
            simulation_score += self._simulate(expand_leaf, copy.deepcopy(expand_model))
        simulation_score /= MCTS_SIMULATIONS
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

    def _select(self, model: Model) -> Tuple[Node, Model]:
        """
        Performs the select phase of the MCTS.

        Args:
            model: the class Model object
        """
        node = self.tree.get_root()
        # model.state.redeterminize_hand(model.state.root_player)
        next_player = model.state.get_next_player_name(node.data.move.player)
        while not node.is_leaf() and self._is_fully_explored(node, model):
            node = self._get_best_child_UCB1(node)
            # make the move that bring us to "node"
            model.make_move(node.data.move, update_saved_hand=True)
            assert next_player == node.data.move.player
            model.restore_hand(node.data.move.player)  # restore hand
            next_player = model.state.get_next_player_name(node.data.move.player)
            model.redeterminize_hand(next_player)  # re-determinize hand
        return node, model

    def _is_fully_explored(self, node: Node, model: Model) -> bool:
        """
        return True if there is no more moves playable at a certain level that has not been tried yet
        """
        return len(self._get_available_plays(node, model)) == 0

    def _get_available_plays(self, node: Node, model: Model) -> List[GameMove]:
        """
        Returns the list of feasible moves from a certain node

        Args:
            node: the current node
            model: the object of class model
        """
        children = self.tree.get_children(node)
        player = model.state.get_next_player_name(node.data.move.player)
        # return only valid moves which haven't been already tried in children
        return list(
            filter(
                lambda move: not find(lambda child: child.data.move == move, children),
                model.valid_moves(player),
            )
        )

    def _expand(self, node: Node, model: Model) -> Tuple[Node, Model]:
        """
        Performs the expand phase of the MCTS.

        Args:
            node: the frontier node returned bu the _select
            model: the object of class model
        """
        expanded_node = None

        # model.check_win should check if the match is over, not if it is won (see simulation and backpropagation function)
        if not model.check_ended()[0]:
            legal_moves = self._get_available_plays(node, model)
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
        """
        Performs the simulate phase of the MCTS.

        Args:
            node: the node returned from the expand phase
            model: the object of class model
        """
        current_player = node.data.move.player

        # here random moves are made until someone wins, then the winning player is passed to backpropagation function
        # the problem is that in hanabi there is no winner (and probably moves can't be random)
        # so this function need some changes (at the end it needs to return the score)
        n_iter = 0
        while not model.check_ended()[0]:
            n_iter += 1
            current_player = model.state.get_next_player_name(current_player)
            # if there are no more legal moves (=> draw)
            if not model.make_random_move(current_player):
                break
        score = model.check_ended()[1]
        assert score is not None

        return score

    # def backpropagate(self, node, winner: int):
    def _backpropagate(self, node: Node, score: int) -> None:
        """
        Performs the backpropagate phase of the MCTS.

        Args:
            node: the 'youngest' node of the explored tree (the one returned from the expand phase)
            score: the score of the simulated game
        """
        # as the simulation function, this one needs to be changed
        # here nodes value is incremented if it leads to a winning game for the agent
        # but in our case need to be evaluated in proportion to the score
        # just to give and idea I implemented a simple version
        while not node.is_root():
            node.data.simulations += 1
            # it maps the score to [0, 1]
            node.data.value += score / 25
            node = self.tree.get_parent(node)
        node.data.simulations += 1

    def _UCB1(self, node: Node, parent: Node, c: float = 0.1) -> float:
        """
        Calculates the Upper Confidence Bound for the MCTS.

        Args:
            node: the node for which it calculates the UCB
            parent: the parent node of `node`
            c: the coefficient of the formula
        """
        exploitation = node.data.value / node.data.simulations
        if parent.data.simulations == 0:
            exploration = 0
        else:
            exploration = np.sqrt(
                np.log(parent.data.simulations) / node.data.simulations
            )
        return exploitation + c * exploration

    def _get_best_child_UCB1(self, node: Node) -> Node:
        """
        Returns the best child of node, based on the UCB calculations.

        Args:
             node: the node whose children are being evaluated
        """
        node_scores = map(
            lambda f: [f, self._UCB1(f, node)], self.tree.get_children(node)
        )
        return reduce(lambda a, b: a if a[1] > b[1] else b, list(node_scores))[0]
