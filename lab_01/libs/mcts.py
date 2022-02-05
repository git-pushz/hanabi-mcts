from libs.model import PLAYER1, PLAYER2, Model, GameMove
from .tree import Tree, Node
from functools import reduce
import numpy as np

def find(pred, iterable):
    for element in iterable:
        if pred(element):
            return element
    return None

class MCTS:
    def __init__(self, model, player=PLAYER1):
        self.model = model
        root = Node(GameNode(GameMove(player*-1, None)))
        self.tree = Tree(root)

    def run_search(self, iterations=50):
        for _ in range(iterations):
            self.run_search_iteration()
        
        best_move_node = reduce(lambda a, b: a if a.data.simulations > b.data.simulations else b, self.tree.get_children(self.tree.get_root()))
        return best_move_node.data.move.position
    
    def run_search_iteration(self):
        select_res = self.select(copy.copy(self.model))
        select_leaf = select_res[0]
        select_model = select_res[1]

        # print('selected node ', select_leaf)
        expand_res = self.expand(select_leaf, select_model)
        expand_leaf = expand_res[0]
        expand_model = expand_res[1]

        simulation_winner = self.simulate(expand_leaf, expand_model)

        self.backpropagate(expand_leaf, simulation_winner)

        # print('children list of ', self.tree.get_root(), ' simulations ', self.tree.get_root().data.simulations)
        # for child in self.tree.get_children(self.tree.get_root()):
        #     print(child)
        #     print('simulations ', child.data.simulations)
        #     print('value ', child.data.value)
        #     print('UCB1 ', self.UCB1(child, self.tree.get_root()))
        #     print('position', child.data.move.position)
        #     print('player', child.data.move.player)
        #     print('---------------------------------------------------------------------------------')
        # input("Enter...")
        return

    def select(self, model):
        node = self.tree.get_root()
        while(not node.is_leaf() and self.is_fully_explored(node, model)):
            node = self.get_best_child_UCB1(node)
            model.make_move(node.data.move)
        
        return node, model
    
    def is_fully_explored(self, node, model):
        return len(self.get_available_plays(node, model)) == 0
    
    def get_available_plays(self, node, model):
        children = self.tree.get_children(node)
        # return only valid moves which haven't been already tried in children
        return list(filter(lambda col: not find(lambda child: child.data.move.position == col, children), model.valid_moves()))
        
    def expand(self, node, model):
        expanded_node = None

        if (not model.check_win()):
            legal_positions = self.get_available_plays(node, model)
            random_pos=np.random.choice(legal_positions)
            other_player = node.data.move.player*-1
            random_move = GameMove(other_player, random_pos)
            model.make_move(random_move)

            expanded_node = Node(GameNode(random_move))
            self.tree.insert(expanded_node, node)
        else:
            expanded_node = node
            # print('winning node')
            # print(np.rot90(model.board.copy()))
            # print('++++++++++++++++++++++++++++++++++++++')
        # print('expanding..')
        # print(np.rot90(model.board.copy()))
        return expanded_node, model
    
    def simulate(self, node, model):
        current_player = node.data.move.player

        while(not model.check_win()):
            current_player = current_player*-1
            # if there are no more legal moves (=> draw)
            if (not model.make_random_move(current_player)):
                break
        winner = model.check_win()
    
        return winner

    def backpropagate(self, node, winner):
        while (not node.is_root()):
            node.data.simulations += 1
            if ((node.data.move.player == 1 and winner == 1) or (node.data.move.player == -1 and winner == -1)):
                node.data.value += 1
            if ((node.data.move.player == 1 and winner == -1) or (node.data.move.player == -1 and winner == 1)):
                node.data.value -= 1
            node = self.tree.get_parent(node)
            # print('parent node ', node)
            # print('is ', node.data.move.position, ' root? ', node.is_root())
        node.data.simulations += 1
        return

    def UCB1(self, node, parent):
        exploitation = node.data.value / node.data.simulations
        if (parent.data.simulations == 0):
            exploration = 0
        else:
            exploration = np.sqrt(2 * np.log(parent.data.simulations) / node.data.simulations)
        return exploitation + exploration
    
    def get_best_child_UCB1(self, node):
        node_scores = map(lambda f: [f, self.UCB1(f, node)], self.tree.get_children(node))
        return reduce(lambda a, b: a if a[1] > b[1] else b, list(node_scores))[0]


class GameNode:
    def __init__(self, move):
        self.move = move
        self.value = 0
        self.simulations = 0

    def copy(self):
        new_game_node = GameNode(None if self.move == None else self.move.copy())
        new_game_node.value = self.value
        new_game_node.simulations = self.simulations
        return new_game_node
