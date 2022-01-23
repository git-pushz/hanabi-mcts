from math import fabs
from libs.model import PLAYER1, PLAYER2, Model, GameMove
from .tree import Tree, Node
from functools import reduce
import numpy as np
DEBUG = False

def find(pred, iterable):
    for element in iterable:
        if pred(element):
            return element
    return None

class GameNode:
    def __init__(self, move: GameMove):
        self.move = move
        self.value = 0
        self.simulations = 0

    def copy(self):
        new_game_node = GameNode(None if self.move == None else self.move.copy())
        new_game_node.value = self.value
        new_game_node.simulations = self.simulations
        return new_game_node

class MCTS:
    # players is the list of players in turn order (Player: name, ready, hand (list of cards in hand order))
    # def __init__(self, model, player=PLAYER1):
    def __init__(self, model: Model, players, current_player: int):
        # model contains all the game specific logic
        self.model = model
        # the root contain the move of the player that played just before the agent (the move itself is not important
        # and it's set to None)
        # the nodes at the first level of the tree will contain the moves of the agent

        ## removed
        # root = Node(GameNode(GameMove(player*-1, None)))
        ##
        ## added
        if current_player == 0:
            player_before_agent_index = len(players) - 1
        else:
            player_before_agent_index = current_player - 1
        ## 
        root = Node(GameNode(GameMove(players[player_before_agent_index], None)))
        self.tree = Tree(root)

    def run_search(self, iterations=50):
        # each iteration represents the select, expand, simulate, backpropagate iteration
        for _ in range(iterations):
            self.run_search_iteration()
        
        # selecting from the direct children of the root the one containing the move with most number of simulations
        best_move_node = reduce(lambda a, b: a if a.data.simulations > b.data.simulations else b, self.tree.get_children(self.tree.get_root()))
        return best_move_node.data.move.position
    
    def run_search_iteration(self):
        select_res = self.select(self.model.copy())
        select_leaf = select_res[0]
        select_model = select_res[1]

        # print('selected node ', select_leaf)
        expand_res = self.expand(select_leaf, select_model)
        expand_leaf = expand_res[0]
        expand_model = expand_res[1]

        ## removed
        # simulation_winner = self.simulate(expand_leaf, expand_model)
        # self.backpropagate(expand_leaf, simulation_winner)

        ## added
        simulation_score = self.simulate(expand_leaf, expand_model)
        self.backpropagate(expand_leaf, simulation_score)
        if DEBUG:
            print('children list of ', self.tree.get_root(), ' simulations ', self.tree.get_root().data.simulations)
            for child in self.tree.get_children(self.tree.get_root()):
                print(child)
                print('simulations ', child.data.simulations)
                print('value ', child.data.value)
                print('UCB1 ', self.UCB1(child, self.tree.get_root()))
                print('position', child.data.move.position)
                print('player', child.data.move.player)
                print('---------------------------------------------------------------------------------')
            input("Enter...")
        return

    def select(self, model: Model):
        node = self.tree.get_root()
        while(not node.is_leaf() and self.is_fully_explored(node, model)):
            node = self.get_best_child_UCB1(node)
            model.make_move(node.data.move)
        
        return node, model
    
    def is_fully_explored(self, node: Node, model: Model):
        '''
        return True if there is no more moves playable at a certain level that has not been tried yet
        '''
        # this function needs to be changed for the hanabi case
        return len(self.get_available_plays(node, model)) == 0
    
    def get_available_plays(self, node: Node, model):
        children = self.tree.get_children(node)
        # return only valid moves which haven't been already tried in children
        return list(filter(lambda col: not find(lambda child: child.data.move.position == col, children), model.valid_moves()))
        
    def expand(self, node: Node, model: Model):
        expanded_node = None

        # model.check_win should check if the match is over, not if it is won (see simulation and backpropagation function)
        if (not model.check_win()):
            legal_positions = self.get_available_plays(node, model)
            random_pos=np.random.choice(legal_positions)
            ## removed
            # other_player = node.data.move.player*-1
            ##
            ## added
            if node.data.move.player == len(model.agent.players):
                next_player = 0
            else:
                next_player = node.data.move.player + 1
            ##
            random_move = GameMove(next_player, random_pos)
            model.make_move(random_move)

            expanded_node = Node(GameNode(random_move))
            self.tree.insert(expanded_node, node)
        else:
            expanded_node = node
            if DEBUG:
                print('winning node')
                print(np.rot90(model.agent.board.copy()))
                print('++++++++++++++++++++++++++++++++++++++')
        if DEBUG:
            print('expanding..')
            print(np.rot90(model.agent.board.copy()))
        return expanded_node, model
    
    def simulate(self, node: Node, model: Model):
        current_player = node.data.move.player

        # here random moves are made until someone wins, then the winning player is passed to backpropagation function
        # the problem is that in hanabi there is no winner (and probably moves can't be random)
        # so this function need some changes (at the end it needs to return the score)

        # only one simulation has been run, probably it is better to run a bunch of simulations
        while(not model.check_win()):
            ## removed
            # current_player = current_player*-1
            ##
            ## added
            if node.data.move.player == len(model.agent.players):
                current_player = 0
            else:
                current_player = node.data.move.player + 1
            ##
            # if there are no more legal moves (=> draw)
            if (not model.make_random_move(current_player)):
                break
        winner = model.check_win()
    
        return winner

    # def backpropagate(self, node, winner: int):
    def backpropagate(self, node: Node, score: int):
        # as the simulation function, this one needs to be changed
        # here nodes value is incremented if it leads to a winning game for the agent
        # but in our case need to be evalueted in proportion to the score
        # just to give and idea I implemented a simple version
        while (not node.is_root()):
            node.data.simulations += 1
            ## removed
            # if ((node.data.move.player == 1 and winner == 1) or (node.data.move.player == -1 and winner == -1)):
            #     node.data.value += 1
            # if ((node.data.move.player == 1 and winner == -1) or (node.data.move.player == -1 and winner == 1)):
            #     node.data.value -= 1
            ## 
            # the problem is that in hanabi there is no winner

            ## added
            # it maps the score to [-1, 1]
            node.data.value += ((score / 99.0) * 2) - 1
            ## 
            node = self.tree.get_parent(node)
            # print('parent node ', node)
            # print('is ', node.data.move.position, ' root? ', node.is_root())
        node.data.simulations += 1
        return

    def UCB1(self, node: Node, parent: Node):
        exploitation = node.data.value / node.data.simulations
        if (parent.data.simulations == 0):
            exploration = 0
        else:
            exploration = np.sqrt(2 * np.log(parent.data.simulations) / node.data.simulations)
        return exploitation + exploration
    
    def get_best_child_UCB1(self, node: Node):
        node_scores = map(lambda f: [f, self.UCB1(f, node)], self.tree.get_children(node))
        return reduce(lambda a, b: a if a[1] > b[1] else b, list(node_scores))[0]


## TODO
# in mcts.py
# adapt function is_fully_explored (medium)
 # actually the problem is when it calls get_available_plays
# adapt function simulate (hard)
# adapt function backpropagate (easy)

# in model.py
# adapt valid_moves (medium)
# adapt make_move (medium)
# adapt make_random_move (that should be changed to make_intentional_move) (hard)
# adapt check_win (that should be changed to check_if_ended) (medium)
# 
