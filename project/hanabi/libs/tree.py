import copy
from typing import List
from model import GameMove


class GameNode:
    def __init__(self, move: GameMove) -> None:
        self.move = move
        self.value = 0
        self.simulations = 0

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        # TODO: copy(None)
        result.move = copy.copy(self.move)
        result.value = self.value
        result.simulations = self.simulations
        return result


class Node:
    def __init__(self, data: GameNode, id=-1, children_ids=[], parent_id=-1):
        self.data = data
        self.id = id
        self.children_ids = children_ids[:]
        self.parent_id = parent_id

    def is_leaf(self):
        return len(self.children_ids) == 0

    def is_root(self):
        return self.id == 0

    def copy(self):
        return Node(self.data, self.id, self.children_ids[:], self.parent_id)


class Tree:
    def __init__(self, root: Node):
        root.id = 0
        self.nodes = [root]

    def insert(self, node: Node, parent: Node):
        node.id = len(self.nodes)
        node.parent_id = parent.id
        self.nodes.append(node)
        self.nodes[node.parent_id].children_ids.append(node.id)

    def get_root(self):
        return self.nodes[0]

    def get_children(self, node: Node) -> List[Node]:
        if not node:
            return []
        arr = []
        for i in range(len(node.children_ids)):
            arr.append(self.nodes[node.children_ids[i]])
        return arr

    def get_siblings(self, node: Node):
        return self.get_children(self.get_parent(node))

    def get_parent(self, node: Node):
        return self.nodes[node.parent_id]
