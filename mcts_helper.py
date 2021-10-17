from montecarlo.montecarlo import MonteCarlo
from montecarlo.node import Node
from rand import rand_move

class MCTS(MonteCarlo):
    def __init__(self, mapstate, p1, p2, model=None):
        self.root_node = Node(mapstate)
        self.root_node.depth = 0
        self.root_node.player_number = p1
        self.root_node.opponent_number = p2
        self.root_node.unapplied_moves = None
        self.max_depth = 25
        self.player = p1
        self.opponent = p2
        self.model = model

    def child_finder(self, node, _):
        if node.state.winner() is not None:
            return
        if node.parent is None or node.parent.expanded:
            n = 20
        else:
            n = 1
        player = node.player_number
        opponent = node.opponent_number
        for i in range(n):
            move = rand_move(node.state, player)
            if node.unapplied_moves is None:
                child = Node(node.state.copy())
                child.unapplied_moves = move
            else:
                move = move.combine(node.unapplied_moves)
                child = Node(move(node.state))
                child.unapplied_moves = None
            child.move = move
            child.player_number = opponent
            child.opponent_number = player
            child.depth = node.depth + 1
            node.add_child(child)

    def node_evaluator(self, node, _):
        winner = node.state.winner()
        if winner is not None:
            return 1 if winner == self.root_node.player_number else -1
        elif self.model is not None:
            value1 =  self.model(*node.state.to_tensor(self.player, self.opponent)).tolist()
            value2 = -self.model(*node.state.to_tensor(self.opponent, self.player)).tolist()
            return 0.5 * (value1 + value2)
        elif node.depth >= self.max_depth:
            return 0

