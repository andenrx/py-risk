from montecarlo.montecarlo import MonteCarlo
from montecarlo.node import Node
from rand import rand_move
from time import time

class MCTS(MonteCarlo):
    def __init__(self, mapstate, p1, p2, model=None, iters=100):
        self.max_depth = 25
        self.player = p1
        self.opponent = p2
        self.model = model
        self.iters = iters
        if mapstate is not None: self.setMapState(mapstate)

    def get_move(self):
        return self.make_choice().move

    def setMapState(self, mapstate):
        self.root_node = Node(mapstate)
        self.root_node.depth = 0
        self.root_node.player_number = self.player
        self.root_node.opponent_number = self.opponent
        self.root_node.unapplied_moves = None

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
            assert -1 <= value1 <= 1
            assert -1 <= value2 <= 1
            return 0.5 * (value1 + value2)
        elif node.depth >= self.max_depth:
            return 0

    def play(self, mapstate):
        assert mapstate.winner() is None
        self.setMapState(mapstate)
        start = time()
        self.simulate(self.iters)
        self.elapsed = time() - start
        assert self.root_node.children
        return self.get_move()

