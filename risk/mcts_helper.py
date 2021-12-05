from montecarlo.montecarlo import MonteCarlo
from montecarlo.node import Node
from .rand import rand_move
from .data_loader import *
from time import time
import math

class MCTS(MonteCarlo):
    def __init__(self, mapstate, p1, p2, model=None, iters=100, max_depth=25, trust_policy=1.0, moves_to_consider=20, timeout=math.inf, exploration=0.35):
        self.max_depth = max_depth
        self.player = p1
        self.opponent = p2
        self.model = model
        self.iters = iters
        self.trust_policy = trust_policy
        self.moves_to_consider = moves_to_consider
        self.timeout = timeout
        self.exploration = exploration
        if mapstate is not None: self.setMapState(mapstate)

    def simulate(self, iters=1, timeout=math.inf):
        finish_time = timeout + time()
        for _ in range(iters):
            if time() >= finish_time:
                break
            super().simulate()

    def get_move(self):
        return self.make_choice().move

    def setMapState(self, mapstate):
        self.root_node = Node(mapstate)
        self.root_node.depth = 0
        self.root_node.player_number = self.player
        self.root_node.opponent_number = self.opponent
        self.root_node.unapplied_moves = None
        self.root_node.discovery_factor = self.exploration

    def child_finder(self, node, _):
        if node.state.winner() is not None:
            return
        if node.parent is None or node.parent.expanded:
            n = self.moves_to_consider
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
            child.discovery_factor = self.exploration
            node.add_child(child)

        if self.model and self.model.predict_policy() and not self.model.batched():
            v, pi = self.model(*node.state.to_tensor(player, opponent), [child.move for child in node.children])
            for prior, child in zip(pi.exp().tolist(), node.children):
                child.update_policy_value(self.trust_policy * prior * len(node.children) + 1 - self.trust_policy)
            node.update_win_value(
                v.tolist() if player == self.player else -v.tolist()
            )

    def node_evaluator(self, node, _):
        winner = node.state.winner()
        if winner is not None:
            return 1 if winner == self.player else -1
        elif self.model and not self.model.predict_policy() and not self.model.batched():
            value1 =  self.model(*node.state.to_tensor(self.player, self.opponent)).tolist()
            value2 = -self.model(*node.state.to_tensor(self.opponent, self.player)).tolist()
            assert -1 <= value1 <= 1
            assert -1 <= value2 <= 1
            return 0.5 * (value1 + value2)
        elif node.depth >= self.max_depth:
            return 0

    def expand(self, node):
        if self.model is None or not self.model.batched():
            return super().expand(node)
        if not node.children:
            self.child_finder(node, self)
        node.expanded = True
        children = []
        for child in node.children:
            win_value = self.node_evaluator(child, self)
            if win_value is not None:
                child.update_win_value(win_value)
            else:
                self.child_finder(child, self)
                children.append(child)

        if not children:
            return
        data = DataLoader([self.prep(child) for child in children], batch_size=100)
        assert data

        vs, pis = [], []
        for batch in data:
            v, pi = self.model(batch)
            vs.append(v)
            pis.append(pi)
        assert vs
        assert pis
        vs = torch.cat(vs, dim=0)
        pis = torch.cat(pis, dim=0)

        for v, pi, child in zip(vs, pis, children):
            player = child.player_number
            opponent = child.opponent_number
            for prior, grandchild in zip(pi.exp().tolist(), child.children):
                grandchild.update_policy_value(self.trust_policy * prior * len(child.children) + 1 - self.trust_policy)
            child.update_win_value(
                v.item() if player == self.player else -v.item()
            )

    def prep(self, node):
        x1, x2, edges = node.state.to_tensor(self.player, self.opponent)
        edges = torch_geometric.utils.to_undirected(edges)
        assert torch_geometric.utils.is_undirected(edges)

        orders = [child.move for child in node.children]
        order_data = build_order_data(orders, node.state, x1)
        return StateData(
            num_nodes=len(node.state.mapstruct),
            num_moves=len(orders),
            graph_data=x1,
            global_data=x2,
            edge_index=edges,
            **order_data,
        )

    def play(self, mapstate):
        assert mapstate.winner() is None
        self.setMapState(mapstate)
        start = time()
        self.simulate(self.iters, timeout=self.timeout)
        self.elapsed = time() - start
        assert self.root_node.children
        return self.get_move()

