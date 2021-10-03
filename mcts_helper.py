from montecarlo.montecarlo import MonteCarlo
from montecarlo.node import Node
from rand import rand_move

def child_finder(node, mcts):
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

def node_evaluator(node, mcts):
    winner = node.state.winner()
    if winner is not None:
        return 1 if winner == mcts.root_node.player_number else -1
    elif node.depth >= mcts.max_depth:
        return 0

def setup_mcts(mapstate, p1, p2):
    mcts = MonteCarlo(Node(mapstate))
    mcts.root_node.depth = 0
    mcts.root_node.player_number = p1
    mcts.root_node.opponent_number = p2
    mcts.root_node.unapplied_moves = None
    mcts.node_evaluator = node_evaluator
    mcts.child_finder = child_finder
    mcts.max_depth = 25
    return mcts

