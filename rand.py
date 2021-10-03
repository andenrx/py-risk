import numpy as np
from orders import AttackTransferOrder, DeployOrder, OrderList

def fixed_round(n): return np.floor(n + 0.5)

def rand_partition(n: int, k: int) -> list:
    """Randomly partition n elements into k buckets (not uniformly)"""
    samples = np.random.randint(k, size=n)
    return np.eye(k, dtype=int)[samples].sum(axis=0)

def rand_deploy(state, player):
    deploy_targets = [
        terr for terr, owner in enumerate(state.owner)
        if owner == player and any(
            state.owner[dst] != player for dst in state.neighbors(terr)
        )
    ]
    assert state.winner() is None
    assert any(owner == player for owner in state.owner)
    assert len(deploy_targets) > 0
    
    part = rand_partition(state.income(player), len(deploy_targets))

    orders = list() # OrderList()
    for terr, n in zip(deploy_targets, part):
        if n > 0:
            orders.append(DeployOrder(player, terr, n))
    return orders

def rand_move(state, player):
    deploy_orders = rand_deploy(state, player)
    state = state.advance(deploy_orders)
    attack_orders = []

    for src in range(len(state)):
        if state.armies[src] <= 0 or state.owner[src] != player:
            continue
        neighbors = [
            neigh for neigh in state.neighbors(src)
            if state.owner[neigh] != player
        ]
        if neighbors:
            # if there is anything to attack, do that
            np.random.shuffle(neighbors)
            neighbors = neighbors[:np.random.randint(len(neighbors))+1]
            armies_available = state.armies[src]
            for dst in neighbors:
                armies_to_take = max(fixed_round(state.armies[dst] / 0.6), 1)
                if armies_available >= armies_to_take:
                    use_armies = np.random.randint(
                        armies_to_take,
                        armies_available+1
                    )
                    armies_available -= use_armies

                    assert armies_to_take <= use_armies <= state.armies[src] 
                    assert armies_available >= 0
                    
                    attack_orders.append(AttackTransferOrder(player, src, dst, use_armies))
        else:
            # if landlocked, move towards border
            # find all shortest paths to a border
            # then move along a random path
            shortest_paths = np.array(state.mapstruct.graph.shortest_paths())
            borders = np.where(state.owner != player)[0]
            dist_src_to_border = shortest_paths[src, borders].min()
            neighbors = np.array(state.mapstruct.graph.neighbors(src))
            assert dist_src_to_border > 0
            assert len(neighbors) > 0
            min_dist_neigh_to_border = shortest_paths[neighbors, :][:,borders].min(axis=1)
            best_neighbors = np.where(min_dist_neigh_to_border == dist_src_to_border-1)[0]
            dst = neighbors[np.random.choice(best_neighbors)]

            attack_orders.append(AttackTransferOrder(player, src, dst, state.armies[src]))

    np.random.shuffle(attack_orders)
    # return deploy_orders + attack_orders
    return OrderList(deploy_orders + attack_orders)

