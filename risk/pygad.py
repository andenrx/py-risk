import risk
from risk.orders import *
from risk.rand import rand_move
from time import time
import numpy as np
import pygad

try:
    import torch
    from torch_geometric.loader import DataLoader
except ImportError:
    pass

def playout(mapstate, p1, p2, max_iter=100):
    for i in range(max_iter):
        if mapstate.winner() is not None:
            return (1 if mapstate.winner() == p1 else -1), i
        m1 = rand_move(mapstate, p1)
        m2 = rand_move(mapstate, p2)
        mapstate = (m1 | m2)(mapstate)
    return 0, max_iter

def create(
        mapstate,
        p1,
        p2,
        model,
        iterations=5,
        pop_size=10,
        max_iter=100,
        rounds=5,
        timeout=np.inf,
        mutation_rate=0.04,
        mirror_model=False,
    ):
    mapstruct = mapstate.mapstruct

    table = np.zeros((pop_size, pop_size))
    table_ready = [False]

    def run_tournament(population):
        table[:] = 0
        if model is not None:
            states = []
            for i in range(pop_size):
                for j in range(pop_size):
                    m1 = OrderList.from_gene(population[0,i], mapstruct, p1)
                    m2 = OrderList.from_gene(population[1,j], mapstruct, p2)
                    result = (m1 | m2)(mapstate)
                    prepped = model.prep(result, p1=p1, p2=p2)
                    prepped.i = i
                    prepped.j = j
                    prepped.player = p1
                    states.append(prepped)

                    if mirror_model:
                        prepped = model.prep(result, p1=p2, p2=p1)
                        prepped.i = i
                        prepped.j = j
                        prepped.player = p2
                        states.append(prepped)

            dl = DataLoader(states, batch_size=400)
            with torch.no_grad():
                for sample in dl:
                    pred = model(sample)
                    for v, i, j, state, player in zip(pred, sample.i, sample.j, sample.state, sample.player):
                        if state.winner() is None:
                            if not mirror_model:
                                table[i, j] = v * 0.995
                            else:
                                table[i, j] += v * 0.995 * (0.5 if player == p1 else -0.5)
                        else:
                            table[i, j] = 1 if state.winner() == p1 else -1
        else:
            for i in range(pop_size):
                for j in range(pop_size):
                    m1 = OrderList.from_gene(population[0,i], mapstruct, p1)
                    m2 = OrderList.from_gene(population[1,j], mapstruct, p2)
                    for k in range(rounds):
                        result, turns = playout((m1 | m2)(mapstate), p1, p2, max_iter=max_iter)
                        table[i, j] += result / rounds * 0.999 ** turns

    def fitness_func(_, index):
        if not table_ready[0]:
            run_tournament(ga.population)
            table_ready[0] = True
        if index[0] == 0:
            # player 1
            return table.mean(1)[index[1]]
        elif index[0] == 1:
            # player 2
            return -table.mean(0)[index[1]]
        else:
            raise KeyError(index)

    def on_mutation(self, offspring):
        edges = mapstruct.edgeLabels()
        offspring[offspring < 0] = 0

        attacks_from = {
                src: np.zeros(len(edges), dtype=bool)
                for src in range(len(mapstate))
        }
        for (src, dst), index in edges.items():
            if src != dst:
                attacks_from[src][index] = True
        for player, off in ((p1, offspring[0]), (p2, offspring[1])):
            for (src, dst), index in edges.items():
                if mapstate.owner[src] != player:
                    # Cannot do attacks from unowned territory
                    off[:, index] = 0

            too_many_deployed = off[:, :len(mapstate)].sum(1) > mapstate.income(player)
            for i in np.where(too_many_deployed)[0]:
                while off[i, :len(mapstate)].sum() > mapstate.income(player):
                    j = np.random.choice(np.where(off[i, :len(mapstate)] > 0)[0])
                    off[i, j] -= 1

            not_enough_deployed = off[:, :len(mapstate)].sum(1) < mapstate.income(player)
            for i in np.where(not_enough_deployed)[0]:
                while off[i, :len(mapstate)].sum() < mapstate.income(player):
                    j = np.random.choice(np.where(mapstate.owner == player)[0])
                    off[i, j] += 1

            for d in off:
                for src in range(len(mapstate)):
                    deployment = d[edges[src, src]]
                    available = mapstate.armies[src] + deployment
                    over_by = d[attacks_from[src]].sum()
                    while over_by > available:
                        k = np.random.choice(np.where(attacks_from[src])[0])
                        d[k] -= min(d[k], over_by)
                        over_by = d[attacks_from[src]].sum()
            assert (off[:, :len(mapstate)].sum(1) == mapstate.income(player)).all()
        return offspring

    def mark_table_as_invalid(ga):
        table_ready[0] = False
        if time() - ga.start_time >= timeout:
            return "stop"

    ga = pygad.GA(
        num_generations=iterations,
        num_parents_mating=pop_size // 2,
        fitness_func=fitness_func,
        initial_population=np.array([
            [rand_move(mapstate, p1).to_gene(mapstruct) for _ in range(pop_size)],
            [rand_move(mapstate, p2).to_gene(mapstruct) for _ in range(pop_size)],
        ]),
        gene_type=int,
        parent_selection_type="rank",
        on_generation=mark_table_as_invalid,
        on_start=mark_table_as_invalid,
        on_mutation=on_mutation,
        mutation_by_replacement=False,
        mutation_type="random",
        mutation_probability=mutation_rate,
        crossover_type="single_point", # TODO: Restructure gene so that crossover keeps same src together
        random_mutation_min_val=-4,
        random_mutation_max_val=4,
        keep_parents=-1,
        cache_fitness=False,
    )
    ga.table = table
    ga.start_time = time()
    return ga

