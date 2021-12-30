import numpy as np
from functools import lru_cache
try:
    import torch
    import torch_geometric
except ImportError:
    torch = None
    torch_geometric = None

class Bonus:
    def __init__(self, name, territories, value):
        self.name = name
        self.terr = territories
        self.value = value

    def ownedby(self, state, player):
        return all(
            state.owner[i] == player
            for i in self.terr
        )

    def __repr__(self):
        return "Bonus(" + repr(self.name) + ", " + repr(self.value) + ")"

class MapStructure:
    def __init__(self, mapid, name, graph, bonuses, ids):
        self.id = mapid
        self.name = name
        self.graph = graph
        self.bonuses = bonuses
        self.ids = ids

    def randState(self):
        armies = np.random.randint(0, 5, len(self))
        owner = np.zeros(len(self), dtype=int)
        owner[np.random.choice(len(self), 4, replace=False)] = [1, 1, 2, 2]
        assert (owner == 1).sum() == 2
        assert (owner == 2).sum() == 2

        return MapState(armies, owner, self)

    @lru_cache(1)
    def edgeTensor(self):
        return torch_geometric.utils.to_undirected(
            torch.tensor(
                [[edge.source, edge.target] for edge in self.graph.es],
                dtype=torch.long
            ).T
        )

    @lru_cache(1)
    def bonusTensor(self):
        return torch.tensor(
            np.array([
                np.isin(np.arange(len(self)), np.array(list(bonus.terr)))
                for bonus in self.bonuses
            ])
        ).T

    def __repr__(self):
        return "MapStructure(" + repr(self.name) + ")"

    def __len__(self):
        return len(self.ids) 

class MapState:
    def __init__(self, armies, owner, mapstruct):
        assert len(armies) == len(owner)
        self.armies = np.array(armies)
        self.owner = np.array(owner)
        self.mapstruct = mapstruct
    
    def __len__(self):
        return len(self.armies)

    def winner(self):
        p = 0
        for owner in self.owner:
            if owner != 0 and p == 0:
                # first non-neutral player
                p = owner
            elif owner != p and owner != 0 != p:
                # two different players are left
                return None
        return p

    def neighbors(self, src):
        return self.mapstruct.graph.neighbors(src)

    def copy(self):
        return MapState(
            self.armies.copy(),
            self.owner.copy(),
            self.mapstruct
        )

    def assertvalid(self):
        assert (self.armies >= 0).all()
        assert len(self.armies) == len(self.owner) == len(self.mapstruct)

    def income(self, player):
        n = 5
        for bonus in self.mapstruct.bonuses:
            if bonus.ownedby(self, player):
                n += bonus.value
        return n

    def advance(self, orders):
        state = self.copy()
        for order in orders:
            order(state, inplace=True)
        return state

    def to_tensor(self, p1, p2, full=True):
        graph_features = torch.tensor(np.array([
            self.owner == p1,
            self.owner == p2,
            self.armies * (self.owner == p1),
            self.armies * (self.owner == p2),
            self.armies * (self.owner == 0),
        ]), dtype=torch.float).T
        if full:
            graph_features = torch.cat([graph_features, self.mapstruct.bonusTensor()], dim=1)
        i1, i2 = self.income(p1), self.income(p2)
        a1, a2 = self.armies[self.owner == p1].sum(), self.armies[self.owner == p2].sum()
        global_features = torch.tensor([
            np.log(i1 / (i1 + i2)),
            np.log(i2 / (i1 + i2)),
            np.log((a1+1) / (a1 + a2 + 2)),
            np.log((a2+1) / (a1 + a2 + 2)),
        ], dtype=torch.float)
        edges = self.mapstruct.edgeTensor()
        return graph_features, global_features, edges

    def __repr__(self):
        return "MapState(" + repr(len(self)) + ")"

