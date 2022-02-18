import numpy as np
from .game_types import MapStructure

def fixed_round(x):
    return int(np.floor(x + 0.5))

def armies_to_take(x):
    if x == 0:
        return 1
    elif x == 1:
        return 2
    else:
        return int(np.floor(x / 0.6))

class Order:
    def __init__(self, player):
        self.player = int(player)

    def __call__(self, state, inplace=False):
        self.assertvalid(state)
        if not inplace:
            state = state.copy()
        state.assertvalid()
        return self.execute(state)
    
    def priority(self): raise NotImplementedError()
    def execute(self, state): raise NotImplementedError()
    def assertvalid(self, state): raise NotImplementedError()
    def encode(self, mapstruct): raise NotImplementedError()
    def to_json(self): raise NotImplementedError()
    def from_json(data):
        order_type = data[0]
        if order_type == "AttackTransferOrder":
            return AttackTransferOrder(*data[1:])
        elif order_type == "DeployOrder":
            return DeployOrder(*data[1:])
        else:
            raise Exception(f"Failed to deserialize '{data}'")

class AttackTransferOrder(Order):
    def __init__(self, player, src, dst, armies):
        super().__init__(player)
        self.src = int(src)
        self.dst = int(dst)
        self.armies = int(armies)

    def combat(attack, defend):
        return fixed_round(attack - defend * 0.7), fixed_round(defend - attack * 0.6)

    def priority(self): return 50

    def assertvalid(self, state):
        assert np.issubdtype(type(self.armies), int)
        assert np.issubdtype(type(self.src), int)
        assert np.issubdtype(type(self.dst), int)
        assert self.dst in state.neighbors(self.src)
        assert self.armies > 0

    def execute(self, state):
        if state.owner[self.src] != self.player:
            # player no longer owns territory
            return state
        if state.owner[self.src] == state.owner[self.dst]:
            # transfer
            armies = min(self.armies, state.armies[self.src])
            state.armies[self.src] -= armies
            state.armies[self.dst] += armies
        else:
            # attack
            attack = min(self.armies, state.armies[self.src])
            defend = state.armies[self.dst]
            attack_survive, defend_survive = AttackTransferOrder.combat(attack, defend)
            if attack_survive > 0 and defend_survive <= 0:
                # attacker wins
                state.armies[self.src] -= attack
                state.armies[self.dst] = attack_survive
                state.owner[self.dst] = self.player
            else:
                # defender wins
                attack_survive = max(attack_survive, 0)
                state.armies[self.src] -= (attack - attack_survive)
                state.armies[self.dst] = max(defend_survive, 0)
        return state

    def encode(self, mapstruct):
        return {
            "type": "GameOrderAttackTransfer",
            "playerID": self.player,
            "from": mapstruct.ids[self.src],
            "to": mapstruct.ids[self.dst],
            "numArmies": str(self.armies),
            "attackTeammates": True
        }
    
    def to_json(self):
        return ["AttackTransferOrder", self.player, self.src, self.dst, self.armies]

    def __repr__(self):
        return f"AttackTransferOrder(player={self.player}, src={self.src}, dst={self.dst}, armies={self.armies})"

class DeployOrder(Order):
    def __init__(self, player, target, armies):
        super().__init__(player)
        self.target = int(target)
        self.armies = int(armies)

    def priority(self): return 25

    def execute(self, state):
        state.armies[self.target] += self.armies
        return state

    def assertvalid(self, state):
        assert np.issubdtype(type(self.armies), int)
        assert np.issubdtype(type(self.target), int)
        assert state.owner[self.target] == self.player
        assert 0 < self.armies <= state.income(self.player)

    def __repr__(self):
        return f"DeployOrder(player={self.player}, target={self.target}, armies={self.armies})"

    def encode(self, mapstruct):
        return {
            "type": "GameOrderDeploy",
            "playerID": self.player,
            "armies": str(self.armies),
            "deployOn": mapstruct.ids[self.target]
        }

    def to_json(self):
        return ["DeployOrder", self.player, self.target, self.armies]

class OrderList(list, Order):
    def __init__(self, orders=None):
        if orders is not None:
            super().__init__(sorted(orders, key=lambda order: order.priority()))
        else:
            super().__init__()

    def execute(self, state):
        for order in self:
            order.execute(state)
        return state

    def assertvalid(self, state):
        for order in self:
            order.assertvalid(state)
        assert all(
            a.priority() <= b.priority() for a, b in zip(self[:-1], self[1:])
        )

    def encode(self, mapstruct):
        return [order.encode(mapstruct) for order in self]

    def to_json(self):
        return [order.to_json() for order in self]

    def from_json(data):
        return OrderList([Order.from_json(entry) for entry in data])

    def __or__(self, other):
        return self.combine(other)

    def __ror__(self, other):
        return self | other

    def combine(self, other):
        i = j = 0
        next_player = np.random.randint(2)
        result = OrderList()
        while i < len(self) and j < len(other):
            if next_player == 0 and self[i].priority() <= other[j].priority():
                next_player = 1-next_player
                result.append(self[i])
                i += 1
            elif next_player == 1 and self[i].priority() >= other[j].priority():
                next_player = 1-next_player
                result.append(other[j])
                j += 1
            elif self[i].priority() < other[j].priority():
                result.append(self[i])
                i += 1
            elif self[i].priority() > other[j].priority():
                result.append(other[j])
                j += 1
            else: assert False
        while i < len(self):
            result.append(self[i])
            i += 1
        while j < len(other):
            result.append(other[j])
            j += 1
        assert len(result) == len(self) + len(other)
        return result

    def embedding(self, mapstate, player):
        N = len(mapstate)
        edges = mapstate.mapstruct.edgeLabels()
        armies = mapstate.armies * (mapstate.owner == player)
        data = np.zeros(len(edges), dtype=int)
        self.assertvalid(mapstate)
        for order in self:
            assert order.player == player
            if isinstance(order, AttackTransferOrder):
                assert mapstate.owner[order.src] == order.player
                armies[order.src] -= order.armies
                data[edges[order.src, order.dst]] += order.armies
            elif isinstance(order, DeployOrder):
                assert mapstate.owner[order.target] == order.player
                armies[order.target] += order.armies
            else: assert False
        assert (armies >= 0).all()
        data[:N] += armies
        assert (data >= 0).all()
        return data / np.linalg.norm(data)

    def to_gene(self, mapstruct):
        edges = mapstruct.edgeLabels()
        data = np.zeros(len(edges), dtype=int)
        for order in self:
            if isinstance(order, AttackTransferOrder):
                data[edges[order.src, order.dst]] += order.armies
            elif isinstance(order, DeployOrder):
                data[edges[order.target, order.target]] += order.armies
            else: assert False
        assert (data >= 0).all()
        return data

    def from_gene(data, mapstruct, player):
        edges = mapstruct.edgeLabels()
        orders = []
        for (src, dst), index in edges.items():
            if data[index] > 0:
                if src == dst:
                    orders.append(DeployOrder(player, src, data[index]))
                else:
                    orders.append(AttackTransferOrder(player, src, dst, data[index]))
        return OrderList(orders)

