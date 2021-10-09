import numpy as np

def fixed_round(x):
    return int(np.floor(x + 0.5))

class Order:
    def __init__(self, player):
        self.player = player

    def __call__(self, state, inplace=False):
        assert self.isvalid(state)
        if not inplace:
            state = state.copy()
        assert state.isvalid()
        return self.execute(state)
    
    def priority(self): raise NotImplementedError()
    def execute(self, state): raise NotImplementedError()
    def isvalid(self, state): raise NotImplementedError()
    def encode(self, mapstruct): raise NotImplementedError()

class AttackTransferOrder(Order):
    def __init__(self, player, src, dst, armies):
        super().__init__(player)
        assert isinstance(src, int)
        assert isinstance(dst, int)
        assert isinstance(armies, int)
        self.src = src
        self.dst = dst
        self.armies = armies

    def combat(attack, defend):
        return fixed_round(attack - defend * 0.7), fixed_round(defend - attack * 0.6)

    def priority(self): return 50

    def isvalid(self, state):
        return (
                isinstance(self.armies, int)
            and isinstance(self.src, int)
            and isinstance(self.dst, int)
            and self.dst in state.neighbors(self.src)
            and self.armies > 0
        )

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
    
    def __repr__(self):
        return f"AttackTransferOrder(player={self.player}, src={self.src}, dst={self.dst}, armies={self.armies})"

class DeployOrder(Order):
    def __init__(self, player, target, armies):
        super().__init__(player)
        self.target = target
        self.armies = armies
    def priority(self): return 25

    def execute(self, state):
        state.armies[self.target] += self.armies
        return state

    def isvalid(self, state):
        return (
                isinstance(self.armies, int)
            and isinstance(self.target, int)
            and state.owner[self.target] == self.player
            and 0 < self.armies <= state.income(self.player)
        )

    def __repr__(self):
        return f"DeployOrder(player={self.player}, target={self.target}, armies={self.armies})"

    def encode(self, mapstruct):
        return {
            "type": "GameOrderDeploy",
            "playerID": self.player,
            "armies": str(self.armies),
            "deployOn": mapstruct.ids[self.target]
        }

class OrderList(list, Order):
    def __init__(self, orders=None):
        if orders is not None:
            super().__init__(orders)
            # super().__init__(sorted(orders))
        else:
            super().__init__()

    def execute(self, state):
        for order in self:
            order.execute(state)
        return state

    def isvalid(self, state):
        return (
                all(
                    order.isvalid(state) for order in self
                )
            and all(
                a.priority() <= b.priority() for a, b in zip(self[:-1], self[1:])
            )
        )

    def encode(self, mapstruct):
        return [order.encode(mapstruct) for order in self]

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

