import risk
from .orders import *
from .utils import weighted_choice
from collections import OrderedDict

import numpy as np
import random
import heapq

from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.csgraph import maximum_flow

class Objective:
    """
    Objective Ideas:
      ✓ Expand
      ☐ Defend
      ☐ Tactical Capture
      ☐ Deal Damage
      ☐ Break Bonus
      ☐ Poke
      ✓ Reposition
    """
    def __init__(self, state: risk.MapState, p1: int, p2: int, available: np.ndarray, owner: np.ndarray, budget: int):
        self.state = state
        self.p1 = p1
        self.p2 = p2
        self.available = available
        self.owner = owner
        self.budget = budget

    def value(self) -> float:
        raise NotImplementedError("Objectives must implement a value method")

    def handle(self, orders: list):
        """Generate the primitive moves to complete this objective

        This method is overriden by subclasses

        orders: list
            The initial list of orders to add to. This is modified in place. 
            Should not add duplicate moves, but should instead modify the original.
        """
        raise NotImplementedError("Objectives must implement a handle method")

    def handle_deploy(self, orders: list, terr: int, amount: int):
        """Increase the number of troops deployed on `terr` by `amount`"""
        assert amount >= 0
        if amount == 0: return
        exists = False
        for order in orders:
            if isinstance(order, DeployOrder) and order.target == terr:
                assert not exists
                exists = True
                order.armies += int(amount)
        if not exists:
            orders.append(DeployOrder(int(self.p1), int(terr), int(amount)))

    def handle_attack(self, orders: list, src: int, dst: int, amount: int):
        """Increase the number of troops attacking from `src` to `dst` by `amount`"""
        assert amount >= 0
        if amount == 0 or src == dst: return
        exists = False
        for order in orders:
            if isinstance(order, AttackTransferOrder) and order.src == src and order.dst == dst:
                assert not exists
                exists = True
                order.armies += int(amount)
        if not exists:
            orders.append(AttackTransferOrder(int(self.p1), int(src), int(dst), int(amount)))

class Expand(Objective):
    def __init__(self, state, p1, p2, available, owner, budget, bonus):
        super().__init__(state, p1, p2, available, owner, budget)
        self.bonus = bonus
        self.pred, self.dist = self.calc_paths()

    def __repr__(self):
        return f"Expand({self.bonus})"

    def value(self) -> float:
        cost = (self.state.armies * (self.owner != self.p1))[list(self.bonus.terr)].sum()
        return self.bonus.value / (cost + 1)

    def handle(self, orders: list):
        attacks, deploys = self.calc_attacks_deploys()
        for src, armies in deploys.items():
            self.handle_deploy(orders, int(src), int(armies))
        for (src, dst), armies in attacks.items():
            self.handle_attack(orders, int(src), int(dst), int(armies))
        return sum(deploys.values())

    def is_reachable(self) -> bool:
        return all(terr in self.pred for terr in self.bonus.terr)

    def allocate(self, srcs, dsts):
        assert len(srcs)
        assert len(dsts)
        matrix = lil_matrix((len(self.state)+2, len(self.state)+2), dtype=int)
        SRC = len(self.state)
        DST = len(self.state) + 1
        for src in srcs:
            matrix[SRC, src] = self.available[src]
        for dst in dsts:
            assert self.state.owner[dst] != self.p1
            assert armies_to_take(self.state.armies[dst]) > 0
            matrix[dst, DST] = armies_to_take(self.state.armies[dst])
            for src in self.state.mapstruct.graph.neighbors(dst):
                if self.state.owner[src] == self.p1:
                    matrix[src, dst] = matrix[dst, DST]
        matrix = matrix.tocsr()
        result = maximum_flow(matrix, SRC, DST).residual

        attacks = {}
        for dst in dsts:
            for src in self.state.mapstruct.graph.neighbors(dst):
                attacks[src, dst] = result[src, dst]
        return attacks

    def calc_attacks_deploys(self):
        # TODO: Maybe prioritize critical before optional
        budget = self.budget
        max_dist = max(self.dist.values())
        assert max_dist > 0
        far_away = {terr for terr in self.bonus.terr if self.dist[terr] == max_dist}
        critical = set()
        for terr in far_away:
            while self.pred[self.pred[terr]] is not None:
                terr = self.pred[terr]
            if self.owner[terr] != self.p1:
                critical.add(terr)
        optional = {terr for terr in self.bonus.terr if self.dist[terr] == 1 and self.owner[terr] != self.p1} - critical
        assert all([src for src in self.state.mapstruct.graph.neighbors(dst) if self.state.owner[src] == self.p1] for dst in critical)
        assert all([src for src in self.state.mapstruct.graph.neighbors(dst) if self.state.owner[src] == self.p1] for dst in optional)
        if not (critical | optional):
            # This bonus should be captured by end of turn
            return {}, {}

        deploys = {}
        attacks = self.allocate(np.where(self.state.owner == self.p1)[0], critical | optional)
        attacks_towards = {dst: {} for dst in critical | optional}
        survivors = {dst: self.state.armies[dst] for dst in critical | optional}
        assert all(val >= 0 for val in attacks.values())

        for (src, dst), armies in attacks.items():
            attacks_towards[dst][src] = armies
            survivors[dst] -= fixed_round(armies * 0.6)
            self.available[src] -= armies
            assert self.available[src] >= 0
        for dst in critical | optional:
            if survivors[dst] > 0:
                # Deploy to the biggest attacker
                assert self.state.mapstruct.graph.neighbors(dst)
                assert [src for src in self.state.mapstruct.graph.neighbors(dst) if self.state.owner[src] == self.p1]
                assert attacks_towards[dst]
                best_src = max(
                    (src for src in attacks_towards[dst] if self.state.owner[src] == self.p1),
                    key=lambda src: attacks[src, dst]
                )
                assert self.state.owner[best_src] == self.p1
                # Due to rounding, this may not be exactly armies_to_take(survivors[dst])
                armies_required = armies_to_take(fixed_round(attacks[best_src, dst] * 0.6) + survivors[dst])
                assert armies_required > attacks[best_src, dst]
                deploy_required = armies_required - attacks[best_src, dst]
                assert deploy_required > 0
                if deploy_required > budget:
                    # Not enough armies to successfully take
                    # Cancel any attacks on this target
                    deploy_recieved = budget
                    for src in attacks_towards[dst]:
                        del attacks[src, dst]
                else:
                    deploy_recieved = deploy_required
                    attacks[best_src, dst] = attacks.get((best_src, dst), 0) + deploy_recieved
                    self.owner[dst] = self.p1
                assert deploy_recieved >= 0
                budget -= deploy_recieved
                assert budget >= 0
                deploys[best_src] = deploys.get(best_src, 0) + deploy_recieved 
            else:
                self.owner[dst] = self.p1

        # Sort attacks smallest to largest
        attacks = OrderedDict(sorted([(edge, value) for edge, value in attacks.items()], key=lambda edge_value: edge_value[1]))
        assert all(val >= 0 for val in attacks.values())
        assert all(val >= 0 for val in deploys.values())

        return attacks, deploys

    def calc_paths(self):
        # TODO: pred should prioritize cheaper paths
        owned = np.where(self.state.owner == self.p1)[0]
        pred = {src: None for src in owned}
        dist = {src: 0 for src in owned}
        heap = [(0, -self.available[src], src) for src in owned]
        heapq.heapify(heap)
        while heap:
            d, troops, src = heapq.heappop(heap)
            for dst in self.state.mapstruct.graph.neighbors(src):
                if dst in self.bonus.terr and dst not in pred:
                    pred[dst] = src
                    dist[dst] = dist[src] + 1
                    heapq.heappush(heap, (dist[dst], troops + self.state.armies[dst], dst))
        return pred, dist

class Defend(Objective):
    def __init__(self, state, p1, p2, available, owner, budget, bonus):
        super().__init__(state, p1, p2, available, owner, budget)
        self.bonus = bonus

    def __repr__(self):
        return f"Defend({self.bonus})"

    def risk(self, terr: int) -> int:
        return sum(
            fixed_round(0.6 * self.state.armies[dst])
            for dst in self.state.neighbors(terr)
            if self.state.owner[dst] == self.p2
        )

    def value(self) -> float:
        owned = set(np.where(self.owner == self.p1)[0])
        v = self.bonus.value * (len(owned & self.bonus.terr) / len(self.bonus.terr)) ** 2
        cost = sum(max(self.risk(terr) + 1 - self.available[terr], 0) for terr in self.bonus.terr if self.owner[terr] == self.p1)
        return v / (cost + 1)

    def handle(self, orders: list):
        # TODO: Maybe mark these troops as unavailable?
        #       But they need to be available for safe_attacks still.
        #       Also need to make sure to not double deploy with overlapping bonuses.
        budget_remaining = self.budget
        for terr in sorted(self.bonus.terr, key=lambda terr: self.available[terr]):
            if self.state.owner[terr] != self.p1 or not any(self.state.owner[dst] == self.p2 for dst in self.state.neighbors(terr)):
                continue
            risk_factor = self.risk(terr)
            if self.available[terr] <= risk_factor:
                deploy = min(risk_factor + 1 - self.available[terr], budget_remaining)
                self.available[terr] += deploy
                self.handle_deploy(orders, terr, deploy)
                budget_remaining -= deploy
            else:
                assert self.available[terr] >= 0
        return self.budget - budget_remaining

class RandDeploy(Objective):
    def handle(self, orders: list):
        terr = random.choice(self.state.borders(self.p1, include_neutrals=True))
        self.available[terr] += self.budget
        self.handle_deploy(orders, terr, self.budget)
        return self.budget

class Reposition(Objective):
    def calc_path_to_border(self):
        borders = {
            src for src in range(len(self.state))
            if self.state.owner[src] == self.p1
            and any(
                self.owner[dst] != self.p1
                for dst in self.state.neighbors(src)
            )
        }
        pred = {}
        curr = list(borders)
        while curr:
            src = curr.pop(0)
            for dst in self.state.mapstruct.graph.neighbors(src):
                if dst not in pred and dst not in borders:
                    pred[dst] = src
                    curr.append(dst)
        return pred

    def handle(self, orders: list):
        pred = self.calc_path_to_border()
        for src, armies in enumerate(self.available):
            if src in pred and self.state.owner[src] == self.p1:
                self.handle_attack(orders, int(src), int(pred[src]), int(armies))

def guess_deploys(state, territories, income, player):
    deploy_amounts = np.random.multinomial(income, [1/len(territories) for _ in territories])
    deploys = []
    for territory, amount in zip(territories, deploy_amounts):
        if amount > 0:
            deploys.append(DeployOrder(player, territory, amount))
    return OrderList(deploys)

def objectives(state, p1, p2, available, owner):
    state = state.copy()
    p1_income, p2_income = state.income(p1), state.income(p2)
    p2_borders = state.borders(p2, include_neutrals=True)
    deploys = guess_deploys(state, p2_borders, p2_income, p2)
    state = deploys(state, inplace=True)

    possible_objectives = []
    for bonus in state.mapstruct.bonuses:
        if  (
                bonus.value > 0
            ) and any( # Player owns at least one territory in/around this bonus
                state.owner[src] == p1
                for dst in bonus.terr
                for src in state.neighbors(dst)
            ) and any( # But does not own the bonus
                state.owner[terr] != p1
                for terr in bonus.terr
            ):
            exp = Expand(state, p1, p2, available, owner, p1_income, bonus)
            if exp.is_reachable(): 
                possible_objectives.append(exp)
    possible_objectives.append(
        Expand(
            state, p1, p2, available, owner, p1_income,
            bonus=risk.game_types.Bonus("Full Map", set(range(len(state))), 1e-7),
        )
    )
    return possible_objectives

def rand_obj(state, p1, p2):
    available = state.armies * (state.owner == p1)
    owner = state.owner.copy()
    possible = objectives(state, p1, p2, available, owner)
    inc = state.income(p1)

    orders = []
    while inc > 0:
        if len(possible) == 0:
            inc -= RandDeploy(state, p1, p2, available, owner, 1).handle(orders)
            continue
        obj = weighted_choice(possible, [obj.value() for obj in possible])
        possible.remove(obj)
        obj.budget = inc
        inc -= obj.handle(orders)
    Reposition(state, p1, p2, available, owner, 0).handle(orders)
    return OrderList(orders)

