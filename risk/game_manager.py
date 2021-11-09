from . import api
from time import time, sleep
from .utils import load_mapstruct

class GameInfo:
    def __init__(self, data):
        self.turn = data["turn"]
        self.players = list(data["players"].keys())
        self.player_data = data["players"]
        self.state = data["state"]

    def get(gameid, botgame=False):
        return GameInfo(api.getGameInfo(gameid, botgame))

    @property
    def winner(self):
        if self.done:
            for player in self.players:
                if self.player_data[player]["state"] != "Eliminated":
                    return player

    @property
    def done(self):
        return self.state == "Finished"

    def ready_for_player_to_make_move(self, player):
        return self.state == "Playing" and not self.player_data[player]["hasCommittedOrders"]

class GameManager:
    def __init__(self, timeout=10):
        self.timeout = timeout

    def sendOrders(self): raise NotImplementedError()
    def getMapState(self, return_turn=False): raise NotImplementedError() # can combine with info
    def sendOrders(self, orders, player, turn=0): raise NotImplementedError()
    def gameInfo(self): raise NotImplementedError()

    def play_loop(self, *bots, callback=None):
        info = self.gameInfo()
        while not info.done:
            mapstate = self.getMapState()
            for bot in bots:
                if not info.ready_for_player_to_make_move(bot.player):
                    continue
                orders = bot.play(mapstate)
                self.sendOrders(orders, player=bot.player, turn=info.turn)
            sleep(self.timeout)
            old_info = info
            info = self.gameInfo()
            if callback and old_info.turn < info.turn:
                callback(bots, mapstate, old_info.turn)
        return info.winner

class RemoteGameManager(GameManager):
    def __init__(self, gameid, p1, p2, botgame=False):
        super().__init__()
        self.gameid = gameid
        self.botgame = botgame
        self.mapstruct = api.getMapStructure(gameid, self.botgame)
        self.p1 = p1
        self.p2 = p2

    def fromMap(mapid, invite, botgame=False):
        gameid = api.createGame(invite, botgame=botgame, mapid=mapid)
        info = api.getGameInfo(gameid, botgame)
        p1, p2 = info["players"].keys()
        return RemoteGameManager(gameid, p1, p2, botgame=botgame)
    
    def getMapState(self, return_turn=False):
        return api.getMapState(self.gameid, self.mapstruct, botgame=self.botgame, playerid=1, return_turn=return_turn)

    def sendOrders(self, orders, player, turn=0):
        api.sendOrders(self.gameid, self.mapstruct, orders, botgame=self.botgame, turn=turn, playerid=player)
    
    def gameInfo(self):
        return GameInfo.get(self.gameid, self.botgame)

class LocalGameManager(GameManager):
    def __init__(self, mapstruct):
        super().__init__(timeout=0)
        self.mapstruct = mapstruct
        self.mapstate = mapstruct.randState()
        self.p1 = 1
        self.p2 = 2
        self.turn = 1
        self.orders = {}
    
    def fromMap(mapid, cache=None):
        return LocalGameManager(load_mapstruct(mapid, cache=cache))

    def getMapState(self, return_turn=False):
        return (self.mapstate, self.turn) if return_turn else self.mapstate

    def sendOrders(self, orders, player, turn=0):
        # should check that player hasn't already sent any
        self.orders[player] = orders
        if len(self.orders) == 2:
            combined_orders = self.orders[self.p1].combine(self.orders[self.p2])
            self.mapstate = combined_orders(self.mapstate)
            self.orders = {}
            self.turn += 1
        else: assert len(self.orders) == 1

    def gameInfo(self):
        winner = self.mapstate.winner()
        return GameInfo({
            "turn": self.turn,
            "players": {
                player: {
                    "state": ("Playing"
                        if winner is None else
                        "Won"
                        if winner == player else
                        "Eliminated"),
                    "hasCommittedOrders": player in self.orders
                }
                for player in [self.p1, self.p2]
            },
            "state": "Playing" if winner is None else "Finished"
        })

