import api
import mcts_helper
from time import time, sleep

class RiskBot:
    def __init__(self, gameid, player, opponent, botgame=False, iterations=100, mapstruct=None):
        self.gameid = gameid
        self.player = player
        self.opponent = opponent
        self.botgame = botgame
        self.mapstruct = api.getMapStructure(gameid, botgame=botgame) if mapstruct is None else mapstruct
        self.iter = iterations

    def play(self):
        mapstate, turn = api.getMapState(
            self.gameid,
            self.mapstruct,
            botgame=self.botgame,
            playerid=self.player,
            return_turn=True
        )
        mcts = mcts_helper.setup_mcts(mapstate, self.player, self.opponent)
        mcts.simulate(self.iter)
        orders = mcts.make_choice().move
        api.sendOrders(
            self.gameid,
            self.mapstruct,
            orders,
            turn+1,
            playerid=self.player,
            botgame=self.botgame
        )
        return turn, mapstate, orders, mcts

    def play_loop(self, callback=None, timeout=10):
        info = api.getGameInfo(self.gameid, botgame=self.botgame)
        while info["state"] == "Playing":
            if info["players"][self.player]["hasCommittedOrders"]:
                sleep(timeout)
                info = api.getGameInfo(self.gameid, botgame=self.botgame)
                continue
            
            start_time = time()
            turn, mapstate, orders, mcts = self.play()

            callback and callback(
                turn=turn,
                mapstate=mapstate,
                orders=orders,
                mcts=mcts,
                time=time()-start_time
            )

            info = api.getGameInfo(self.gameid, botgame=self.botgame)
        return info["players"][self.player]["state"] == "Won"

