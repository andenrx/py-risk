from joblib import Parallel, delayed
from functools import partial
import datetime as dt
import argparse
import json
from time import sleep

import api
from bot import RiskBot

def __main__(args):
    mapid = api.MapID[args.map]
    gameid = api.createGame([1,2], botgame=True, mapid=mapid)
    print(f"Starting game {gameid} on {args.map}")

    data = {
        "self-play": True,
        "map": int(mapid),
        "turns": []
    }
    def callback(bot, mcts, turn, time, mapstate, **kwargs):
        winrate = 0.5 * mcts.root_node.win_value / mcts.root_node.visits + 0.5
        print(f"Turn {turn+1:2} (Player {bot.player}):")
        print(f"  Time: {time:8.2f}s")
        print(f"  Winrate:{100*winrate:6.2f}%")

        if args.output_dir:
            data["turns"].append({
                "player": bot.player,
                "owner": mapstate.owner.tolist(),
                "armies": mapstate.armies.tolist(),
                "win_value": mcts.root_node.win_value,
                "visits": mcts.root_node.visits,
            })

    bot1 = RiskBot(gameid, 1, 2, botgame=True)
    bot2 = RiskBot(gameid, 2, 1, botgame=True)
    result = Parallel(2)(
        delayed(bot.play_loop)(partial(callback, bot), timeout=3) for bot in (bot1, bot2)
    )
    data["winner"] = 1 if result[0] else 2
    print(f"Game complete: Player {1 if result[0] else 2} Won")
    if args.output_dir:
        json.dump(data, open(f"{args.output_dir}/{dt.datetime.now()}.json", "w"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play game")
    parser.add_argument("--map", type=str, default="ITALY", choices=[map.name for map in api.MapID], help="Map to play on")
    parser.add_argument("--iter", type=int, default=100, help="Number of iterations to run per turn")
    parser.add_argument("--output-dir", type=str, default=None)
    __main__(parser.parse_args())

