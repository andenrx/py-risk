import datetime as dt
import argparse
import json
from time import sleep

import api
from bot import RiskBot
from nn import *
import pickle

def __main__(args):
    botgame = args.player is None
    mapid = api.MapID[args.map]
    if args.resume is None:
        if botgame:
            invite = [1, "AI@warlight.net"]
        else:
            invite = ["me", args.player]
        gameid = api.createGame(invite, botgame=botgame, mapid=mapid)
        print(f"Starting game {gameid} on {args.map}")
    else:
        gameid = args.resume
        print(f"Resuming game {gameid}")
    if botgame:
        p1, p2 = 1, 2
    else:
        info = api.getGameInfo(gameid, botgame=botgame)
        p1 = 633947
        p2 = [player for player in info["players"].keys() if player != 633947][0]

        if info["state"] == "WaitingForPlayers":
            print("Waiting for players to join the game")
        while info["state"] == "WaitingForPlayers":
            sleep(10)
            info = api.getGameInfo(gameid, botgame=botgame)
    if args.model is None:
        model = None
    else:
        model = pickle.load(open(args.model, "rb"))

    data = {
        "map": int(mapid),
        "turns": []
    }
    def callback(mcts, turn, time, mapstate, **kwargs):
        winrate = 0.5 * mcts.root_node.win_value / mcts.root_node.visits + 0.5
        print(f"Turn {turn+1:2}:")
        print(f"  Time: {time:8.2f}s")
        print(f"  Winrate:{100*winrate:6.2f}%")

        if args.output_dir:
            data["turns"].append({
                "owner": mapstate.owner.tolist(),
                "armies": mapstate.armies.tolist(),
                "win_value": mcts.root_node.win_value,
                "visits": mcts.root_node.visits,
            })

    bot = RiskBot(gameid, p1, p2, botgame=botgame, model=model)
    result = bot.play_loop(callback)
    data["win"] = result
    print("Game complete:", "Win" if result else "Lose")
    if args.output_dir:
        json.dump(data, open(f"{args.output_dir}/{dt.datetime.now()}.json", "w"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play game")
    parser.add_argument("--resume", type=int, default=None)
    parser.add_argument("--map", type=str, default="ITALY", choices=[map.name for map in api.MapID], help="Map to play on")
    parser.add_argument("--player", type=str, default=None)
    parser.add_argument("--iter", type=int, default=100, help="Number of iterations to run per turn")
    parser.add_argument("--model", type=str, default=None, help="")
    parser.add_argument("--output-dir", type=str, default=None)
    __main__(parser.parse_args())

