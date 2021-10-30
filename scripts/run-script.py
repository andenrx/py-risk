import datetime as dt
import argparse
import json
import pickle
from time import sleep
import os

import risk
try:
    from risk.nn import *
except ImportError:
    pass

def __main__(args):
    botgame = args.player is None
    mapid = risk.api.MapID[args.map]
    if args.resume is None:
        if botgame:
            invite = [1, "AI@warlight.net"]
        else:
            invite = ["me", args.player]
        gameid = risk.api.createGame(invite, botgame=botgame, mapid=mapid)
        print(f"Starting game {gameid} on {args.map}")
    else:
        gameid = args.resume
        print(f"Resuming game {gameid}")
    if botgame:
        p1, p2 = 1, 2
    else:
        info = risk.api.getGameInfo(gameid, botgame=botgame)
        p1 = 633947
        p2 = [player for player in info["players"].keys() if player != 633947][0]

        if info["state"] == "WaitingForPlayers":
            print("Waiting for players to join the game")
        while info["state"] == "WaitingForPlayers":
            sleep(10)
            info = risk.api.getGameInfo(gameid, botgame=botgame)
    if args.model is None:
        model = None
    else:
        model = pickle.load(open(args.model, "rb"))

    data = {
        "self-play": False,
        "map": int(mapid),
        "turns": [],
        "winner": None
    }

    bot = risk.MCTS(None, p1, p2, model, iters=args.iter)
    game = risk.RemoteGameManager(gameid, p1, p2, botgame=botgame)
    result = game.play_loop(
        bot,
        callback=(
            risk.compose_callbacks(risk.standard_callback, risk.record_data_callback(data))
            if args.output_dir else
            risk.standard_callback
        )
    )

    data["win"] = result
    print("Game complete:", "Win" if result else "Lose")
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        json.dump(data, open(f"{args.output_dir}/{dt.datetime.now()}.json", "w"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play game")
    parser.add_argument("--map", type=str, default="ITALY", choices=[map.name for map in risk.api.MapID], help="The map to play on")
    parser.add_argument("--player", type=str, default=None, help="Email of the player to start the game against, default plays against the built in AI")
    parser.add_argument("--resume", type=int, default=None, help="The game id to resume playing, default starts a new game")
    parser.add_argument("--iter", type=int, default=100, help="Number of iterations to run per turn")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to store run data in")
    parser.add_argument("--model", type=str, default=None, help="")
    __main__(parser.parse_args())

