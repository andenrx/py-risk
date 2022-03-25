import datetime as dt
import argparse
import json
import pickle
from time import sleep
import os
from distutils.util import strtobool
import asyncio

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

    bot = args.model_type(
            None,
            p1,
            p2,
            model,
            iters=args.iter,
            max_depth=args.max_depth,
            trust_policy=args.policy_trust,
            moves_to_consider=args.moves_consider,
            timeout=args.time_limit,
            exploration=args.exploration,
            cache_opponent_moves=args.cache_opponent_moves,
            obj_rand=args.obj_rand,
            alpha=args.alpha,
            pop_size=args.pop_size,
            mirror_model=args.mirror_model,
    )
    game = risk.RemoteGameManager(gameid, p1, p2, botgame=botgame, timeout=0.1 if botgame else 10.0)

    callbacks = [risk.standard_callback]
    if args.output_dir:
        callbacks.append(risk.record_data_callback(data))
    if args.save_replay:
        os.makedirs(f"{args.output_dir}/replays", exist_ok=True)
        data["replay"] = f"replays/{gameid}.xml"
    try:
        def ping():
            if args.ping == 0: return
            if args.save_replay:
                risk.api.saveReplay(gameid, f"{args.output_dir}/replays/{gameid}.xml")
            elif botgame:
                game.gameInfo()

        result = asyncio.run(
            risk.utils.repeat_until_done(
                game.play_loop_async(bot, callback=risk.compose_callbacks(*callbacks)),
                ping, # keep alive by calling GetGameInfo every 5 seconds
                delay=args.ping
            )
        )
        print("Game complete:", "Win" if result == p1 else "Lose")
        data["winner"] = result
        if args.save_replay:
            risk.api.saveReplay(gameid, f"{args.output_dir}/replays/{gameid}.xml")
    except Exception as ex:
        print(ex)
        data["error"] = repr(ex)
        raise ex
    finally:
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
    parser.add_argument("--save-replay", type=strtobool, default=False, help="")
    parser.add_argument("--ping", type=int, default=0, help="")
    parser.add_argument("--model", type=str, default=None, help="")
    parser.add_argument("--max-depth", type=int, default=25, help="")
    parser.add_argument("--policy-trust", type=float, default=1.0, help="")
    parser.add_argument("--moves-consider", type=int, default=20, help="")
    parser.add_argument("--time-limit", type=float, default=float("inf"), help="")
    parser.add_argument("--exploration", type=float, default=0.35, help="")
    parser.add_argument("--cache-opponent-moves", type=strtobool, default=False, help="")
    parser.add_argument("--obj-rand", type=strtobool, default=False, help="")
    parser.add_argument("--alpha", type=float, default="inf", help="")
    parser.add_argument("--model-type", type=risk.mcts_helper.model_builder, default="MCTS", help="")
    parser.add_argument("--pop-size", type=int, default=50, help="")
    parser.add_argument("--mirror-model", type=strtobool, default=False, help="")
    __main__(parser.parse_args())

