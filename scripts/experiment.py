import risk
import importlib
import argparse
import yaml
import json
import os
import math
import random
import pandas as pd
from risk.mcts_helper import model_builder
try:
    from risk.nn import *
except ImportError:
    pass

default_player_args = {
    "model": None,
    "iter": 100,
    "moves-consider": 20,
    "max-depth": 25,
    "policy-trust": 1.0,
    "time-limit": float("inf"),
    "exploration": 0.35,
    "cache-opponent-moves": False,
    "obj-rand": False,
    "alpha": float("inf"),
    "model-type": "MCTS",
    "pop-size": 50,
    "mirror-model": False,
}

def run(args):
    config = yaml.safe_load(open(f"{args.dir}/config.yaml"))

    local = config.pop("local", True)
    if local:
        play = importlib.import_module("self-play").__main__
        if config["player 2"] == "default":
            raise Exception(f"'player 2: default' is not allowed for local runs")

        for player_num in [1, 2]:
            player_id = f"player {player_num}"
            if isinstance(config[player_id], str):
                config[player_id] = yaml.safe_load(open(f"{args.dir}/{config[player_id]}"))
            player_settings = default_player_args.copy()
            player_settings.update(config.pop(player_id))
            player_settings["model-type"] = model_builder(player_settings["model-type"])
            for key, value in player_settings.items():
                config[f"{key}-{player_num}"] = value
    else:
        play = importlib.import_module("run-script").__main__

        if isinstance(config["player 1"], str):
            config["player 1"] = yaml.safe_load(open(f"{args.dir}/{config['player 1']}"))
        player_settings = default_player_args.copy()
        player_settings.update(config.pop("player 1"))
        player_settings["model-type"] = model_builder(player_settings["model-type"])
        for key, value in player_settings.items():
            config[key] = value
        if config.pop("player 2") != "default":
            raise Exception(f"Expecting 'local: True' or 'player 2: default'")
        config["player"] = None
        config["resume"] = None
        config["ping"] = args.ping

    config["map-cache"] = args.map_cache
    config["output-dir"] = args.dir
    config.setdefault("surrender-thresh", 0.0)
    maps = config["map"]

    config = { key.replace("-", "_"): value for key, value in config.items() }
    for i in range(1, args.games + 1):
        if isinstance(maps, list):
            config["map"] = random.choice(maps)
        if args.games > 1:
            print(f"Round {i}")
        try:
            play(DotDict(config))
        except risk.api.ServerException as ex:
            print(ex)

class DotDict(dict):
    def __getattr__(self, attr):
        if attr in self:
            return self[attr]
        else:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

def analyze(args):
    df = pd.DataFrame([], columns=["game", "pred", "result"])
    game_results = pd.DataFrame([], columns=["game", "won"])
    for i, file in enumerate(os.listdir(args.dir)):
        if not file.endswith(".json"):
            continue
        game_data = json.load(open(f"{args.dir}/{file}"))
        game_results = game_results.append({"game": i, "won": game_data["winner"] == 1}, ignore_index=True)
        for entry in game_data["turns"]:
            df = df.append(
                {
                    "game": i,
                    "pred": entry["win_value"][0] / entry["visits"][0],
                    "result": (3-2*game_data["winner"])
                },
                ignore_index=True
            )
    print(f"Experiment: {args.dir}")
    print(f"==========")
    try:
        config = open(f"{args.dir}/config.yaml").readlines()
        print("".join("|   " + line for line in config), end="")
    except FileNotFoundError:
        print(f"Missing '{args.dir}/config.yaml'")
    print(f"==========")
    if game_results.empty:
        print("No results found")
    else:
        print("Games:", len(game_results))
        print(f"Win Percent:    {100 * game_results.won.mean():.2f}% ± {100 * 1.96 * 0.5 / math.sqrt(len(game_results)):.2f}%")
        actual_value = (df["result"]/2+0.5).sum()
        predicted_value = (df["pred"]/2+0.5).sum()
        print(f"Predicted Wins: {predicted_value:.2f}")
        print(f"Actual Wins:    {actual_value:.2f}")
        print(f"Difference:     {100 * (actual_value - predicted_value) / predicted_value :-.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment")
    subparser = parser.add_subparsers(title="subparse", dest="subparser_name")
    run_parse = subparser.add_parser("run")
    run_parse.add_argument("dir", type=str, help="")
    run_parse.add_argument("--games", type=int, default=1, help="")
    run_parse.add_argument("--map-cache", type=str, default="cached-maps", help="")
    run_parse.add_argument("--ping", type=int, default=0, help="")
    analyze_parse = subparser.add_parser("analyze")
    analyze_parse.add_argument("dir", type=str, help="")
    args = parser.parse_args()
    if args.subparser_name == "run":
        run(args)
    elif args.subparser_name == "analyze":
        analyze(args)

