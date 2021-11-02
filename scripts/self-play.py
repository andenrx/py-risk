import datetime as dt
import argparse
import json
import pickle
import os

import risk
try:
    from risk.nn import *
except ImportError:
    pass

def __main__(args):
    mapid = risk.api.MapID[args.map]
    if args.model_1 is not None:
        model1 = pickle.load(open(args.model_1, "rb"))
    else:
        model1 = None
    if args.model_2 is not None:
        model2 = pickle.load(open(args.model_2, "rb"))
    else:
        model2 = None

    print(f"Starting game on {args.map}")

    data = {
        "self-play": True,
        "map": int(mapid),
        "turns": [],
        "winner": None
    }

    bot1 = risk.MCTS(None, 1, 2, model1, iters=args.iter_1, max_depth=args.max_depth_1, trust_policy=args.policy_trust_1, moves_to_consider=args.moves_consider_1)
    bot2 = risk.MCTS(None, 2, 1, model2, iters=args.iter_2, max_depth=args.max_depth_2, trust_policy=args.policy_trust_2, moves_to_consider=args.moves_consider_2)
    game = risk.LocalGameManager.fromMap(mapid, cache=args.map_cache)

    result = game.play_loop(
        bot1,
        bot2,
        callback=(
            risk.compose_callbacks(risk.standard_callback, risk.record_data_callback(data))
            if args.output_dir else
            risk.standard_callback
        )
    )

    data["winner"] = result
    print(f"Game complete: Player {result} Won")
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        json.dump(data, open(f"{args.output_dir}/{dt.datetime.now()}.json", "w"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play game")
    parser.add_argument("--map", type=str, default="ITALY", choices=[map.name for map in risk.api.MapID], help="The map to play on")
    parser.add_argument("--iter-1", type=int, default=100, help="Number of iterations to run per turn for player 1")
    parser.add_argument("--iter-2", type=int, default=100, help="Number of iterations to run per turn for player 2")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to store run data in")
    parser.add_argument("--map-cache", type=str, default=None, help="Directory to use for map caches")
    parser.add_argument("--model-1", type=str, default=None, help="Pickle of the model to use for player 1")
    parser.add_argument("--model-2", type=str, default=None, help="Pickle of the model to use for player 2")
    parser.add_argument("--max-depth-1", type=int, default=25, help="")
    parser.add_argument("--max-depth-2", type=int, default=25, help="")
    parser.add_argument("--policy-trust-1", type=float, default=1.0, help="")
    parser.add_argument("--policy-trust-2", type=float, default=1.0, help="")
    parser.add_argument("--moves-consider-1", type=int, default=20, help="")
    parser.add_argument("--moves-consider-2", type=int, default=20, help="")
    __main__(parser.parse_args())

