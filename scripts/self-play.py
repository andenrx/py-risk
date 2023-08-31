import datetime as dt
import argparse
import json
import pickle
import os
from distutils.util import strtobool

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

    bot1 = args.model_type_1(
            None, 1, 2, model1,
            iters=args.iter_1,
            max_depth=args.max_depth_1,
            trust_policy=args.policy_trust_1,
            moves_to_consider=args.moves_consider_1,
            timeout=args.time_limit_1,
            exploration=args.exploration_1,
            cache_opponent_moves=args.cache_opponent_moves_1,
            obj_rand=args.obj_rand_1,
            alpha=args.alpha_1,
            pop_size=args.pop_size_1,
            mirror_model=args.mirror_model_1,
            rounds=args.rounds_1,
    )
    bot2 = args.model_type_2(
            None, 2, 1, model2,
            iters=args.iter_2,
            max_depth=args.max_depth_2,
            trust_policy=args.policy_trust_2,
            moves_to_consider=args.moves_consider_2,
            timeout=args.time_limit_2,
            exploration=args.exploration_2,
            cache_opponent_moves=args.cache_opponent_moves_2,
            obj_rand=args.obj_rand_2,
            alpha=args.alpha_2,
            pop_size=args.pop_size_2,
            mirror_model=args.mirror_model_2,
            rounds=args.rounds_2,
    )
    game = risk.LocalGameManager.fromMap(mapid, cache=args.map_cache)

    callbacks = [risk.standard_callback]
    if args.output_dir:
        callbacks.append(risk.record_data_callback(data))
    if args.surrender_thresh > 0:
        callbacks.append(risk.early_terminate_callback(args.surrender_thresh))
    result = game.play_loop(
        bot1,
        bot2,
        callback=risk.compose_callbacks(*callbacks),
    )

    data["winner"] = result
    print(f"Game complete: Player {result} Won")
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        json.dump(data, open(f"{args.output_dir}/{dt.datetime.now()}.json", "w"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play game")
    parser.add_argument("--map", type=str, default="ITALY", choices=[map.name for map in risk.api.MapID], help="The map to play on")
    parser.add_argument("--surrender-thresh", type=float, default=0.0, help="Terminate early if a player has this probability of winning")
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
    parser.add_argument("--time-limit-1", type=float, default=float("inf"), help="")
    parser.add_argument("--time-limit-2", type=float, default=float("inf"), help="")
    parser.add_argument("--exploration-1", type=float, default=0.35, help="")
    parser.add_argument("--exploration-2", type=float, default=0.35, help="")
    parser.add_argument("--cache-opponent-moves-1", type=strtobool, default=False, help="")
    parser.add_argument("--cache-opponent-moves-2", type=strtobool, default=False, help="")
    parser.add_argument("--obj-rand-1", type=strtobool, default=False, help="")
    parser.add_argument("--obj-rand-2", type=strtobool, default=False, help="")
    parser.add_argument("--alpha-1", type=float, default="inf", help="")
    parser.add_argument("--alpha-2", type=float, default="inf", help="")
    parser.add_argument("--model-type-1", type=risk.mcts_helper.model_builder, default="MCTS", help="")
    parser.add_argument("--model-type-2", type=risk.mcts_helper.model_builder, default="MCTS", help="")
    parser.add_argument("--pop-size-1", type=int, default=50, help="")
    parser.add_argument("--pop-size-2", type=int, default=50, help="")
    parser.add_argument("--mirror-model-1", type=strtobool, default=False, help="")
    parser.add_argument("--mirror-model-2", type=strtobool, default=False, help="")
    parser.add_argument("--rounds-1", type=int, default=1, help="")
    parser.add_argument("--rounds-2", type=int, default=1, help="")
    __main__(parser.parse_args())

