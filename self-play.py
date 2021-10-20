from joblib import Parallel, delayed
from functools import partial
import datetime as dt
import argparse
import json
import pickle
import os
from time import time, sleep

import api
from bot import RiskBot
import mcts_helper
from nn import *

def __main__(args):
    mapid = api.MapID[args.map]
    os.makedirs(args.maps_dir, exist_ok=True)
    if os.path.isfile(f"{args.maps_dir}/{mapid}.pkl"):
        mapstruct = pickle.load(open(f"{args.maps_dir}/{mapid}.pkl", "rb"))
    else:
        print("Downloading map")
        gameid = api.createGame([1,2], botgame=True, mapid=mapid)
        mapstruct = api.getMapStructure(gameid, botgame=True)
        pickle.dump(mapstruct, open(f"{args.maps_dir}/{mapid}.pkl", "wb"))
    mapstate = mapstruct.randState()

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
    def callback(mcts1, mcts2, turn, time, mapstate, **kwargs):
        winrate1 = 0.5 * mcts1.root_node.win_value / mcts1.root_node.visits + 0.5
        winrate2 = 0.5 * mcts2.root_node.win_value / mcts2.root_node.visits + 0.5
        print(f"Turn {turn:2}:")
        print(f"  Time: {time:8.2f}s")
        print(f"  Winrate:{100*winrate1:6.2f}%")
        print(f"  Winrate:{100*winrate2:6.2f}%")

        if args.output_dir:
            data["turns"].append({
                "owner": mapstate.owner.tolist(),
                "armies": mapstate.armies.tolist(),
                "p1_win_value": int(mcts1.root_node.win_value),
                "p1_visits": int(mcts1.root_node.visits),
                "p2_win_value": int(mcts2.root_node.win_value),
                "p2_visits": int(mcts2.root_node.visits),
                "p1_moves": [child.move.to_json() for child in mcts1.root_node.children],
                "p1_move_probs": [child.visits / mcts1.root_node.visits for child in mcts1.root_node.children],
                "p2_moves": [child.move.to_json() for child in mcts2.root_node.children],
                "p2_move_probs": [child.visits / mcts2.root_node.visits for child in mcts2.root_node.children],
            })
    def helper(mapstate, player, opponent, model, iters):
        mcts = mcts_helper.MCTS(mapstate, player, opponent, model)
        mcts.simulate(iters)
        return mcts

    turn = 0
    with Parallel(2) as parallel:
        while mapstate.winner() is None:
            start_time = time()
            turn += 1

            mcts1, mcts2 = parallel([
                    delayed(helper)(mapstate, 1, 2, model1, args.iter_1),
                    delayed(helper)(mapstate, 2, 1, model2, args.iter_2)
            ])
            orders1 = mcts1.make_choice().move
            orders2 = mcts2.make_choice().move

            mapstate = orders1.combine(orders2)(mapstate)

            callback and callback(
                turn=turn,
                mapstate=mapstate,
                orders1=orders1,
                orders2=orders2,
                mcts1=mcts1,
                mcts2=mcts2,
                time=time()-start_time
            )

    data["winner"] = int(mapstate.winner())
    print(f"Game complete: Player {mapstate.winner()} Won")
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        json.dump(data, open(f"{args.output_dir}/{dt.datetime.now()}.json", "w"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play game")
    parser.add_argument("--map", type=str, default="ITALY", choices=[map.name for map in api.MapID], help="The map to play on")
    parser.add_argument("--iter-1", type=int, default=100, help="Number of iterations to run per turn for player 1")
    parser.add_argument("--iter-2", type=int, default=100, help="Number of iterations to run per turn for player 2")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to store run data in")
    parser.add_argument("--maps-dir", type=str, default="/tmp/risk-maps", help="Directory to store the map in")
    parser.add_argument("--model-1", type=str, default=None, help="Pickle of the model to use for player 1")
    parser.add_argument("--model-2", type=str, default=None, help="Pickle of the model to use for player 2")
    __main__(parser.parse_args())

