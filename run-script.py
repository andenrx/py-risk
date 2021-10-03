import argparse
import api
import mcts_helper
from time import time, sleep

def __main__(args):
    botgame = args.player is None
    if args.resume is None:
        if botgame:
            invite = [1, "AI@warlight.net"]
        else:
            invite = ["me", args.player]
        gameid = api.createGame(invite, botgame=botgame, mapid=api.MapID[args.map])
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

    mapstruct = api.getMapStructure(gameid, botgame=botgame)
    mapstate = None
    info = api.getGameInfo(gameid, botgame=botgame)
    while info["state"] == "Playing" or info["state"] == "WaitingForPlayers":
        if info["state"] == "WaitingForPlayers" or info["players"][p1]["hasCommittedOrders"]:
            sleep(10)
            info = api.getGameInfo(gameid, botgame=botgame)
            continue
        mapstate, turn = api.getMapState(gameid, mapstruct, botgame=botgame, playerid=p1, return_turn=True)

        mcts = mcts_helper.setup_mcts(mapstate, p1, p2)
        start = time()
        mcts.simulate(args.iter)

        winrate = 0.5 * mcts.root_node.win_value / mcts.root_node.visits + 0.5
        print(f"Turn {turn+1:2}:")
        print(f"  Time: {time() - start:8.2f}s")
        print(f"  Winrate:{100*winrate:6.2f}%")

        orders = mcts.make_choice().move
        api.sendOrders(gameid, mapstruct, orders, turn+1, playerid=p1, botgame=botgame) 

        info = api.getGameInfo(gameid, botgame=botgame)

    print("Game complete:", info["players"][p1]["state"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play game")
    parser.add_argument("--resume", type=int, default=None)
    parser.add_argument("--map", type=str, default="ITALY", choices=[map.name for map in api.MapID], help="Map to play on")
    parser.add_argument("--player", type=str, default=None)
    parser.add_argument("--iter", type=int, default=100, help="Number of iterations to run per turn")
    __main__(parser.parse_args())

