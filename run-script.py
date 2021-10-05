import argparse
import api
from bot import RiskBot
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

    def print_callback(mcts, turn, time, **kwargs):
        winrate = 0.5 * mcts.root_node.win_value / mcts.root_node.visits + 0.5
        print(f"Turn {turn+1:2}:")
        print(f"  Time: {time:8.2f}s")
        print(f"  Winrate:{100*winrate:6.2f}%")

    bot = RiskBot(gameid, p1, p2, botgame=botgame)
    result = bot.play_loop(print_callback)
    print("Game complete:", "Win" if result else "Lose")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play game")
    parser.add_argument("--resume", type=int, default=None)
    parser.add_argument("--map", type=str, default="ITALY", choices=[map.name for map in api.MapID], help="Map to play on")
    parser.add_argument("--player", type=str, default=None)
    parser.add_argument("--iter", type=int, default=100, help="Number of iterations to run per turn")
    __main__(parser.parse_args())

