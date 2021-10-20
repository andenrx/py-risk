import pickle
import os

import api
from game_types import MapStructure

def load_mapstruct(mapid: int, cache=None) -> MapStructure:
    """Download a map or load it from a cache"""
    if cache is not None:
        os.makedirs(cache, exist_ok=True)
        if os.path.isfile(f"{cache}/{mapid}.pkl"):
            return pickle.load(open(f"{cache}/{mapid}.pkl", "rb"))

    # The only way to download map data seems to be by creating a game
    gameid = api.createGame([1,2], botgame=True, mapid=mapid)
    mapstruct = api.getMapStructure(gameid, botgame=True)
    
    if cache is not None:
        pickle.dump(mapstruct, open(f"{cache}/{mapid}.pkl", "wb"))
    return mapstruct

