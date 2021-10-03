from enum import IntEnum
import requests
import json
import igraph
from uuid import uuid4
from game_types import Bonus, MapStructure, MapState

ROOT = "http://aiserver.warzone.com/api/"

class MapID(IntEnum):
    # Map ids, sorted from smallest to largest
    BANANA = 29633 # 12, 4
    NEW_ZEALAND_SMALL = 38436 # 18, 8
    ITALY = 3448 # 20, 10
    ICELAND = 24221 # 24, 7
    FINLAND = 34615 # 31, 6
    APPLE = 32414 # 36, 13
    SMALL_EARTH = 70306 # 42, 6
    MEDIUM_EARTH = 19785 # 129, 27

def createGame(players, botgame=False, mapid=MapID.SMALL_EARTH):
    return call(
        "CreateBotGame" if botgame else "CreateGame",
        {
            "gameName": str(uuid4()),
            "players": [
                { "token": handleToken(player) }
                for player in players
            ],
            "settings": {
                "Fog": "NoFog",
                "MaxCardsHold": 999,
                "ReinforcementCard": "none",
                "OrderPriorityCard": "none",
                "OrderDelayCard": "none",
                "BlockadeCard": "none",
                "AutomaticTerritoryDistribution": "Automatic",
                "OneArmyStandsGuard": False,
                "TerritoryLimit": 2,
                "InitialPlayerArmiesPerTerritory": 2,
                "Wastelands": {
                    "NumberOfWastelands": 0,
                    "WastelandSize": 10,
                },
                "Map": mapid,
            }
        }
    )["gameID"]

def getMapStructure(gameid, botgame=False):
    return create_map_structure(
        call(
            "GetBotGameSettings" if botgame else "GetGameSettings",
            {"gameID": gameid}
        )["map"]
    )

def getMapState(gameid, mapstruct, playerid=633947, botgame=False, return_turn=False):
    response = call(
        "GetBotGameInfo" if botgame else "GetGameInfo",
        {
            "gameID": gameid,
            "playerID": playerid
        }
    )
    if isinstance(response["gameInfo"], str):
        raise Exception(response["gameInfo"])

    standing = response["gameInfo"]["latestStanding"]
    if return_turn:
        return create_map_state(standing, mapstruct), int(response["game"]["numberOfTurns"])
    else:
        return create_map_state(standing, mapstruct)

def getGameInfo(gameid, botgame=False):
    response = call(
        "GetBotGameInfo" if botgame else "GetGameInfo",
        {"gameID": gameid}
    )
    return {
        "turn": int(response["game"]["numberOfTurns"]),
        "players": {
            player["id"]: {
                "state": player["state"],
                "hasCommittedOrders": player["hasCommittedOrders"] == "True"
            }
            for player in response["game"]["players"]
        },
        "state": response["game"]["state"]
    }

def sendOrders(gameid, mapstruct, orders, turn, playerid=633947, botgame=False):
    response = call(
        "SendOrdersBotGame" if botgame else "SendOrders",
        {
            "gameID": gameid,
            "turnNumber": turn,
            "orders": orders.encode(mapstruct),
            "playerID": playerid
        }
    )
    return response

def call(api, data):
    response = json.loads(requests.post(ROOT + api, json=data).text)#.content.decode("utf-8"))
    if "error" in response:
        raise Exception(response["error"])
    return response

def handleToken(token):
    if isinstance(token, int):
        return f"00{token}00"
    else:
        return token

def create_map_structure(data):
    g = igraph.Graph()

    old_id_to_new_id = {}
    new_id_to_old_id = {}
    for i, src in enumerate(data["territories"]):
        g.add_vertex(src["name"])
        old_id_to_new_id[int(src["id"])] = i
        new_id_to_old_id[i] = int(src["id"])

    for i, src in enumerate(data["territories"]):
        for dst in src["connectedTo"]:
            g.add_edge(i, old_id_to_new_id[dst])
    g = g.as_undirected().simplify()

    bonuses = []
    for bonus_data in data["bonuses"]:
        bonuses.append(Bonus(
            bonus_data["name"],
            { old_id_to_new_id[i] for i in bonus_data["territoryIDs"] },
            int(bonus_data["value"])
        ))


    return MapStructure(data["name"], g, bonuses, new_id_to_old_id)

def create_map_state(data, mapstruct):
    assert len(mapstruct) == len(data)
    data_dict = {terr["terrID"]: terr for terr in data}
    data = [data_dict[mapstruct.ids[i]] for i in range(len(data))]
    return MapState(
            [int(terr["armies"]) for terr in data],
            [parseOwner(terr["ownedBy"]) for terr in data],
            mapstruct
    )

def parseOwner(owner):
    if owner == "Neutral":
        return 0
    return int(owner)

