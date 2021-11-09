def compose_callbacks(*cbs):
    def func(*args, **kwargs):
        for cb in cbs:
            cb(*args, **kwargs)
    return func

def standard_callback(bots, mapstate, turn):
    winrates = {bot: 0.5 * bot.root_node.win_value / bot.root_node.visits + 0.5 for bot in bots}
    times = {bot: bot.elapsed for bot in bots}
    print(f"Turn {turn:2}:")
    for bot in bots:
        print(f"  Winrate:{100*winrates[bot]:6.2f}% ({times[bot]:.2f}s)")


def record_data_callback(data):
    def callback(bots, mapstate, turn):
        data["turns"].append({
            "owner": mapstate.owner.tolist(),
            "armies": mapstate.armies.tolist(),
            "win_value": [int(bot.root_node.win_value) for bot in bots],
            "visits": [int(bot.root_node.visits) for bot in bots],
            "moves": [[child.move.to_json() for child in bot.root_node.children] for bot in bots],
            "move_probs": [[child.visits / bot.root_node.visits for child in bot.root_node.children] for bot in bots],
            "time": [bot.elapsed for bot in bots],
        })
    return callback

