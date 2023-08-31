import numpy as np
import argparse
import os
from tqdm import tqdm
import collections
import json
import tabulate as tb
from risk.api import MapID
tb.PRESERVE_WHITESPACE = True

def win_prob(exp_scores, i, j):
    return exp_scores[i] / (exp_scores[i] + exp_scores[j])

def log_match_prob(exp_scores, i, j, n_wins, n_games):
    assert 0 <= n_wins <= n_games
    return n_wins * np.log(win_prob(exp_scores, i, j)) + (n_games - n_wins) * np.log(win_prob(exp_scores, j, i))

def log_data_prob(scores, matches):
    exp_scores = np.exp(scores)
    log_p = 0
    for (n1, n2), (wins, games) in matches.items():
        i = get_index(n1)
        j = get_index(n2)
        log_p += log_match_prob(exp_scores, i, j, wins, games)
    return log_p

def get_index(name):
    return player_dict[name]

def mcmc_sample(matches, samples, skip_every=10, burn_in=0):
  x = np.zeros(len(player_dict))
  log_p = log_data_prob(x, matches)

  results = []
  i = 0
  bar = tqdm(total=samples)
  while len(results) < samples:
    x_new = x + np.random.randn(len(player_dict)) / 10
    log_p_new = log_data_prob(x_new, matches)
    if np.log(np.random.rand()) < log_p_new - log_p:
      x = x_new
      log_p = log_p_new
    i += 1
    if i > burn_in and i % skip_every == 0:
      results.append(x)
      bar.update(1)
  return np.array(results)

player_dict = {}

def mapid_from_name(name):
    return MapID[name].value

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("tournament", type=str)
    parser.add_argument("--map", default=None, type=lambda str:
        None if not str else [mapid_from_name(name) for name in str.split(',')]
    )
    parser.add_argument("--agents", default=None, type=lambda str: str.split(','))
    parser.add_argument("--std", default=1.0, type=float)
    # mcmc args
    parser.add_argument("--samples", default=10000, type=int)
    parser.add_argument("--burn-in", default=1000, type=int)
    parser.add_argument("--skip-every", default=10, type=int)
    # elo args
    parser.add_argument("--elo-factor", default=np.log(10)/400, type=float)
    parser.add_argument("--elo-base", default=1000, type=float)
    parser.add_argument("--elo-peg", default=None, type=str)
    args = parser.parse_args()

    players = set()
    matches = collections.defaultdict(lambda: (0, 0))
    matches_on_map = collections.defaultdict(lambda: collections.defaultdict(int))
    fail_count = 0
    inferred = 0
    for dir_name in os.listdir(args.tournament):
        if os.path.isfile(os.path.join(args.tournament, dir_name)):
            continue
        assert "-vs-" in dir_name
        p1, p2 = dir_name.split("-vs-")
        if args.agents is not None and (p1 not in args.agents or p2 not in args.agents):
            continue
        players.add(p1)
        players.add(p2)
        for file_name in sorted(os.listdir(os.path.join(args.tournament, dir_name)))[::]:
            if not file_name.endswith(".json"):
                continue
            data = json.load(open(os.path.join(args.tournament, dir_name, file_name)))
            if matches_on_map[p1, p2][data["map"]] >= 5:
                continue
            if args.map is None or data["map"] in args.map:
                wins, games = matches[p1, p2]
                if data["winner"] == 1:
                    matches[p1, p2] = wins+1, games+1
                    matches_on_map[p1, p2][data["map"]] += 1
                    if file_name.endswith("inferred-winner.json"):
                        inferred += 1
                elif data["winner"] == -1 or data["winner"] == 2:
                    matches[p1, p2] = wins, games+1
                    matches_on_map[p1, p2][data["map"]] += 1
                    if file_name.endswith("inferred-winner.json"):
                        inferred += 1
                elif data["winner"] is None:
                    fail_count += 1
                else:
                    assert False, data
    assert players
    assert matches
    print(f"Found {sum(games for wins, games in matches.values())} games ({fail_count} failed, {inferred} inferred)")
    players = sorted(players)
    if args.agents:
        players = args.agents
    short_names = []
    short_indices = []
    for i, player in enumerate(players):
        player_dict[player] = i
        if True:
            short_names.append(player)
            short_indices.append(i)

    results = mcmc_sample(matches, args.samples, skip_every=args.skip_every, burn_in=args.burn_in)
    if args.elo_peg:
        results -= results[:,player_dict[args.elo_peg]][:,None]
    else:
        results -= results[:,short_indices].mean(1)[:,None]

    print("Games Played:")
    tbl = []
    win_tbl = np.zeros((len(players), len(players)))
    games_tbl = np.zeros((len(players), len(players)))
    for i, p1 in enumerate(players):
        assert player_dict[p1] == i
        row = [p1]
        player_wins = player_games = 0
        for j, p2 in enumerate(players):
            if i == j:
                row.append("")
                continue
            assert player_dict[p2] == j
            
            wins = games = 0
            if (p1, p2) in matches:
                w, g = matches[p1, p2]
                wins += w
                games += g
            if (p2, p1) in matches:
                w, g = matches[p2, p1]
                wins += g - w
                games += g
            player_wins += wins
            player_games += games
            row.append(f"{wins:3d} / {games:3d}")
            win_tbl[i, j] = wins
            games_tbl[i, j] = games
            # print(f"  {i} vs {j}: {win_probs.mean():.1f} ± {win_probs.std():.1f}%")
        row.append(f"{player_wins:4d} / {player_games:4d}")
        tbl.append(row)
    print(tb.tabulate(tbl, disable_numparse=True, headers=list(player_dict)+["total"]))
    print()

    print("Elo:")
    tbl = []
    for i, player in enumerate(player_dict):
        #if 'fast' in player: continue
        assert player_dict[player] == i
        temp = results[:,i] / args.elo_factor + args.elo_base
        # print(f"  {i}: {temp.mean():.3f} ± {temp.std():.3f}")
        tbl.append([player, f"{temp.mean():6.1f} ± {args.std * temp.std():4.1f}"])
    print(tb.tabulate(tbl, disable_numparse=True))
    print()

    print("Win Prob:")
    tbl = []
    win_prob_tbl = np.zeros((len(players), len(players)))
    for i, p1 in enumerate(players):
        assert player_dict[p1] == i
        row = [p1]
        for j, p2 in enumerate(players):
            if i == j:
                row.append("")
                continue
            assert player_dict[p2] == j

            exp_score_1 = np.exp(results[:, i])
            exp_score_2 = np.exp(results[:, j])
            win_probs = 100 * exp_score_1 / (exp_score_1 + exp_score_2)
            if True:
                row.append(f"{win_probs.mean():4.1f} ± {args.std * win_probs.std():.1f}%")
            win_prob_tbl[i, j] = win_probs.mean()/100
            # print(f"  {i} vs {j}: {win_probs.mean():.1f} ± {win_probs.std():.1f}%")
        tbl.append(row)
    print(tb.tabulate(tbl, disable_numparse=True, headers=list(short_names)))
    print()

    print("Probability of Superiority:")
    tbl = []
    for i, p1 in enumerate(players):
        assert player_dict[p1] == i
        row = [p1]
        for j, p2 in enumerate(players):
            if i == j:
                row.append("")
                continue
            assert player_dict[p2] == j
            temp = 100*(results[:,i] > results[:, j])
            row.append(f"{temp.mean():4.1f}%")
            # print(f"  {i} vs {j}: {temp.mean():.1f}%")
        tbl.append(row)
    print(tb.tabulate(tbl, disable_numparse=True, headers=list(short_names)))
    print()

    print("Probability of Best:")
    tbl = []
    for i, player in enumerate(players):
        assert player_dict[player] == i
        temp = 100*(results[:,i] >= results[:,short_indices].max(1))
        # print(f"  {i}: {temp.mean():.1f}%")
        tbl.append([player, f"{temp.mean():4.1f}%"])
    print(tb.tabulate(tbl, disable_numparse=True))
    print()

    exp_win_tbl = games_tbl * win_prob_tbl
    diff_squared = (exp_win_tbl - win_tbl)**2
    stat_tbl = np.where(
        games_tbl > 0, diff_squared / exp_win_tbl, 0
    )
    chi_stat = stat_tbl.sum()
    df = (games_tbl > 0).sum() // 2 - 1
    from scipy import stats
    print("p =", 1 - stats.chi2.cdf(chi_stat, df))
    

