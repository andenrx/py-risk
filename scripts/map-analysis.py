import risk
import argparse
import tabulate as tb

metric_list = {
    "Territories": lambda mapstruct: len(list(mapstruct.graph.vs)),
    "Bonuses":     lambda mapstruct: len(mapstruct.bonuses),
    "Edges":       lambda mapstruct: len(list(mapstruct.graph.es)),
    "Max Income":  lambda mapstruct: sum(b.value for b in mapstruct.bonuses),

    "Avg Distance":lambda mapstruct: f"{mapstruct.graph.average_path_length():.4f}",
    "Radius":      lambda mapstruct: mapstruct.graph.radius(),
    "Diameter":    lambda mapstruct: mapstruct.graph.diameter(),
    "Clustering":  lambda mapstruct: f"{mapstruct.graph.transitivity_undirected():.4f}",
    "Bonus Clust.":lambda mapstruct: f"{bonus_clustering(mapstruct):.4f}",
}
# average bonus size
# average income per bonus
# average degree
# modularity on bonuses

def __main__(mapids, metrics_to_use, output_format="simple"):
    tbl = []
    for mapid in mapids:
        mapstruct = risk.utils.load_mapstruct(mapid, "cached-maps")
        row = [mapid.name]
        for metric in metrics_to_use:
            row.append(metric_list[metric](mapstruct))
        tbl.append(row)
    print(tb.tabulate(tbl, headers=["Name"] + metrics_to_use, tablefmt=output_format))

def bonus_clustering(mapstruct):
    crossing_edges = 0
    for e in mapstruct.graph.es:
        for bonus in mapstruct.bonuses:
            if e.source in bonus.terr and e.target not in bonus.terr \
                    or e.source not in bonus.terr and e.target in bonus.terr:
                crossing_edges += 1
                break
    return 1 - crossing_edges / len(list(mapstruct.graph.es))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mapid", nargs='+', type=lambda name: risk.api.MapID[name],
        help=f"Maps to analyze, choices: " \
            + ", ".join(f"'{map.name}'" for map in risk.api.MapID)
    )
    parser.add_argument("--metrics", default=list(metric_list), type=lambda s: s.split(","))
    parser.add_argument("--format", default="simple")
    args = parser.parse_args()
    __main__(args.mapid, args.metrics, output_format=args.format)

