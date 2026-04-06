import os
import pandas as pd
import math
from collections import defaultdict
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def pcss_mapper(v, neighbours, output_dict):
    """
    Emits candidate edges (u,v) with adjacency lists for similarity computation.
    Skips self-loops.
    """

    for n in neighbours:
        if v < n:
            key = (v,n)
        elif v > n:
            key = (n,v)
        else:         #skip self-loop (when n=v)
            continue
        output_dict[key].append(neighbours)


def structural_similarity(set1, set2, epsilon):
    """
    Helper function for PCSS Reducer.
    Computes PSCAN structural similarity between two nodes.
    Uses early pruning for efficiency.
    """
    degree1 = len(set1)
    degree2 = len(set2)

    # avoid division by zero
    if degree1 == 0 or degree2 == 0:
        return 0

    # upper bound pruning (optimization)
    max_possible = min(degree1, degree2) / math.sqrt(degree1 * degree2)
    if max_possible < epsilon:
        return 0

    # compute intersection size
    intersection = len(set1.intersection(set2))

    return intersection / math.sqrt(degree1 * degree2)


def pcss_reducer(edge, lists, output_dict, epsilon):
    """
    Computes similarity for each edge and filters by similarity threshold (epsilon).
    """
    if len(lists) < 2:
        return  # need both endpoints' adjacency lists

    sigma = structural_similarity(lists[0], lists[1], epsilon)

    if sigma >= epsilon:
        output_dict[edge] = sigma


def lpcc_mapper(v, struct, output_dict):
    """
    Propagates labels from active nodes to their neighbours.
    """
    if struct["status"] == "active":
        for n in struct["adj_list"]:
            output_dict[n]["labels"].add(struct["label"])

    # always emit node structure
    output_dict[v]["struct"] = struct


def lpcc_reducer(v, data, output_dict):
    """
    Updates node label based on received labels (min-label propagation).
    """
    current_label = data["struct"]["label"]
    incoming_labels = data["labels"]

    # include own label when choosing minimum
    new_label = min(incoming_labels.union({current_label}))
    
    # update status
    new_status = "inactive" if new_label == current_label else "active"

    # update node struct with new status and label
    output_dict[v] = {
        "status": new_status,
        "label": new_label,
        "adj_list": data["struct"]["adj_list"]
    }


def compute_modularity(adj_list, clusters, deg, m):
    """
    Computes modularity for a given graph.
    Used to help determine optimal similarity threshold (epsilon).
    """
    Q = 0

    for nodes in clusters.values():
        length = len(nodes)
        for i, u in enumerate(nodes):
            for j in range(i + 1, length):
                v = nodes[j]
                A_ij = 1 if v in adj_list[u] else 0
                Q += A_ij - (deg[u] * deg[v]) / (2 * m)

    Q /= m   # divide by m, not 2m, since each bidirectional edge only counted once
    return Q


def load_graph(file_path):
    """
    Loads graph in either CSV or NSE file format.
    """
    dict_adj_list = defaultdict(set)
    edges = []

    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".csv":
        df = pd.read_csv(file_path)
        df = df[['source', 'target']]

        for row in df.itertuples(index=False):
            u, v = row.source, row.target
            dict_adj_list[u].add(v)
            dict_adj_list[v].add(u)
            if u != v:
                edges.append(tuple(sorted((u, v))))

    elif ext == ".nse":
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                parts = line.split("\t")
                if len(parts) < 2:
                    continue

                u, v = int(parts[0]), int(parts[1])
                dict_adj_list[u].add(v)
                dict_adj_list[v].add(u)

                if u != v:
                    edges.append(tuple(sorted((u, v))))
    else:
        raise ValueError("Unsupported file type: must be .csv or .nse")

    # add self-loops for PSCAN
    for v in dict_adj_list:
        dict_adj_list[v].add(v)

    edges = list(set(edges))
    m = len(edges)
    deg = {v: len(neigh) - 1 for v, neigh in dict_adj_list.items()}

    print(f"Graph loaded: {len(dict_adj_list)} nodes, {m} edges")

    return dict_adj_list, edges, deg, m


def run_pscan(epsilon, dict_adj_list):
    """
    Full pipeline for MapReduce PSCAN.
    """

    # --- PCSS ---
    pcss_map_dict = defaultdict(list)

    for v, neighbours in dict_adj_list.items():
        pcss_mapper(v, neighbours, pcss_map_dict)

    pcss_reduce_dict = {}

    for e, lists in pcss_map_dict.items():
        pcss_reducer(e, lists, pcss_reduce_dict, epsilon)

    # --- build filtered graph ---
    new_adj_list = defaultdict(set)

    for (u, v) in pcss_reduce_dict:
        new_adj_list[u].add(v)
        new_adj_list[v].add(u)


    # --- LPCC initialization ---
    lpcc_list = {}

    for v, neighbours in new_adj_list.items():
        lpcc_list[v] = {
            "status":"active",
            "label":v,
            "adj_list":neighbours
        }

    # --- LPCC iterations ---
    max_iterations = 250
    iteration = 0
    num_active = 1

    while num_active and iteration < max_iterations:
        iteration += 1

        lpcc_map_dict = defaultdict(lambda: {"labels": set(), "struct": None})

        for v, struct in lpcc_list.items():
            lpcc_mapper(v, struct, lpcc_map_dict)

        updated_lpcc_list = {}
        for v, data in lpcc_map_dict.items():
            lpcc_reducer(v, data, updated_lpcc_list)

        lpcc_list = updated_lpcc_list

        num_active = sum(1 for x in lpcc_list.values() if x["status"] == "active")

    if num_active == 0:
        print(f"Algorithm converged after {iteration} iterations")
    else:
        print(f"Maximum iterations ({max_iterations}) reached without full convergence. Active nodes: {num_active}")

    # --- build clusters ---
    clusters = defaultdict(list)
    for v, struct in lpcc_list.items():
        clusters[struct["label"]].append(v)

    return clusters, pcss_reduce_dict


def load_ground_truth(file_path):
    """
    Loads ground truth clusters.
    Used to evaluate NMI and ARI scores.
    """
    gt_labels = {}

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            v, c = int(parts[0]), int(parts[1])
            gt_labels[v] = c

    return gt_labels


def main():
    """
    Full pipeline for benchmarking performance tests.
    Also used to determine the optimal similarity threshold for our Amazon co-purchasing dataset (without ARI/NMI evaluation).
    """

    graph_path = "/Users/kaitlynn/Documents/UVic/CSC502/Project/LFR_benchmark/Graph-40k.nse"       # option: .csv or .nse
    gt_path = "/Users/kaitlynn/Documents/UVic/CSC502/Project/LFR_benchmark/Graph-40k.nmc"

    # graph_path = "/Users/kaitlynn/Documents/UVic/CSC502/Project/edges_co.csv"

    dict_adj_list, edges, deg, m = load_graph(graph_path)

    eps_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    # eps_values = [0.38, 0.39, 0.40, 0.41, 0.42]
    results = []

    for eps in eps_values:
        print(f"\nRunning PSCAN with epsilon={eps}")

        clusters, _ = run_pscan(eps, dict_adj_list)
        modularity = compute_modularity(dict_adj_list, clusters, deg, m)

        results.append((eps, len(clusters), modularity))

        print("clusters:", len(clusters))
        print("modularity:", modularity)

    best = max(results, key=lambda x: x[2])

    print("\nBest epsilon:", best[0])
    print("Clusters:", best[1])
    print("Modularity:", best[2])

    # --- ARI and NMI evaluation ---
    gt_labels = load_ground_truth(gt_path)

    clusters, _ = run_pscan(best[0], dict_adj_list)

    pred_labels = {}
    for cluster_id, nodes in clusters.items():
        for n in nodes:
            pred_labels[n] = cluster_id

    y_true, y_pred = [], []
    for node in gt_labels:
        if node in pred_labels:
            y_true.append(gt_labels[node])
            y_pred.append(pred_labels[node])

    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)

    print(f"\nFinal Results: epsilon={best[0]} | ARI={ari:.4f} | NMI={nmi:.4f}")


if __name__ == "__main__":
    main()