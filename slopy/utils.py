from typing import List, Dict
import copy as cp

import slopy.loader as loader

import numpy as np


def progressbar(iteration, total, prefix = '', suffix = '', filler = '█', printEnd = "\r") -> None:
    """ Show a progress bar indicating downloading progress """
    percent = f'{round(100 * (iteration / float(total)), 1)}'
    add = int(100 * iteration // total)
    bar = filler * add + '-' * (100 - add)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    if iteration == total: 
        print()


def topo_sort(batches: Dict[str, loader.SeqSlope]):
    """
    Kahn's algorithm for topological sorting of SeqSlope graph.
    """
    import copy as cp

    copy_batches = cp.deepcopy(batches)
    L = []
    S = [seq.node for seq in copy_batches.values() if sum(len(v) for v in seq.ancestor.values()) == 0]

    while S:
        new_node = S.pop(0)
        L.append(new_node)

        size, _ = new_node.split('_')
        for nxt in copy_batches[new_node].targets.get(int(size), []):
            try:
                copy_batches[nxt].ancestor[int(size)].remove(new_node)
            except (KeyError, ValueError):
                pass

            total_ancestor = sum(len(v) for v in copy_batches[nxt].ancestor.values())
            if total_ancestor == 0:
                S.append(nxt)

    return L, copy_batches


def topo_sort(batches: Dict[str, loader.SeqSlope]):
    """
    Kahn's algorithm for topological sorting of SeqSlope graph.
    """
    import copy as cp

    copy_batches = cp.deepcopy(batches)
    L = []
    S = [seq.node for seq in copy_batches.values() if sum(len(v) for v in seq.ancestor.values()) == 0]

    while S:
        new_node = S.pop(0)
        L.append(new_node)
        size, _ = new_node.split('_')
        size = int(size)
        for _, nxt_list in copy_batches[new_node].targets.items():
            for nxt in nxt_list:
                ancestors = copy_batches[nxt].ancestor.get(size, [])

                if new_node in ancestors:
                    copy_batches[nxt].ancestor[size].remove(new_node)

                total_ancestor = sum(len(v) for v in copy_batches[nxt].ancestor.values())
                if total_ancestor == 0 and nxt not in S:
                    S.append(nxt)


    return L, copy_batches


def find_longest_path(batches: Dict[str, loader.SeqSlope]):
    """
    Finds the longest path (most edges) in the DAG.
    Returns the path as a list of node names.
    """
    S = [seq.node for seq in batches.values() if sum(len(v) for v in seq.targets.values()) != 0] 
    topo_order, _ = topo_sort(batches)

    for i, s in enumerate(S):
        progressbar(iteration=i, total=len(S))
        dist = {node: -np.inf for node in topo_order}
        pred = {node: None for node in topo_order}
        dist[s] = 0

        for u in topo_order:
            if dist[u] == -np.inf:
                continue

            for size, target in batches[u].targets.items():
                for t in target:
                    if dist[t] < dist[u] + 1:
                        dist[t] = dist[u] + 1
                        pred[t] = u

        # Find node with longest distance
        end_node = max(dist, key=dist.get)
        path = []
        while end_node is not None:
            path.append(end_node)
            end_node = pred[end_node]
        path.reverse()
        batches[s].path = path
        batches[s].lenpath = len(path)
    return batches
