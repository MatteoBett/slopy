import os
from typing import Dict, List

import numpy as np

import slopy.loader as loader
import slopy.utils as utils

def truth_table(A : int, B : int, A_ss : int, B_ss : int):
    if A == B and A_ss == B_ss:
        return 0
    elif A == B and A_ss != B_ss:
        return 1
    elif A != B and A_ss == B_ss:
        return 0
    elif A != B and A_ss != B_ss:
        return 0
    
def jaccard_distance(lstA : List[int], lstB: List[int], A_ss: List[int], B_ss: List[int]):
    idxA = 0
    idxB = 0

    intersection = 0
    union = 0

    while idxA < len(lstA) and idxB < len(lstB):
        union += 1
        if lstA[idxA] == lstB[idxB] or A_ss == B_ss:
            intersection += truth_table(A=lstA[idxA], B=lstB[idxB], A_ss=A_ss[idxA], B_ss=B_ss[idxB])
            idxA += 1
            idxB += 1
        elif lstA[idxA] < lstB[idxB]:
            idxA += 1
        else:
            idxB += 1

    union += len(lstA) - idxA
    union += len(lstB) - idxB

    return intersection / union


def stream_batches_similarity(batches : Dict[int, List[loader.SeqSlope]]):
    for index, (size, batch) in enumerate(batches.items()):
        utils.progressbar(iteration=index+1, total=len(batches))
        distmat = np.zeros((len(batch), len(batch)))
        for i in range(0, len(batch)):
            for j in range(i, len(batch)):
                if i == j:
                    distmat[i,j] = 0
                
                else:
                    distmat[i,j] = jaccard_distance(lstA=batch[i].encoded, 
                                                    lstB=batch[j].encoded,
                                                    A_ss=batch[i].ss_encoded,
                                                    B_ss=batch[j].ss_encoded)
        yield size, distmat

def fill_clusters(distmat : np.matrix, batch : List[loader.SeqSlope], threshold : float = 0.3):
    nrows, _ = distmat.shape
    for row in range(nrows):
        sim = np.argwhere((distmat[row,:] > threshold)).squeeze().tolist()
        batch[row].cluster_seq += sim if isinstance(sim, list) else [sim]

def stream_batches_smaller(batches : Dict[int, List[loader.SeqSlope]], threshold : int = 0.3):
    keys = list(batches.keys())
    for index, (size, batch) in enumerate(batches.items()):
        utils.progressbar(iteration=index+1, total=len(batches))
        for key in keys[index+1:len(keys)]:
            for i in range(0, len(batch)):
                for j in range(0, len(batches[key])):
                    dist_ij = jaccard_distance(lstA=batch[i].encoded, 
                                                lstB=batches[key][j].encoded,
                                                A_ss=batch[i].ss_encoded,
                                                B_ss=batches[key][j].ss_encoded)
                
                    if dist_ij > threshold:
                        if key not in batches[size][i].targets.keys():
                            batches[size][i].targets[key] = [j] + batches[key][j].cluster_seq
                        else:
                            batches[size][i].targets[key].append(j)



        