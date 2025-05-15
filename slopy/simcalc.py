import sys
from typing import Dict, List

import numpy as np
from collections import defaultdict, deque

import slopy.loader as loader
import slopy.utils as utils

sys.setrecursionlimit(100000)

def truth_table(A : int, B : int, A_ss : int, B_ss : int):
    if A == B and A_ss == B_ss:
        return 1
    elif A == B and A_ss != B_ss:
        return 0.25
    elif A != B and A_ss == B_ss:
        return 0.5
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

def fill_clusters(distmat : np.matrix, batch : List[loader.SeqSlope], threshold : float = 0.1):
    nrows, _ = distmat.shape
    for row in range(nrows):
        sim = np.argwhere((distmat[row,:] > threshold)).squeeze().tolist()
        if not isinstance(sim, list):
            sim = [sim]
        
        for i in sim:
            batch[row].cluster_seq.append(batch[i].node)
            batch[i].cluster_seq.append(batch[row].node)
            

def stream_batches_smaller(batches : Dict[int, List[loader.SeqSlope]], size_fam : int, threshold : int = 0.2, rep_thr : float = 0.01):
    A = np.zeros((size_fam, size_fam))
    keys = list(batches.keys())
    for index, (size, batch) in enumerate(batches.items()):
        utils.progressbar(iteration=index+1, total=len(batches))
        for key in keys[index+1:len(keys)]:
            if ((len(batches[key]) + len(batch))/size_fam < rep_thr) and (keys[index+1] == key):
                for i in range(0, len(batch)):
                    for j in range(0, len(batches[key])):
                        if key not in batches[size][i].targets.keys():
                            batches[size][i].targets[key] = [batches[key][j].node]
                            if size not in batches[size][i].targets.keys():
                                batches[key][j].ancestor[size] = [batch[i].node]
                            else:
                                batches[key][j].ancestor[size].append(batch[i].node)
                        else:
                            batches[size][i].targets[key].append(batches[key][j].node)
                            if size not in batches[size][i].targets.keys():
                                batches[key][j].ancestor[size] = [batch[i].node]
                            else:
                                batches[key][j].ancestor[size].append(batch[i].node) 
                    
            else:
                for i in range(0, len(batch)):
                    tmp = []
                    for j in range(0, len(batches[key])):
                        dist_ij = jaccard_distance(lstA=batch[i].encoded, 
                                                    lstB=batches[key][j].encoded,
                                                    A_ss=batch[i].ss_encoded,
                                                    B_ss=batches[key][j].ss_encoded)

                        A[batch[i]._id, batches[key][j]._id] = np.exp(dist_ij) + len(batch[i].seq)*0.01
                        A[batches[key][j]._id, batch[i]._id] = np.exp(dist_ij) + len(batches[key][j].seq)*0.01
                        tmp.append(dist_ij)
                        if dist_ij > threshold:
                            
                            if key not in batches[size][i].targets.keys():
                                batches[size][i].targets[key] = [batches[key][j].node] + batches[key][j].cluster_seq
                                if size not in batches[size][i].targets.keys():
                                    batches[key][j].ancestor[size] = [batch[i].node]
                                else:
                                    batches[key][j].ancestor[size].append(batch[i].node)
                            else:
                                batches[size][i].targets[key].append(batches[key][j].node) 
                                batches[size][i].targets[key] += batches[key][j].cluster_seq
                                if size not in batches[size][i].targets.keys():
                                    batches[key][j].ancestor[size] = [batch[i].node]
                                else:
                                    batches[key][j].ancestor[size].append(batch[i].node)

                    batches[size][i].targets_sim[key] = np.mean(tmp)
            

    D = A.sum(axis=0)
    L = np.identity(size_fam)-D**(-1/2)*A*D**(1/2)
    return L

def make_laplacian(batches : Dict[int, List[loader.SeqSlope]], Laplacian : np.matrix, path_output: str):
    for index, (_, batch) in enumerate(batches.items()):
        for elt in batch:
            i = elt._id
            for key, seqs in elt.targets.items():
                Laplacian[i,i] += len(seqs)
            for key, seqs in elt.ancestor.items():
                Laplacian[i,i] += len(seqs)

    np.save(file=path_output, arr=Laplacian)
    return Laplacian


def reformat(batches : Dict[int, List[loader.SeqSlope]]):
    nw = {}
    for key_size, batch in batches.items():
        for idx, seq in enumerate(batch):
            for nxt_size, target in seq.targets.items():
                seq.targets[nxt_size] = sorted(list(set(target)))
            for prev_size, ancestry in seq.ancestor.items():
                seq.ancestor[prev_size] = sorted(list(set(ancestry)))
            
            nw[f"{key_size}_{seq._id}"] = seq

    return batches, nw

def get_seqpaths(batches : Dict[int, List[loader.SeqSlope]], target_list:List[int], next_size : int, path : list = []):
    if len(target_list) == 0:
        return path, next_size
    else:
        for index, target in enumerate(target_list):
            nextlist = batches[next_size][index].targets
            if len(path) == 0:
                first = [(next_size, target)]
                path.append(first)
                next_size = max(batches[next_size][index].targets.keys())
                return get_seqpaths(batches=batches, target_list=nextlist, next_size=next_size, path=first)
            
            else:
                path.append((next_size, target))
                return get_seqpaths(batches=batches, target_list=nextlist, next_size=next_size, path=path)

def stream_seq(batches : Dict[int, List[loader.SeqSlope]], size_fam : int):
    Seqpath = {}

def avg_path_len(G : Dict[str, loader.SeqSlope]):
    degrees = np.zeros((len(G,)))

    for key, seq in G.items():
        degrees[seq._id] = len(seq.ancestor) + len(seq.targets)

    topo = []
    queue = deque([key for key in G if len(G[key].targets) == 0])
    pass