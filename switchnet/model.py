from typing import Dict, List, Tuple, Any
import itertools as it

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as func

import switchnet.loader as loader

class SuperEncoder:
    def __init__(self, layers_list : List[int], q : int = 4):
        self.SuperEncoder = {f"{idx1}_{idx2}":UnitEncoder(ln_in=idx1*q, ln_out=idx2*q) for idx1, idx2 in it.combinations(layers_list, 2) if idx1 > idx2}

    def make_Encoder(self, seqpath: List[int]) -> Tuple[List[Any], List[optim.Adam]]:
        encoders = []
        optimizers = []
        for idx in range(len(seqpath)-1):
            encoder = self.SuperEncoder[f"{seqpath[idx].split("_")[0]}_{seqpath[idx+1].split("_")[0]}"]
            encoders.append(encoder)
            optimizers.append(optim.Adam(encoder.parameters(), lr=0.005))

        return encoders, optimizers

class UnitEncoder(nn.Module):
    def __init__(self, ln_in : int, ln_out : int):
        """ 
        ln_in: length of the sequences entering the layer
        ln_out: length of the sequences exiting the layer
        """
        super(UnitEncoder, self).__init__()
        self.linear1 = nn.Linear(ln_in, ln_out)
        self.linear2 = nn.Linear(ln_out, ln_out)

    def forward(self, x : torch.Tensor, num_samples : int = 1000) -> torch.Tensor:
        device = next(self.parameters()).device  # Récupère l'appareil du modèle
        x = x.to(device)
        _, q = x.shape

        x = x.ravel()
        x = func.relu(self.linear1(x))
        x = func.relu(self.linear2(x))

        l = len(x)
        x = x.reshape(l//q, q)
        x = torch.multinomial(torch.softmax(x, dim=-1), num_samples=num_samples, replacement=True)
        return x

def load_model(batches : Dict[int, loader.SeqSlope]):
    all_edges = []
    for _, seq in batches.items():
        for node in seq.path:
            layer = int(node.split('_')[0])
            all_edges.append(layer)

    layers = sorted(set(all_edges), reverse=True)
    supEncoder = SuperEncoder(layers_list=layers)

    return supEncoder