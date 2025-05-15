import re
from dataclasses import dataclass
from typing import List, Dict
import pickle

from Bio import SeqIO
import numpy as np
import torch

from slopy.loader import SeqSlope

def load_params(path_params : str, path_seqslope : str):
    with open(path_seqslope, "rb") as f:
        Seqslope = pickle.load(f)

    Lap = np.load(file=path_params)
    return Lap, Seqslope


def stream_slopes(Seqslopes : Dict[str, SeqSlope], device : str = "cpu") -> Dict[int, SeqSlope]:
    """
    Batches the sequences contained in the family file by their size. 
    """
    for _, seq in Seqslopes.items():
        seq.oh_seq = one_hot(encode_sequence(seq=seq.seq), device=device)

    return Seqslopes

@torch.jit.script
def _one_hot(x: torch.Tensor, num_classes: int = -1, dtype: torch.dtype = torch.int32):

    if num_classes < 0:
        num_classes = x.max() + 1

    res = torch.zeros(x.shape[0],  num_classes, device=x.device, dtype=dtype)
    tmp = torch.meshgrid(
        torch.arange(x.shape[0], device=x.device),
        indexing="ij",
    )

    index = (tmp[0], x)
    values = torch.ones(x.shape[0], device=x.device, dtype=dtype)
    res.index_put_(index, values)
    
    return res


def one_hot(mat: List[List[int]] | torch.Tensor, device : str = 'cpu',  num_classes: int = -1, dtype: torch.dtype = torch.float32):
    """
    A fast one-hot encoding function faster than the PyTorch one working with torch.int32 and returning a float Tensor.
    Works only for 2D tensors.
    """
    if isinstance(mat, List):
        mat = torch.tensor(mat, device=device, dtype=torch.int32)
    return _one_hot(mat, num_classes, dtype)

def encode_sequence(seq :str):
    dico = {'A':0,'U':1,'C':2,'G':3}
    new = []
    for nuc in seq:
        try :
            new.append(dico[nuc])
        except KeyError:
            return False
    
    return new
