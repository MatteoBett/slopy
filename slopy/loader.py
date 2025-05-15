import os, re
from dataclasses import dataclass
from typing import List, Generator, Any, Dict

from Bio import SeqIO
import RNA
import torch

import slopy.struct as struct

@dataclass
class SeqSlope:
    header : str
    _id : int
    seq : str
    oh_seq : torch.Tensor
    secondary_structure : str
    encoded : List[int] | List[str]
    ss_encoded : List[int] | List[str]
    cluster_seq : List[int]
    ancestor: Dict[int, List[int]]
    targets : Dict[int, List[int]]
    targets_sim : Dict[int, float]
    node:str
    path: List[str]
    lenpath : int


def get_k(seq : str, ss : str):
    L1_ss = RNA.abstract_shapes(ss, 1)
    helix_count = len(re.sub("_", "", L1_ss))/2
    return int(len(seq)//helix_count)

def family_stream(family_dir : str):
    """ Yield the output of load_msa function for each family directory """
    for family_file in os.listdir(family_dir):
        yield family_file, os.path.join(family_dir, family_file, f"{family_file}.fasta")


def stream_batches(family_dir : str) -> Generator[str, Dict[int, List[SeqSlope]], int]:
    """
    Batches the sequences contained in the family file by their size. 
    """
    for family_name, infile_path in family_stream(family_dir=family_dir):
        batches = {}
        k = []
        record_list = [record for record in SeqIO.parse(handle=infile_path, format="fasta-pearson")]
        record_list.sort(key=lambda x:len(x.seq)-str(x.seq).count('-'), reverse=True)

        for index, record in enumerate(record_list):
            
            seq = str(record.seq)
            size = len(seq) - seq.count('-')
            if size not in batches.keys():
                seq = re.sub("-", "", seq)
                ss = struct.get_ss(seq=seq)[0]
                k.append(get_k(seq=seq, ss=ss))
                batches[size] = [SeqSlope(
                    header=record.description,
                    _id=index,
                    seq=seq,
                    secondary_structure=ss,
                    encoded=[],
                    ss_encoded=[],
                    cluster_seq=[],
                    ancestor={},
                    targets={},
                    targets_sim={},
                    node=f'{size}_{index}',
                    path=[],
                    lenpath=0,
                    oh_seq=torch.tensor([])
                )]
            else:
                seq = re.sub("-", "", seq)
                ss = struct.get_ss(seq=seq)[0]
                k.append(get_k(seq=seq, ss=ss))
                batches[size].append(SeqSlope(
                    header=record.description,
                    _id=index,
                    seq=seq,
                    secondary_structure=ss,
                    encoded=[],
                    ss_encoded=[],
                    cluster_seq=[],
                    ancestor={},
                    targets={},
                    targets_sim={},
                    node=f"{size}_{index}",
                    path=[],
                    lenpath=0,
                    oh_seq=torch.tensor([])
                ))
        yield family_name, batches, min(k), len(k)