import os
from typing import List, Dict

import torch
import torch.nn as nn
import numpy as np

from switchnet.model import SuperEncoder
from slopy.loader import SeqSlope

def makeref(seq : SeqSlope, Seqslopes : Dict[str, SeqSlope]):
    ref = {}
    for target in seq.path:
        cluster = Seqslopes[target].cluster_seq 
        ref[target.split('_')[0]] = [Seqslopes[k].oh_seq for k in (cluster+[target])]
    return ref

def make_traintest(Seqslopes : Dict[str, SeqSlope], prop : float = 0.8):
    train, test = {}, {}
    for k, seq in Seqslopes.items():
        if np.random.random() > prop:
            test[k] = seq
        else:
            train[k] = seq
    return train, test

def train_seq(traindata : Dict[str, SeqSlope], Seqslopes : Dict[str, SeqSlope], testdata : Dict[str, SeqSlope], fullnet : SuperEncoder):
    for k, seq in traindata.items():
        models, optimizers = fullnet.make_Encoder(seq.path)
        ref = makeref(seq=seq, Seqslopes=Seqslopes)

        for idx, model in enumerate(models):
            opt = optimizers[idx]
            opt.zero_grad()
            
            x_hat = model(seq.oh_seq)
            print(x_hat.shape)

            device = x_hat.device
            loss = torch.tensor([0])
            
            loss.backward()
            opt.step()
            return


def main_train(Seqslopes : Dict[str, SeqSlope], fullnet : SuperEncoder):
    train, test = make_traintest(Seqslopes=Seqslopes)
    train_seq(traindata=train, Seqslopes=Seqslopes, testdata=test, fullnet=fullnet)