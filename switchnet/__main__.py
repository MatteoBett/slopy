import os

import switchnet.loader as loader 
import switchnet.model as model
import switchnet.train as train
import slopy

def main(params_path : str, family_data_path : str, params_seqslope : str):
    params, Seqslopes = loader.load_params(path_params=params_path,path_seqslope=params_seqslope)
    Seqslopes = loader.stream_slopes(Seqslopes)

    fullnet = model.load_model(batches=Seqslopes)

    train.main_train(Seqslopes=Seqslopes, fullnet=fullnet)

if __name__ == "__main__":
    params_path = r'/home/mbettiati/LBE_MatteoBettiati/code/slopy/output/outdata/Azoarcus.npy'
    params_seqslope = r'/home/mbettiati/LBE_MatteoBettiati/code/slopy/output/outdata/Azoarcus_seqslope.pkl'
    family_data_path = r'/home/mbettiati/LBE_MatteoBettiati/code/slopy/data/test/Azoarcus/Azoarcus.fasta'
    main(params_path=params_path, family_data_path=family_data_path, params_seqslope=params_seqslope)