import os

import RNA

def get_ss(seq : str):
    return RNA.fold(seq)

def main_struct(seq : str):
    ss = get_ss(seq=seq)