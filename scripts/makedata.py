import os

def makefasta(filein : str, fileout : str):
    with open(file=filein, mode='r') as content:
        content = content.readlines()

    with open(file=fileout, mode='w') as out:
        for i, seq in enumerate(content):
            seq = seq.split(';')[0]
            out.write(f">seq_{i}\n{seq}\n")

makefasta(filein=r'/home/mbettiati/LBE_MatteoBettiati/code/slopy/data/raw/Artificial/ss_same-A1_var-length.out',
          fileout=r'/home/mbettiati/LBE_MatteoBettiati/code/slopy/data/raw/Artificial/Artificial.fasta')