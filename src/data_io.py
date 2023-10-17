import os
import pickle
import numpy as np
from scipy import sparse
import pandas as pd

datasets = 'homo'
namespace = ['mf','bp','cc']
def df2fasta(df,dir):
    fasta = open(dir, 'w')
    for id,row in enumerate(df.itertuples()):
        p_id = row.proteins
        seq = row.sequences
        fasta.writelines('>' + p_id + '\n')
        fasta.writelines(seq.upper()+'\n')

    fasta.close()


train_df = pd.read_pickle(f'data/{datasets}/train_data.pkl')
test_df = pd.read_pickle(f'data/{datasets}/test_df.pkl')
df2fasta(test_df,f'data/{datasets}/test.fa')