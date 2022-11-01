import numpy as np
import pickle as pkl
import pdb
import json
import random
import os
from tqdm import tqdm
'''
create map pidx to all blocks pidx
'''
def create_map(passage_ids_new,old_pid2pidx):
    newpidx2oldpidx = []
    for id_new in passage_ids_new:
        newpidx2oldpidx.append(old_pid2pidx[id_new])

    return newpidx2oldpidx

if __name__ == '__main__':

    passage_ids_new_path = '/home/xbli/ConvDR-main/datasets_passage_set_s/datasets_bm25_top5/passage_ids.pickle'
    passage_ids_old_path = '/home/xbli/ConvDR-main/datasets/raw/or-quac/passage_ids.pkl'


    with open(passage_ids_new_path ,'rb') as handle:
        passage_ids_new = pkl.load(handle)

    with open(passage_ids_old_path, 'rb') as handle:
        passage_ids_old = pkl.load(handle)

    old_pid2pidx = {id:idx for idx,id in enumerate(passage_ids_old)}
    newpidx2oldpidx = create_map(passage_ids_new,old_pid2pidx)
    newpidx2oldpidx = np.array(newpidx2oldpidx)
    with open('datasets_bm25_top5/newpidx2oldpidx.pickle','wb') as handle:
        pkl.dump(newpidx2oldpidx,handle)