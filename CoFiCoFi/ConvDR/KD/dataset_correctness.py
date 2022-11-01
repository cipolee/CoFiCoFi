
import json
import logging
import math
import collections
import linecache
import numpy as np
from io import open
from tqdm import tqdm
from torch.utils.data import Dataset
import pdb
import pickle as pkl
from transformers.tokenization_bert import BasicTokenizer, whitespace_tokenize
logger = logging.getLogger(__name__)
def get_passage(i):
    # def get_passage(i, args):
    line = linecache.getline('/orconvqa/all_blocks.txt', i + 1)
    line = json.loads(line.strip())
    return line
if __name__ == '__main__':
    logger.info('read q_p_s.pickle')
    with open('/ConvDR-main/KD/q_p_s1.pickle','rb') as handle:
        q_p_s = pkl.load(handle)

    logger.info('read qrels.txt')
    with open('/orconvqa/qrels.txt','r') as handle:
        qrels = json.load(handle)
    # pdb.set_trace()
    for qid , pids in tqdm(q_p_s.items()):
        
        # if len(pids)<40:
        #     pdb.set_trace()
        
        # if get_passage()
        tmp_dict = {v:k for k,v in pids.items()}
        ppidx = tmp_dict[max(tmp_dict)]
        ppid = get_passage(ppidx)['id'] 
        if ppid!=list(qrels[qid].keys())[0]:
            pdb.set_trace()
        ######## 2022/4/22 正确性得到证明