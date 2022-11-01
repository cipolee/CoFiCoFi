import numpy as np
import pickle as pkl
import pdb
import json
import random
import os
from tqdm import tqdm
'''
only has passage id
'''
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def load_positive(qrels_path):
    '''
    qrels : len 40527
    format : {qid0:{pid0:1,pid1:1,...},....}
    '''
    positive_passages = set()
    qrels = {}

    with open(qrels_path) as handle:
        qrels = json.load(handle)
    # print('here')
    for qid,pid_labels in qrels.items():
        for pid , label in pid_labels.items():
            positive_passages.add(pid)
    return positive_passages

def load_high_confused(passage_ids_path,predict_files):
    '''
    passage_ids_path
    predict_files : list
    predict_files[0] : trec
    '''
    with open(passage_ids_path, 'rb') as handle:
        passage_ids = pkl.load(handle)
    
    high_confused = set()
    for predict_file in predict_files:
        passages_per_question = 0
        re = ''
        with open(predict_file,'r') as f:
            for line in tqdm(f.readlines()):
                
                qid=line.split(" ")[0]
                if re == '':
                    re = qid
                    passages_per_question += 1
                    pidx = int(line.split(" ")[2])
                    # pdb.set_trace()
                    pid = passage_ids[pidx]
                    high_confused.add(pid)
                elif re ==qid and passages_per_question < 30:
                    passages_per_question +=1
                    pidx = int(line.split(" ")[2])
                    pid = passage_ids[pidx]
                    high_confused.add(pid)
                elif re ==qid and passages_per_question > 30:
                    continue
                else:
                    re = qid
                    passages_per_question = 1
                    pidx = int(line.split(" ")[2])
                    pid = passage_ids[pidx]
                    high_confused.add(pid)
    return high_confused



                # if qid not in DR:
                #     DR[qid] = {line.split(" ")[2]:sigmoid(int(line.split(" ")[4])/200)}
                # else:
                #     DR[qid][line.split(" ")[2]] = sigmoid(int(line.split(" ")[4])/200)

def random_choice(passage_ids_path):
    '''
    choose 200W pids randomly
    later rewrited by 120W positives and remain 80W
    '''

    # pidxs = np.random.choice(np.arange(11377951), (2000000)) # 重复采样
    pidxs = np.random.choice(np.arange(11377951), (2000000),replace = False)

    with open(passage_ids_path, 'rb') as handle:
        passage_ids = pkl.load(handle)
    
    pids = passage_ids[pidxs]
    
    return pids

def merge(intergration,part,part1,rewrite_mark,update_ids):
    print('merging')
    increasing_idx  = rewrite_mark
    hash_set = set(intergration)

    a = 0
    part_set = set(part1)
    for i in tqdm(part):
        if i not in hash_set:
            while intergration[increasing_idx] in part_set:
                increasing_idx += 1
            hash_set.remove(intergration[increasing_idx])
            hash_set.add(i)
            update_ids.add(i)
            intergration[increasing_idx] = i
            increasing_idx += 1
            a += 1


    print(a)
    return intergration,increasing_idx,update_ids
def check_all_positive(qrels_path,passage_ids_path):
    with open(qrels_path,'rb') as handle:
        qrels= json.load(handle)
    with open(passage_ids_path,'rb') as handle:
        passage_ids= pkl.load(handle)
    print('passage_ids before len {}'.format(len(passage_ids)))

    passage_ids = set(passage_ids)
    no_match = 0
    print('passage_ids after len {}'.format(len(passage_ids)))
    for qid,pids in qrels.items():
        for pid ,score in pids.items():
            if pid not in passage_ids:
                # print('here')
                no_match+=1
            else:
                continue
    print(no_match)
if __name__ == '__main__':
    # passage_list = []
    
    set_seed(42)
    passage_ids_path='/home/xbli/ConvDR-main/datasets/raw/or-quac/passage_ids.pkl'
    qrels_path = '/home/xbli/ConvDR-main/thu_results/qrels.txt'


    # new_dataset_path = '/home/xbli/ConvDR-main/datasets_passage_set_s/datasets/'
    new_dataset_path = '/home/xbli/ConvDR-main/datasets_passage_set_s/datasets_bm25_top5/'



    predict_files = [new_dataset_path+file for file in os.listdir(new_dataset_path) if 'trec' in file]
    print(predict_files) 
    passage_ndarray = random_choice(passage_ids_path)
    print(len(passage_ndarray))
    print(len(set(passage_ndarray)))
    # pdb.set_trace()
    positive_pids = load_positive(qrels_path)
    passage_ndarray,increasing_idx,update_ids = merge(passage_ndarray,positive_pids,positive_pids,0,set())
    high_confused = load_high_confused(passage_ids_path,predict_files)
    passage_ndarray,increasing_idx,update_ids = merge(passage_ndarray,high_confused,positive_pids,increasing_idx,update_ids)
    # pdb.set_trace()
    
    np.random.shuffle(passage_ndarray)#first dim shuffle
    with open('datasets_bm25_top5/my_passage.pkl','wb') as handle:
        pkl.dump(passage_ndarray,handle)
    check_all_positive(qrels_path,'datasets_bm25_top5/my_passage.pkl')
    # pdb.set_trace()
    # pass