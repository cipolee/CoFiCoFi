import numpy as np
import pickle as pkl
import pdb
import json
import os
import logging
import linecache
'''
check is consistence between id and embedding
step1 choose a pidx and get its embedding 'A' by old_passage_reps, get its pidx by passage_embedding_ids, 
    offset2pid,searching all_blocks.txt
step2 search pid's embedding 'B'
step3 compare 'A' and 'B'
'''
# def get_passage_id(all_blocks.txt,idx):
#     return offset2pid[passage_embedding2id[per_pid]]
def get_passage_id(idx):
    # def get_passage(i, args):
    line = linecache.getline('/home/xbli/ConvDR-main/datasets/raw/or-quac/all_blocks.txt', idx + 1)
    line = json.loads(line.strip())
    return line['id']
def get_offset2pid(idx):
    return offset2pid[idx]
def step1(ann_data_dir,block_id):
    get_passages_id = np.vectorize(get_passage_id)
    get_offset2pids = np.vectorize(get_offset2pid)
    with open(
            os.path.join(
                ann_data_dir,
                "passage__emb_p__data_obj_" + str(block_id) + ".pb"),
            'rb') as handle:
        passage_embedding = pkl.load(handle)
    with open(
            os.path.join(
                ann_data_dir,
                "passage__embid_p__data_obj_" + str(block_id) + ".pb"),
            'rb') as handle:
        passage_embedding2id = pkl.load(handle)
    random_idx = np.random.choice(np.arange(1000000),1000)
    embeddings = passage_embedding[random_idx]
    # pids = get_passages_id(offset2pid[passage_embedding2id[random_idx]])
    hidden_idx = passage_embedding2id[random_idx]
    pidxs = get_offset2pids(hidden_idx)
    pids = get_passages_id(pidxs)
    return pids,embeddings
def get_new_pidx(id):
    if id in new_id2idx:

        return new_id2idx[id]
    else:
        return -1
def step2(pids,embedding_ids_new_path):
    # np.arange(11377951), (2000000))
    get_new_pidxs = np.vectorize(get_new_pidx)
    with open(embedding_ids_new_path,'rb') as handle:
        passage_embeddings = pkl.load(handle)
    
    pidxs = get_new_pidxs(pids)
    # pids = pids[]
    print('len pidx with bad key {}'.format(len(pidxs)))
    return_pidxs = pidxs.copy()
    pidxs = pidxs[pidxs!=-1]
    print('len pidx without bad key {}'.format(len(pidxs)))
    embeddings = passage_embeddings[pidxs]
    return embeddings,return_pidxs

def step3(embeddings1,embeddings2,return_pidxs):
    idx = (return_pidxs!=-1)
    idx_2dim = np.expand_dims(idx,axis=-1)
    idx_2dim_repeat = np.repeat(idx_2dim,768,axis=-1)
    print('the shape of idx_2dim_repeat is {}'.format(idx_2dim_repeat.shape))
    if embeddings1[idx_2dim_repeat].shape==embeddings2.reshape(-1,1)[:,0].shape:
        print(embeddings1[idx_2dim_repeat] == embeddings2.reshape(-1,1)[:,0])
        print('checked, no error here')
    else:
        embeddings3 = embeddings1[idx_2dim_repeat]
        pdb.set_trace()
    # pass

if __name__ == '__main__':
    embedding_ids_new_path = 'datasets/passage_reps.pickle'
    offset2pid_path = '/home/xbli/ConvDR-main/datasets/or-quac/tokenized/offset2pid.pickle'
    ann_data_dir = '/home/xbli/ConvDR-main/datasets/or-quac/embeddings'
    passage_ids_new_path = '/home/xbli/ConvDR-main/datasets_passage_set_s/datasets/passage_ids.pickle'
    # offset2pid
    with open(offset2pid_path,'rb') as handle:
        offset2pid = pkl.load(handle)
    # pdb.set_trace()
    with open(passage_ids_new_path,'rb') as handle:
        passage_ids_new = pkl.load(handle)
    new_id2idx = {id:idx for idx,id in enumerate(passage_ids_new)}
    passage_ids_old,embeddings_old = step1(ann_data_dir,0)
    embeddings_new,return_pidxs = step2(passage_ids_old,embedding_ids_new_path)
    step3(embeddings_old,embeddings_new,return_pidxs)
    pdb.set_trace()
    

