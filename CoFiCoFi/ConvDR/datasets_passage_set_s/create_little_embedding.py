import numpy as np
import pickle as pkl
import pdb
import json
import os
import logging
'''
map passage id to embedding
save as passage_reps.
'''

logger = logging.getLogger(__name__)

def cread_round_by_round(pid2pidx,passage_embeddings,pid2offset,passage_ids,my_passages,candidate_embeddings,block_id):
    print('cread round by round')
    updating_my_passages = []
    for passage_id in my_passages:
        passage_idx_global = pid2pidx[passage_id]
        if (pid2offset[passage_idx_global]-block_id)%4==0:
            embedding = passage_embeddings[pid2offset[passage_idx_global]//4]
            candidate_embeddings.append(embedding)
            passage_ids.append(passage_id)
        else:
            updating_my_passages.append(passage_id)

    return passage_ids , updating_my_passages ,candidate_embeddings
def search_one_by_one(pid2pidx,ann_data_dir,pid2offset,my_passages):
    '''
    将passage id转换为embeddings
    '''
    print('searching one by one')
    passage_ids = []
    candidate_embeddings = []
    for block_id in range(8):
        logger.info("Loading passage reps " + str(block_id))
        passage_embedding = None
        passage_embedding2id = None
        try:

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

            
            passage_ids , my_passages,candidate_embeddings = cread_round_by_round(pid2pidx,passage_embedding,pid2offset,passage_ids,my_passages,candidate_embeddings,block_id)
                                                                                #passage_embeddings,pid2offset,passage_ids,my_passages,candidate_embeddings,block_id
            # pdb.set_trace()
        except:
            break

    return passage_ids,candidate_embeddings

if __name__ =='__main__':
    ann_data_dir = '/home/xbli/ConvDR-main/datasets/or-quac/embeddings'
    tokenized_dir = '/home/xbli/ConvDR-main/datasets/or-quac/tokenized'
    passage_ids_old_path = '/home/xbli/ConvDR-main/datasets/raw/or-quac/passage_ids.pkl'

    # my_passage_path = '/home/xbli/ConvDR-main/datasets_passage_set_s/datasets/my_passage.pkl'
    # passage_ids_new_path = 'datasets/passage_ids.pickle'
    # embedding_ids_new_path = 'datasets/passage_reps.pickle'


    my_passage_path = '/home/xbli/ConvDR-main/datasets_passage_set_s/datasets_bm25_top5/my_passage.pkl'
    passage_ids_new_path = 'datasets_bm25_top5/passage_ids.pickle'
    embedding_ids_new_path = 'datasets_bm25_top5/passage_reps.pickle'

    with open(os.path.join(
                        tokenized_dir,
                        "pid2offset.pickle"),
                    'rb') as handle:
        pid2offset = pkl.load(handle)

    with open(my_passage_path,'rb') as handle:
        my_passages = pkl.load(handle)
        
    with open(passage_ids_old_path,'rb') as handle:
        passage_ids_old = pkl.load(handle)
    passage_ids_old = {id:idx for idx,id in enumerate(passage_ids_old)}
    passage_ids,candidate_embeddings = search_one_by_one(passage_ids_old,ann_data_dir,pid2offset,my_passages)
    passage_ids = np.array(passage_ids,dtype='<U334')
    candidate_embeddings = np.array(candidate_embeddings)

    with open(passage_ids_new_path,'wb') as handle:
        pkl.dump(passage_ids,handle)
    with open(embedding_ids_new_path,'wb') as handle:
        pkl.dump(candidate_embeddings,handle)
    # pdb.set_trace()

    