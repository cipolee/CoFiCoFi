import argparse
import logging
import os
import torch
import random
import csv
import copy
from tensorboardX import SummaryWriter
import pickle
from utils.util import pad_input_ids_with_mask, getattr_recursive


from torch.utils.data.sampler import SequentialSampler
from torch.utils.data import DataLoader, RandomSampler


from tqdm import tqdm, trange
from transformers import get_linear_schedule_with_warmup,BertConfig,BertTokenizer
from torch import nn
import numpy as np
from model.models import MSMarcoConfigDict,BertForOrconvqaGlobal
# from utils.util import ConvSearchDataset, NUM_FOLD, set_seed, load_model
from utils.dpr_utils import CheckpointState, get_model_obj, get_optimizer
import json
from utils.util import ConvSearchDataset, NUM_FOLD, set_seed, load_model, load_collection
logger = logging.getLogger(__name__)

# Model_dict = {}
MODEL_CLASSES = {
    'reader': (BertConfig, BertForOrconvqaGlobal, BertTokenizer),
    # 'retriever': (AlbertConfig, AlbertForRetrieverOnlyPositivePassage, AlbertTokenizer),
}


def _save_checkpointRT(args,
                     model,
                     output_dir,
                     optimizer=None,
                     scheduler=None,
                     step=0) -> str:
    offset = step
    epoch = 0
    model_to_save = get_model_obj(model)
    cp = os.path.join(output_dir, 'checkpointRetriver-' + str(offset))

    meta_params = {}
    state = CheckpointState(model_to_save.state_dict(), optimizer.state_dict(),
                            scheduler.state_dict(), offset, epoch, meta_params)
    torch.save(state._asdict(), cp)
    logger.info('Saved checkpoint at %s', cp)
    return cp

def _save_checkpointRD(args,
                     model,
                     output_dir,
                     optimizer=None,
                     scheduler=None,
                     step=0) -> str:
    offset = step
    epoch = 0
    model_to_save = get_model_obj(model)
    cp = os.path.join(output_dir, 'checkpointReader-' + str(offset))

    meta_params = {}
    state = CheckpointState(model_to_save.state_dict(), optimizer.state_dict(),
                            scheduler.state_dict(), offset, epoch, meta_params)
    torch.save(state._asdict(), cp)
    logger.info('Saved checkpoint at %s', cp)
    return cp



############## 作为全局变量的部分


qrels = 'datasets/or-quac/qrels.tsv'
raw_data_dir = 'datasets/or-quac'
output_query_type = 'train.raw'
processed_data_dir = 'datasets/or-quac/tokenized'


gpu_indexs,passage_embedding2ids=None,None#
# gpu_indexs 4*gpuindex ;passage_embedding2ids 4*passage_embedding2id

# gpu_indexs 更改成8个


logger.info("Reading queries and passages...")
queries = {}

with open(
        os.path.join(raw_data_dir,
                        "queries." + output_query_type + ".tsv"), "r") as f:
    for line in f:
        qid, query = line.strip().split("\t")
        queries[qid] = query

with open(os.path.join(processed_data_dir, "offset2pid.pickle"),
            "rb") as f:
    offset2pid = pickle.load(f)
collection = os.path.join(raw_data_dir, "collection.jsonl")
if not os.path.exists(collection):
    collection = os.path.join(raw_data_dir, "collection.tsv")
    if not os.path.exists(collection):
        raise FileNotFoundError(
            "Neither collection.tsv nor collection.jsonl found in {}".
            format(raw_data_dir))
all_passages = load_collection(collection)

dev_query_positive_id = {}
if qrels is not None:
    with open(qrels, 'r', encoding='utf8') as f:
        tsvreader = csv.reader(f, delimiter="\t")
        for [topicid, _, docid, rel] in tsvreader:
            topicid = str(topicid)
            docid = int(docid)
            rel = int(rel)
            if topicid not in dev_query_positive_id:
                if rel > 0:
                    dev_query_positive_id[topicid] = {}
                    dev_query_positive_id[topicid][docid] = rel
            else:
                dev_query_positive_id[topicid][docid] = rel

###############



def retrieve(qids,ann_data_dir,batch_query_embedding,topN,raw_sequences,query_embedding2id):
    merged_candidate_matrix = None
    # batch_query_embedding [b,]
    for idx , gpu_index in enumerate(gpu_indexs):
        D, I = gpu_index.search(batch_query_embedding,topN)
        candidate_id_matrix = passage_embedding2ids[idx][
            I]  # passage_idx -> passage_id
        D = D.tolist()
        candidate_id_matrix = candidate_id_matrix.tolist()
        candidate_matrix = []
        for score_list, passage_list in zip(D, candidate_id_matrix):
            candidate_matrix.append([])
            for score, passage in zip(score_list, passage_list):
                candidate_matrix[-1].append((score, passage))
            assert len(candidate_matrix[-1]) == len(passage_list)
        assert len(candidate_matrix) == I.shape[0]
        if merged_candidate_matrix == None:
            merged_candidate_matrix = candidate_matrix
            continue
    
        ############# merge
        merged_candidate_matrix_tmp = copy.deepcopy(merged_candidate_matrix)
        merged_candidate_matrix = []
        for merged_list, cur_list in zip(merged_candidate_matrix_tmp,
                                            candidate_matrix):
            p1, p2 = 0, 0
            merged_candidate_matrix.append([])
            while p1 < topN and p2 < topN:
                if merged_list[p1][0] >= cur_list[p2][0]:
                    merged_candidate_matrix[-1].append(merged_list[p1])
                    p1 += 1
                else:
                    merged_candidate_matrix[-1].append(cur_list[p2])
                    p2 += 1
            while p1 < topN:
                merged_candidate_matrix[-1].append(merged_list[p1])
                p1 += 1
            while p2 < topN:
                merged_candidate_matrix[-1].append(cur_list[p2])
                p2 += 1
    merged_D, merged_I = [], []
    for merged_list in merged_candidate_matrix:
        merged_D.append([])
        merged_I.append([])
        for candidate in merged_list:
            merged_D[-1].append(candidate[0])
            merged_I[-1].append(candidate[1])
    merged_D, merged_I = np.array(merged_D), np.array(merged_I)
    I_nearest_neighbor = merged_I
    prediction = {}

    qids_to_ranked_candidate_passages = {}
    qids_to_ranked_candidate_passages_ori = {}
    qids_to_raw_sequences = {}
    for query_idx in range(len(I_nearest_neighbor)):
        seen_pid = set()
        query_id = query_embedding2id[query_idx]


        top_ann_pid = I_nearest_neighbor[query_idx].copy()
        top_ann_score = merged_D[query_idx].copy()
        selected_ann_idx = top_ann_pid[:topN]
        selected_ann_score = top_ann_score[:topN].tolist()
        rank = 0

        if query_id in qids_to_ranked_candidate_passages:
            pass
        else:
            tmp = [(0, 0)] * topN
            tmp_ori = [0] * topN
            qids_to_ranked_candidate_passages[query_id] = tmp
            qids_to_ranked_candidate_passages_ori[query_id] = tmp_ori
        # qids_to_raw_sequences[query_id] = inputs

        for idx, score in zip(selected_ann_idx, selected_ann_score):
            pred_pid = offset2pid[idx]

            if not pred_pid in seen_pid:
                qids_to_ranked_candidate_passages[query_id][rank] = (pred_pid,
                                                                     score)
                qids_to_ranked_candidate_passages_ori[query_id][
                    rank] = pred_pid

                rank += 1
                prediction[query_id][pred_pid] = -rank
                seen_pid.add(pred_pid)

   

    # Write to file
    # with open(output_file, "w") as f, open(output_trec_file, "w") as g:
    return_dict = {}
    query , doc, label , query_id , doc_id = [], [], [], [], []
    for qid, passages in qids_to_ranked_candidate_passages.items():
        ori_qid = qid
        query_text = queries[ori_qid]
        query.append([])
        doc.append([])
        label.append([])
        query_id.append([])
        doc_id.append([])
        sequences = qids_to_raw_sequences[ori_qid]
        for i in range(topN):
            pid, score = passages[i]
            ori_pid = pid
            passage_text = all_passages[ori_pid]
            label = 0 if qid not in dev_query_positive_id else (
                dev_query_positive_id[qid][ori_pid]
                if ori_pid in dev_query_positive_id[qid] else 0)
            # f.write(
            #     json.dumps({
            #         "query": query_text,
            #         "doc": passage_text,
            #         "label": label,
            #         "query_id": str(ori_qid),
            #         "doc_id": str(ori_pid),
            #         "retrieval_score": score,
            #         "input": sequences
            #     }) + "\n")
            query[-1].append(query_text)
            doc[-1].append(passage_text)
            label[-1].append(label)
            query_id[-1].append(str(ori_qid))
            doc_id[-1].append(str(ori_pid))

    return_dict ={
        'query':np.asarray(query),
        'doc':np.asarray(doc),
        'label':np.asarray(label),
        'query_id':np.asarray(query_id),
        'doc_id':np.asarray(doc_id)

    }
    return return_dict

def train(args, train_dataset, model, logger,topN =5):
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # eval_sampler = SequentialSampler(train_dataset)
    train_sampler = RandomSampler(train_dataset)
    eval_dataloader = DataLoader(train_dataset,
                                 sampler=(train_dataset),
                                 batch_size=args.eval_batch_size,
                                 collate_fn=train_dataset.get_collate_fn(
                                     args, "train"))

    model.zero_grad()
    set_seed(
        args)  # Added here for reproducibility (even between python 2 and 3)
    embedding = []
    embedding2id = []
    raw_sequences = []
    epoch_iterator = eval_dataloader
    for batch in epoch_iterator:
        # pdb.set_trace()
        qids = batch["qid"]
        ids, id_mask = (
            ele.to(args.device)
            for ele in [batch["concat_ids"], batch["concat_id_mask"]])
        model.eval()

        with torch.no_grad():
            embs = model(ids, id_mask)
        embs = embs.detach().cpu().numpy()
        embedding.append(embs)
        for qid in qids:
            embedding2id.append(qid)

        sequences = batch["history_utterances"]
        raw_sequences.extend(sequences)
        retrieval_results = retrieve(ann_data_dir='',batch_query_embedding=embs,
        topN=topN,raw_sequences=raw_sequences,
        query_embedding2id=embedding2id)

 