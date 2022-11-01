import sys

sys.path += ['../']
import sys
sys.path.append('/**/ConvDR-main')
import csv
import logging
import json
from model.models import MSMarcoConfigDict
import os
import pickle
import time
import copy
import faiss
import torch
import numpy as np
from torch.utils.data.sampler import SequentialSampler
from torch.utils.data import DataLoader
import torch
import os
from utils.util import (
    barrier_array_merge,
    StreamingDataset,
    EmbeddingCache,)
# )
from data.tokenizing import GetProcessingFn
from model.models import MSMarcoConfigDict
from torch import nn
import torch.distributed as dist
from tqdm import tqdm
from torch.utils.data import DataLoader
# import numpy as np
import argparse
from utils.util import ConvSearchDataset, NUM_FOLD, set_seed, load_model, load_collection
import pdb
# import logging
from utils.dpr_utils import load_states_from_checkpoint, get_model_obj
import re
import random

torch.multiprocessing.set_sharing_strategy('file_system')

logger = logging.getLogger(__name__)

def pid2passage(embedding2id,idx):
    # 也是从4个passage_embeddings里面读取
    # 然后随机做内积 得到qid, pid和score
    pid = embedding2id[idx]
    # return ;
    pass
def qid2pid():
    pass
    # passimport argparse



logger = logging.getLogger(__name__)


def EvalDevQuery(query_embedding2id,
                 merged_D,
                 dev_query_positive_id,
                 I_nearest_neighbor,
                 topN,
                 output_file,
                 output_trec_file,
                 offset2pid,
                 raw_data_dir,
                 output_query_type,
                 raw_sequences=None):
    prediction = {}

    qids_to_ranked_candidate_passages = {}
    qids_to_ranked_candidate_passages_ori = {}
    qids_to_raw_sequences = {}
    for query_idx in range(len(I_nearest_neighbor)):
        seen_pid = set()
        inputs = raw_sequences[query_idx]
        query_id = query_embedding2id[query_idx]
        prediction[query_id] = {}

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
        qids_to_raw_sequences[query_id] = inputs

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

    logger.info("Reading queries and passages...")
    queries = {}
    with open(
            os.path.join(raw_data_dir,
                         "queries." + output_query_type + ".tsv"), "r") as f:
        for line in f:
            qid, query = line.strip().split("\t")
            queries[qid] = query
    collection = os.path.join(raw_data_dir, "collection.jsonl")
    if not os.path.exists(collection):
        collection = os.path.join(raw_data_dir, "collection.tsv")
        if not os.path.exists(collection):
            raise FileNotFoundError(
                "Neither collection.tsv nor collection.jsonl found in {}".
                format(raw_data_dir))
    all_passages = load_collection(collection)

    # Write to file
    with open(output_file, "w") as f, open(output_trec_file, "w") as g:
        for qid, passages in qids_to_ranked_candidate_passages.items():
            ori_qid = qid
            query_text = queries[ori_qid]
            sequences = qids_to_raw_sequences[ori_qid]
            for i in range(topN):
                pid, score = passages[i]
                ori_pid = pid
                passage_text = all_passages[ori_pid]
                label = 0 if qid not in dev_query_positive_id else (
                    dev_query_positive_id[qid][ori_pid]
                    if ori_pid in dev_query_positive_id[qid] else 0)
                f.write(
                    json.dumps({
                        "query": query_text,
                        "doc": passage_text,
                        "label": label,
                        "query_id": str(ori_qid),
                        "doc_id": str(ori_pid),
                        "retrieval_score": score,
                        "input": sequences
                    }) + "\n")
                g.write(
                    str(ori_qid) + " Q0 " + str(ori_pid) + " " + str(i + 1) +
                    " " + str(-i - 1 + 200) + " ance\n")


def evaluate(args, eval_dataset, model, logger):
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset,
                                 sampler=eval_sampler,
                                 batch_size=args.eval_batch_size,
                                 collate_fn=eval_dataset.get_collate_fn(
                                     args, "inference"))

    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Instantaneous batch size per GPU = %d",
                args.per_gpu_eval_batch_size)

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

    embedding = np.concatenate(embedding, axis=0)
    return embedding, embedding2id, raw_sequences


def search_one_by_one(ann_data_dir, query_embedding,query_embedding2id, topN,dev_query_positive_id,pid2offset,offset2pid):
    # def get_passage_rep(per_pid):
    #     return passage_embedding[per_pid]
    def get_passage_id(per_pid):
        return offset2pid[passage_embedding2id[per_pid]]
    # vectorize()不能再返回numpy数组了
    merged_candidate_matrix = None
    get_passage_ids = np.vectorize(get_passage_id)
    # get_passage_reps = np.vectorize(get_passage_rep)
    dict_qid_pid_2_score = {}
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
                passage_embedding = pickle.load(handle)
            with open(
                    os.path.join(
                        ann_data_dir,
                        "passage__embid_p__data_obj_" + str(block_id) + ".pb"),
                    'rb') as handle:
                passage_embedding2id = pickle.load(handle)
        except:
            break
        print('passage embedding shape: ' + str(passage_embedding.shape))
        print("query embedding shape: " + str(query_embedding.shape))
        n = len(passage_embedding)

        ######## 
        # pdb.set_trace()
        # passage_embedding2id_r = {v:k for k,v in passage_embedding2id}
        # 直接按照4*x+1的关系干
        for i,query in enumerate(query_embedding):
            
            qid = query_embedding2id[i]
             
            rd_pids = np.random.randint(0,n,10) 

            ####### 获取正例
            positive_pid = pid2offset[list(dev_query_positive_id[qid].keys())[0]] 
            # pid.list()
            # offset2pid[]
            if (positive_pid-block_id)%4==0:
                rd_pids[-1] = int((positive_pid-block_id)//4)
            pids = get_passage_ids(rd_pids)
            # pdb.set_trace()
            # p_reps = get_passage_reps(rd_pids)
            # if qid in ['C_13aadb0e20b8470a9990f35dc1f181c8_1_q#4','C_28244e74dfdd44ca935842c2a62fe4a5_1_q#1',
            # 'C_8360e6920a814dc2a7b81eb7fa17d25e_1_q#6','C_9a66feafbabb477794d7dfd53a0e8954_1_q#0',
            # 'C_5d5d2ebdcd5140e1a5ae51e28ba1113a_1_q#4']:
            #     # 总共错误20个左右
            #     pdb.set_trace()
            p_reps = []
            for rd_pid in rd_pids:
                p_reps.append(passage_embedding[rd_pid])
            query = torch.tensor(query,dtype=torch.float)
            #np.ndarray()
            p_reps = torch.tensor(p_reps,dtype=torch.float)
            inner_scores = dot_inner_dot(query,p_reps)
            if (positive_pid-block_id)%4==0:
                if qid in ['C_13aadb0e20b8470a9990f35dc1f181c8_1_q#4','C_28244e74dfdd44ca935842c2a62fe4a5_1_q#1',
            'C_8360e6920a814dc2a7b81eb7fa17d25e_1_q#6','C_9a66feafbabb477794d7dfd53a0e8954_1_q#0',
            'C_5d5d2ebdcd5140e1a5ae51e28ba1113a_1_q#4']:
                    print(inner_scores)
                # 总共错误20个左右
                    # pdb.set_trace()
                # if random.random()>0.9:
                #     print(inner_scores)
                # pdb.set_trace()
            for j,pid in enumerate(pids):
                if qid not in dict_qid_pid_2_score:
                    dict_qid_pid_2_score[qid] = {pid:inner_scores[j]}
                else:
                    dict_qid_pid_2_score[qid][pid] = inner_scores[j]

            
        del passage_embedding
        del passage_embedding2id
        # del passage_embedding2id_r

    # pdb.set_trace()
    return dict_qid_pid_2_score
    # return merged_D, merged_I

def dot_inner_dot(query_rep,passage_rep):
    query_rep = query_rep.unsqueeze(-1)
    query_rep = query_rep.expand(768,10) #proj_size, num_blocks)
    query_rep = query_rep.transpose(-1, -2) # query_rep (num_blocks, proj_size)
    retrieval_logits = query_rep * passage_rep # num_blocks, proj_size
    # pdb.set_trace()
    retrieval_logits = torch.sum(retrieval_logits, dim=-1) # num_blocks
    # retrieval_probs = F.softmax(retrieval_logits, dim=1)
    # pdb.set_trace()
    return retrieval_logits
    '''
        (Pdb) p inner_scores
    tensor([58.2449, 56.8529, 51.9905, 59.6154, 57.2407, 55.1135, 55.9141, 55.5914,
            49.6275, 45.6968])
    (Pdb) c
    >/KD/create_triplet.py(256)search_one_by_one()
    -> pdb.set_trace()
    (Pdb) p inner_scores
    tensor([54.0755, 37.4977, 54.3959, 52.8689, 46.7409, 56.6896, 50.5956, 46.2102,
            47.7414, 38.0816])
    (Pdb) c
    > /KD/create_triplet.py(257)search_one_by_one()
    -> for j,pid in enumerate(pids):
    (Pdb) p inner_scores
    tensor([53.9300, 47.2031, 54.2636, 59.9774, 48.5900, 59.4715, 57.8755, 43.8277,
            53.4675, 38.0777])
    (Pdb) c
    >/ConvDR-main/KD/create_triplet.py(256)search_one_by_one()
    -> pdb.set_trace()
    (Pdb) p inner_scores
    tensor([54.7027, 45.9688, 47.4182, 44.1137, 50.6359, 45.9822, 54.2095, 55.5345,
            48.9173, 39.6282])
    '''
def calculate_inner_dot():
    dict_qid_pid_2_score = {}
    # pass
    return dict_qid_pid_2_score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="The model checkpoint.")
    parser.add_argument("--eval_file",
                        type=str,
                        help="The evaluation dataset.")
    parser.add_argument(
        "--max_concat_length",
        default=256,
        type=int,
        help="Max input concatenated query length after tokenization.")
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument("--max_query_length",
                        default=64,
                        type=int,
                        help="Max input query length after tokenization."
                        "This option is for single query input.")
    parser.add_argument("--cross_validate",
                        action='store_true',
                        help="Set when doing cross validation.")
    parser.add_argument("--per_gpu_eval_batch_size",
                        default=4,
                        type=int,
                        help="Batch size per GPU/CPU.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Avoid using CUDA when available (for pytorch).")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="Random seed for initialization.")
    parser.add_argument("--fp16",
                        type=bool,
                        help="is used fp16")
    parser.add_argument("--cache_dir", type=str)
    parser.add_argument("--ann_data_dir",
                        type=str,
                        help="Path to ANCE embeddings.")
    parser.add_argument("--use_gpu",
                        action='store_true',
                        help="Whether to use GPU for Faiss.")
    parser.add_argument("--qrels", type=str, help="The qrels file.")
    parser.add_argument("--processed_data_dir",
                        type=str,
                        help="Path to tokenized documents.")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                    help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                         "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--raw_data_dir", type=str, help="Path to dataset.")
    # parser.add_argument("--output_file",
    #                     type=str,
    #                     help="Output file for OpenMatch reranking.")
    # parser.add_argument(
    #     "--output_trec_file",
    #     type=str,
    #     help="TREC-style run file, to be evaluated by the trec_eval tool.")
    parser.add_argument(
        "--query",
        type=str,
        default="no_res",
        choices=["no_res", "man_can", "auto_can", "target", "output", "raw"],
        help="Input query format.")
    parser.add_argument("--output_query_type",
                        type=str,
                        help="Query to be written in the OpenMatch file.")
    parser.add_argument("--output_path",
                        type=str,
                        help="Query to be written in the OpenMatch file.")
    parser.add_argument(
        "--fold",
        type=int,
        default=-1,
        help="Fold to evaluate on; set to -1 to evaluate all folds.")
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " +
        ", ".join(MSMarcoConfigDict.keys()),
    )
    ############## top_n
    parser.add_argument("--top_n",
                        default=100,
                        type=int,
                        help="Number of retrieved documents for each query.")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 1
    args.device = device

    ngpu = faiss.get_num_gpus()
    gpu_resources = []
    tempmem = -1

    for i in range(ngpu):
        res = faiss.StandardGpuResources()
        if tempmem >= 0:
            res.setTempMemory(tempmem)
        gpu_resources.append(res)

    # Setup logging
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO)
    logger.warning("device: %s, n_gpu: %s", device, args.n_gpu)

    # Set seed
    set_seed(args)

    with open(os.path.join(args.processed_data_dir, "offset2pid.pickle"),
              "rb") as f:
        offset2pid = pickle.load(f)

    logger.info("Building Build qid pid score")

    dev_query_positive_id = {}
    if args.qrels is not None:
        with open(args.qrels, 'r', encoding='utf8') as f:
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
    with open(os.path.join(args.processed_data_dir, "offset2pid.pickle"),
              "rb") as f:
        offset2pid = pickle.load(f)
    # pdb.set_trace()
    pid2offset = {v:k for k,v in enumerate(offset2pid)}
    total_embedding = []
    total_embedding2id = []
    total_raw_sequences = []

    if not args.cross_validate:
        if args.fp16:
            try:
                import apex
                apex.amp.register_half_function(torch, 'einsum')
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        config, tokenizer, model = load_model(args, args.model_path)
        # pdb.set_trace()
        if args.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model= amp.initialize(
                model,None, opt_level=args.fp16_opt_level)
        if args.max_concat_length <= 0:
            args.max_concat_length = tokenizer.max_len_single_sentence
        args.max_concat_length = min(args.max_concat_length,
                                     tokenizer.max_len_single_sentence)

        # eval
        logger.info("Training/evaluation parameters %s", args)
        eval_dataset = ConvSearchDataset([args.eval_file],
                                         tokenizer,
                                         args,
                                         mode="inference")
        total_embedding, total_embedding2id, raw_sequences = evaluate(
            args, eval_dataset, model, logger)
        total_raw_sequences.extend(raw_sequences)
        del model
        torch.cuda.empty_cache()

    else:
        # K-Fold Cross Validation

        for i in range(NUM_FOLD):
            if args.fold != -1 and i != args.fold:
                continue

            logger.info("Testing Fold #{}".format(i))
            suffix = ('-' + str(i))
            config, tokenizer, model = load_model(args,
                                                  args.model_path + suffix)

            if args.max_concat_length <= 0:
                args.max_concat_length = tokenizer.max_len_single_sentence
            args.max_concat_length = min(args.max_concat_length,
                                         tokenizer.max_len_single_sentence)

            logger.info("Training/evaluation parameters %s", args)
            eval_file = "%s.%d" % (args.eval_file, i)
            logger.info("eval_file: {}".format(eval_file))
            eval_dataset = ConvSearchDataset([eval_file],
                                             tokenizer,
                                             args,
                                             mode="inference")
            embedding, embedding2id, raw_sequences = evaluate(
                args, eval_dataset, model, logger)
            total_embedding.append(embedding)
            total_embedding2id.extend(embedding2id)
            total_raw_sequences.extend(raw_sequences)

            del model
            torch.cuda.empty_cache()

        total_embedding = np.concatenate(total_embedding, axis=0)
        total_embedding2id = np.concatenate(total_embedding2id,axis = 0)
    q_p_s = search_one_by_one(args.ann_data_dir,
                                           total_embedding,total_embedding2id, args.top_n,dev_query_positive_id,pid2offset,offset2pid)
    with open(os.path.join(args.output_path,'q_p_s1.pickle'), 'wb') as handle:
        pickle.dump(q_p_s, handle)
    # logger.info("start EvalDevQuery...")
    # EvalDevQuery(total_embedding2id,
    #              merged_D,
    #              dev_query_positive_id=dev_query_positive_id,
    #              I_nearest_neighbor=merged_I,
    #              topN=args.top_n,
    #              output_file=args.output_file,
    #              output_trec_file=args.output_trec_file,
    #              offset2pid=offset2pid,
    #              raw_data_dir=args.raw_data_dir,
    #              output_query_type=args.output_query_type,
    #              raw_sequences=total_raw_sequences)


if __name__ == "__main__":
    main()



# 是否我需要对打分，内积进行除以根号下维度处理，不然内积大小和维度有关系
# 768维度的放缩点积集中在2.7到3
# 128维度缩放点积在1左右
# 使用缩放点积蒸馏
'''
[[ 8.087805   8.08669    7.871638   7.8575134  7.7606287]
 [14.901171  13.557567  13.282486  13.024028  12.577696 ]
 [15.100792  14.267819  14.043733  13.932785  13.238268 ]
 [13.98053   12.818369  12.553574  12.534168  12.316328 ]
 [13.332914  11.959454  11.957876  11.635626  11.538331 ]
 [11.779741  11.712401  11.347803  11.254755  11.148792 ]
 [11.150782  10.858947  10.637745  10.464769  10.400422 ]
 [ 9.694717   9.668701   9.497084   9.486417   9.439333 ]]
 [[10.794081 10.552587  9.730876  9.686491  9.635757]
 [10.580913 10.417297  9.743341  9.567007  9.481488]
 [ 9.619566  9.426853  9.317619  8.931681  8.807878]
 [ 9.53434   9.385593  9.372788  8.888012  8.865193]
 [ 9.562826  9.484079  9.004158  8.946127  8.851354]
 [10.265808 10.225644 10.121735 10.062891  9.968314]
 [11.800749 11.329281 11.123982 11.007555 10.991575]
 [11.402956 11.322878 10.837725 10.782764 10.684713]]
 [[13.227348  12.751058  12.510188  12.295488  12.1051035]
 [11.460298  11.013689  10.920542  10.61216   10.523015 ]
 [10.855182  10.538092  10.460547  10.203725  10.098548 ]
 [10.052838   9.835017   9.738856   9.5184765  9.4485235]
 [ 9.878335   9.848821   9.769752   9.562151   9.492159 ]
 [ 9.8534975  9.833921   9.694505   9.590285   9.386878 ]
 [12.87647   12.370258  12.114713  11.239535  11.04531  ]
 [11.926411  11.207533  10.873091  10.515838  10.173242 ]]
 [[7.83144   7.5787735 7.48454   7.280225  7.134349 ]
 [7.6555166 7.3506975 7.2970715 7.1924486 7.0370073]
 [7.327962  6.8224974 6.7772455 6.685478  6.576191 ]
 [7.2609925 6.7556767 6.688114  6.6663003 6.6470013]
 [7.2005653 6.686643  6.6295714 6.5806265 6.5054026]
 [6.4514036 6.378797  6.360803  6.2899103 6.177807 ]
 [7.748239  7.603876  7.601399  7.483007  7.4814157]
 [8.34083   8.238213  8.219971  8.1879225 8.137467 ]]
 [ 8.818202   8.8016205  8.784694   8.775724   8.697016 ]
 [ 9.031652   8.934038   8.9063     8.732094   8.65967  ]
 [ 9.12056    9.03863    8.818106   8.775708   8.6474285]
 [ 8.697321   8.6066065  8.43277    8.343704   8.238728 ]
 [ 8.907586   8.73078    8.567696   8.492966   8.491354 ]
 [ 8.5059185  8.259537   8.1014     8.059162   8.011908 ]
 [11.352497  11.159584  11.003747  10.914047  10.862484 ]]


'''


##### qps作为文件 封装进dataset。创建query tokenized的文件，pid到问题的映射，pid到passage的文件