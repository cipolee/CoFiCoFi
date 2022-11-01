import argparse
import sys
sys.path.append('/home/xbli/ConvDR-main')
# sys.path.append('/home/xbli/ConvDR-main/utils')
import csv
import logging
import pdb
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
from data.gen_ranking_data import create_rank_data
from utils.util import ConvSearchDataset, load_collection
from drivers.run_convdr_inference import search_one_by_one,EvalDevQuery,evaluate

logger = logging.getLogger(__name__)

# @dataclass
class Arg:
    qrels = '/home/xbli/ConvDR-main/datasets/or-quac/qrels.tsv'
    eval_file = '/home/xbli/ConvDR-main/datasets/or-quac/train.jsonl'
    max_concat_length = 256
    max_query_length = 64
    per_gpu_eval_batch_size = 8
    seed = 42
    no_cuda =False
    use_gpu =True
    raw_data_dir = '/home/xbli/ConvDR-main/datasets/or-quac'
    processed_data_dir = '/home/xbli/ConvDR-main/datasets/or-quac/tokenized'
    ann_data_dir = '/home/xbli/ConvDR-main/datasets/or-quac/embeddings'
    query = 'no_res'
    model_type = 'dpr'
    output_query_type='train.raw'
    top_n = 100
def read_and_create_pos_and_neg(output_rank_file):
    q2_pos = {}
    q2_neg = {}
    with open(output_rank_file, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            qid = record["qid"]
            doc_pos = record["doc_pos"]
            doc_negs = record["doc_negs"]
            q2_pos[qid] = doc_pos
            q2_neg[qid] = doc_negs
            # pdb.set_trace()
    # pass
    return q2_pos,q2_neg

def data_channel(model,tokenizer,output_file,output_trec_file,output_rank_file):

    #1.接受checkpoints train file
    #2.离线检索 # train file
    #2.给输出trec
    main(model,tokenizer,output_file,output_trec_file)
    
    #3.给出rank.jsonl

    create_rank_data(run=output_trec_file,output=output_rank_file)

    # 读取rank.jsonl
    q2_pos,q2_neg = read_and_create_pos_and_neg(output_rank_file)
    # 返回q2pos，q2neg
    return  q2_pos,q2_neg
    
def create_pos_and_negs(qids,q2_pos,q2_neg):
    pos_and_negs = []
    for qid in qids:
        pos_and_negs.append([])
        # pos_and_negs[-1].append()
        pos_and_negs[-1] = [q2_pos[qid]] + q2_neg[qid]
    return pos_and_negs



def main(model,tokenizer,output_file,output_trec_file):


    # parser = argparse.ArgumentParser()
    # parser.add_argument("--model_path", type=str, help="The model checkpoint.")
    # parser.add_argument("--eval_file",
    #                     type=str,
    #                     help="The evaluation dataset.")
    # parser.add_argument(
    #     "--max_concat_length",
    #     default=256,
    #     type=int,
    #     help="Max input concatenated query length after tokenization.")
    # parser.add_argument("--max_query_length",
    #                     default=64,
    #                     type=int,
    #                     help="Max input query length after tokenization."
    #                     "This option is for single query input.")
    # parser.add_argument("--cross_validate",
    #                     action='store_true',
    #                     help="Set when doing cross validation.")
    # parser.add_argument("--per_gpu_eval_batch_size",
    #                     default=4,
    #                     type=int,
    #                     help="Batch size per GPU/CPU.")
    # parser.add_argument("--no_cuda",
    #                     action='store_true',
    #                     help="Avoid using CUDA when available (for pytorch).")
    # parser.add_argument('--seed',
    #                     type=int,
    #                     default=42,
    #                     help="Random seed for initialization.")
    # parser.add_argument("--cache_dir", type=str)
    # parser.add_argument("--ann_data_dir",
    #                     type=str,
    #                     help="Path to ANCE embeddings.")
    # parser.add_argument("--use_gpu",
    #                     action='store_true',
    #                     help="Whether to use GPU for Faiss.")
    # parser.add_argument("--qrels", type=str, help="The qrels file.")
    # parser.add_argument("--processed_data_dir",
    #                     type=str,
    #                     help="Path to tokenized documents.")
    # parser.add_argument("--raw_data_dir", type=str, help="Path to dataset.")
    # parser.add_argument("--output_file",
    #                     type=str,
    #                     help="Output file for OpenMatch reranking.")
    # parser.add_argument(
    #     "--output_trec_file",
    #     type=str,
    #     help="TREC-style run file, to be evaluated by the trec_eval tool.")
    # parser.add_argument(
    #     "--query",
    #     type=str,
    #     default="no_res",
    #     choices=["no_res", "man_can", "auto_can", "target", "output", "raw"],
    #     help="Input query format.")
    # parser.add_argument("--output_query_type",
    #                     type=str,
    #                     help="Query to be written in the OpenMatch file.")
    # parser.add_argument(
    #     "--fold",
    #     type=int,
    #     default=-1,
    #     help="Fold to evaluate on; set to -1 to evaluate all folds.")
    # parser.add_argument(
    #     "--model_type",
    #     default=None,
    #     type=str,
    #     required=True,
    #     help="Model type selected in the list: " +
    #     ", ".join(MSMarcoConfigDict.keys()),
    # )
    # ############## top_n
    # parser.add_argument("--top_n",
    #                     default=100,
    #                     type=int,
    #                     help="Number of retrieved documents for each query.")
    # args = parser.parse_args()
    args = Arg()
    args.output_file = output_file
    args.output_trec_file = output_trec_file

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 1
    args.device = device
    ###########
    #xbli
    model.eval()
    ##########

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


    with open(os.path.join(args.processed_data_dir, "offset2pid.pickle"),
              "rb") as f:
        offset2pid = pickle.load(f)

    logger.info("Building index")
    # faiss.omp_set_num_threads(16)
    cpu_index = faiss.IndexFlatIP(768)
    index = None
    if args.use_gpu:
        co = faiss.GpuMultipleClonerOptions()
        co.shard = True
        co.usePrecomputed = False
        # gpu_vector_resources, gpu_devices_vector
        vres = faiss.GpuResourcesVector()
        vdev = faiss.Int32Vector()
        for i in range(0, ngpu):
            vdev.push_back(i)
            vres.push_back(gpu_resources[i])
        gpu_index = faiss.index_cpu_to_gpu_multiple(vres,
                                                    vdev,
                                                    cpu_index, co)
        index = gpu_index
    else:
        index = cpu_index

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

    total_embedding = []
    total_embedding2id = []
    total_raw_sequences = []



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

    merged_D, merged_I = search_one_by_one(args.ann_data_dir, index,
                                           total_embedding, args.top_n)
    logger.info("start EvalDevQuery...")
    EvalDevQuery(total_embedding2id,
                 merged_D,
                 dev_query_positive_id=dev_query_positive_id,
                 I_nearest_neighbor=merged_I,
                 topN=args.top_n,
                 output_file=args.output_file,
                 output_trec_file=args.output_trec_file,
                 offset2pid=offset2pid,
                 raw_data_dir=args.raw_data_dir,
                 output_query_type=args.output_query_type,
                 raw_sequences=total_raw_sequences)


if __name__ == "__main__":
    read_and_create_pos_and_neg('/home/xbli/ConvDR-main/datasets/or-quac/train.rank_multitask_5.jsonl')
    pdb.set_trace()


    arg = Arg()
    arg.train_file = 'xbli'
    pdb.set_trace()
    main()
