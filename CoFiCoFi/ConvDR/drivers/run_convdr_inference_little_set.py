import argparse
import sys
sys.path.append('/home/xbli/ConvDR-main')
import csv
import logging
import json
from model.models import MSMarcoConfigDict
import os
import pickle as pkl
import time
import copy
import faiss
import torch
import linecache
import pdb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib

# We import seaborn to make nice plots.

from torch.utils.data.sampler import SequentialSampler
from torch.utils.data import DataLoader
import sklearn
from sklearn.manifold import TSNE
from utils.util import ConvSearchDataset, NUM_FOLD, set_seed, load_model, load_collection,load_model_V2,load_model_V3
import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})
logger = logging.getLogger(__name__)


def mydraw(I,qid2gold_pids,qids,passage_origin_ids,passage_ids_idx,passage_origin_ids_idx,passage_ids,passage_embedding):
    donthit_id = 0
    donthit_id_0 = 0
    for idx,qid in enumerate(qids):
        gold_pids = np.asarray(list(qid2gold_pids[qid].keys()))
        qid2top100p = I[idx]
        gold_ids = passage_origin_ids[gold_pids]
        gold_passages_mapped = []
        for id in gold_ids:
            gold_passages_mapped.append(passage_ids_idx[id])
        gold_passages_mapped = set(gold_passages_mapped)
        # pdb.set_trace()
        if any (pid in gold_passages_mapped for pid in qid2top100p):


            ############

            # x_y = create_draw_data_3dimons(np.asarray(qid2top100p),gold_pids,gold_passages_mapped,passage_origin_ids_idx,passage_ids)
            # # pdb.set_trace()
            # dolowdim_and_scatter(x_y,passage_embedding,'find{}'.format(donthit_id_0))


            

            ############

            if donthit_id_0 in [296,317,318]:
                x_y = create_draw_data_3dimons(np.asarray(qid2top100p),gold_pids,gold_passages_mapped,passage_origin_ids_idx,passage_ids)
            # pdb.set_trace()
                dolowdim_and_scatter(x_y,passage_embedding,'find{}'.format(donthit_id_0))

                pdb.set_trace()
            donthit_id_0 += 1
            
            # pdb.set_trace()
        else:
            # pdb.set_trace()
            x_y = create_draw_data_3dimons(np.asarray(qid2top100p),gold_pids,gold_passages_mapped,passage_origin_ids_idx,passage_ids)
            # pdb.set_trace()

            dolowdim_and_scatter(x_y,passage_embedding,donthit_id)
            donthit_id += 1


 
def dolowdim_and_scatter(x,passage_embedding,donthit_id):
    # We choose a color palette with seaborn.
    # palette = np.array(sns.color_palette("hls", 10))
    ######## idx of x to 768 dim vector, lower the dim to 2 and concatenate label
    cdict = {0:'red',1:'yellow',2:'green',3:'blue',4:'purple',5:'brown'}
    RS = 20220704
    # pdb.set_trace()
    # 1 gold passage , 0 not a document , same document but not a passage
    if len(np.where(x[:,1] == 1)[0])>4:
        selected_embedding = passage_embedding[x[:,0]]
        
        digits_proj = TSNE(random_state=RS).fit_transform(selected_embedding)
        # pdb.set_trace()
        # x = np.column_stack((digits_proj,x[:,1]))
        x = np.column_stack((digits_proj,x[:,1]))
        # pdb.set_trace()
        
            
        fig, ax = plt.subplots()
        for g in np.unique([1,0,2,3,4]):
            ix = np.where(x[:,2] == g)
            choiced_by_label = x[ix]
            ax.scatter(choiced_by_label[:,0], choiced_by_label[:,1], c = cdict[g], label = g, s = 100)
        ax.legend()
        plt.show()
        plt.savefig('/home/xbli/ConvDR-main/figures/scatter/{}.jpg'.format(donthit_id))
def get_title(pidx,blocks_path='/home/xbli/ConvDR-main/datasets/raw/or-quac/all_blocks.txt'):
    line = linecache.getline(blocks_path, pidx + 1)
    line = json.loads(line.strip())
    return line['title'] #,line['bid']
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
    for query_idx in range(len(I_nearest_neighbor)):
        seen_pid = set()
        query_id = query_embedding2id[query_idx]
        prediction[query_id] = {}

        top_ann_pid = I_nearest_neighbor[query_idx].copy()
        top_ann_score = merged_D[query_idx].copy()
        selected_ann_idx = top_ann_pid[:topN]
        selected_ann_score = top_ann_score[:topN]
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
            # pred_pid = offset2pid[idx]
            pred_pid = idx

            if not pred_pid in seen_pid:
                qids_to_ranked_candidate_passages[query_id][rank] = (pred_pid,
                                                                     score)
                qids_to_ranked_candidate_passages_ori[query_id][
                    rank] = pred_pid

                rank += 1
                prediction[query_id][pred_pid] = -rank
                seen_pid.add(pred_pid)

    # pdb.set_trace()
    with  open(output_trec_file, "w") as g:
         for qid, passages in qids_to_ranked_candidate_passages.items():
            ori_qid = qid
            # query_text = queries[ori_qid]
            # sequences = qids_to_raw_sequences[ori_qid]
            for i in range(topN):
                pid, score = passages[i]
                ori_pid = pid
                g.write(
                str(ori_qid) + " Q0 " + str(ori_pid) + " " + str(i + 1) +
                " " + str(-i - 1 + 200) + " ance\n")


    #################################################  xbli
    # all_passages = load_collection(collection)
    # Write to file
    # with open(output_file, "w") as f, open(output_trec_file, "w") as g:
    #     for qid, passages in qids_to_ranked_candidate_passages.items():
    #         ori_qid = qid
    #         query_text = queries[ori_qid]
    #         sequences = qids_to_raw_sequences[ori_qid]
    #         for i in range(topN):
    #             pid, score = passages[i]
    #             ori_pid = pid
    #             passage_text = all_passages[ori_pid]
    #             label = 0 if qid not in dev_query_positive_id else (
    #                 dev_query_positive_id[qid][ori_pid]
    #                 if ori_pid in dev_query_positive_id[qid] else 0)
    #             f.write(
    #                 json.dumps({
    #                     "query": query_text,
    #                     "doc": passage_text,
    #                     "label": label,
    #                     "query_id": str(ori_qid),
    #                     "doc_id": str(ori_pid),
    #                     "retrieval_score": score,
    #                     "input": sequences
    #                 }) + "\n")
    #             g.write(
    #                 str(ori_qid) + " Q0 " + str(ori_pid) + " " + str(i + 1) +
    #                 " " + str(-i - 1 + 200) + " ance\n")

    ####################################


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

def is_positive_passage_exists(I,dev_query_positive_id):
    ans  = []
    for i in I:
        if i in dev_query_positive_id:
            ans.append(i)
    return ans

def  create_draw_data_3dimons(x,gold_passages=None,gold_passages_mapped=None,passage_origin_ids_idx=None,passage_ids=None):
    
    y = []
    had_gold = 0
    one_of_glod_passage = gold_passages[0]
    gold_title = get_title(one_of_glod_passage)
    for i in range(len(x)):
        if x[i] in gold_passages_mapped:
            if i<5:
                y.append(0)
            elif get_title(passage_origin_ids_idx[passage_ids[x[i]]])==gold_title:
                y.append(3)
            else:
                y.append(4)
            had_gold += 1
        elif get_title(passage_origin_ids_idx[passage_ids[x[i]]])==gold_title:
            y.append(1)
        else:
            y.append(2)
    if not had_gold:
        y.append(5)
        x = np.append(x,list(gold_passages_mapped)[0])
    # pdb.set_trace()
    return np.column_stack((x,y))


def mysearch(embedding_path, gpu_index, query_embedding, topN,passage_embedding):

    # pdb.set_trace()
    gpu_index.add(passage_embedding)
    ts = time.time()
    D, I = gpu_index.search(query_embedding, topN)
    te = time.time()
    elapsed_time = te - ts
    print({
        "total": elapsed_time,
        "data": query_embedding.shape[0],
        "per_query": elapsed_time / query_embedding.shape[0]
    })
    # candidate_id_matrix = passage_embedding2id[
    #     I]  # passage_idx -> passage_id
    D = D.tolist()
    # candidate_id_matrix = candidate_id_matrix.tolist()
    I = I.tolist()
    # pdb.set_trace()
    # dev_query_positive_ids = dev_query_positive_ids

    ####################



    #judge is positive passage exists



    
    ####################


    # pdb.set_trace()
    # return D , candidate_id_matrix
    return D,I


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
    parser.add_argument("--raw_data_dir", type=str, help="Path to dataset.")
    parser.add_argument("--output_file",
                        type=str,
                        help="Output file for OpenMatch reranking.")
    parser.add_argument(
        "--output_trec_file",
        type=str,
        help="TREC-style run file, to be evaluated by the trec_eval tool.")
    parser.add_argument(
        "--query",
        type=str,
        default="no_res",
        choices=["no_res", "man_can", "auto_can", "target", "output", "raw"],
        help="Input query format.")
    parser.add_argument("--output_query_type",
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
    dataset_name = 'datasets'

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
        offset2pid = pkl.load(f)

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
    # eval
    logger.info("Training/evaluation parameters %s", args)
    embedding_path = '/home/xbli/ConvDR-main/datasets_passage_set_s/{}/passage_reps.pickle'.format(dataset_name)
    with open(embedding_path,'rb') as handle:
        passage_embedding = pkl.load(handle)
    with open('/home/xbli/ConvDR-main/datasets_passage_set_s/datasets/passage_ids.pickle', 'rb') as handle:
    # with open('/home/xbli/ConvDR-main/datasets_passage_set_s/datasets/passage_ids.pickle', 'rb') as handle:
        passage_ids = pkl.load(handle)
    with open('/home/xbli/ConvDR-main/datasets/raw/or-quac/passage_ids.pkl','rb') as handle:
        passage_origin_ids = pkl.load(handle)

    passage_ids_idx = {id:idx for idx,id in enumerate(passage_ids)}
    passage_origin_ids_idx = {id:idx for idx,id in enumerate(passage_origin_ids)}
    # del passage_origin_ids
    if not args.cross_validate:

        config, tokenizer, model = load_model(args, args.model_path)
        

        if args.max_concat_length <= 0:
            args.max_concat_length = tokenizer.max_len_single_sentence
        args.max_concat_length = min(args.max_concat_length,
                                     tokenizer.max_len_single_sentence)

        
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
            # config, tokenizer, model = load_model(args,
            #                                       args.model_path + suffix)
            config, tokenizer, model = load_model_V3(args, args.model_path)
            # config, tokenizer, model = load_model(args, args.model_path)
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

    # merged_D, merged_I = search_one_by_one(args.ann_data_dir, index,
    #                                        total_embedding, args.top_n)
    
    merged_D, merged_I = mysearch(embedding_path, index,
                                           total_embedding, args.top_n,passage_embedding)


    mydraw(merged_I,dev_query_positive_id,total_embedding2id,passage_origin_ids,passage_ids_idx,passage_origin_ids_idx,passage_ids,passage_embedding)
    





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
    main()
