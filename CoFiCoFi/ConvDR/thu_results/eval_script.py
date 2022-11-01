import pytrec_eval
import numpy as np
import json
from tqdm import tqdm
import linecache
import scipy as sp
import pdb
import pickle as pkl
import os

def sigmoid(x):
    return 1/(1+np.exp(-x))
def get_retrieval_metrics(evaluator, all_predictions, eval_retriever_probs=True):
    rerank_run = {}
    # for qid, preds in all_predictions.items():
    #     rerank_run[qid] = {}
    #     for pred in preds:
    #         pid = pred['example_id'].split('*')[1] if eval_retriever_probs else pred['example_id']
    #         rerank_run[qid][pid] = pred['retrieval_logit']
    # # print('rerank_run', rerank_run)
    # rerank_metrics = evaluator.evaluate(rerank_run)
    # rerank_mrr_list = [v['recip_rank'] for v in rerank_metrics.values()]
    # rerank_recall_list = [v['recall_5'] for v in rerank_metrics.values()]    
    # return_dict = {'rerank_mrr': np.average(rerank_mrr_list), 'rerank_recall': np.average(rerank_recall_list)}
    return_dict = {}
    if eval_retriever_probs:
        retriever_run = {}
        for qid, preds in all_predictions.items():
            retriever_run[qid] = {}
            for pred in preds.items():
                # pid = pred['example_id'].split('*')[1]
                retriever_run[qid][pred[0]] = pred[1]
            # pdb.set_trace()
        # print('retriever_run', retriever_run)
        retriever_metrics = evaluator.evaluate(retriever_run)
        # pdb.set_trace()
        retriever_mrr_list = [v['recip_rank'] for v in retriever_metrics.values()]
        retriever_recall_list = [v['recall_5'] for v in retriever_metrics.values()]    
        return_dict.update({'retriever_mrr': np.average(retriever_mrr_list), 
                            'retriever_recall': np.average(retriever_recall_list)})
        
    return return_dict
def check_all_positive(qrels,passage_ids):
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
def   evaluate_origin(evaluator,base_path):
    for j in os.listdir(base_path):
        i = base_path +j
        with open(i,'r') as f:
            # multi_train_task
            for line in tqdm(f.readlines()):
                qid=line.split(" ")[0]
                did=passage_ids[int(line.split(" ")[2])]
                score=sigmoid(int(line.split(" ")[4])/200)
                if qid not in DR_Result:
                    DR_Result[qid] ={did:score}
                else:
                    DR_Result[qid][did] = score
                # DR_Result[qid][0].append(did)
                # DR_Result[qid][1].append(score)
        print(get_retrieval_metrics(evaluator,DR_Result))
        print(i)
def evaluate_online(evaluator,base_path):
    for j in os.listdir(base_path):
        DR_Result = {}
        i = base_path +j
        with open(i,'r') as f:
            for line in tqdm(f.readlines()):
                # pdb.set_trace()

                qid=line.split(" ")[0]
                if qid not in DR_Result:
                    # DR_Result[qid] = {passage_ids[int(line.split(" ")[2])]:sigmoid(int(line.split(" ")[4])/200)}
                    DR_Result[qid] = {passage_ids[int(line.split(" ")[2])]:int(line.split(" ")[4])}
                else:
                    # DR_Result[qid][passage_ids[int(line.split(" ")[2])]] = sigmoid(int(line.split(" ")[4])/200)
                    DR_Result[qid][passage_ids[int(line.split(" ")[2])]] = int(line.split(" ")[4])
        print(get_retrieval_metrics(evaluator,DR_Result))
        print(i)
def evaluate_thu(evaluator,base_path):
    with open(base_path,'r') as f:
        DR_Result = {}
        for line in tqdm(f.readlines()):
            # pdb.set_trace()

            qid=line.split(" ")[0]
            if qid not in DR_Result:
                # DR_Result[qid] = {passage_ids[int(line.split(" ")[2])]:sigmoid(int(line.split(" ")[4])/200)}
                DR_Result[qid] = {passage_ids[int(line.split(" ")[2])]:int(line.split(" ")[4])}
            else:
                # DR_Result[qid][passage_ids[int(line.split(" ")[2])]] = sigmoid(int(line.split(" ")[4])/200)
                DR_Result[qid][passage_ids[int(line.split(" ")[2])]] = int(line.split(" ")[4])
        # pdb.set_trace()
        print(get_retrieval_metrics(evaluator,DR_Result))
        print(base_path)

if __name__ == '__main__':


    # with open(args.qrels) as handle:
    # qrels = json.load(handle)
    qrels = {}
    with open('/home/xbli/ConvDR-main/thu_results/qrels.txt') as handle:
        qrels = json.load(handle)

    DR = {}
    with open('/home/xbli/ConvDR-main/thu_results/convdr_mapped.trec','r') as f:
    
        
        for line in tqdm(f.readlines()):
            # pdb.set_trace()

            qid=line.split(" ")[0]
            if qid not in DR:
                DR[qid] = {line.split(" ")[2]:int(line.split(" ")[4])}
            else:
                DR[qid][line.split(" ")[2]] = int(line.split(" ")[4])
    # with open('/home/xbli/ConvDR-main/thu_results/passage_ids.pkl', 'rb') as handle:
    #     passage_ids = pkl.load(handle)
    # pdb.set_trace()
    
    with open('/home/xbli/ConvDR-main/datasets_passage_set_s/datasets/passage_ids.pickle','rb') as handle:
        passage_ids = pkl.load(handle)
    # check_all_positive(qrels,passage_ids)
   
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'recip_rank', 'recall'})
    
    print(get_retrieval_metrics(evaluator,DR))
    print(evaluate_thu(evaluator,'/home/xbli/ConvDR-main/results/online/multi_test_task-test.trec'))
    # print(get_retrieval_metrics(evaluator,DR_Result))
    print(evaluate_online(evaluator,'/home/xbli/ConvDR-main/results/online/'))
    # sp.specail
    # sp.special
    '''
    test:{'retriever_mrr': 0.5827675486745029, 'retriever_recall': 0.7460061030335667}
    ance test:
    train:{'retriever_mrr': 0.7973401253103126, 'retriever_recall': 0.9557825287064645}
    14999 A train {'retriever_mrr': 0.8810866684877932, 'retriever_recall': 0.97744718644928}
    5099 C train {'retriever_mrr': 0.8524688382110387, 'retriever_recall': 0.9667258770538603}
    # mrr +5
    ance train: {'retriever_mrr': 0.5785437394386225, 'retriever_recall': 0.7687305715917021}
    lr5e-6Dropout02-A 8000 {'retriever_mrr': 0.6128856382741497, 'retriever_recall': 0.7569556632561479}

    '''