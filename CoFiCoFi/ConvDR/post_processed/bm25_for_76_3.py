import pytrec_eval
import numpy as np
import json
from tqdm import tqdm
import linecache
import scipy as sp
import pdb
import pickle as pkl
import os
from rank_bm25 import BM25Okapi
import linecache

def get_title(pidx,blocks_path='****/raw/or-quac/all_blocks.txt'):
    line = linecache.getline(blocks_path, pidx + 1)
    line = json.loads(line.strip())
    return line['title'],line['bid']

def bm25_1_topk(top_passage_file,q2ner,k=10):
    map_bm = {x:(10-x)/400 for x in range(20)}
    cnt = 0
    with open(top_passage_file,'r') as f:
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
    for qid,pid2scores in tqdm(DR_Result.items()):
        flag = True
        corpus_idx = []
        q_ner = q2ner[qid]
        if q_ner == '':
            continue
        for pid,score in pid2scores.items():
            if score>194:

                continue
            corpus_idx.append(pid)
        topktitle = get_title_score_topk(q_ner,corpus_idx,k)
        for pid,score in pid2scores.items():
            passage_title,bid = get_title(id2idx[pid])
            if not any(x in passage_title.split() for x in q_ner.split()):
                # DR_Result[qid][pid] = 99
                flag = False
                
            if score<195 and not flag:
                try:
                    title_idx=topktitle.index(passage_title)
                except:
                    title_idx = -1
                if title_idx!=-1:
                    cnt += 1
                    DR_Result[qid][pid] = 195+1/(bid+2)+map_bm[title_idx]
                    # DR_Result[qid][pid] = 195+1/(bid+2)
    print(cnt)
    return DR_Result





        

def ensamble_1(top_passage_file,q2ner):
    with open(top_passage_file,'r') as f:
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
        cnt = 1
        hash_q_set = set()
        for qid,pid2scores in tqdm(DR_Result.items()):
            q_ner = q2ner[qid]
            if q_ner == '':
                continue
            flag = True
            for pid,score in pid2scores.items():
                # pdb.set_trace()
                passage_title,bid = get_title(id2idx[pid])
                if not any(x in passage_title.split() for x in q_ner.split()) or bid>6:
                    # DR_Result[qid][pid] = 99
                    flag = False
                    if bid>6:
                        DR_Result[qid][pid]=100
                
                if score<195 and not flag:
                    hash_q_set.add(qid)
                    passage_title,bid = get_title(id2idx[pid])
                    # pdb.set_trace()
                    if any(x in passage_title.split() for x in q_ner.split()):
                        # print('hi')
                        cnt += 1

                        DR_Result[qid][pid] = 195+1/(bid+2)
            DR_Result[qid] = post_process_dict(DR_Result[qid])
        print('---------------{}------------'.format(cnt))
        print(len(hash_q_set))
        pdb.set_trace()
    return DR_Result
def post_process_dict(pid_score):
    scores = sorted(list(set(pid_score.values())),reverse = True)
    score_pid = {}
    ans_dict = {}
    cnt = 0
    for item in pid_score.items():
        if str(item[1]) in score_pid:
            score_pid[str(item[1])].append(item[0])
        else:
            score_pid[str(item[1])] = [item[0]]
    for idx,item in enumerate(scores):
        
        for pid in score_pid[str(item)]:
            cnt += 1
            ans_dict[pid] = 200-cnt
            # ans_dict[pid] = 200-idx
    return ans_dict
def create_corpus(top_passage_file):
    with open(top_passage_file,'r') as f:
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
        
    
def get_title_score_topk(query,corpus_idx,k=2):
    hash_title_set = set()
    for pid in corpus_idx:
        # hash_title_set.add(get_title(idx))
        passage_title,bid = get_title(id2idx[pid])
        hash_title_set.add(passage_title)
    corpus = list(hash_title_set)
    tokenized_corpus = [doc.split(" ") for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    
    tokenized_query = query.split()
    top_title = bm25.get_top_n(tokenized_query, corpus, n=k)
    # pdb.set_trace()
    return top_title


def write_function(DR_Result,output_trec_file):
    with  open(output_trec_file, "w") as g:
        for qid, passages in DR_Result.items():
            ori_qid = qid
            # query_text = queries[ori_qid]
            # sequences = qids_to_raw_sequences[ori_qid]
            for pid, score in passages.items():
                ori_pid = smallid2idx[pid]
                g.write(
                str(ori_qid) + " Q0 " + str(ori_pid) + " " + str(pid + '1') +
                " " + str(score) + " ance\n")
# pass
if __name__ =='__main__':
    with open('****/passage_ids.pickle','rb') as handle:
        passage_ids = pkl.load(handle)
    with open('****/or-quac/passage_ids.pkl','rb') as handle:
        passage_ids_orgin = pkl.load(handle)
    # q2ner_file = '/home/xbli/ConvDR-main/datasets/raw/or-quac/q2ner.txt'
    # with open(q2ner_file,'r') as handle:
    #     str_data = handle.read()
    #     q2ner_list = json.loads(str_data)
    # q2ner = {}
    # for item in q2ner_list:
    #     q2ner.update(item)


    q2ner_file = '**/datasets/q2dialogner.txt'
    with open(q2ner_file,'r') as handle:
        str_data = handle.read()
        q2ner = json.loads(str_data)
    
    id2idx = {v:k for k,v in enumerate(passage_ids_orgin)}
    del passage_ids_orgin
    smallid2idx = {v:k for k,v in enumerate(passage_ids)}
    qrels = {}
    with open('**/thu_results/qrels.txt') as handle:
        qrels = json.load(handle)
    for i in range(8):
        qid = 'C_06717dc55d604bd69728387cee4c14bc_0_q#'+str(i)
        pid = list(qrels[qid].keys())[0]
        print(get_title(id2idx[pid]))
    pdb.set_trace()
    # for i in range(8):
    #     print(get_title(id2idx[list(qrels['C_999f9d9ff37c4fa4a5d16a2a9777cd01_1_q#{}'.format(i)].keys())[0]]))
    # print(get_title(id2idx[list(qrels['C_999f9d9ff37c4fa4a5d16a2a9777cd01_1_q#0'].keys())[0]]))

    # print(get_title(id2idx[list(qrels['C_999f9d9ff37c4fa4a5d16a2a9777cd01_1_q#7'].keys())[0]]))

    # base_path = 'retriever_5e-6reader2e-5_epoch1_weakreaderdrop01'
    # base_path = 'online_5e-62e-5_epoch1_weakreaderDp01'
    # base_path = 'retriever_5e-6reader2e-5_epoch1_weakreaderdrop01_seed52'
    # base_path = 'retriever_5e-6reader2e-5_epoch1_weakreaderdrop01_seed32'
    # No_data = '15000'
    # No_data = '15763'
    base_path = '***'
    No_data = '***'
    DR_Result = ensamble_1('**/results/{}/multi_test_task-{}.trec'.format(base_path,No_data),q2ner)
    # DR_Result = bm25_1_topk('/home/xbli/ConvDR-main/results/{}/multi_test_task-{}.trec'.format(base_path,No_data),q2ner)
    write_function(DR_Result,'**/results/{}/multi_test_task-{}_3.trec'.format(base_path,No_data))


    ### 写入文件 1.trec 规则的判断
    ### 写入文件 2.trec bm first question
    ### 写入文件 3.trec dialog question
    ### bid 6


    #ensamble first 763597 dialog 763597
    # 75893 75946 42
    # 75668   756237 32
    # 758840 75697 52
    # 758840 755594  256