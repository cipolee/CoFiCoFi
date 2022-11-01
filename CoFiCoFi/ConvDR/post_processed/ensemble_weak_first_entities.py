import pickle as pkl
import pdb
import json
import os
from tqdm import tqdm

def load_pid2score(ensemble_file1,ensemble_file2):
    DR = {}
    files = [ensemble_file1,ensemble_file2]
    for i,file in enumerate(files):
        print(file)
        with open(file,'r') as f:
            for line in tqdm(f.readlines()):
                
                qid=line.split(" ")[0]

                pidx = int(line.split(" ")[2])

                score = int(line.split(" ")[4])
                if qid not in DR:
                    DR[qid] = {}
                if pidx not in DR[qid]:
                    DR[qid][pidx] = 0
                
                DR[qid][pidx] += 0.1*score if i==0 else 5*score
    return DR


                

def sort_ensemble(q2pscores):
    q2sortedps = {}
    for qid,sortedps in q2pscores.items():
        pids = list(sortedps.keys())
        list_p_score = [[pid,sortedps[pid]] for pid in pids]
        list_p_score = sorted(list_p_score,key = lambda x:x[1],reverse=True)
        q2sortedps[qid]=[i[0] for i in list_p_score]
    return q2sortedps

def write_trec(q2sortedps,output_trec_file):
    with open(output_trec_file,'w') as g:
        for qid,sortedps in q2sortedps.items():
            for i,pid in enumerate(sortedps):
                g.write(
                str(qid) + " Q0 " + str(pid) + " " + str(i + 1) +
                " " + str(-i - 1 + 200) + " ance\n")




if __name__ == '__main__':
    # base_name = '/home/xbli/ConvDR-main/results/online_5e-62e-5_epoch1_weakreaderDp01/'
    base_name = '**/results/retriever_5e-6reader2e-5_epoch1_weakreaderdrop01_seed32/'
    # base_name = '/home/xbli/ConvDR-main/results/retriever_5e-6reader2e-5_epoch1_weakreaderdrop01_seed256/'
    # base_name = '/home/xbli/ConvDR-main/results/retriever_5e-6reader2e-5_epoch1_weakreaderdrop01_seed52/'
    output_trec_file = base_name + 'multi_test_task-15763_4.trec'
    ensemble_file1 =  base_name + 'multi_test_task-15763.trec'
    ensemble_file2 =  base_name + 'multi_test_task-15763_3.trec'
    DR = load_pid2score(ensemble_file1,ensemble_file2)
    q2sortedps = sort_ensemble(DR)
    write_trec(q2sortedps,output_trec_file)

    pass