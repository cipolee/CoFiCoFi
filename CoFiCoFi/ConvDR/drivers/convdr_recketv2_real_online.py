import sys
sys.path.append('/home/xbli/ConvDR-main')
import argparse
import pdb
import gc
import numpy as np
import logging
import os
import time
import torch
import random
import linecache
from tensorboardX import SummaryWriter
import pickle as pkl
from utils.util import pad_input_ids_with_mask, getattr_recursive,gen_reader_features
import json
import faiss
import scipy as sp
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm, trange
from transformers import get_linear_schedule_with_warmup
from torch import nn
from transformers import BertConfig,BertTokenizer
from model.models import MSMarcoConfigDict,Pipeline,BertForOrconvqaGlobal#,BertForOrconvqaGlobal_C_TpLoss,BertForOrconvqaGlobal_D_Dropout
from utils.util import ConvSearchDataset, NUM_FOLD, set_seed, load_model,ConvSearchDataset_V2,Pipeline_DP
from utils.dpr_utils import CheckpointState, get_model_obj, get_optimizer,get_optimizer_defferentlr

logger = logging.getLogger(__name__)
# MODEL_CLASSES = {'reader': (BertConfig, BertForOrconvqaGlobal, BertTokenizer)}
# MODEL_CLASSES = {'reader': (BertConfig, BertForOrconvqaGlobal_C_TpLoss, BertTokenizer)}
# BertForOrconvqaGlobal_C_TpLoss 需要更改预训练语言模型再试一次
# MODEL_CLASSES = {'reader': (BertConfig, BertForOrconvqaGlobal_D_Dropout, BertTokenizer)}
def mysave(model,output_dir,step):
    pass

def _save_checkpoint(args,
                     model,
                     output_dir,
                     optimizer=None,
                     scheduler=None,
                     step=0) -> str:
    offset = step
    epoch = 0
    model_to_save = get_model_obj(model)
    cp = os.path.join(output_dir, 'checkpoint-' + str(offset))

    meta_params = {}
    state = CheckpointState(model_to_save.state_dict(), optimizer.state_dict(),
                            scheduler.state_dict(), offset, epoch, meta_params)
    torch.save(state._asdict(), cp)
    logger.info('Saved checkpoint at %s', cp)
    return cp

def retrieve(args, qids, qid_to_idx, query_reps,
             passage_ids, passage_id_to_idx, passage_reps,
             qrels, qrels_sparse_matrix,
             gpu_index, include_positive_passage=False,is_evaluating = False):
    query_reps = query_reps.detach().cpu().numpy()
    D, I = gpu_index.search(query_reps, 10)

    pidx_for_retriever = np.copy(I)
    qidx = [qid_to_idx[qid] for qid in qids]
    qidx_expanded = np.expand_dims(qidx, axis=1)
    qidx_expanded = np.repeat(qidx_expanded, 10, axis=1)
    try:
        labels_for_retriever = qrels_sparse_matrix[qidx_expanded, pidx_for_retriever].toarray()
    except:
        pdb.set_trace()
    # print('labels_for_retriever before', labels_for_retriever)
    if include_positive_passage:
        for i, (qid, labels_per_query) in enumerate(zip(qids, labels_for_retriever)):
                has_positive = np.sum(labels_per_query)
                if not has_positive:
                    positive_pid = list(qrels[qid].keys())[0]
                    positive_pidx = passage_id_to_idx[positive_pid]
                    pidx_for_retriever[i][-1] = positive_pidx
        labels_for_retriever = qrels_sparse_matrix[qidx_expanded, pidx_for_retriever].toarray()
        # print('labels_for_retriever after', labels_for_retriever)
        assert np.sum(labels_for_retriever) >= len(labels_for_retriever)
    pids_for_retriever = passage_ids[pidx_for_retriever]
    passage_reps_for_retriever = passage_reps[pidx_for_retriever]
    
    args.top_k_for_reader = 10
    scores = D[:, :args.top_k_for_reader]
    retriever_probs = sp.special.softmax(scores, axis=1)
    pidx_for_reader = I[:, :args.top_k_for_reader]
    


        
    qidx_expanded = np.expand_dims(qidx, axis=1)
    qidx_expanded = np.repeat(qidx_expanded, args.top_k_for_reader, axis=1)
    # print('qidx_expanded', qidx_expanded)
    
    labels_for_reader = qrels_sparse_matrix[qidx_expanded, pidx_for_reader].toarray()
    # pdb.set_trace()
    # positive_num.append(np.sum(labels_for_reader))
    # assert np.sum(labels_for_reader) >= len(labels_for_reader)
    # print('labels_for_reader before', labels_for_reader)
    # print('labels_for_reader before', labels_for_reader)
    if include_positive_passage:
        for i, (qid, labels_per_query) in enumerate(zip(qids, labels_for_reader)):
                has_positive = np.sum(labels_per_query)
                if not has_positive:
                    positive_pid = list(qrels[qid].keys())[0]
                    positive_pidx = passage_id_to_idx[positive_pid]
                    pidx_for_reader[i][-1] = positive_pidx
        labels_for_reader = qrels_sparse_matrix[qidx_expanded, pidx_for_reader].toarray()
        # print('labels_for_reader after', labels_for_reader)
        assert np.sum(labels_for_reader) >= len(labels_for_reader)
    # print('labels_for_reader after', labels_for_reader)
    # pos_neg_embeddings = 
    pids_for_reader = passage_ids[pidx_for_reader]
    passage_reps_for_reader = passage_reps[pidx_for_reader]
    # print('pids_for_reader', pids_for_reader)
    passages_for_reader = get_passages(pidx_for_reader, args)
    # we do not need to modify scores and probs matrices because they will only be
    # needed at evaluation, where include_positive_passage will be false

    return {'qidx': qidx,
            'pidx_for_retriever': pidx_for_retriever,
            'pids_for_retriever': pids_for_retriever,
            'passage_reps_for_retriever': passage_reps_for_retriever,
            'labels_for_retriever': labels_for_retriever,
            'retriever_probs': retriever_probs,
            'pidx_for_reader': pidx_for_reader,
            'pids_for_reader': pids_for_reader,
            'passage_reps_for_reader': passage_reps_for_reader,
            'passages_for_reader': passages_for_reader, 
            'labels_for_reader': labels_for_reader}


# In[8]:


def get_passage(i, args):
    mapped_i = pidx2allblocks[i]
    line = linecache.getline(args.blocks_path, mapped_i + 1)
    line = json.loads(line.strip())
    return line['text']
get_passages = np.vectorize(get_passage)
def train(args,
          train_dataset,
          gpu_index,
          passage_reps,
          model,
          teacher_model,
          loss_fn,
          logger,
          writer: SummaryWriter,
          cross_validate_id=-1,
          loss_fn_2=None,
          tokenizer=None,
          reader_tokenizer=None):
    # args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu) # 为了占用0号卡
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu-1)#- faiss的卡
    
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset,
                                  sampler=train_sampler,
                                  batch_size=args.train_batch_size,
                                  collate_fn=train_dataset.get_collate_fn(
                                      args, "train"))

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (
            len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(
            train_dataloader
        ) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    # optimizer = get_optimizer(args, model, weight_decay=args.weight_decay)
    optimizer = get_optimizer_defferentlr(args, model, weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.fp16_opt_level)
    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d",
                args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d",
                args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    tr_loss1, tr_loss2 = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    set_seed(
        args)  # Added here for reproducibility (even between python 2 and 3)
    passage_id_pseudo =0
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            if step == 5000:
            # pdb.set_trace() 
                # for param in model.reader.bert.encoder.layer[args.freezed_reader_layer].parameters():
                for param in model.reader.bert.encoder.parameters():
                    param.requires_grad = False
                # pdb.set_trace()
            concat_ids, concat_id_mask, target_ids, target_id_mask = (ele.to(
                args.device) for ele in [
                    batch["concat_ids"], batch["concat_id_mask"],
                    batch["target_ids"], batch["target_id_mask"]
                ])
            model.train()
            teacher_model.eval()
            # embs = model.module.retriever(concat_ids, concat_id_mask) # 修改
            embs = model(encoder_name='retriever',concat_ids=concat_ids, concat_id_mask=concat_id_mask)
            with torch.no_grad():
                teacher_embs = teacher_model(target_ids,
                                             target_id_mask).detach()
            loss1 = None
            if not args.no_mse:
                loss1 = loss_fn(embs, teacher_embs)
            loss = loss1
            # pdb.set_trace()
            loss2 = None
            if args.ranking_task:
                
                bs = concat_ids.shape[0]  # real batch size
                pos_and_negs = batch["documents"]  #.numpy()  # (B, K)
                
                answer_starts = np.asarray(
                batch['answer_starts']).reshape(-1).tolist()
                question_texts = batch['question_texts']
                qids = batch['qid']
                answer_texts = batch['answer_texts']
                
                

                retrieval_results = retrieve(args, qids, qid_to_idx, embs,
                                         passage_ids, passage_id_to_idx, passage_reps,
                                         qrels, qrels_sparse_matrix,
                                         gpu_index, include_positive_passage=True)
                passage_reps_for_retriever = retrieval_results['passage_reps_for_retriever']
                labels_for_retriever = retrieval_results['labels_for_retriever']

                pids_for_reader = retrieval_results['pids_for_reader']
                pidxs_for_reader = retrieval_results['pidx_for_reader']
                # print(pids_for_reader)
                passages_for_reader = retrieval_results['passages_for_reader']
                labels_for_reader = retrieval_results['labels_for_reader']




                #############################
                # pos_and_negs_embeddings = torch.cat(pos_and_negs_embeddings,
                #                                     dim=0)  # (B * K, E)
                # pos_and_negs_embeddings = pos_and_negs_embeddings.view(
                #     bs, args.num_negatives + 1, -1)  # (B, K, E)
                pos_and_negs_embeddings = retrieval_results['passage_reps_for_retriever']
                embs_for_ranking = embs.unsqueeze(-1)  # (B, E, 1)
                embs_for_ranking = embs_for_ranking.expand(
                    bs, 768, args.num_negatives + 1)  # (B, E, K)
                embs_for_ranking = embs_for_ranking.transpose(1,
                                                              2)  # (B, K, E)
                # pdb.set_trace()
                logits = embs_for_ranking * torch.tensor(pos_and_negs_embeddings,dtype=torch.float).to(args.device)
                


                




                logits = torch.sum(logits, dim=-1)  # (B, K)
                # pdb.set_trace()
                if torch.min(logits) >60:
                    logits = logits - 60
                else:
                    logits = logits - torch.min(logits) +5
                # pdb.set_trace()
                ######## xbli
                # 阅读理解任务
                # question_texts [B]
                # passage_texts [B,K]
                # retrieval_label [B,K] 需要传进去，用于判断is_impossible类型
                # answer [B] answer_start [B]
                
                # passages_for_reader = batch_passage_texts
                # labels_for_reader = torch.zeros((bs,args.num_negatives + 1),dtype=torch.long)
                # # bs随着n_gpu变化而变化
                # labels_for_reader[:,0]=1

                # pdb.set_trace()
                # if 'CANNOTANSWER' in answer_texts:
                #     pdb.set_trace()
                reader_batch = gen_reader_features(qids, question_texts, answer_texts, answer_starts,pids_for_reader,
                                        passages_for_reader, labels_for_reader,
                                        reader_tokenizer, args.reader_max_seq_length, is_training=True)


                # if 'CANNOTANSWER' in answer_texts:
                #     pdb.set_trace()
                reader_batch = {k: v.to(args.device) for k, v in reader_batch.items()}

                inputs = {'input_ids':       reader_batch['input_ids'],
                        'attention_mask':  reader_batch['input_mask'],
                        'token_type_ids':  reader_batch['segment_ids'],
                        'start_positions': reader_batch['start_position'],
                        'end_positions':   reader_batch['end_position'],
                        'retrieval_label': reader_batch['retrieval_label'],
                        'encoder_name':'reader'#新加的
                        }
                # pdb.set_trace()

                ###### 内存节省
                # pdb.set_trace()
                # del embs_for_ranking
                # del pos_and_negs_embeddings
                # del batch_ids
                # del batch_mask
                # del reader_batch
                # gc.collect()
                # time.sleep(5)
                ##########3
                # pdb.set_trace()
                # reader_outputs = model.module.reader(**inputs)
                reader_outputs = model(**inputs)
                
                reader_loss, qa_loss, rerank_loss, reader_logits = reader_outputs[0:4]
                p = torch.log_softmax(logits, dim=-1)
                q_tec = torch.softmax(reader_logits, dim=-1)
                # pdb.set_trace()
                retriever_loss = torch.nn.functional.kl_div(p, q_tec, reduction='none').sum()
                # pdb.set_trace()
                reader_loss = reader_loss.mean()
                loss2 = retriever_loss + reader_loss
                if random.random()>0.9:
                    print('loss1 : {}, retriever_loss : {}, qa_loss : {}'.format(loss1,retriever_loss,reader_loss))
                # pdb.set_trace()


                ######## xbli
                # labels = torch.zeros(bs, dtype=torch.long).to(args.device)
                # loss2 = loss_fn_2(logits, labels)
                loss = loss1 + loss2 if loss1 != None else loss2

            if args.n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            # loss.backward()
            
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            tr_loss += loss.item()
            if not args.no_mse:
                tr_loss1 += loss1.item()
            if args.ranking_task:
                tr_loss2 += loss2.item()
            del loss
            torch.cuda.empty_cache()
            # time.sleep(2)

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if global_step % args.log_steps == 0:
                    writer.add_scalar(
                        str(cross_validate_id) + "/loss",
                        tr_loss / args.log_steps, global_step)
                    if not args.no_mse:
                        writer.add_scalar(
                            str(cross_validate_id) + "/mse_loss",
                            tr_loss1 / args.log_steps, global_step)
                    if args.ranking_task:
                        writer.add_scalar(
                            str(cross_validate_id) + "/ranking_loss",
                            tr_loss2 / args.log_steps, global_step)
                    tr_loss = 0.0
                    tr_loss1 = 0.0
                    tr_loss2 = 0.0

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    checkpoint_prefix = 'checkpoint'
                    output_dir = args.output_dir + (
                        ('-' + str(cross_validate_id))
                        if cross_validate_id != -1 else "")
                    if args.model_type == "rdot_nll":
                        output_dir = os.path.join(
                            output_dir, '{}-{}'.format(checkpoint_prefix,
                                                       global_step))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = model.module.retriever if hasattr(
                            model, 'module') else model
                        model_to_save.save_pretrained(output_dir)
                        torch.save(
                            args, os.path.join(output_dir,
                                               'training_args.bin'))
                    else:
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        _save_checkpoint(args, model, output_dir, optimizer,
                                         scheduler, global_step)
                    logger.info("Saving model checkpoint to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.model_type == "dpr":
        output_dir = args.output_dir + (
            ('-' + str(cross_validate_id)) if cross_validate_id != -1 else "")
        # output_dir = os.path.join(output_dir, '{}-{}'.format(checkpoint_prefix, global_step))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        _save_checkpoint(args, model, output_dir, optimizer, scheduler,
                         global_step)

    return global_step, tr_loss / global_step


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help=
        "The output directory where the model predictions and checkpoints will be written."
    )
    parser.add_argument("--reader_config_name", default="", type=str,
                    help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--reader_model_name_or_path", default='/home/jqian/pre-train_model/bert-base-uncased', type=str, required=False,
                        help="reader model name")
    parser.add_argument("--reader_model_type", default='bert', type=str, required=False,
                        help="reader model type")
    parser.add_argument("--reader_tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    # parser.add_argument("--reader_tokenizer_name", default="bert-base-uncased", type=str,
    #                     help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="The model checkpoint for weights initialization.")
    parser.add_argument(
        "--max_concat_length",
        default=256,
        type=int,
        help="Optional input sequence length after tokenization."
        "The training dataset will be truncated in block of this size for training."
        "Default to the model max input length for single sentence inputs (take into account special tokens)."
    )
    parser.add_argument(
        "--max_query_length", 
        default=64, 
        type=int,
        help="Max input query length after tokenization."
             "This option is for single query input."
    )
    parser.add_argument("--do_lower_case", default=True, type=bool,
                    help="Set this flag if you are using an uncased model.")
    parser.add_argument("--qa_loss_factor", default=1.0, type=float,
                    help="total_loss = qa_loss_factor * qa_loss + retrieval_loss_factor * retrieval_loss")
    parser.add_argument("--retrieval_loss_factor", default=1.0, type=float,
                    help="total_loss = qa_loss_factor * qa_loss + retrieval_loss_factor * retrieval_loss")
    parser.add_argument(
        "--train_file",
        default=None,
        type=str,
        required=True,
        help=
        "Path of training file. Do not add fold suffix when cross validate, i.e. use 'data/eval_topics.jsonl' instead of 'data/eval_topics.jsonl.0'"
    )
    parser.add_argument(
        "--cross_validate",
        action='store_true',
        help="Set when doing cross validation"
    )
    parser.add_argument(
        "--init_from_multiple_models",
        action='store_true',
        help=
        "Set when initialize from different models during cross validation (Model-based+CV)"
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " +
        ", ".join(MSMarcoConfigDict.keys()),
    )

    parser.add_argument(
        "--ranking_task", 
        action='store_true',
        help="Whether to use ranking loss."
    )
    parser.add_argument(
        "--no_mse", 
        action="store_true",
        help="Whether to disable KD loss."
    )
    parser.add_argument(
        "--num_negatives", 
        type=int, 
        default=9,
        help="Number of negative documents per query."
    )
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                    help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                         "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--fp16', default=False, type=bool,
                    help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument(
        "--per_gpu_train_batch_size",
        default=4,
        type=int,
        help="Batch size per GPU/CPU for training."
    )
    parser.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=1,
        help=
        "Number of updates steps to accumulate before performing a backward/update pass."
    )
    parser.add_argument(
        "--learning_rate",
        default=1e-5,
        type=float,
        help="The initial learning rate for Adam."
    )
    parser.add_argument(
        "--weight_decay",
        default=0.0,
        type=float,
        help="Weight deay if we apply some."
    )
    parser.add_argument(
        "--include_first_question",
        default=True,
        type=bool,
        help="include the first or not."
    )
    parser.add_argument(
        "--adam_epsilon",
        default=1e-8,
        type=float,
        help="Epsilon for Adam optimizer."
    )
    parser.add_argument(
        "--max_grad_norm",
        default=1.0,
        type=float,
        help="Max gradient norm."
    )
    parser.add_argument(
        "--num_train_epochs",
        default=1.0,
        type=float,
        help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help=
        "If > 0: set total number of training steps to perform. Override num_train_epochs."
    )
    parser.add_argument(
        "--warmup_steps",
        default=0,
        type=int,
        help="Linear warmup over warmup_steps."
    )
    parser.add_argument("--local_rank", type=int, default=-1,
                    help="local_rank for distributed training on gpus")
    parser.add_argument(
        '--save_steps',
        type=int,
        default=-1,
        help="Save checkpoint every X updates steps."
    )
    parser.add_argument("--reader_max_seq_length", default=512, type=int,
                    help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                         "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument(
        "--no_cuda",
        action='store_true',
        help="Avoid using CUDA when available"
    )
    parser.add_argument(
        '--overwrite_output_dir',
        action='store_true',
        help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help="random seed for initialization"
    )
    parser.add_argument(
        "--log_dir", 
        type=str,
        help="Directory for tensorboard logging."
    )
    parser.add_argument(
        "--log_steps", 
        type=int, 
        default=1,
        help="Log loss every x steps."
    )
    parser.add_argument(
        "--cache_dir", 
        type=str
    )
    parser.add_argument(
        "--teacher_model", 
        type=str,
        help="The teacher model. If None, use `model_name_or_path` as teacher."
    )
    parser.add_argument(
        "--query",
        type=str,
        default="no_res",
        choices=["no_res", "man_can", "auto_can", "target", "output", "raw"],
        help="Input query format."
    )
    args = parser.parse_args()


    ################add
    args.blocks_path = '/home/xbli/ConvDR-main/datasets/raw/or-quac/all_blocks.txt'



    #####################
    tb_writer = SummaryWriter(log_dir=args.log_dir)
    
    # Setup CUDA, GPU & distributed training
    # we now only support joint training on a single card
    # we will request two cards, one for torch and the other one for faiss
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        # args.n_gpu = torch.cuda.device_count()
        args.n_gpu = 1
        # torch.cuda.set_device(0)
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome."
            .format(args.output_dir))
    if args.fp16:
        try:
            import apex
            apex.amp.register_half_function(torch, 'einsum')
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

    # device = torch.device(
    #     "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()-1
    # args.device = device

    # Setup logging
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO)
    logger.warning("device: %s, n_gpu: %s", device, args.n_gpu)

    # Set seed
    set_seed(args)

    loss_fn = nn.MSELoss()
    loss_fn.to(args.device)
    loss_fn_2 = nn.CrossEntropyLoss()
    loss_fn_2.to(args.device)

    if args.teacher_model == None:
        args.teacher_model = args.model_name_or_path
        # args.teacher_model = ''
    _, __, teacher_model = load_model(args, args.teacher_model)

    if not args.cross_validate:
        # if args.local_rank == -1 or args.no_cuda:
        #     device = torch.device(
        #     "cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        #     # args.n_gpu = torch.cuda.device_count()
        #     args.n_gpu = 1
        #     # torch.cuda.set_device(0)
        # else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        #     torch.cuda.set_device(args.local_rank)
        #     device = torch.device("cuda", args.local_rank)
        #     torch.distributed.init_process_group(backend='nccl')
        #     args.n_gpu = 1
        # args.device = device
        #### xbli
        # model = Pipeline()
        
        config, tokenizer, _,reader_tokenizer = load_model(args, args.model_name_or_path,is_pipeline=True)
        model = Pipeline_DP(args)
        if args.query in ["man_can", "auto_can"]:
            tokenizer.add_tokens(["<response>"])
            model.retriever.resize_token_embeddings(len(tokenizer))
            # model.resize_token_embeddings(len(tokenizer))
        if args.max_concat_length <= 0:
            args.max_concat_length = tokenizer.max_len_single_sentence
        args.max_concat_length = min(args.max_concat_length,
                                     tokenizer.max_len_single_sentence)

        ####### 改写pipeline——dp后加的 
        model.to(args.device)
        #
        # Training
        logger.info("Training/evaluation parameters %s", args)
        # train_dataset = ConvSearchDataset([args.train_file],
        #                                   tokenizer,
        #                                   args,
        #                                   mode="train")
        qid2answer_file = '/home/xbli/ConvDR-main/datasets/or-quac/q2answers.pickle'
        if args.include_first_question:
            q2ner_file = '/home/xbli/ConvDR-main/datasets/raw/or-quac/q2ner.txt'
            with open(q2ner_file,'r') as handle:
                str_data = handle.read()
                q2ner_list = json.loads(str_data)
            q2ner = {}
            for item in q2ner_list:
                q2ner.update(item)
            train_dataset = ConvSearchDataset_V2([args.train_file],qid2answer_file,
                                          tokenizer,
                                          args,
                                          mode="train",q2ner=q2ner)
        else:
            train_dataset = ConvSearchDataset_V2([args.train_file],qid2answer_file,
                                          tokenizer,
                                          args,
                                          mode="train")
        logger.info('constructing passage faiss_index')
        faiss_res = faiss.StandardGpuResources() 
        index = faiss.IndexFlatIP(768)
        index.add(passage_reps)
        gpu_index = faiss.index_cpu_to_gpu(faiss_res, 1, index)
        # model.to('cuda')
        global_step, tr_loss = train(args,
                                     train_dataset,
                                     gpu_index,
                                     passage_reps,
                                     model,
                                     teacher_model,
                                     loss_fn,
                                     logger,
                                     tb_writer,
                                     cross_validate_id=0,
                                     loss_fn_2=loss_fn_2,
                                     tokenizer=tokenizer,
                                     reader_tokenizer=reader_tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step,
                    tr_loss)

        # Saving
        # Create output directory if needed

        if args.model_type == "rdot_nll":
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)
            logger.info("Saving model checkpoint to %s", args.output_dir)
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            torch.save(args, os.path.join(args.output_dir,
                                          'training_args.bin'))

    else:
        # K-Fold Cross Validation
        for i in range(NUM_FOLD):
            logger.info("Training Fold #{}".format(i))
            suffix = ('-' + str(i)) if args.init_from_multiple_models else ''
            config, tokenizer, model = load_model(
                args, args.model_name_or_path + suffix)
            if args.query in ["man_can", "auto_can"]:
                tokenizer.add_tokens(["<response>"])
                model.resize_token_embeddings(len(tokenizer))

            if args.max_concat_length <= 0:
                args.max_concat_length = tokenizer.max_len_single_sentence
            args.max_concat_length = min(args.max_concat_length,
                                         tokenizer.max_len_single_sentence)

            logger.info("Training/evaluation parameters %s", args)
            train_files = [
                "%s.%d" % (args.train_file, j) for j in range(NUM_FOLD)
                if j != i
            ]
            logger.info("train_files: {}".format(train_files))
            train_dataset = ConvSearchDataset(train_files,
                                              tokenizer,
                                              args,
                                              mode="train")
            global_step, tr_loss = train(args,
                                         train_dataset,
                                         model,
                                         teacher_model,
                                         loss_fn,
                                         logger,
                                         tb_writer,
                                         cross_validate_id=i,
                                         loss_fn_2=loss_fn_2,
                                         tokenizer=tokenizer)
            logger.info(" global_step = %s, average loss = %s", global_step,
                        tr_loss)

            if args.model_type == "rdot_nll":
                output_dir = args.output_dir + '-' + str(i)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                logger.info("Saving model checkpoint to %s", output_dir)
                model_to_save = model.module if hasattr(model,
                                                        'module') else model
                model_to_save.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)
                torch.save(args, os.path.join(output_dir, 'training_args.bin'))

            del model
            torch.cuda.empty_cache()

    tb_writer.close()


if __name__ == "__main__":
    passage_ids_path = '/home/xbli/ConvDR-main/datasets_passage_set_s/datasets/passage_ids.pickle'
    passage_reps_path = '/home/xbli/ConvDR-main/datasets_passage_set_s/datasets/passage_reps.pickle'
    qrels_path = '/home/xbli/ConvDR-main/datasets/raw/or-quac/qrels.txt'
    pidx2allblocks_path = '/home/xbli/ConvDR-main/datasets_passage_set_s/datasets/newpidx2oldpidx.pickle'
    logger.info(f'loading passage ids from {passage_ids_path}')
    with open(passage_ids_path, 'rb') as handle:
        passage_ids = pkl.load(handle)
    logger.info(f'loading passage reps from {passage_reps_path}')
    with open(passage_reps_path, 'rb') as handle:
        passage_reps = pkl.load(handle)
    
    logger.info(f'loading qrels from {qrels_path}')
    with open(qrels_path) as handle:
        qrels = json.load(handle)
    with open(pidx2allblocks_path,'rb') as handle:
        pidx2allblocks = pkl.load(handle)
    passage_id_to_idx = {}
    for i, pid in enumerate(passage_ids):
        passage_id_to_idx[pid] = i
    # pdb.set_trace()
    qrels_data, qrels_row_idx, qrels_col_idx = [], [], []
    qid_to_idx = {}
    max_ = 0
    for i, (qid, v) in enumerate(qrels.items()):
        qid_to_idx[qid] = i
        for pid in v.keys():
            qrels_data.append(1)
            qrels_row_idx.append(i)
            max_ = max(max_,passage_id_to_idx[pid])
            qrels_col_idx.append(passage_id_to_idx[pid])
    print(max_)
    qrels_data.append(0)
    qrels_row_idx.append(len(qrels))
    # max_ = max(max_,passage_id_to_idx[pid])
    qrels_col_idx.append(len(passage_id_to_idx)-1)
    qrels_sparse_matrix = sp.sparse.csr_matrix(
        (qrels_data, (qrels_row_idx, qrels_col_idx)))
    # pdb.set_trace()
    main()
