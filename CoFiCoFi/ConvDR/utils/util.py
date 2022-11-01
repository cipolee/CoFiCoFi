import sys

sys.path += ['../']
import pdb
# import pandas as pd
# from sklearn.metrics import roc_curve, auc
import gzip
import copy
import torch
from torch import nn
import torch.distributed as dist
from tqdm import tqdm, trange
import os
import pickle as pkl
from os import listdir
from os.path import isfile, join
import json
import logging
import random
import pytrec_eval
import collections
# import pickle
import numpy as np
import torch
from transformers.tokenization_bert import BasicTokenizer, whitespace_tokenize
from transformers import BertTokenizer,BertConfig
torch.multiprocessing.set_sharing_strategy('file_system')
from multiprocessing import Process
from torch.utils.data import DataLoader, Dataset, TensorDataset, IterableDataset
from utils.dpr_utils import get_model_obj, load_states_from_checkpoint
import re
from model.models import MSMarcoConfigDict, ALL_MODELS,Pipeline
from typing import List, Set, Dict, Tuple, Callable, Iterable, Any

logger = logging.getLogger(__name__)
NUM_FOLD = 5
from model.models import MSMarcoConfigDict,Pipeline,BertForOrconvqaGlobal_DR_01,BertForOrconvqaGlobal0,BertForOrconvqaGlobal #,BertForOrconvqaGlobal_C_TpLoss
# MODEL_CLASSES = {'reader': (BertConfig, BertForOrconvqaGlobal_DR_01, BertTokenizer)}
MODEL_CLASSES = {'reader': (BertConfig, BertForOrconvqaGlobal, BertTokenizer)}

class InputFeaturesPair(object):
    """
    A single set of features of data.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        label: Label corresponding to the input
    """
    def __init__(self,
                 input_ids_a,
                 attention_mask_a=None,
                 token_type_ids_a=None,
                 input_ids_b=None,
                 attention_mask_b=None,
                 token_type_ids_b=None,
                 label=None):

        self.input_ids_a = input_ids_a
        self.attention_mask_a = attention_mask_a
        self.token_type_ids_a = token_type_ids_a

        self.input_ids_b = input_ids_b
        self.attention_mask_b = attention_mask_b
        self.token_type_ids_b = token_type_ids_b

        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def getattr_recursive(obj, name):
    for layer in name.split("."):
        if hasattr(obj, layer):
            obj = getattr(obj, layer)
        else:
            return None
    return obj


def barrier_array_merge(args,
                        data_array,
                        merge_axis=0,
                        prefix="",
                        load_cache=False,
                        only_load_in_master=False,
                        merge=True):
    # data array: [B, any dimension]
    # merge alone one axis

    if args.local_rank == -1:
        return data_array

    if not load_cache:
        rank = args.rank
        if is_first_worker():
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)

        dist.barrier()  # directory created
        pickle_path = os.path.join(
            args.output_dir, "{1}_data_obj_{0}.pb".format(str(rank), prefix))
        with open(pickle_path, 'wb') as handle:
            pkl.dump(data_array, handle, protocol=4)

        # make sure all processes wrote their data before first process
        # collects it
        dist.barrier()

    data_array = None

    data_list = []

    if not merge:
        return None

    # return empty data
    if only_load_in_master:
        if not is_first_worker():
            dist.barrier()
            return None

    for i in range(args.world_size
                   ):  # TODO: dynamically find the max instead of HardCode
        pickle_path = os.path.join(
            args.output_dir, "{1}_data_obj_{0}.pb".format(str(i), prefix))
        try:
            with open(pickle_path, 'rb') as handle:
                b = pkl.load(handle)
                data_list.append(b)
        except BaseException:
            continue

    data_array_agg = np.concatenate(data_list, axis=merge_axis)
    dist.barrier()
    return data_array_agg


def pad_input_ids(input_ids, max_length, pad_on_left=False, pad_token=0):
    padding_length = max_length - len(input_ids)
    padding_id = [pad_token] * padding_length

    # attention_mask = [1] * len(input_ids) + [0] * padding_length

    if padding_length <= 0:
        input_ids = input_ids[:max_length]
    else:
        if pad_on_left:
            input_ids = padding_id + input_ids
        else:
            input_ids = input_ids + padding_id

    return input_ids


def pad_input_ids_with_mask(input_ids,
                            max_length,
                            pad_on_left=False,
                            pad_token=0):
    padding_length = max_length - len(input_ids)
    padding_id = [pad_token] * padding_length

    attention_mask = []

    if padding_length <= 0:
        input_ids = input_ids[:max_length]
        attention_mask = [1] * max_length
    else:
        if pad_on_left:
            input_ids = padding_id + input_ids
        else:
            attention_mask = [1] * len(input_ids) + [0] * padding_length
            input_ids = input_ids + padding_id

    assert len(input_ids) == max_length
    assert len(attention_mask) == max_length

    return input_ids, attention_mask


def pad_ids(input_ids,
            attention_mask,
            token_type_ids,
            max_length,
            pad_on_left=False,
            pad_token=0,
            pad_token_segment_id=0,
            mask_padding_with_zero=True):
    padding_length = max_length - len(input_ids)
    padding_id = [pad_token] * padding_length
    padding_type = [pad_token_segment_id] * padding_length
    padding_attention = [0 if mask_padding_with_zero else 1] * padding_length

    if padding_length <= 0:
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        token_type_ids = token_type_ids[:max_length]
    else:
        if pad_on_left:
            input_ids = padding_id + input_ids
            attention_mask = padding_attention + attention_mask
            token_type_ids = padding_type + token_type_ids
        else:
            input_ids = input_ids + padding_id
            attention_mask = attention_mask + padding_attention
            token_type_ids = token_type_ids + padding_type

    return input_ids, attention_mask, token_type_ids


# to reuse pytrec_eval, id must be string
def convert_to_string_id(result_dict):
    string_id_dict = {}

    # format [string, dict[string, val]]
    for k, v in result_dict.items():
        _temp_v = {}
        for inner_k, inner_v in v.items():
            _temp_v[str(inner_k)] = inner_v

        string_id_dict[str(k)] = _temp_v

    return string_id_dict


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
def load_model_V2(args, checkpoint_path):
    label_list = ["0", "1"]
    num_labels = len(label_list)
    args.model_type = args.model_type.lower()
    configObj = MSMarcoConfigDict[args.model_type]
    args.model_path = checkpoint_path
    config, tokenizer, model = None, None, None

    model = configObj.model_class(args)
    
    tokenizer = configObj.tokenizer_class.from_pretrained(
        "bert-base-uncased",
        do_lower_case=True,
        cache_dir=None,
    )
    Hybrid_model = Pipeline()
    reader_config_class, reader_model_class, reader_tokenizer_class = MODEL_CLASSES['reader']
    reader_config = reader_config_class.from_pretrained('bert-base-uncased')
    reader_config.num_qa_labels = 2
    # this not used for BertForOrconvqaGlobal
    reader_config.num_retrieval_labels = 2
    reader_config.qa_loss_factor = 1.0
    reader_config.retrieval_loss_factor = 1.0

    reader_model = reader_model_class.from_pretrained('bert-base-uncased',config=reader_config)

    Hybrid_model.retriever = model
    Hybrid_model.reader = reader_model

    saved_state = load_states_from_checkpoint(checkpoint_path)
    model_to_load = get_model_obj(Hybrid_model)
    logger.info('Loading saved model state ...')
    model_to_load.load_state_dict(saved_state.model_dict)

    Hybrid_model.to(args.device)
    return config,tokenizer,Hybrid_model.retriever
        

def load_model(args, checkpoint_path,is_pipeline=False,is_todevice=True):
    label_list = ["0", "1"]
    num_labels = len(label_list)
    args.model_type = args.model_type.lower()
    configObj = MSMarcoConfigDict[args.model_type]
    args.model_path = checkpoint_path
    config, tokenizer, model = None, None, None
    if args.model_type != "dpr":
        config = configObj.config_class.from_pretrained(
            args.model_path,
            num_labels=num_labels,
            finetuning_task="MSMarco",
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        tokenizer = configObj.tokenizer_class.from_pretrained(
            args.model_path,
            do_lower_case=True,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        model = configObj.model_class.from_pretrained(
            args.model_path,
            from_tf=bool(".ckpt" in args.model_path),
            config=config,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
    else:  # dpr

        model = configObj.model_class(args)
        saved_state = load_states_from_checkpoint(checkpoint_path)
        model_to_load = get_model_obj(model)
        logger.info('Loading saved model state ...')
        model_to_load.load_state_dict(saved_state.model_dict)
        tokenizer = configObj.tokenizer_class.from_pretrained(
            "bert-base-uncased",
            do_lower_case=True,
            cache_dir=None,
        )
        if is_pipeline:
            Hybrid_model = Pipeline()
            reader_config_class, reader_model_class, reader_tokenizer_class = MODEL_CLASSES['reader']
            reader_config = reader_config_class.from_pretrained(args.reader_model_name_or_path)
            reader_config.num_qa_labels = 2
            # this not used for BertForOrconvqaGlobal
            reader_config.num_retrieval_labels = 2
            reader_config.qa_loss_factor = args.qa_loss_factor
            reader_config.retrieval_loss_factor = args.retrieval_loss_factor

            reader_tokenizer = reader_tokenizer_class.from_pretrained(args.reader_tokenizer_name if args.reader_tokenizer_name else args.reader_model_name_or_path,
                                                                    do_lower_case=args.do_lower_case)
            reader_model = reader_model_class.from_pretrained(args.reader_model_name_or_path,
                                                            config=reader_config)

            Hybrid_model.retriever = model
            Hybrid_model.reader = reader_model

            Hybrid_model.to(args.device)
            return config,tokenizer,Hybrid_model,reader_tokenizer
        else:
            if is_todevice:
                model.to(args.device)
            return config, tokenizer, model


def is_first_worker():
    return not dist.is_available() or not dist.is_initialized(
    ) or dist.get_rank() == 0


def concat_key(all_list, key, axis=0):
    return np.concatenate([ele[key] for ele in all_list], axis=axis)


def get_checkpoint_no(checkpoint_path):
    return int(re.findall(r'\d+', checkpoint_path)[-1])


def get_latest_ann_data(ann_data_path):
    ANN_PREFIX = "ann_ndcg_"
    if not os.path.exists(ann_data_path):
        return -1, None, None
    files = list(next(os.walk(ann_data_path))[2])
    num_start_pos = len(ANN_PREFIX)
    data_no_list = [
        int(s[num_start_pos:]) for s in files
        if s[:num_start_pos] == ANN_PREFIX
    ]
    if len(data_no_list) > 0:
        data_no = max(data_no_list)
        with open(os.path.join(ann_data_path, ANN_PREFIX + str(data_no)),
                  'r') as f:
            ndcg_json = json.load(f)
        return data_no, os.path.join(ann_data_path, "ann_training_data_" +
                                     str(data_no)), ndcg_json
    return -1, None, None

def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False
  


class QuacExample(object):
    """
    A single training/test example for the QuAC dataset.
    For examples without an answer, the start and end position are -1.
    """

    def __init__(self,
                 example_id,
                 qas_id,
                 question_text,
                 doc_tokens,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None,
                 is_impossible=None, 
                 followup=None, 
                 yesno=None, 
                 retrieval_label=None, 
                 history=None,
                 start_position_his=None,
                 end_position_his=None,
                 last_answer_text=None):
        self.example_id = example_id
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible
        self.followup = followup
        self.yesno = yesno
        self.retrieval_label = retrieval_label
        self.history = history
        self.start_position_his=start_position_his
        self.end_position_his=end_position_his
        self.last_answer_text=last_answer_text
    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "example_id: %s" % (self.example_id)
        s += "qas_id: %s" % (self.qas_id)
        s += ", question_text: %s" % (
            self.question_text)
        s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
        if self.start_position:
            s += ", start_position: %d" % (self.start_position)
        if self.end_position:
            s += ", end_position: %d" % (self.end_position)
        if self.is_impossible:
            s += ", is_impossible: %r" % (self.is_impossible)
        s += ', followup: {}'.format(self.followup)
        s += ', yesno: {}'.format(self.yesno)
        if self.retrieval_label:
            s += ', retrieval_label: {}'.format(self.retrieval_label)
        s += ', history: {}'.format(self.history)
            
        return s


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_id,
                 doc_span_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 segment_ids,
                 cls_index,
                 p_mask,
                 paragraph_len,
                 start_position=None,
                 end_position=None,
                 is_impossible=None, 
                 retrieval_label=None):
        # we have exactly 1 feature for every example,
        # so the unique id is the same with the example id
        self.unique_id = unique_id 
        self.example_id = example_id
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.cls_index = cls_index
        self.p_mask = p_mask
        self.paragraph_len = paragraph_len
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible
        self.retrieval_label = retrieval_label


def numbered_byte_file_generator(base_path, file_no, record_size):
    for i in range(file_no):
        with open('{}_split{}'.format(base_path, i), 'rb') as f:
            while True:
                b = f.read(record_size)
                if not b:
                    # eof
                    break
                yield b


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,#少许修改
                         orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""

    # The SQuAD annotations are character based. We first project them to
    # whitespace-tokenized words. But then after WordPiece tokenization, we can
    # often find a "better match". For example:
    #
    #   Question: What year was John Smith born?
    #   Context: The leader was John Smith (1895-1943).
    #   Answer: 1895
    #
    # The original whitespace-tokenized answer will be "(1895-1943).". However
    # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
    # the exact answer, 1895.
    #
    # However, this is not always possible. Consider the following:
    #
    #   Question: What country is the top exporter of electornics?
    #   Context: The Japanese electronics industry is the lagest in the world.
    #   Answer: Japan
    #
    # In this case, the annotator chose "Japan" as a character sub-span of
    # the word "Japanese". Since our WordPiece tokenizer does not split
    # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
    # in SQuAD, but does happen.
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))
    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:

                return (new_start, new_end)

    return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def convert_example_to_feature(example, tokenizer, max_seq_length=512,
                                 doc_stride=384, max_query_length=125, is_training=True,
                                 cls_token_at_end=False,
                                 cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                 sequence_a_segment_id=0, sequence_b_segment_id=1,
                                 cls_token_segment_id=0, pad_token_segment_id=0,
                                 mask_padding_with_zero=True,
                                 sequence_a_is_doc=False):
    """Convert a single QuacExample to features (model input)"""

    query_tokens = tokenizer.tokenize(example.question_text)

    if len(query_tokens) > max_query_length:
        query_tokens = query_tokens[-max_query_length:]

    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    for (i, token) in enumerate(example.doc_tokens):
        orig_to_tok_index.append(len(all_doc_tokens))
        sub_tokens = tokenizer.tokenize(token)
        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)

    tok_start_position = None
    tok_end_position = None
    if is_training and example.is_impossible:
        tok_start_position = -1
        tok_end_position = -1
    if is_training and not example.is_impossible:
        tok_start_position = orig_to_tok_index[example.start_position]
        if example.end_position < len(example.doc_tokens) - 1:
            tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
        else:
            tok_end_position = len(all_doc_tokens) - 1
        (tok_start_position, tok_end_position) = _improve_answer_span(
            all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
            example.orig_answer_text)

    # The -3 accounts for [CLS], [SEP] and [SEP]
    max_tokens_for_doc = max_seq_length - len(query_tokens) - 3
    assert max_tokens_for_doc >= 384, max_tokens_for_doc

    # We can have documents that are longer than the maximum sequence length.
    # To deal with this we do a sliding window approach, where we take chunks
    # of the up to our max length with a stride of `doc_stride`.
    
    # we set the doc_stride to 384, which is the max length of evidence text,
    # meaning that each evidence has exactly one _DocSpan
    _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
        "DocSpan", ["start", "length"])
    doc_spans = []
    start_offset = 0
    while start_offset < len(all_doc_tokens):
        length = len(all_doc_tokens) - start_offset
        if length > max_tokens_for_doc:
            length = max_tokens_for_doc
        doc_spans.append(_DocSpan(start=start_offset, length=length))
        if start_offset + length == len(all_doc_tokens):
            break
        start_offset += min(length, doc_stride)
    assert len(doc_spans) == 1, (max_tokens_for_doc, example)
    # if len(doc_spans) > 1:
        # print(len(doc_spans), example)
    #     doc_spans = [doc_spans[0]]

    for (doc_span_index, doc_span) in enumerate(doc_spans):
        tokens = []
        token_to_orig_map = {}
        token_is_max_context = {}
        segment_ids = []

        # p_mask: mask with 1 for token than cannot be in the answer (0 for token which can be in an answer)
        # Original TF implem also keep the classification token (set to 0) (not sure why...)
        p_mask = []

        # CLS token at the beginning
        if not cls_token_at_end:
            tokens.append(cls_token)
            segment_ids.append(cls_token_segment_id)
            p_mask.append(0)
            cls_index = 0

        # XLNet: P SEP Q SEP CLS
        # Others: CLS Q SEP P SEP
        if not sequence_a_is_doc:
            # Query
            tokens += query_tokens
            segment_ids += [sequence_a_segment_id] * len(query_tokens)
            p_mask += [1] * len(query_tokens)

            # SEP token
            tokens.append(sep_token)
            segment_ids.append(sequence_a_segment_id)
            p_mask.append(1)

        # Paragraph
        for i in range(doc_span.length):
            split_token_index = doc_span.start + i
            token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

            is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                   split_token_index)
            token_is_max_context[len(tokens)] = is_max_context
            tokens.append(all_doc_tokens[split_token_index])
            if not sequence_a_is_doc:
                segment_ids.append(sequence_b_segment_id)
            else:
                segment_ids.append(sequence_a_segment_id)
            p_mask.append(0)
        paragraph_len = doc_span.length

        if sequence_a_is_doc:
            # SEP token
            tokens.append(sep_token)
            segment_ids.append(sequence_a_segment_id)
            p_mask.append(1)

            tokens += query_tokens
            segment_ids += [sequence_b_segment_id] * len(query_tokens)
            p_mask += [1] * len(query_tokens)

        # SEP token
        tokens.append(sep_token)
        segment_ids.append(sequence_b_segment_id)
        p_mask.append(1)

        # CLS token at the end
        if cls_token_at_end:
            tokens.append(cls_token)
            segment_ids.append(cls_token_segment_id)
            p_mask.append(0)
            cls_index = len(tokens) - 1  # Index of classification token

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(pad_token)
            input_mask.append(0 if mask_padding_with_zero else 1)
            segment_ids.append(pad_token_segment_id)
            p_mask.append(1)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        span_is_impossible = example.is_impossible
        start_position = None
        end_position = None
        if is_training and not span_is_impossible:
            # For training, if our document chunk does not contain an annotation
            # we throw it out, since there is nothing to predict.
            doc_start = doc_span.start
            doc_end = doc_span.start + doc_span.length - 1
            out_of_span = False
            if not (tok_start_position >= doc_start and
                    tok_end_position <= doc_end):
                out_of_span = True
            if out_of_span:
                start_position = 0
                end_position = 0
                span_is_impossible = True
            else:
                if sequence_a_is_doc:
                    doc_offset = 0
                else:
                    doc_offset = len(query_tokens) + 2
                start_position = tok_start_position - doc_start + doc_offset
                end_position = tok_end_position - doc_start + doc_offset

        if is_training and span_is_impossible:
            start_position = cls_index
            end_position = cls_index

        if False:
            logger.info("*** Example ***")
            logger.info("unique_id: %s" % (example.example_id))
            logger.info("example_id: %s" % (example.example_id))
            logger.info("qid of the example: %s" % (example.qas_id))
            logger.info("doc_span_index: %s" % (doc_span_index))
            logger.info("tokens: %s" % " ".join(tokens))
            logger.info("token_to_orig_map: %s" % " ".join([
                "%d:%d" % (x, y) for (x, y) in token_to_orig_map.items()]))
            logger.info("token_is_max_context: %s" % " ".join([
                "%d:%s" % (x, y) for (x, y) in token_is_max_context.items()
            ]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info(
                "input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            if is_training and span_is_impossible:
                logger.info("impossible example")
            if is_training and not span_is_impossible:
                answer_text = " ".join(tokens[start_position:(end_position + 1)])
                logger.info("start_position: %d" % (start_position))
                logger.info("end_position: %d" % (end_position))
                logger.info("retrieval_label: %d" % (example.retrieval_label))
                logger.info(
                    "answer: %s" % (answer_text))

        feature = InputFeatures(
                    unique_id=example.example_id,
                    example_id=example.example_id,
                    doc_span_index=doc_span_index,
                    tokens=tokens,
                    token_to_orig_map=token_to_orig_map,
                    token_is_max_context=token_is_max_context,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    cls_index=cls_index,
                    p_mask=p_mask,
                    paragraph_len=paragraph_len,
                    start_position=start_position,
                    end_position=end_position,
                    is_impossible=span_is_impossible, 
                    retrieval_label=example.retrieval_label)

    return feature



################ xbli
def gen_reader_features(qids, question_texts, answer_texts, answer_starts, passage_ids,
                        passages, all_retrieval_labels, reader_tokenizer, max_seq_length, is_training=False):
    # print('all_retrieval_labels', all_retrieval_labels, type(all_retrieval_labels))
    batch_features = []
    all_examples, all_features = {}, {}
    for (qas_id, question_text, answer_text, answer_start, pids_per_query,
         paragraph_texts, retrieval_labels) in zip(qids, question_texts, answer_texts, answer_starts, 
                                                   passage_ids, passages, all_retrieval_labels):
        # print('retrieval_labels', retrieval_labels)
        per_query_features = []
        for i, (pid, paragraph_text, retrieval_label) in enumerate(zip(pids_per_query, paragraph_texts, retrieval_labels)):
            # print('retrieval_label', retrieval_label)
            example_id = f'{qas_id}*{pid}'
            doc_tokens = []
            char_to_word_offset = []
            prev_is_whitespace = True
            for c in paragraph_text:
                if is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        doc_tokens.append(c)
                    else:
                        doc_tokens[-1] += c
                    prev_is_whitespace = False
                char_to_word_offset.append(len(doc_tokens) - 1)

            start_position = None
            end_position = None
            orig_answer_text = None
            is_impossible = False

            if is_training:
                if answer_text in ['CANNOTANSWER', 'NOTRECOVERED'] or retrieval_label == 0:
                    is_impossible = True

                if not is_impossible:
                    # pdb.set_trace()
                    orig_answer_text = answer_text
                    answer_offset = answer_start
                    answer_length = len(orig_answer_text)
                    start_position = char_to_word_offset[answer_offset]
                    end_position = char_to_word_offset[answer_offset +
                                                       answer_length - 1]
                    # Only add answers where the text can be exactly recovered from the
                    # document. If this CAN'T happen it's likely due to weird Unicode
                    # stuff so we will just skip the example.
                    #
                    # Note that this means for training mode, every example is NOT
                    # guaranteed to be preserved.
                    actual_text = " ".join(
                        doc_tokens[start_position:(end_position + 1)])
                    cleaned_answer_text = " ".join(
                        whitespace_tokenize(orig_answer_text))
                    if actual_text.find(cleaned_answer_text) == -1:
                        logger.warning("Could not find answer: '%s' vs. '%s'",
                                       actual_text, cleaned_answer_text)
                else:
                    start_position = -1
                    end_position = -1
                    orig_answer_text = ""

            example = QuacExample(
                example_id=example_id,
                qas_id=qas_id,
                question_text=question_text,
                doc_tokens=doc_tokens,
                orig_answer_text=orig_answer_text,
                start_position=start_position,
                end_position=end_position,
                is_impossible=is_impossible,
                retrieval_label=retrieval_label)

            feature = convert_example_to_feature(
                example, reader_tokenizer, is_training=is_training)

            # when evaluating, we save all examples and features
            # so that we can recover answer texts
            if not is_training:
                all_examples[example_id] = example
                all_features[example_id] = feature

            if is_training:
                if retrieval_label:
                    per_query_feature = {'input_ids': np.asarray(feature.input_ids),
                                     'segment_ids': np.asarray(feature.segment_ids),
                                     'input_mask': np.asarray(feature.input_mask),
                                     'start_position': feature.start_position + i * max_seq_length,
                                     'end_position': feature.end_position + i * max_seq_length,
                                     'retrieval_label': feature.retrieval_label}

                else:
                    per_query_feature = {'input_ids': np.asarray(feature.input_ids),
                                     'segment_ids': np.asarray(feature.segment_ids),
                                     'input_mask': np.asarray(feature.input_mask),
                                     'start_position': -1,
                                     'end_position': -1,
                                     'retrieval_label': feature.retrieval_label}
            else:
                per_query_feature = {'input_ids': np.asarray(feature.input_ids),
                                 'segment_ids': np.asarray(feature.segment_ids),
                                 'input_mask': np.asarray(feature.input_mask),
                                 'example_id': feature.example_id}

            per_query_features.append(per_query_feature)

        collated = {}

        keys = per_query_features[0].keys()
        for key in keys:
            if key != 'example_id':
                collated[key] = np.vstack([dic[key] for dic in per_query_features])
        if 'example_id' in keys:
            collated['example_id'] = [dic['example_id'] for dic in per_query_features]

        batch_features.append(collated)
    
    # print('batch_features', batch_features)
    batch = {}
    keys = batch_features[0].keys()
    for key in keys:
        if key != 'example_id':
            batch[key] = np.stack([dic[key] for dic in batch_features], axis=0)
            batch[key] = torch.from_numpy(batch[key])
    if 'example_id' in keys:
        batch['example_id'] = []
        for item in batch_features:
            batch['example_id'].extend(item['example_id'])
    # print('batch', batch)
    if is_training:
        return batch
    else:
        return batch, all_examples, all_features




################
def load_collection(collection_file):
    all_passages = ["[INVALID DOC ID]"] * 5000_0000
    ext = collection_file[collection_file.rfind(".") + 1:]
    if ext not in ["jsonl", "tsv"]:
        raise TypeError("Unrecognized file type")
    with open(collection_file, "r") as f:
        if ext == "jsonl":
            for line in f:
                line = line.strip()
                obj = json.loads(line)
                pid = int(obj["id"])
                passage = obj["title"] + "[SEP]" + obj["text"]
                all_passages[pid] = passage
        else:
            for line in f:
                line = line.strip()
                try:
                    line_arr = line.split("\t")
                    pid = int(line_arr[0])
                    passage = line_arr[1].rstrip()
                    all_passages[pid] = passage
                except IndexError:
                    print("bad passage")
                except ValueError:
                    print("bad pid")
    return all_passages


class EmbeddingCache:
    def __init__(self, base_path, seed=-1):
        self.base_path = base_path
        with open(base_path + '_meta', 'r') as f:
            meta = json.load(f)
            self.dtype = np.dtype(meta['type'])
            self.total_number = meta['total_number']
            self.record_size = int(
                meta['embedding_size']) * self.dtype.itemsize + 4
        if seed >= 0:
            self.ix_array = np.random.RandomState(seed).permutation(
                self.total_number)
        else:
            self.ix_array = np.arange(self.total_number)
        self.f = None

    def open(self):
        self.f = open(self.base_path, 'rb')

    def close(self):
        self.f.close()

    def read_single_record(self):
        record_bytes = self.f.read(self.record_size)
        passage_len = int.from_bytes(record_bytes[:4], 'big')
        passage = np.frombuffer(record_bytes[4:], dtype=self.dtype)
        return passage_len, passage

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def __getitem__(self, key):
        if key < 0 or key > self.total_number:
            raise IndexError(
                "Index {} is out of bound for cached embeddings of size {}".
                format(key, self.total_number))
        self.f.seek(key * self.record_size)
        return self.read_single_record()

    def __iter__(self):
        self.f.seek(0)
        for i in range(self.total_number):
            new_ix = self.ix_array[i]
            yield self.__getitem__(new_ix)

    def __len__(self):
        return self.total_number


class StreamingDataset(IterableDataset):
    def __init__(self, elements, fn):
        super().__init__()
        self.elements = elements
        self.fn = fn
        self.num_replicas = -1

    def __iter__(self):
        if dist.is_initialized():
            self.num_replicas = dist.get_world_size()
            self.rank = dist.get_rank()
            print("Rank:", self.rank, "world:", self.num_replicas)
        else:
            print("Not running in distributed mode")
        for i, element in enumerate(self.elements):
            if self.num_replicas != -1 and i % self.num_replicas != self.rank:
                continue
            records = self.fn(element, i)
            for rec in records:
                # print("yielding record")
                # print(rec)
                yield rec

class ConvSearchExample_V2:
    def __init__(self,
                 qid,
                 question_text,
                 answer_start,
                 answer_text,
                 concat_ids,
                 concat_id_mask,
                 target_ids,
                 target_id_mask,
                 doc_pos=None,
                 doc_negs=None,
                 raw_sequences=None):
        self.qid = qid
        self.question_text = question_text
        self.answer_start = answer_start
        self.answer_text = answer_text
        self.concat_ids = concat_ids
        self.concat_id_mask = concat_id_mask
        self.target_ids = target_ids
        self.target_id_mask = target_id_mask
        self.doc_pos = doc_pos
        self.doc_negs = doc_negs
        self.raw_sequences = raw_sequences

class ConvSearchExample:
    def __init__(self,
                 qid,
                 concat_ids,
                 concat_id_mask,
                 target_ids,
                 target_id_mask,
                 doc_pos=None,
                 doc_negs=None,
                 raw_sequences=None):
        self.qid = qid
        self.concat_ids = concat_ids
        self.concat_id_mask = concat_id_mask
        self.target_ids = target_ids
        self.target_id_mask = target_id_mask
        self.doc_pos = doc_pos
        self.doc_negs = doc_negs
        self.raw_sequences = raw_sequences


class ConvSearchDataset(Dataset):
    def __init__(self, filenames, tokenizer, args, mode="train"):
        self.examples = []
        for filename in filenames:
            with open(filename, encoding="utf-8") as f:
                for line in f:
                    record = json.loads(line)
                    input_sents = record['input']
                    target_sent = record['target']
                    auto_sent = record.get('output', "no")
                    raw_sent = record["input"][-1]
                    responses = record[
                        "manual_response"] if args.query == "man_can" else (
                            record["automatic_response"]
                            if args.query == "auto_can" else [])
                    topic_number = record.get('topic_number', None)
                    query_number = record.get('query_number', None)
                    qid = str(topic_number) + "_" + str(
                        query_number) if topic_number != None else str(
                            record["qid"])
                    sequences = record['input']
                    concat_ids = []
                    concat_id_mask = []
                    target_ids = None
                    target_id_mask = None
                    doc_pos = None
                    doc_negs = None
                    if mode == "train" and args.ranking_task:
                        doc_pos = record["doc_pos"]
                        doc_negs = record["doc_negs"]

                    if mode == "train" or args.query in [
                            "no_res", "man_can", "auto_can"
                    ]:
                        if args.model_type == "dpr":
                            concat_ids.append(
                                tokenizer.cls_token_id
                            )  # dpr (on OR-QuAC) uses BERT-style sequence [CLS] q1 [SEP] q2 [SEP] ...
                        for sent in input_sents[:-1]:  # exlude last one
                            if args.model_type != "dpr":
                                concat_ids.append(
                                    tokenizer.cls_token_id
                                )  # RoBERTa-style sequence <s> q1 </s> <s> q2 </s> ...
                            concat_ids.extend(
                                tokenizer.convert_tokens_to_ids(
                                    tokenizer.tokenize(sent)))
                            concat_ids.append(tokenizer.sep_token_id)

                        if args.query in [
                                "man_can", "auto_can"
                        ] and len(responses) >= 2:  # add response
                            if args.model_type != "dpr":
                                concat_ids.append(tokenizer.cls_token_id)
                            concat_ids.extend(
                                tokenizer.convert_tokens_to_ids(["<response>"
                                                                 ]))
                            concat_ids.extend(
                                tokenizer.convert_tokens_to_ids(
                                    tokenizer.tokenize(responses[-2])))
                            concat_ids.append(tokenizer.sep_token_id)
                            sequences.insert(-1, responses[-2])

                        if args.model_type != "dpr":
                            concat_ids.append(tokenizer.cls_token_id)
                        concat_ids.extend(
                            tokenizer.convert_tokens_to_ids(
                                tokenizer.tokenize(input_sents[-1])))
                        concat_ids.append(tokenizer.sep_token_id)

                        # We do not use token type id for BERT (and for RoBERTa, of course)
                        concat_ids, concat_id_mask = pad_input_ids_with_mask(
                            concat_ids, args.max_concat_length)
                        assert len(concat_ids) == args.max_concat_length

                    elif args.query == "target":  # manual

                        concat_ids = tokenizer.encode(
                            target_sent,
                            add_special_tokens=True,
                            max_length=args.max_query_length)
                        concat_ids, concat_id_mask = pad_input_ids_with_mask(
                            concat_ids, args.max_query_length)
                        assert len(concat_ids) == args.max_query_length

                    elif args.query == "output":  # reserved for query rewriter output

                        concat_ids = tokenizer.encode(
                            auto_sent,
                            add_special_tokens=True,
                            max_length=args.max_query_length)
                        concat_ids, concat_id_mask = pad_input_ids_with_mask(
                            concat_ids, args.max_query_length)
                        assert len(concat_ids) == args.max_query_length

                    elif args.query == "raw":

                        concat_ids = tokenizer.encode(
                            raw_sent,
                            add_special_tokens=True,
                            max_length=args.max_query_length)
                        concat_ids, concat_id_mask = pad_input_ids_with_mask(
                            concat_ids, args.max_query_length)
                        assert len(concat_ids) == args.max_query_length

                    else:
                        raise KeyError("Unsupported query type")

                    if mode == "train":
                        target_ids = tokenizer.encode(
                            target_sent,
                            add_special_tokens=True,
                            max_length=args.max_query_length)
                        target_ids, target_id_mask = pad_input_ids_with_mask(
                            target_ids, args.max_query_length)
                        assert len(target_ids) == args.max_query_length

                    self.examples.append(
                        ConvSearchExample(qid, concat_ids, concat_id_mask,
                                          target_ids, target_id_mask, doc_pos,
                                          doc_negs, sequences))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]

    @staticmethod
    def get_collate_fn(args, mode):
        def collate_fn(batch_dataset: list):
            collated_dict = {
                "qid": [],
                "concat_ids": [],
                "concat_id_mask": [],
            }
            if mode == "train":
                collated_dict.update({"target_ids": [], "target_id_mask": []})
                if args.ranking_task:
                    collated_dict.update({"documents": []})
            else:
                collated_dict.update({"history_utterances": []})
            for example in batch_dataset:
                collated_dict["qid"].append(example.qid)
                collated_dict["concat_ids"].append(example.concat_ids)
                collated_dict["concat_id_mask"].append(example.concat_id_mask)
                if mode == "train":
                    collated_dict["target_ids"].append(example.target_ids)
                    collated_dict["target_id_mask"].append(
                        example.target_id_mask)
                    if args.ranking_task:
                        collated_dict["documents"].append([example.doc_pos] +
                                                          example.doc_negs)
                else:
                    collated_dict["history_utterances"].append(
                        example.raw_sequences)
            should_be_tensor = [
                "concat_ids", "concat_id_mask", "target_ids", "target_id_mask"
            ]
            for key in should_be_tensor:
                if key in collated_dict:
                    collated_dict[key] = torch.tensor(collated_dict[key],
                                                      dtype=torch.long)

            return collated_dict

        return collate_fn

class ConvSearchDataset_V2(Dataset):
    def __init__(self, filenames,q2answers_file, tokenizer, args, mode="train",q2ner=None):
        self.examples = []
        with open(q2answers_file,'rb') as handle:
            q2answers = pkl.load(handle)
        for filename in filenames:
            with open(filename, encoding="utf-8") as f:
                for line in f:
                    record = json.loads(line)
                    input_sents = record['input']
                    target_sent = record['target']
                    auto_sent = record.get('output', "no")
                    raw_sent = record["input"][-1]
                    answer_start = q2answers[record['qid']][0]
                    answer_text = q2answers[record['qid']][1]
                    responses = record[
                        "manual_response"] if args.query == "man_can" else (
                            record["automatic_response"]
                            if args.query == "auto_can" else [])
                    topic_number = record.get('topic_number', None)
                    query_number = record.get('query_number', None)
                    qid = str(topic_number) + "_" + str(
                        query_number) if topic_number != None else str(
                            record["qid"])
                    sequences = record['input']
                    question_text = '[SEP]'.join(sequences[-6:])
                    concat_ids = []
                    concat_id_mask = []
                    target_ids = None
                    target_id_mask = None
                    doc_pos = None
                    doc_negs = None
                    if mode == "train" and args.ranking_task:
                        doc_pos = record["doc_pos"]
                        doc_negs = record["doc_negs"]

                    if mode == "train" or args.query in [
                            "no_res", "man_can", "auto_can"
                    ]:
                        if args.model_type == "dpr":
                            concat_ids.append(
                                tokenizer.cls_token_id
                            )  # dpr (on OR-QuAC) uses BERT-style sequence [CLS] q1 [SEP] q2 [SEP] ...
                        for idx,sent in enumerate(input_sents[:-1]):  # exlude last one
                            if args.include_first_question and idx == 0:
                                first_ner = q2ner[qid]
                                sent = first_ner
                                # sent = first_ner if len(first_ner) else sent
                                # pdb.set_trace()
                            if args.model_type != "dpr":
                                concat_ids.append(
                                    tokenizer.cls_token_id
                                )  # RoBERTa-style sequence <s> q1 </s> <s> q2 </s> ...
                            concat_ids.extend(
                                tokenizer.convert_tokens_to_ids(
                                    tokenizer.tokenize(sent)))
                            concat_ids.append(tokenizer.sep_token_id)

                        if args.query in [
                                "man_can", "auto_can"
                        ] and len(responses) >= 2:  # add response
                            if args.model_type != "dpr":
                                concat_ids.append(tokenizer.cls_token_id)
                            concat_ids.extend(
                                tokenizer.convert_tokens_to_ids(["<response>"
                                                                 ]))
                            concat_ids.extend(
                                tokenizer.convert_tokens_to_ids(
                                    tokenizer.tokenize(responses[-2])))
                            concat_ids.append(tokenizer.sep_token_id)
                            sequences.insert(-1, responses[-2])

                        if args.model_type != "dpr":
                            concat_ids.append(tokenizer.cls_token_id)
                        concat_ids.extend(
                            tokenizer.convert_tokens_to_ids(
                                tokenizer.tokenize(input_sents[-1])))
                        concat_ids.append(tokenizer.sep_token_id)

                        # We do not use token type id for BERT (and for RoBERTa, of course)
                        concat_ids, concat_id_mask = pad_input_ids_with_mask(
                            concat_ids, args.max_concat_length)
                        assert len(concat_ids) == args.max_concat_length

                    elif args.query == "target":  # manual

                        concat_ids = tokenizer.encode(
                            target_sent,
                            add_special_tokens=True,
                            max_length=args.max_query_length)
                        concat_ids, concat_id_mask = pad_input_ids_with_mask(
                            concat_ids, args.max_query_length)
                        assert len(concat_ids) == args.max_query_length

                    elif args.query == "output":  # reserved for query rewriter output

                        concat_ids = tokenizer.encode(
                            auto_sent,
                            add_special_tokens=True,
                            max_length=args.max_query_length)
                        concat_ids, concat_id_mask = pad_input_ids_with_mask(
                            concat_ids, args.max_query_length)
                        assert len(concat_ids) == args.max_query_length

                    elif args.query == "raw":

                        concat_ids = tokenizer.encode(
                            raw_sent,
                            add_special_tokens=True,
                            max_length=args.max_query_length)
                        concat_ids, concat_id_mask = pad_input_ids_with_mask(
                            concat_ids, args.max_query_length)
                        assert len(concat_ids) == args.max_query_length

                    else:
                        raise KeyError("Unsupported query type")

                    if mode == "train":
                        target_ids = tokenizer.encode(
                            target_sent,
                            add_special_tokens=True,
                            max_length=args.max_query_length)
                        target_ids, target_id_mask = pad_input_ids_with_mask(
                            target_ids, args.max_query_length)
                        assert len(target_ids) == args.max_query_length

                    self.examples.append(
                        ConvSearchExample_V2(qid,question_text, answer_start,answer_text,concat_ids, concat_id_mask,
                                          target_ids, target_id_mask, doc_pos,
                                          doc_negs, sequences))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]

    @staticmethod
    def get_collate_fn(args, mode):
        def collate_fn(batch_dataset: list):
            collated_dict = {
                "qid": [],
                "concat_ids": [],
                "concat_id_mask": [],
                "answer_texts":[],
                "answer_starts":[],
                "question_texts":[]

            }
            if mode == "train":
                collated_dict.update({"target_ids": [], "target_id_mask": []})
                if args.ranking_task:
                    collated_dict.update({"documents": []})
            else:
                collated_dict.update({"history_utterances": []})
            for example in batch_dataset:
                collated_dict["qid"].append(example.qid)
                collated_dict["concat_ids"].append(example.concat_ids)
                collated_dict["concat_id_mask"].append(example.concat_id_mask)
                if mode == "train":
                    collated_dict["answer_texts"].append(example.answer_text)
                    collated_dict["answer_starts"].append(example.answer_start)
                    collated_dict["question_texts"].append(example.question_text)
                    collated_dict["target_ids"].append(example.target_ids)
                    collated_dict["target_id_mask"].append(
                        example.target_id_mask)
                    if args.ranking_task:
                        collated_dict["documents"].append([example.doc_pos] +
                                                          example.doc_negs)
                else:
                    collated_dict["history_utterances"].append(
                        example.raw_sequences)
            should_be_tensor = [
                "concat_ids", "concat_id_mask", "target_ids", "target_id_mask"
            ]
            for key in should_be_tensor:
                if key in collated_dict:

                    collated_dict[key] = torch.tensor(collated_dict[key],
                                                      dtype=torch.long) if key in ['input_ids','input_mask','segment_ids',
                                                      'start_position','end_position','retrieval_label','concat_ids',
                                                      'concat_id_mask','target_id_mask','target_ids'] else np.asarray(collated_dict[key])

            return collated_dict

        return collate_fn

def tokenize_to_file(args, i, num_process, in_path, out_path, line_fn):

    configObj = MSMarcoConfigDict[args.model_type]
    tokenizer = configObj.tokenizer_class.from_pretrained(
        args.model_name_or_path,
        do_lower_case=True,
        cache_dir=None,
    )

    with open(in_path, 'r', encoding='utf-8') if in_path[-2:] != "gz" else gzip.open(in_path, 'rt', encoding='utf8') as in_f,\
            open('{}_split{}'.format(out_path, i), 'wb') as out_f:
        for idx, line in enumerate(in_f):
            if idx % num_process != i:
                continue
            try:
                res = line_fn(args, line, tokenizer)
            except ValueError:
                print("Bad passage.")
            else:
                out_f.write(res)


#                      args, 32,        , collection.tsv, passages,
def multi_file_process(args, num_process, in_path, out_path, line_fn):
    processes = []
    for i in range(num_process):
        p = Process(target=tokenize_to_file,
                    args=(
                        args,
                        i,
                        num_process,
                        in_path,
                        out_path,
                        line_fn,
                    ))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return [data]

    world_size = dist.get_world_size()
    # serialized to a Tensor
    buffer = pkl.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.LongTensor([tensor.numel()]).to("cuda")
    size_list = [torch.LongTensor([0]).to("cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.ByteTensor(size=(max_size, )).to("cuda"))
    if local_size != max_size:
        padding = torch.ByteTensor(size=(max_size - local_size, )).to("cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pkl.loads(buffer))

    return data_list
def load_model_V3(args, checkpoint_path,is_pipeline=False,is_todevice=True):
    label_list = ["0", "1"]
    num_labels = len(label_list)
    args.model_type = args.model_type.lower()
    configObj = MSMarcoConfigDict[args.model_type]
    args.model_path = checkpoint_path
    config, tokenizer, model = None, None, None
    if args.model_type != "dpr":
        config = configObj.config_class.from_pretrained(
            args.model_path,
            num_labels=num_labels,
            finetuning_task="MSMarco",
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        tokenizer = configObj.tokenizer_class.from_pretrained(
            args.model_path,
            do_lower_case=True,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        model = configObj.model_class.from_pretrained(
            args.model_path,
            from_tf=bool(".ckpt" in args.model_path),
            config=config,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
    else:  # dpr

        # model = configObj.model_class(args)
        args.model_name_or_path = '/home/xbli/ConvDR-main/thu_checkpoints/convdr-multi-orquac.cp'
        args.reader_model_name_or_path = 'bert-base-uncased'
        args.reader_tokenizer_name = False
        args.do_lower_case = True
        model = Pipeline_DP(args)
        saved_state = load_states_from_checkpoint(checkpoint_path)
        # pdb.set_trace()
        model_to_load = get_model_obj(model)
        logger.info('Loading saved model state ...')
        model_to_load.load_state_dict(saved_state.model_dict)
        tokenizer = configObj.tokenizer_class.from_pretrained(
            "bert-base-uncased",
            do_lower_case=True,
            cache_dir=None,
        )
        if is_pipeline:

            Hybrid_model = Pipeline()
            reader_config_class, reader_model_class, reader_tokenizer_class = MODEL_CLASSES['reader']
            reader_config = reader_config_class.from_pretrained(args.reader_model_name_or_path)
            reader_config.num_qa_labels = 2
            # this not used for BertForOrconvqaGlobal
            reader_config.num_retrieval_labels = 2
            reader_config.qa_loss_factor = args.qa_loss_factor
            reader_config.retrieval_loss_factor = args.retrieval_loss_factor

            reader_tokenizer = reader_tokenizer_class.from_pretrained(args.reader_tokenizer_name if args.reader_tokenizer_name else args.reader_model_name_or_path,
                                                                    do_lower_case=args.do_lower_case)
            reader_model = reader_model_class.from_pretrained(args.reader_model_name_or_path,
                                                            config=reader_config)

            Hybrid_model.retriever = model
            Hybrid_model.reader = reader_model

            Hybrid_model.to(args.device)
            return config,tokenizer,Hybrid_model,reader_tokenizer
        else:
            if is_todevice:
                model.to(args.device)
            return config, tokenizer, model.retriever


class Pipeline_DP(nn.Module):
    def __init__(self,args):
        super(Pipeline_DP, self).__init__()
        reader_config = BertConfig.from_pretrained('bert-base-uncased')
        reader_config.num_qa_labels = 2
        # this not used for BertForOrconvqaGlobal
        reader_config.num_retrieval_labels = 2
        reader_config.qa_loss_factor = 1.0
        reader_config.retrieval_loss_factor = 1.0
        self.reader = BertForOrconvqaGlobal_DR_01.from_pretrained(args.reader_model_name_or_path,config=reader_config)
        # self.reader = BertForOrconvqaGlobal0.from_pretrained(args.reader_model_name_or_path,config=reader_config)
        # pdb.set_trace()
        # self.reader = BertForOrconvqaGlobal.from_pretrained(args.reader_model_name_or_path,config=reader_config)
        _, _, retriever = load_model(args, args.model_name_or_path)
        self.retriever = retriever
    def forward(self,encoder_name,concat_ids=None, concat_id_mask=None,input_ids=None, attention_mask=None, token_type_ids=None, 
                start_positions=None, end_positions=None, retrieval_label=None):
        #encoder_name reader retriever
        if encoder_name=='retriever':
            return self.retriever(concat_ids, concat_id_mask)
        else:
            return self.reader(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, 
                start_positions=start_positions, end_positions=end_positions, retrieval_label=retrieval_label)
