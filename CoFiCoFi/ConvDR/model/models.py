from csv import reader
import sys


sys.path += ['../']
# from turtle import forward
import random
import pdb
import torch
from torch import nn
from torch.nn import CrossEntropyLoss,TripletMarginLoss
from transformers import (RobertaConfig, RobertaModel,BertPreTrainedModel,
                          RobertaForSequenceClassification, RobertaTokenizer,
                          BertModel, BertTokenizer, BertConfig)
import torch.nn.functional as F
from data.process_fn import triple_process_fn, triple2dual_process_fn
# from utils.util import load_model


class EmbeddingMixin:
    """
    Mixin for common functions in most embedding models. Each model should define its own bert-like backbone and forward.
    We inherit from RobertaModel to use from_pretrained 
    """
    def __init__(self, model_argobj):
        if model_argobj is None:
            self.use_mean = False
        else:
            self.use_mean = model_argobj.use_mean
        print("Using mean:", self.use_mean)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)

    def masked_mean(self, t, mask):
        s = torch.sum(t * mask.unsqueeze(-1).float(), axis=1)
        d = mask.sum(axis=1, keepdim=True).float()
        return s / d

    def masked_mean_or_first(self, emb_all, mask):
        # emb_all is a tuple from bert - sequence output, pooler
        assert isinstance(emb_all, tuple)
        if self.use_mean:
            return self.masked_mean(emb_all[0], mask)
        else:
            return emb_all[0][:, 0]

    def query_emb(self, input_ids, attention_mask):
        raise NotImplementedError("Please Implement this method")

    def body_emb(self, input_ids, attention_mask):
        raise NotImplementedError("Please Implement this method")


class NLL(EmbeddingMixin):
    def forward(self,
                query_ids,
                attention_mask_q,
                input_ids_a=None,
                attention_mask_a=None,
                input_ids_b=None,
                attention_mask_b=None,
                is_query=True):
        if input_ids_b is None and is_query:
            return self.query_emb(query_ids, attention_mask_q)
        elif input_ids_b is None:
            return self.body_emb(query_ids, attention_mask_q)

        q_embs = self.query_emb(query_ids, attention_mask_q)
        a_embs = self.body_emb(input_ids_a, attention_mask_a)
        b_embs = self.body_emb(input_ids_b, attention_mask_b)

        logit_matrix = torch.cat([(q_embs * a_embs).sum(-1).unsqueeze(1),
                                  (q_embs * b_embs).sum(-1).unsqueeze(1)],
                                 dim=1)  # [B, 2]
        lsm = F.log_softmax(logit_matrix, dim=1)
        loss = -1.0 * lsm[:, 0]
        return (loss.mean(), )


class NLL_MultiChunk(EmbeddingMixin):
    def forward(self,
                query_ids,
                attention_mask_q,
                input_ids_a=None,
                attention_mask_a=None,
                input_ids_b=None,
                attention_mask_b=None,
                is_query=True):
        if input_ids_b is None and is_query:
            return self.query_emb(query_ids, attention_mask_q)
        elif input_ids_b is None:
            return self.body_emb(query_ids, attention_mask_q)

        q_embs = self.query_emb(query_ids, attention_mask_q)
        a_embs = self.body_emb(input_ids_a, attention_mask_a)
        b_embs = self.body_emb(input_ids_b, attention_mask_b)

        [batchS, full_length] = input_ids_a.size()
        chunk_factor = full_length // self.base_len

        # special handle of attention mask -----
        attention_mask_body = attention_mask_a.reshape(
            batchS, chunk_factor, -1)[:, :, 0]  # [batchS, chunk_factor]
        inverted_bias = ((1 - attention_mask_body) * (-9999)).float()

        a12 = torch.matmul(q_embs.unsqueeze(1),
                           a_embs.transpose(1, 2))  # [batch, 1, chunk_factor]
        logits_a = (a12[:, 0, :] + inverted_bias).max(
            dim=-1, keepdim=False).values  # [batch]
        # -------------------------------------

        # special handle of attention mask -----
        attention_mask_body = attention_mask_b.reshape(
            batchS, chunk_factor, -1)[:, :, 0]  # [batchS, chunk_factor]
        inverted_bias = ((1 - attention_mask_body) * (-9999)).float()

        a12 = torch.matmul(q_embs.unsqueeze(1),
                           b_embs.transpose(1, 2))  # [batch, 1, chunk_factor]
        logits_b = (a12[:, 0, :] + inverted_bias).max(
            dim=-1, keepdim=False).values  # [batch]
        # -------------------------------------

        logit_matrix = torch.cat(
            [logits_a.unsqueeze(1),
             logits_b.unsqueeze(1)], dim=1)  # [B, 2]
        lsm = F.log_softmax(logit_matrix, dim=1)
        loss = -1.0 * lsm[:, 0]
        return (loss.mean(), )


class RobertaDot_NLL_LN(NLL, RobertaForSequenceClassification):
    """None
    Compress embedding to 200d, then computes NLL loss.
    """
    def __init__(self, config, model_argobj=None):
        NLL.__init__(self, model_argobj)
        RobertaForSequenceClassification.__init__(self, config)
        self.embeddingHead = nn.Linear(config.hidden_size, 768)
        self.norm = nn.LayerNorm(768)
        self.apply(self._init_weights)

    def query_emb(self, input_ids, attention_mask):
        outputs1 = self.roberta(input_ids=input_ids,
                                attention_mask=attention_mask)
        full_emb = self.masked_mean_or_first(outputs1, attention_mask)
        query1 = self.norm(self.embeddingHead(full_emb))
        return query1

    def body_emb(self, input_ids, attention_mask):
        return self.query_emb(input_ids, attention_mask)


class RobertaDot_NLL_LN_Inference(RobertaDot_NLL_LN):
    def __init__(self, config, model_argobj=None):
        RobertaDot_NLL_LN.__init__(self, config, model_argobj=model_argobj)

    def forward(self, input_ids, attention_mask):
        return self.query_emb(input_ids, attention_mask)


class RobertaDot_CLF_ANN_NLL_MultiChunk(NLL_MultiChunk, RobertaDot_NLL_LN):
    def __init__(self, config):
        RobertaDot_NLL_LN.__init__(self, config)
        self.base_len = 512

    def body_emb(self, input_ids, attention_mask):
        [batchS, full_length] = input_ids.size()
        chunk_factor = full_length // self.base_len

        input_seq = input_ids.reshape(batchS, chunk_factor,
                                      full_length // chunk_factor).reshape(
                                          batchS * chunk_factor,
                                          full_length // chunk_factor)
        attention_mask_seq = attention_mask.reshape(
            batchS, chunk_factor,
            full_length // chunk_factor).reshape(batchS * chunk_factor,
                                                 full_length // chunk_factor)

        outputs_k = self.roberta(input_ids=input_seq,
                                 attention_mask=attention_mask_seq)

        compressed_output_k = self.embeddingHead(
            outputs_k[0])  # [batch, len, dim]
        compressed_output_k = self.norm(compressed_output_k[:, 0, :])

        [batch_expand, embeddingS] = compressed_output_k.size()
        complex_emb_k = compressed_output_k.reshape(batchS, chunk_factor,
                                                    embeddingS)

        return complex_emb_k  # size [batchS, chunk_factor, embeddingS]


class HFBertEncoder(BertModel):
    def __init__(self, config):
        BertModel.__init__(self, config)
        assert config.hidden_size > 0, 'Encoder hidden_size can\'t be zero'
        self.init_weights()

    @classmethod
    def init_encoder(cls, args, dropout: float = 0.1):
        cfg = BertConfig.from_pretrained("bert-base-uncased")
        if dropout != 0:
            cfg.attention_probs_dropout_prob = dropout
            cfg.hidden_dropout_prob = dropout
        return cls.from_pretrained("bert-base-uncased")

    def forward(self, input_ids, attention_mask):
        hidden_states = None

        sequence_output, pooled_output = super().forward(
            input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = sequence_output[:, 0, :]
        return sequence_output, pooled_output, hidden_states

    def get_out_size(self):
        if self.encode_proj:
            return self.encode_proj.out_features
        return self.config.hidden_size


class BiEncoder(nn.Module):
    """ Bi-Encoder model component. Encapsulates query/question and context/passage encoders.
    """
    def __init__(self, args):
        super(BiEncoder, self).__init__()
        self.question_model = HFBertEncoder.init_encoder(args)
        self.ctx_model = HFBertEncoder.init_encoder(args)

    def query_emb(self, input_ids, attention_mask):
        sequence_output, pooled_output, hidden_states = self.question_model(
            input_ids, attention_mask)
        # print('input_ids.shape')
        # print(input_ids.shape)
        # pdb.set_trace()
        return pooled_output

    def body_emb(self, input_ids, attention_mask):
        sequence_output, pooled_output, hidden_states = self.ctx_model(
            input_ids, attention_mask)
        return pooled_output

    def forward(self,
                query_ids,
                attention_mask_q,
                input_ids_a=None,
                attention_mask_a=None,
                input_ids_b=None,
                attention_mask_b=None,
                is_query=True):
        if input_ids_b is None:
            if input_ids_a is None:
                return self.query_emb(
                    query_ids,
                    attention_mask_q) if is_query else self.body_emb(
                        query_ids, attention_mask_q)
            q_embs = self.query_emb(query_ids, attention_mask_q)
            a_embs = self.body_emb(input_ids_a, attention_mask_a)

            return (q_embs, a_embs)
        q_embs = self.query_emb(query_ids, attention_mask_q)
        a_embs = self.body_emb(input_ids_a, attention_mask_a)
        b_embs = self.body_emb(input_ids_b, attention_mask_b)
        # print('q_embs shape')
        # print(q_embs.shape)
        # pdb.set_trace()
        logit_matrix = torch.cat([(q_embs * a_embs).sum(-1).unsqueeze(1),
                                  (q_embs * b_embs).sum(-1).unsqueeze(1)],
                                 dim=1)  #[B, 2]
        lsm = F.log_softmax(logit_matrix, dim=1)
        loss = -1.0 * lsm[:, 0]
        return (loss.mean(), )
    
class BertForOrconvqaGlobal(BertPreTrainedModel):
    r"""
        **start_positions**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, num_blocks,)``:
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
        **end_positions**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, num_blocks,)``:
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
        **retrieval_label**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, num_blocks,)``:
            Whether the retrieved evidence is the true evidence. For computing the sentece classification loss.
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
        **start_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length,)``
            Span-start scores (before SoftMax).
        **end_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length,)``
            Span-end scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape 
            ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.


    """
    def __init__(self, config):
        super(BertForOrconvqaGlobal, self).__init__(config)
        self.num_qa_labels = config.num_qa_labels

        self.bert = BertModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_qa_labels)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)
        
        self.qa_loss_factor = config.qa_loss_factor
        self.retrieval_loss_factor = config.retrieval_loss_factor
        


        # add interactive
        # self.attentionlayer = BertLayer(config)
        # self.pos_emb = nn.Embedding(5, config.hidden_size)   # num_blocks, hidden_size
        # self.layernorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, 
                position_ids=None, head_mask=None, inputs_embeds=None,
                start_positions=None, end_positions=None, retrieval_label=None):
        
        batch_size, num_blocks, seq_len = input_ids.size()
        input_ids = input_ids.view(-1, seq_len)
        attention_mask = attention_mask.view(-1, seq_len)
        token_type_ids = token_type_ids.view(-1, seq_len)               

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)

        sequence_output = outputs[0]
        pooled_output = outputs[1]


        qa_logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = qa_logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1) # (batch_size * num_blocks, seq_len)
        # print('start_logits', start_logits.size())
        end_logits = end_logits.squeeze(-1)
                
        pooled_output = self.dropout(pooled_output)
        retrieval_logits = self.classifier(pooled_output) # (batch_size * num_blocks, 1)
        # print('retrieval_logits', retrieval_logits.size())
        
        outputs = (start_logits, end_logits, retrieval_logits) + outputs[2:]
        if start_positions is not None and end_positions is not None and retrieval_label is not None:
            start_logits = start_logits.view(batch_size, -1)
            end_logits = end_logits.view(batch_size, -1)
            
            retrival_logits = retrieval_logits.squeeze(-1)
            retrieval_logits = retrieval_logits.view(batch_size, -1)
        
            start_positions = start_positions.squeeze(-1).max(dim=1).values
            end_positions = end_positions.squeeze(-1).max(dim=1).values
            retrieval_label = retrieval_label.squeeze(-1).argmax(dim=1)
            
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            qa_loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = qa_loss_fct(start_logits, start_positions)
            end_loss = qa_loss_fct(end_logits, end_positions)
            qa_loss = (start_loss + end_loss) / 2
            
            retrieval_loss_fct = CrossEntropyLoss()
            retrieval_loss = retrieval_loss_fct(retrieval_logits, retrieval_label)
            
            total_loss = self.qa_loss_factor * qa_loss + self.retrieval_loss_factor * retrieval_loss
                               
            outputs = (total_loss, qa_loss, retrieval_loss, retrieval_logits) + outputs

        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)
class BertForOrconvqaGlobal0(BertPreTrainedModel):
    r"""
        **start_positions**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, num_blocks,)``:
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
        **end_positions**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, num_blocks,)``:
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
        **retrieval_label**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, num_blocks,)``:
            Whether the retrieved evidence is the true evidence. For computing the sentece classification loss.
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
        **start_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length,)``
            Span-start scores (before SoftMax).
        **end_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length,)``
            Span-end scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape 
            ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.


    """
    def __init__(self, config):
        super(BertForOrconvqaGlobal0, self).__init__(config)
        self.num_qa_labels = config.num_qa_labels

        self.bert = BertModel(config)
        ########## xbli
        # self.mid_size = 8
        # self.qa_outputs0 = nn.Linear(config.hidden_size, self.mid_size)
        # self.qa_outputs1 = nn.Linear(self.mid_size//2,1)
        # self.qa_outputs3 = nn.Linear(config.hidden_size,self.num_qa_labels)
        self.qa_outputs = nn.Linear(config.hidden_size,self.num_qa_labels)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)
        
        self.qa_loss_factor = config.qa_loss_factor
        self.retrieval_loss_factor = config.retrieval_loss_factor
        self.contrastive_loss_factor = 0.2
        self.init_weights()
    ##################xbli
    '''
    对history_answer_indexs进行对比损失
    '''
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, \
                position_ids=None, head_mask=None, inputs_embeds=None, \
                start_positions=None, end_positions=None, retrieval_label=None,start_positions_his=None, \
                end_positions_his=None,targets=None,question_lengths=None):
        
        batch_size, num_blocks, seq_len = input_ids.size()
        input_ids = input_ids.view(-1, seq_len)
        attention_mask = attention_mask.view(-1, seq_len)
        token_type_ids = token_type_ids.view(-1, seq_len)               

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)

        sequence_output = outputs[0]
        pooled_output = outputs[1]
        qa_logits = self.qa_outputs(sequence_output)
        # start_logits_mid,end_logits_mid = qa_logits.split(self.mid_size//2, dim=-1)
        start_logits,end_logits = qa_logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1) # (batch_size * num_blocks, seq_len)
        # print('start_logits', start_logits.size())
        # qa_logits = self.qa_outputs0(sequence_output)
        # start_logits_mid,end_logits_mid = qa_logits.split(self.mid_size//2, dim=-1)
        # start_logits_mid = F.normalize(start_logits_mid,dim = -1)
        # end_logits_mid = F.normalize(end_logits_mid,dim = -1)
        # start_logits = self.qa_outputs1(start_logits_mid)
        # end_logits = self.qa_outputs1(end_logits_mid)
        # start_logits = start_logits.squeeze(-1) # (batch_size * num_blocks, seq_len)
        # print('start_logits', start_logits.size())
        end_logits = end_logits.squeeze(-1)
        # pdb.set_trace()
        pooled_output = self.dropout(pooled_output)
        retrieval_logits = self.classifier(pooled_output) # (batch_size * num_blocks, 1)
        # print('retrieval_logits', retrieval_logits.size())
        
        outputs = (start_logits, end_logits, retrieval_logits) + outputs[2:]
        # plt.plot(range(len(start_logits[0])),start_logits[0],label='s')
        # plt.plot(range(len(end_logits[0])),end_logits[0],label='e')
        # # start_positions
        # plt.legend()
        # plt.savefig('{}.jpg'.format(input_ids[0][20]))
        # plt.close('all')
        # pdb.set_trace()
        if start_positions is not None and end_positions is not None and retrieval_label is not None:
            # pdb.set_trace()


            start_logits_b_top_reader = start_logits.unsqueeze(0).reshape((batch_size,10,-1))
            end_logits_b_top_reader = end_logits.unsqueeze(0).reshape((batch_size,10,-1))
            x = (retrieval_label==1).expand((start_logits_b_top_reader.shape))
            start_logits_b_retrieve_reader = start_logits_b_top_reader[x].unsqueeze(0).reshape(batch_size,-1)
            end_logits_b_retrieve_reader = end_logits_b_top_reader[x].unsqueeze(0).reshape(batch_size,-1)
            idx_s=start_logits_b_retrieve_reader.detach().cpu().numpy().argsort()
            idx_s = idx_s[:,::-1]
            top_k_idx_s = idx_s[:,:2]
            idx_e=end_logits_b_retrieve_reader.detach().cpu().numpy().argsort()
            idx_e = idx_e[:,::-1]
            top_k_idx_e = idx_e[:,:2]

            # top_k_idx_s = top_k_idx_s.copy()

            start_logits = start_logits.view(batch_size, -1)
            end_logits = end_logits.view(batch_size, -1)
            # start_logits_mid = start_logits_mid.view(batch_size,-1,self.mid_size//2)
            # end_logits_mid = end_logits_mid.view(batch_size,-1,self.mid_size//2)

            retrival_logits = retrieval_logits.squeeze(-1)
            retrieval_logits = retrieval_logits.view(batch_size, -1)
            # start_positions_his = start_positions_his.squeeze(-1).max(dim=1).values
            # end_positions_his = end_positions_his.squeeze(-1).max(dim=1).values
            start_positions = start_positions.squeeze(-1).max(dim=1).values
            end_positions = end_positions.squeeze(-1).max(dim=1).values
            retrieval_label = retrieval_label.squeeze(-1).argmax(dim=1)
            # pdb.set_trace()
            #加一个softmax
            # start_logits = F.softmax(start_logits,dim = -1)
            # end_logits =  F.softmax(end_logits,dim = -1)
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)
            contrastive_loss = 0.0
            contrastive_loss_fct = TripletMarginLoss(margin = 0.1,p=2)#contrastive6_2_stage margin=0.1
            ######### predict
            EPS = 0.01
            # predict_start_logits = start_logits_b_retrieve_reader.max(dim = -1).values.detach().unsqueeze(-1).view(-1,1) + EPS
            # predict_end_logits = end_logits_b_retrieve_reader.max(dim = -1).values.detach().unsqueeze(-1).view(-1,1) + EPS
            predict_start_logits = start_logits_b_retrieve_reader.max(dim = -1).values.detach().unsqueeze(-1).view(-1,1) + EPS
            predict_end_logits = end_logits_b_retrieve_reader.max(dim = -1).values.detach().unsqueeze(-1).view(-1,1) + EPS
            for i in range(start_logits.shape[0]):
                true_position_start = start_positions[i]%512
                true_position_end  = end_positions[i]%512
                # pdb.set_trace()
                ###

                    #把bert的输出，先换成Out*Out.T[batch,seq,hidden]*[batch,hidden,seq]->[batch,seq,seq]
                    # simcse




                ###
                for j in range(2):
                    if  true_position_start+j <= true_position_end and true_position_start+j != top_k_idx_s[i][j]:
                            # pdb.set_trace()
                        contrastive_loss+=contrastive_loss_fct(predict_start_logits[i],start_logits_b_retrieve_reader[i][true_position_start+j].unsqueeze(-1).view(-1,1),start_logits_b_retrieve_reader[i][top_k_idx_s[i][j]].unsqueeze(-1).view(-1,1))
                    
                    if true_position_end - j>=true_position_start and true_position_end - j != top_k_idx_e[i][j]:
                        contrastive_loss+=contrastive_loss_fct(predict_end_logits[i],end_logits_b_retrieve_reader[i][true_position_end - j].unsqueeze(-1).view(-1,1),end_logits_b_retrieve_reader[i][top_k_idx_e[i][j]].unsqueeze(-1).view(-1,1))

            qa_loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = qa_loss_fct(start_logits, start_positions)
            end_loss = qa_loss_fct(end_logits, end_positions)
            qa_loss = (start_loss + end_loss) / 2
            # contrastive_loss_fct = SupervisedContrastiveLoss()
            # start_loss_contrastive=contrastive_loss_fct()
            
            retrieval_loss_fct = CrossEntropyLoss()
            retrieval_loss = retrieval_loss_fct(retrieval_logits, retrieval_label)
            
            total_loss = self.qa_loss_factor * qa_loss + self.retrieval_loss_factor * retrieval_loss + \
            self.contrastive_loss_factor*contrastive_loss
            if random.random()>0.9:
                print('modeling_hybrid_reader.py line 249 qa_loss: {} ,retrieval_loss : {}, contrastive_loss : {}'.format(qa_loss,\
                retrieval_loss,contrastive_loss))
                               
            outputs = (total_loss, qa_loss, retrieval_loss,retrieval_logits) + outputs

        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)
class BertForOrconvqaGlobal_DR_02(BertPreTrainedModel):#02表示dropout0.2
    r"""
        **start_positions**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, num_blocks,)``:
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
        **end_positions**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, num_blocks,)``:
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
        **retrieval_label**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, num_blocks,)``:
            Whether the retrieved evidence is the true evidence. For computing the sentece classification loss.
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
        **start_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length,)``
            Span-start scores (before SoftMax).
        **end_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length,)``
            Span-end scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape 
            ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.


    """
    def __init__(self, config):
        super(BertForOrconvqaGlobal_DR_02, self).__init__(config)
        self.num_qa_labels = config.num_qa_labels

        self.bert = BertModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_qa_labels)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)
        
        self.qa_loss_factor = config.qa_loss_factor
        self.retrieval_loss_factor = config.retrieval_loss_factor
        self.reader_dropout = nn.Dropout(0.2)
        # add interactive
        # self.attentionlayer = BertLayer(config)
        # self.pos_emb = nn.Embedding(5, config.hidden_size)   # num_blocks, hidden_size
        # self.layernorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, 
                position_ids=None, head_mask=None, inputs_embeds=None,
                start_positions=None, end_positions=None, retrieval_label=None):
        
        batch_size, num_blocks, seq_len = input_ids.size()
        input_ids = input_ids.view(-1, seq_len)
        attention_mask = attention_mask.view(-1, seq_len)
        token_type_ids = token_type_ids.view(-1, seq_len)               

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)

        sequence_output = outputs[0]
        pooled_output = outputs[1]
        ######## xbli
        sequence_output = self.reader_dropout(sequence_output)
        ########
        qa_logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = qa_logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1) # (batch_size * num_blocks, seq_len)
        # print('start_logits', start_logits.size())
        end_logits = end_logits.squeeze(-1)
                
        pooled_output = self.dropout(pooled_output)
        retrieval_logits = self.classifier(pooled_output) # (batch_size * num_blocks, 1)
        # print('retrieval_logits', retrieval_logits.size())
        
        outputs = (start_logits, end_logits, retrieval_logits) + outputs[2:]
        if start_positions is not None and end_positions is not None and retrieval_label is not None:
            start_logits = start_logits.view(batch_size, -1)
            end_logits = end_logits.view(batch_size, -1)
            
            retrival_logits = retrieval_logits.squeeze(-1)
            retrieval_logits = retrieval_logits.view(batch_size, -1)
        
            start_positions = start_positions.squeeze(-1).max(dim=1).values
            end_positions = end_positions.squeeze(-1).max(dim=1).values
            retrieval_label = retrieval_label.squeeze(-1).argmax(dim=1)
            
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            qa_loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = qa_loss_fct(start_logits, start_positions)
            end_loss = qa_loss_fct(end_logits, end_positions)
            qa_loss = (start_loss + end_loss) / 2
            
            retrieval_loss_fct = CrossEntropyLoss()
            retrieval_loss = retrieval_loss_fct(retrieval_logits, retrieval_label)
            if random.random()>0.9:
                print('reader loss {} , reranker loss {}'.format(qa_loss,retrieval_loss))
            total_loss = self.qa_loss_factor * qa_loss + self.retrieval_loss_factor * retrieval_loss
                               
            outputs = (total_loss, qa_loss, retrieval_loss, retrieval_logits) + outputs

        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)

# --------------------------------------------------
ALL_MODELS = sum(
    (tuple(conf.pretrained_config_archive_map.keys())
     for conf in (RobertaConfig, )),
    (),
)

default_process_fn = triple_process_fn


class MSMarcoConfig:
    def __init__(self,
                 name,
                 model,
                 process_fn=default_process_fn,
                 use_mean=True,
                 tokenizer_class=RobertaTokenizer,
                 config_class=RobertaConfig):
        self.name = name
        self.process_fn = process_fn
        self.model_class = model
        self.use_mean = use_mean
        self.tokenizer_class = tokenizer_class
        self.config_class = config_class


configs = [
    MSMarcoConfig(
        name="rdot_nll",
        model=RobertaDot_NLL_LN,
        use_mean=False,
    ),
    MSMarcoConfig(
        name="rdot_nll_multi_chunk",
        model=RobertaDot_CLF_ANN_NLL_MultiChunk,
        use_mean=False,
    ),
    MSMarcoConfig(
        name="dpr",
        model=BiEncoder,
        tokenizer_class=BertTokenizer,
        config_class=BertConfig,
        use_mean=False,
    ),
]

MSMarcoConfigDict = {cfg.name: cfg for cfg in configs}
class Pipeline(nn.Module):
    def __init__(self):
        super().__init__()
        self.reader = None
        self.retriever = None
     