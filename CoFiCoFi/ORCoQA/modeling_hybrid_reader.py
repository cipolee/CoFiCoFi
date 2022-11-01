import os
import logging
import collections
import torch
from transformers import BertModel, BertPreTrainedModel, AlbertModel
from transformers.modeling_bert import (BertEncoder, BertOutput, BertAttention, 
                                        BertIntermediate, BertLayer, BertEmbeddings,
                                        BertPooler, BertLayerNorm)
from transformers.modeling_albert import AlbertPreTrainedModel

from torch import nn
from torch.nn import CrossEntropyLoss , TripletMarginLoss
import torch.nn.functional as F
from copy import deepcopy
import numpy as np
from transformers.configuration_utils import PretrainedConfig
from transformers.file_utils import (TF2_WEIGHTS_NAME, TF_WEIGHTS_NAME, WEIGHTS_NAME, 
                         cached_path)
import  numpy as np
import matplotlib.pyplot as plt
# from pytorch_metric_learning import losses
import random
import pdb
SMALL_NUM = np.log(1e-45)
logger = logging.getLogger(__name__)

class BertForOrconvqa(BertPreTrainedModel):
    r"""
        **start_positions**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
        **end_positions**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
        **retrieval_label**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
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
        super(BertForOrconvqa, self).__init__(config)
        self.num_qa_labels = config.num_qa_labels
        self.num_retrieval_labels = config.num_retrieval_labels

        self.bert = BertModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_qa_labels)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_retrieval_labels)
        
        self.qa_loss_factor = config.qa_loss_factor
        self.retrieval_loss_factor = config.retrieval_loss_factor

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, 
                position_ids=None, head_mask=None, inputs_embeds=None,
                start_positions=None, end_positions=None, retrieval_label=None):

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
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        
        pooled_output = self.dropout(pooled_output)
        retrieval_logits = self.classifier(pooled_output)

        outputs = (start_logits, end_logits, retrieval_logits) + outputs[2:]
        if start_positions is not None and end_positions is not None and retrieval_label is not None:
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
            retrieval_loss = retrieval_loss_fct(retrieval_logits.view(-1, self.num_retrieval_labels), retrieval_label.view(-1))
            
            total_loss = self.qa_loss_factor * qa_loss + self.retrieval_loss_factor * retrieval_loss
                               
            outputs = (total_loss, qa_loss, retrieval_loss,) + outputs

        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)
    
    
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
        ########## xbli
        self.mid_size = 8
        # self.qa_outputs0 = nn.Linear(config.hidden_size, self.mid_size)
        # self.qa_outputs1 = nn.Linear(self.mid_size//2,1)
        self.qa_outputs3 = nn.Linear(config.hidden_size,self.num_qa_labels)
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
                end_positions_his=None,targets=None):
        
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
        qa_logits = self.qa_outputs3(sequence_output)
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


            start_logits_b_top_reader = start_logits.unsqueeze(0).reshape((batch_size,5,-1))
            end_logits_b_top_reader = end_logits.unsqueeze(0).reshape((batch_size,5,-1))
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
            contrastive_loss_fct = TripletMarginLoss(margin = 0.1,p=2)
            ######### predict
            EPS = 0.01
            # predict_start_logits = start_logits_b_retrieve_reader.max(dim = -1).values.detach().unsqueeze(-1).view(-1,1) + EPS
            # predict_end_logits = end_logits_b_retrieve_reader.max(dim = -1).values.detach().unsqueeze(-1).view(-1,1) + EPS
            predict_start_logits = start_logits_b_retrieve_reader.max(dim = -1).values.detach().unsqueeze(-1).view(-1,1) + EPS
            predict_end_logits = end_logits_b_retrieve_reader.max(dim = -1).values.detach().unsqueeze(-1).view(-1,1) + EPS
            for i in range(start_logits.shape[0]):
                # true_position_start = start_positions[i]%512
                # true_position_end  = start_positions[i]%512
                for j in range(2):
                    
                    if top_k_idx_s[i][j]!=start_positions[i]+j and start_positions[i]+j<start_logits.shape[1]:
                        # pdb.set_trace()
                        contrastive_loss+=contrastive_loss_fct(predict_start_logits[i],start_logits[i][start_positions[i]+j].unsqueeze(-1).view(-1,1),start_logits_b_retrieve_reader[i][top_k_idx_s[i][j]].unsqueeze(-1).view(-1,1))
                    if top_k_idx_e[i][j]!=end_positions[i]-j and end_positions[i]-j>=0:
                        # pdb.set_trace()
                        contrastive_loss+=contrastive_loss_fct(predict_end_logits[i],end_logits[i][end_positions[i]-j].unsqueeze(-1).view(-1,1),end_logits_b_retrieve_reader[i][top_k_idx_e[i][j]].unsqueeze(-1).view(-1,1))
            

            # pdb.set_trace()
            # if start_positions[0]>0 and end_positions[0]>0 and start_positions_his[0] != -1 and  end_positions_his[0] != -1 and not \
            #     (start_positions_his[0] <= start_positions[0] <= end_positions_his[0] or start_positions_his[0] <= end_positions[0]<= end_positions_his[0]):
                # triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
                # anchor = torch.randn(100, 128, requires_grad=True)
                # positive = torch.randn(100, 128, requires_grad=True)
                # negative = torch.randn(100, 128, requires_grad=True)
                # output = triplet_loss(anchor, positive, negative)
                ################# answer span
                # num_positive = end_positions[0]-start_positions[0]+1
                # num_nagetive = end_positions_his[0]-start_positions_his[0]+1
                # targets = torch.tensor([1]*(num_positive)+[0]*(num_nagetive),dtype=torch.long).to(targets.device)
                # contrastive_loss_fct = MarginContrastiveLoss(margin=0.5)# 43.4性能margin=0.5
                ###################### 多维度
                # pdb.set_trace()
                # contrastive_loss = (contrastive_loss_fct(start_logits_mid[:,start_positions[0],:].expand(2,4),
                # torch.cat((end_logits_mid[:,end_positions[0],:],start_logits_mid[:,start_positions_his[0],:]),dim=0),
                # targets = targets) + contrastive_loss_fct(end_logits_mid[:,end_positions[0],:].expand(2,4),
                # torch.cat((start_logits_mid[:,start_positions[0],:],end_logits_mid[:,end_positions_his[0],:]),dim=0),
                # targets = targets))/2
                ################# 单维度
                # contrastive_loss = (contrastive_loss_fct(start_logits[:,start_positions[0]].expand(2,1),
                # torch.cat((end_logits[:,end_positions[0]],start_logits[:,start_positions_his[0]]),dim=0),
                # targets = targets) + contrastive_loss_fct(end_logits[:,end_positions[0]].expand(2,1),
                # torch.cat((start_logits[:,start_positions[0]],end_logits[:,end_positions_his[0]]),dim=0),
                # targets = targets))/2
                ################## 单维度answer span
                # contrastive_loss = (contrastive_loss_fct(start_logits[:,start_positions[0]].expand(num_positive+num_nagetive,1),
                # torch.cat((end_logits[:,start_positions[0]:end_positions[0]+1].view(-1,1),start_logits[:,start_positions_his[0]:end_positions_his[0]+1].view(-1,1)),dim=0),
                # targets = targets) + contrastive_loss_fct(end_logits[:,end_positions[0]].expand(num_positive+num_nagetive,1),
                # torch.cat((start_logits[:,start_positions[0]:end_positions[0]+1].view(-1,1),end_logits[:,start_positions_his[0]:end_positions_his[0]+1].view(-1,1)),dim=0),
                # targets = targets))/2
                ################### 单维度 tripletmarginloss
                # pdb.set_trace()
                # contrastive_loss_fct = TripletMarginLoss(margin = 0.5,p=2)
                # contrastive_loss = (contrastive_loss_fct(predict_start_logits, start_logits[:,start_positions[0]], start_logits[:,start_positions_his[0]]) + \
                #     contrastive_loss_fct(predict_end_logits, end_logits[:,end_positions[0]], end_logits[:,end_positions_his[0]]))/2
            
            
            # ######## 多batch topk 进行对比loss

            # contrastive_loss_fct = TripletMarginLoss(margin = 0.1,p=2)

            # ################ start部分

            # contrastive_loss = (contrastive_loss_fct(predict_start_logits, start_logits[:,start_positions[0]], start_logits[:,start_positions_his[0]]) + \
            #         contrastive_loss_fct(predict_end_logits, end_logits[:,end_positions[0]], end_logits[:,end_positions_his[0]]))/2
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
    
class AlbertForOrconvqaGlobal(AlbertPreTrainedModel):
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
        super(AlbertForOrconvqaGlobal, self).__init__(config)
        self.num_qa_labels = config.num_qa_labels

        self.albert = AlbertModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_qa_labels)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)
        
        self.qa_loss_factor = config.qa_loss_factor
        self.retrieval_loss_factor = config.retrieval_loss_factor

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, 
                position_ids=None, head_mask=None, inputs_embeds=None,
                start_positions=None, end_positions=None, retrieval_label=None):
        
        batch_size, num_blocks, seq_len = input_ids.size()
        input_ids = input_ids.view(-1, seq_len)
        attention_mask = attention_mask.view(-1, seq_len)
        token_type_ids = token_type_ids.view(-1, seq_len)               

        outputs = self.albert(input_ids,
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
                               
            outputs = (total_loss, qa_loss, retrieval_loss,) + outputs

        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)
 
    
class BertForRetriever(BertPreTrainedModel):
    r"""
    
    """
    def __init__(self, config):
        super(BertForRetriever, self).__init__(config)

        self.query_encoder = BertModel(config)
        self.query_proj = nn.Linear(config.hidden_size, config.proj_size)
        
        self.passage_encoder = BertModel(config)
        self.passage_proj = nn.Linear(config.hidden_size, config.proj_size)
        self.proj_size = config.proj_size
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        self.init_weights()

    def forward(self, query_input_ids=None, query_attention_mask=None, query_token_type_ids=None, 
                passage_input_ids=None, passage_attention_mask=None, passage_token_type_ids=None, 
                retrieval_label=None):
        outputs = ()
        
        if query_input_ids is not None:
            query_outputs = self.query_encoder(query_input_ids,
                                attention_mask=query_attention_mask,
                                token_type_ids=query_token_type_ids)
            
            query_pooled_output = query_outputs[1]
            query_pooled_output = self.dropout(query_pooled_output)
            query_rep = self.query_proj(query_pooled_output) # batch_size, proj_size    
            # print(query_rep[:, 0])
            outputs = (query_rep, ) + outputs
        
        if passage_input_ids is not None:
            if len(passage_input_ids.size()) == 3:
                # this means we are pretraining
                batch_size, num_blocks, seq_len = passage_input_ids.size()
                passage_input_ids = passage_input_ids.view(-1, seq_len) # batch_size * num_blocks, seq_len
                passage_attention_mask = passage_attention_mask.view(-1, seq_len)
                passage_token_type_ids = passage_token_type_ids.view(-1, seq_len) 

            passage_outputs = self.passage_encoder(passage_input_ids,
                                attention_mask=passage_attention_mask,
                                token_type_ids=passage_token_type_ids) 

            passage_pooled_output = passage_outputs[1] 
            passage_pooled_output = self.dropout(passage_pooled_output)
            passage_rep = self.passage_proj(passage_pooled_output) # batch_size * num_blocks, proj_size
            # print(passage_rep[:, 0])
            outputs = (passage_rep, ) + outputs
                       
        if query_input_ids is not None and passage_input_ids is not None and retrieval_label is not None:
            passage_rep = passage_rep.view(batch_size, num_blocks, -1) # batch_size, num_blocks, proj_size      
            query_rep = query_rep.unsqueeze(-1) # query_rep (batch_size, proj_size, 1)
            query_rep = query_rep.expand(batch_size, self.proj_size, num_blocks) # batch_size, proj_size, num_blocks)
            query_rep = query_rep.transpose(1, 2) # query_rep (batch_size, num_blocks, proj_size)
            retrieval_logits = query_rep * passage_rep # batch_size, num_blocks, proj_size
            retrieval_logits = torch.sum(retrieval_logits, dim=-1) # batch_size, num_blocks
            retrieval_probs = F.softmax(retrieval_logits, dim=1)
            # print('retrieval_label before', retrieval_label.size(), retrieval_label)
            retrieval_label = retrieval_label.squeeze(-1).argmax(dim=1)
            # print('retrieval_label after', retrieval_label.size(), retrieval_label)
            retrieval_loss_fct = CrossEntropyLoss()
            # print('retrieval_logits', retrieval_logits.size(), retrieval_logits)
            # print('retrieval_label', retrieval_label.size(), retrieval_label)
            retrieval_loss = retrieval_loss_fct(retrieval_logits, retrieval_label)
            
            retrieval_logits = retrieval_logits.view(-1)
            outputs = (retrieval_loss, retrieval_logits, retrieval_probs) + outputs

        return outputs
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        r"""
        """
        if pretrained_model_name_or_path is not None and (
                "albert" in pretrained_model_name_or_path and "v2" in pretrained_model_name_or_path):
            logger.warning("There is currently an upstream reproducibility issue with ALBERT v2 models. Please see " +
                           "https://github.com/google-research/google-research/issues/119 for more information.")

        config = kwargs.pop('config', None)
        state_dict = kwargs.pop('state_dict', None)
        cache_dir = kwargs.pop('cache_dir', None)
        from_tf = kwargs.pop('from_tf', False)
        force_download = kwargs.pop('force_download', False)
        resume_download = kwargs.pop('resume_download', False)
        proxies = kwargs.pop('proxies', None)
        output_loading_info = kwargs.pop('output_loading_info', False)

        # Load config
        if config is None:
            config, model_kwargs = cls.config_class.from_pretrained(
                pretrained_model_name_or_path, *model_args,
                cache_dir=cache_dir, return_unused_kwargs=True,
                force_download=force_download,
                proxies=proxies,
                **kwargs
            )
        else:
            model_kwargs = kwargs

        # Load model
        if pretrained_model_name_or_path is not None:
            if pretrained_model_name_or_path in cls.pretrained_model_archive_map:
                archive_file = cls.pretrained_model_archive_map[pretrained_model_name_or_path]
            elif os.path.isdir(pretrained_model_name_or_path):
                if from_tf and os.path.isfile(os.path.join(pretrained_model_name_or_path, TF_WEIGHTS_NAME + ".index")):
                    # Load from a TF 1.0 checkpoint
                    archive_file = os.path.join(pretrained_model_name_or_path, TF_WEIGHTS_NAME + ".index")
                elif from_tf and os.path.isfile(os.path.join(pretrained_model_name_or_path, TF2_WEIGHTS_NAME)):
                    # Load from a TF 2.0 checkpoint
                    archive_file = os.path.join(pretrained_model_name_or_path, TF2_WEIGHTS_NAME)
                elif os.path.isfile(os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)):
                    # Load from a PyTorch checkpoint
                    archive_file = os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)
                else:
                    raise EnvironmentError("Error no file named {} found in directory {} or `from_tf` set to False".format(
                        [WEIGHTS_NAME, TF2_WEIGHTS_NAME, TF_WEIGHTS_NAME + ".index"],
                        pretrained_model_name_or_path))
            elif os.path.isfile(pretrained_model_name_or_path + ".index"):
                assert from_tf, "We found a TensorFlow checkpoint at {}, please set from_tf to True to load from this checkpoint".format(
                    pretrained_model_name_or_path + ".index")
                archive_file = pretrained_model_name_or_path + ".index"
            

  
            # redirect to the cache, if necessary
            try:
                resolved_archive_file = cached_path(archive_file, cache_dir=cache_dir, force_download=force_download,
                                                    proxies=proxies)
            except EnvironmentError:
                if pretrained_model_name_or_path in cls.pretrained_model_archive_map:
                    msg = "Couldn't reach server at '{}' to download pretrained weights.".format(
                            archive_file)
                else:
                    msg = "Model name '{}' was not found in model name list ({}). " \
                        "We assumed '{}' was a path or url to model weight files named one of {} but " \
                        "couldn't find any such file at this path or url.".format(
                            pretrained_model_name_or_path,
                            ', '.join(cls.pretrained_model_archive_map.keys()),
                            archive_file,
                            [WEIGHTS_NAME, TF2_WEIGHTS_NAME, TF_WEIGHTS_NAME])
                raise EnvironmentError(msg)
                
            if resolved_archive_file == archive_file:
                logger.info("loading weights file {}".format(archive_file))
            else:
                logger.info("loading weights file {} from cache at {}".format(
                    archive_file, resolved_archive_file))
        else:
            resolved_archive_file = None

        # Instantiate model.
        model = cls(config, *model_args, **model_kwargs)

        if state_dict is None and not from_tf:
            state_dict = torch.load(resolved_archive_file, map_location='cpu')

        missing_keys = []
        unexpected_keys = []
        error_msgs = []

        if from_tf:
            if resolved_archive_file.endswith('.index'):
                # Load from a TensorFlow 1.X checkpoint - provided by original authors
                model = cls.load_tf_weights(model, config, resolved_archive_file[:-6])  # Remove the '.index'
            else:
                # Load from our TensorFlow 2.0 checkpoints
                try:
                    from transformers import load_tf2_checkpoint_in_pytorch_model
                    model = load_tf2_checkpoint_in_pytorch_model(model, resolved_archive_file, allow_missing_keys=True)
                except ImportError as e:
                    logger.error("Loading a TensorFlow model in PyTorch, requires both PyTorch and TensorFlow to be installed. Please see "
                        "https://pytorch.org/ and https://www.tensorflow.org/install/ for installation instructions.")
                    raise e
        else:
            # Convert old format to new format if needed from a PyTorch state_dict
            old_keys = []
            new_keys = []
            for key in state_dict.keys():
                new_key = None
                if 'gamma' in key:
                    new_key = key.replace('gamma', 'weight')
                if 'beta' in key:
                    new_key = key.replace('beta', 'bias')
                if key == 'lm_head.decoder.weight':
                    new_key = 'lm_head.weight'
                if new_key:
                    old_keys.append(key)
                    new_keys.append(new_key)
            for old_key, new_key in zip(old_keys, new_keys):
                state_dict[new_key] = state_dict.pop(old_key)

            # copy state_dict so _load_from_state_dict can modify it
            metadata = getattr(state_dict, '_metadata', None)
            # print('orig state dict', state_dict.keys(), len(state_dict))
            customized_state_dict = collections.OrderedDict()
            for k, v in state_dict.items():
                k_split = k.split('.')
                if k_split[0] == 'bert':
                    k_split[0] = 'query_encoder'
                    customized_state_dict['.'.join(k_split)] = v
                    k_split[0] = 'passage_encoder'
                    customized_state_dict['.'.join(k_split)] = v
                    
            if len(customized_state_dict) == 0:
                # loading from our trained model
                state_dict = state_dict.copy()
                # print('using orig state dict', state_dict.keys())
            else:
                # loading from original bert model
                state_dict = customized_state_dict.copy()
                # print('using custome state dict', state_dict.keys())
            
            # print('modified state dict', state_dict.keys(), len(state_dict))
            if metadata is not None:
                state_dict._metadata = metadata

            # PyTorch's `_load_from_state_dict` does not copy parameters in a module's descendants
            # so we need to apply the function recursively.
            def load(module, prefix=''):
                local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
                module._load_from_state_dict(
                    state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
                for name, child in module._modules.items():
                    if child is not None:
                        load(child, prefix + name + '.')

            # Make sure we are able to load base models as well as derived models (with heads)
            start_prefix = ''
            model_to_load = model
#             if not hasattr(model, cls.base_model_prefix) and any(s.startswith(cls.base_model_prefix) for s in state_dict.keys()):
#                 start_prefix = cls.base_model_prefix + '.'
#             if hasattr(model, cls.base_model_prefix) and not any(s.startswith(cls.base_model_prefix) for s in state_dict.keys()):
#                 model_to_load = getattr(model, cls.base_model_prefix)

#             load(model_to_load, prefix=start_prefix)
            load(model_to_load, prefix='')
            if len(missing_keys) > 0:
                logger.info("Weights of {} not initialized from pretrained model: {}".format(
                    model.__class__.__name__, missing_keys))
            if len(unexpected_keys) > 0:
                logger.info("Weights from pretrained model not used in {}: {}".format(
                    model.__class__.__name__, unexpected_keys))
            if len(error_msgs) > 0:
                raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                                model.__class__.__name__, "\n\t".join(error_msgs)))

        model.tie_weights()  # make sure word embedding weights are still tied if needed

        # Set model in evaluation mode to desactivate DropOut modules by default
        model.eval()

        if output_loading_info:
            loading_info = {"missing_keys": missing_keys, "unexpected_keys": unexpected_keys, "error_msgs": error_msgs}
            return model, loading_info

        return model
    
    
class BertForRetrieverOnlyPositivePassage(BertForRetriever):
    r"""
    
    """
    def __init__(self, config):
        super(BertForRetriever, self).__init__(config)

        self.query_encoder = BertModel(config)
        self.query_proj = nn.Linear(config.hidden_size, config.proj_size)
        
        self.passage_encoder = BertModel(config)
        self.passage_proj = nn.Linear(config.hidden_size, config.proj_size)
        self.proj_size = config.proj_size
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        self.init_weights()

    def forward(self, query_input_ids=None, query_attention_mask=None, query_token_type_ids=None, 
                passage_input_ids=None, passage_attention_mask=None, passage_token_type_ids=None, 
                retrieval_label=None):
        outputs = ()
        
        if query_input_ids is not None:
            query_outputs = self.query_encoder(query_input_ids,
                                attention_mask=query_attention_mask,
                                token_type_ids=query_token_type_ids)
            
            query_pooled_output = query_outputs[1]
            query_pooled_output = self.dropout(query_pooled_output)
            query_rep = self.query_proj(query_pooled_output) # batch_size, proj_size    
            # print(query_rep[:, 0])
            outputs = (query_rep, ) + outputs
        
        if passage_input_ids is not None:
            passage_outputs = self.passage_encoder(passage_input_ids,
                                attention_mask=passage_attention_mask,
                                token_type_ids=passage_token_type_ids) 

            passage_pooled_output = passage_outputs[1] 
            passage_pooled_output = self.dropout(passage_pooled_output)
            passage_rep = self.passage_proj(passage_pooled_output) # batch_size, proj_size
            # print(passage_rep[:, 0])
            outputs = (passage_rep, ) + outputs
                       
        if query_input_ids is not None and passage_input_ids is not None:
            passage_rep_t = passage_rep.transpose(0, 1) # proj_size, batch_size
            retrieval_logits = torch.matmul(query_rep, passage_rep_t) # batch_size, batch_size
            retrieval_label = torch.arange(query_rep.size(0), device=query_rep.device, dtype=retrieval_label.dtype)
            # print('retrieval_label after', retrieval_label.size(), retrieval_label)
            retrieval_loss_fct = CrossEntropyLoss()
            # print('retrieval_logits', retrieval_logits.size(), retrieval_logits)
            # print('retrieval_label', retrieval_label.size(), retrieval_label)
            retrieval_loss = retrieval_loss_fct(retrieval_logits, retrieval_label)
            
            outputs = (retrieval_loss, ) + outputs

        return outputs
    
class AlbertForRetrieverOnlyPositivePassage(AlbertPreTrainedModel):
    r"""
    
    """
    def __init__(self, config):
        super(AlbertForRetrieverOnlyPositivePassage, self).__init__(config)

        self.query_encoder = AlbertModel(config)
        self.query_proj = nn.Linear(config.hidden_size, config.proj_size)
        
        self.passage_encoder = AlbertModel(config)
        self.passage_proj = nn.Linear(config.hidden_size, config.proj_size)
        self.proj_size = config.proj_size
        self.temperature = 0.1
        self.sigma = 0.5
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        self.init_weights()

    def forward(self, query_input_ids=None, query_attention_mask=None, query_token_type_ids=None, 
                passage_input_ids=None, passage_attention_mask=None, passage_token_type_ids=None, 
                negative_input_ids=None, negative_attention_mask=None, negative_token_type_ids=None,
                retrieval_label=None, query_rep=None, passage_rep=None):
        outputs = ()
        
        if query_input_ids is not None:
            query_outputs = self.query_encoder(query_input_ids,
                                attention_mask=query_attention_mask,
                                token_type_ids=query_token_type_ids)
            
            query_pooled_output = query_outputs[1]
            query_pooled_output = self.dropout(query_pooled_output)
            query_rep = self.query_proj(query_pooled_output) # batch_size, proj_size    
            # print(query_rep[:, 0])
            outputs = (query_rep, ) + outputs
        
        if passage_input_ids is not None:
            passage_outputs = self.passage_encoder(passage_input_ids,
                                attention_mask=passage_attention_mask,
                                token_type_ids=passage_token_type_ids) 

            passage_pooled_output = passage_outputs[1] 
            passage_pooled_output = self.dropout(passage_pooled_output)
            passage_rep = self.passage_proj(passage_pooled_output) # batch_size, proj_size
            # print(passage_rep[:, 0])
            outputs = (passage_rep, ) + outputs
        
        if negative_input_ids is not None:
            negative_outputs = self.passage_encoder(negative_input_ids,
                                attention_mask=negative_attention_mask,
                                token_type_ids=negative_token_type_ids) 

            negative_pooled_output = negative_outputs[1] 
            negative_pooled_output = self.dropout(negative_pooled_output)
            negative_rep = self.passage_proj(negative_pooled_output) # batch_size, proj_size
            outputs = (negative_rep, ) + outputs

        if query_input_ids is not None and passage_input_ids is not None:
            

            ## in-batch negative   /  query 
            if negative_input_ids is not None:    ## 增加BM25/TF-IDF负例
                passage_rep = torch.cat((passage_rep, negative_rep), dim=0)   # 2*batch_size, proj_size 

            passage_rep_t = passage_rep.transpose(0, 1) # proj_size, batch_size / 2*batch_size
            retrieval_logits = torch.matmul(query_rep, passage_rep_t) # batch_size, batch_size / 2 * batch_size
            retrieval_label = torch.arange(query_rep.size(0), device=query_rep.device, dtype=retrieval_label.dtype)
            retrieval_loss_fct = CrossEntropyLoss()
            retrieval_loss = retrieval_loss_fct(retrieval_logits, retrieval_label)

            ## in-batch negative   /  passage   暂时没用过
            # query_rep_t = query_rep.transpose(0, 1)
            # retrieval_logits_p = torch.matmul(passage_rep, query_rep_t)
            # retrieval_label_p = torch.arange(passage_rep.size(0), device=passage_rep.device, dtype=retrieval_label.dtype)
            # retrieval_loss += retrieval_loss_fct(retrieval_logits_p, retrieval_label_p)

            ## in-batch negative   /  query  Decouple CLW
            # cross_view_distance = torch.mm(query_rep, passage_rep.t())
            # positive_loss = -torch.diag(cross_view_distance) / self.temperature

            # # weight_fn = lambda z1, z2: 2 - z1.size(0) * torch.nn.functional.softmax((z1 * z2).sum(dim=1) / self.sigma, dim=0).squeeze()
            # # positive_loss = positive_loss * weight_fn(query_rep, passage_rep)

            # neg_similarity = torch.cat((cross_view_distance, torch.mm(query_rep, negative_rep.t())), dim=1) / self.temperature
            # neg_mask = torch.cat((torch.eye(query_rep.size(0), device=query_rep.device), torch.zeros((query_rep.size(0), query_rep.size(0)), device=query_rep.device)), dim=1)
            
            # negative_loss = torch.logsumexp(neg_similarity + neg_mask * SMALL_NUM, dim=1, keepdim=False)
            # retrieval_loss = (positive_loss + negative_loss).mean()


            outputs = (retrieval_loss, ) + outputs
            
        if query_input_ids is not None and passage_rep is not None and retrieval_label is not None and len(passage_rep.size()) == 3:
            # this is during fine tuning
            # passage_rep: batch_size, num_blocks, proj_size      
            query_outputs = self.query_encoder(query_input_ids,
                                attention_mask=query_attention_mask,
                                token_type_ids=query_token_type_ids)
            
            query_pooled_output = query_outputs[1]
            query_pooled_output = self.dropout(query_pooled_output)
            query_rep = self.query_proj(query_pooled_output) # batch_size, proj_size  
            
            batch_size, num_blocks, proj_size = passage_rep.size()
            query_rep = query_rep.unsqueeze(-1) # query_rep (batch_size, proj_size, 1)
            query_rep = query_rep.expand(batch_size, self.proj_size, num_blocks) # batch_size, proj_size, num_blocks)
            query_rep = query_rep.transpose(1, 2) # query_rep (batch_size, num_blocks, proj_size)
            retrieval_logits = query_rep * passage_rep # batch_size, num_blocks, proj_size
            retrieval_logits = torch.sum(retrieval_logits, dim=-1) # batch_size, num_blocks
            retrieval_probs = F.softmax(retrieval_logits, dim=1)
            # print('retrieval_label before', retrieval_label.size(), retrieval_label)
            retrieval_label = retrieval_label.squeeze(-1).argmax(dim=1)
            # print('retrieval_label after', retrieval_label.size(), retrieval_label)
            retrieval_loss_fct = CrossEntropyLoss()
            # print('retrieval_logits', retrieval_logits.size(), retrieval_logits)
            # print('retrieval_label', retrieval_label.size(), retrieval_label)
            retrieval_loss = retrieval_loss_fct(retrieval_logits, retrieval_label)
            ############xbli0 idea rocketv2
            # outputs = (retrieval_loss, ) + outputs
            outputs = (retrieval_logits,) + outputs

        return outputs
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        r"""
        """
        if pretrained_model_name_or_path is not None and (
                "albert" in pretrained_model_name_or_path and "v2" in pretrained_model_name_or_path):
            logger.warning("There is currently an upstream reproducibility issue with ALBERT v2 models. Please see " +
                           "https://github.com/google-research/google-research/issues/119 for more information.")

        config = kwargs.pop('config', None)
        state_dict = kwargs.pop('state_dict', None)
        cache_dir = kwargs.pop('cache_dir', None)
        from_tf = kwargs.pop('from_tf', False)
        force_download = kwargs.pop('force_download', False)
        resume_download = kwargs.pop('resume_download', False)
        proxies = kwargs.pop('proxies', None)
        output_loading_info = kwargs.pop('output_loading_info', False)

        # Load config
        if config is None:
            config, model_kwargs = cls.config_class.from_pretrained(
                pretrained_model_name_or_path, *model_args,
                cache_dir=cache_dir, return_unused_kwargs=True,
                force_download=force_download,
                proxies=proxies,
                **kwargs
            )
        else:
            model_kwargs = kwargs

        # Load model
        if pretrained_model_name_or_path is not None:
            if pretrained_model_name_or_path in cls.pretrained_model_archive_map:
                archive_file = cls.pretrained_model_archive_map[pretrained_model_name_or_path]
            elif os.path.isdir(pretrained_model_name_or_path):
                if from_tf and os.path.isfile(os.path.join(pretrained_model_name_or_path, TF_WEIGHTS_NAME + ".index")):
                    # Load from a TF 1.0 checkpoint
                    archive_file = os.path.join(pretrained_model_name_or_path, TF_WEIGHTS_NAME + ".index")
                elif from_tf and os.path.isfile(os.path.join(pretrained_model_name_or_path, TF2_WEIGHTS_NAME)):
                    # Load from a TF 2.0 checkpoint
                    archive_file = os.path.join(pretrained_model_name_or_path, TF2_WEIGHTS_NAME)
                elif os.path.isfile(os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)):
                    # Load from a PyTorch checkpoint
                    archive_file = os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)
                else:
                    raise EnvironmentError("Error no file named {} found in directory {} or `from_tf` set to False".format(
                        [WEIGHTS_NAME, TF2_WEIGHTS_NAME, TF_WEIGHTS_NAME + ".index"],
                        pretrained_model_name_or_path))
            elif os.path.isfile(pretrained_model_name_or_path + ".index"):
                assert from_tf, "We found a TensorFlow checkpoint at {}, please set from_tf to True to load from this checkpoint".format(
                    pretrained_model_name_or_path + ".index")
                archive_file = pretrained_model_name_or_path + ".index"
            

  
            # redirect to the cache, if necessary
            try:
                resolved_archive_file = cached_path(archive_file, cache_dir=cache_dir, force_download=force_download,
                                                    proxies=proxies)
            except EnvironmentError:
                if pretrained_model_name_or_path in cls.pretrained_model_archive_map:
                    msg = "Couldn't reach server at '{}' to download pretrained weights.".format(
                            archive_file)
                else:
                    msg = "Model name '{}' was not found in model name list ({}). " \
                        "We assumed '{}' was a path or url to model weight files named one of {} but " \
                        "couldn't find any such file at this path or url.".format(
                            pretrained_model_name_or_path,
                            ', '.join(cls.pretrained_model_archive_map.keys()),
                            archive_file,
                            [WEIGHTS_NAME, TF2_WEIGHTS_NAME, TF_WEIGHTS_NAME])
                raise EnvironmentError(msg)
                
            if resolved_archive_file == archive_file:
                logger.info("loading weights file {}".format(archive_file))
            else:
                logger.info("loading weights file {} from cache at {}".format(
                    archive_file, resolved_archive_file))
        else:
            resolved_archive_file = None

        # Instantiate model.
        model = cls(config, *model_args, **model_kwargs)

        if state_dict is None and not from_tf:
            state_dict = torch.load(resolved_archive_file, map_location='cpu')

        missing_keys = []
        unexpected_keys = []
        error_msgs = []

        if from_tf:
            if resolved_archive_file.endswith('.index'):
                # Load from a TensorFlow 1.X checkpoint - provided by original authors
                model = cls.load_tf_weights(model, config, resolved_archive_file[:-6])  # Remove the '.index'
            else:
                # Load from our TensorFlow 2.0 checkpoints
                try:
                    from transformers import load_tf2_checkpoint_in_pytorch_model
                    model = load_tf2_checkpoint_in_pytorch_model(model, resolved_archive_file, allow_missing_keys=True)
                except ImportError as e:
                    logger.error("Loading a TensorFlow model in PyTorch, requires both PyTorch and TensorFlow to be installed. Please see "
                        "https://pytorch.org/ and https://www.tensorflow.org/install/ for installation instructions.")
                    raise e
        else:
            # Convert old format to new format if needed from a PyTorch state_dict
            old_keys = []
            new_keys = []
            for key in state_dict.keys():
                new_key = None
                if 'gamma' in key:
                    new_key = key.replace('gamma', 'weight')
                if 'beta' in key:
                    new_key = key.replace('beta', 'bias')
                if key == 'lm_head.decoder.weight':
                    new_key = 'lm_head.weight'
                if new_key:
                    old_keys.append(key)
                    new_keys.append(new_key)
            for old_key, new_key in zip(old_keys, new_keys):
                state_dict[new_key] = state_dict.pop(old_key)

            # copy state_dict so _load_from_state_dict can modify it
            metadata = getattr(state_dict, '_metadata', None)
            # print('orig state dict', state_dict.keys(), len(state_dict))
            customized_state_dict = collections.OrderedDict()
            for k, v in state_dict.items():
                k_split = k.split('.')
                if k_split[0] == 'albert':
                    k_split[0] = 'query_encoder'
                    customized_state_dict['.'.join(k_split)] = v
                    k_split[0] = 'passage_encoder'
                    customized_state_dict['.'.join(k_split)] = v
                    
            if len(customized_state_dict) == 0:
                # loading from our trained model
                state_dict = state_dict.copy()
                # print('using orig state dict', state_dict.keys())
            else:
                # loading from original bert model
                state_dict = customized_state_dict.copy()
                # print('using custome state dict', state_dict.keys())
            
            # print('modified state dict', state_dict.keys(), len(state_dict))
            if metadata is not None:
                state_dict._metadata = metadata

            # PyTorch's `_load_from_state_dict` does not copy parameters in a module's descendants
            # so we need to apply the function recursively.
            def load(module, prefix=''):
                local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
                module._load_from_state_dict(
                    state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
                for name, child in module._modules.items():
                    if child is not None:
                        load(child, prefix + name + '.')

            # Make sure we are able to load base models as well as derived models (with heads)
            start_prefix = ''
            model_to_load = model
#             if not hasattr(model, cls.base_model_prefix) and any(s.startswith(cls.base_model_prefix) for s in state_dict.keys()):
#                 start_prefix = cls.base_model_prefix + '.'
#             if hasattr(model, cls.base_model_prefix) and not any(s.startswith(cls.base_model_prefix) for s in state_dict.keys()):
#                 model_to_load = getattr(model, cls.base_model_prefix)

#             load(model_to_load, prefix=start_prefix)
            load(model_to_load, prefix='')
            if len(missing_keys) > 0:
                logger.info("Weights of {} not initialized from pretrained model: {}".format(
                    model.__class__.__name__, missing_keys))
            if len(unexpected_keys) > 0:
                logger.info("Weights from pretrained model not used in {}: {}".format(
                    model.__class__.__name__, unexpected_keys))
            if len(error_msgs) > 0:
                raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                                model.__class__.__name__, "\n\t".join(error_msgs)))

        model.tie_weights()  # make sure word embedding weights are still tied if needed

        # Set model in evaluation mode to desactivate DropOut modules by default
        model.eval()

        if output_loading_info:
            loading_info = {"missing_keys": missing_keys, "unexpected_keys": unexpected_keys, "error_msgs": error_msgs}
            return model, loading_info

        return model
    
class Pipeline(nn.Module):
    def __init__(self):
        super(Pipeline, self).__init__()
        # super().__init__()
        
        self.reader = None
        self.retriever = None

    
# class QueryEncoder(BertPreTrainedModel):
#     r"""
    
#     """
#     def __init__(self, config):
#         super(QueryEncoder, self).__init__(config)

#         self.query_encoder = BertModel(config)
#         self.query_proj = nn.Linear(config.hidden_size, config.proj_size)        
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.proj_size = config.proj_size
        
#         self.init_weights()

#     def forward(self, query_input_ids=None, query_attention_mask=None, query_token_type_ids=None):

#         query_outputs = self.query_encoder(query_input_ids,
#                             attention_mask=query_attention_mask,
#                             token_type_ids=query_token_type_ids)

#         query_pooled_output = query_outputs[1]
#         query_pooled_output = self.dropout(query_pooled_output)
#         query_rep = self.query_proj(query_pooled_output) # batch_size, proj_size            
#         outputs = (query_rep, ) + query_outputs[2:]
        
#         return outputs
    
# class PassageEncoder(BertPreTrainedModel):
#     r"""
    
#     """
#     def __init__(self, config):
#         super(PassageEncoder, self).__init__(config)
        
#         self.passage_encoder = BertModel(config)
#         self.passage_proj = nn.Linear(config.hidden_size, config.proj_size)
#         self.proj_size = config.proj_size       
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
#         self.init_weights()

#     def forward(self, passage_input_ids=None, passage_attention_mask=None, 
#                 passage_token_type_ids=None, retrieval_label=None):
        
#         batch_size, num_blocks, seq_len = passage_input_ids.size()
#         passage_input_ids = passage_input_ids.view(-1, seq_len) # batch_size * num_blocks, seq_len
#         passage_attention_mask = passage_attention_mask.view(-1, seq_len)
#         passage_token_type_ids = passage_token_type_ids.view(-1, seq_len) 

#         passage_outputs = self.passage_encoder(passage_input_ids,
#                             attention_mask=passage_attention_mask,
#                             token_type_ids=passage_token_type_ids) 

#         passage_pooled_output = passage_outputs[1] 
#         passage_pooled_output = self.dropout(passage_pooled_output)
#         passage_rep = self.passage_proj(passage_pooled_output) # batch_size * num_blocks, proj_size
#         passage_rep = passage_rep.view(batch_size, num_blocks, -1) # batch_size, num_blocks, proj_size

#         outputs = (passage_rep, ) + passage_outputs[2:]
        
#         return outputs
    
# class Retriever(nn.Module):
#     r"""
    
#     """

#     def forward(self, query_rep, passage_rep, retrieval_label=None):
#         batch_size, num_blocks, proj_size = passage_rep.size()
#         query_rep = query_rep.unsqueeze(-1) # query_rep (batch_size, proj_size, 1)
#         query_rep = query_rep.expand(batch_size, proj_size, num_blocks) # batch_size, proj_size, num_blocks)
#         query_rep = query_rep.transpose(1, 2) # query_rep (batch_size, num_blocks, proj_size)
#         retrieval_logits = query_rep * passage_rep # batch_size, num_blocks, proj_size
#         retrieval_logits = torch.sum(retrieval_logits, dim=-1) # batch_size, num_blocks
#         retrieval_probs = F.softmax(retrieval_logits, dim=1)
        
#         if retrieval_label is not None:
#             retrieval_label = retrieval_label.squeeze(-1).argmax(dim=1)
#             retrieval_loss_fct = CrossEntropyLoss()
#             retrieval_loss = retrieval_loss_fct(retrieval_logits, retrieval_label)

#             retrieval_logits = retrieval_logits.view(-1)
#             outputs = (retrieval_loss, retrieval_logits, retrieval_probs)
#         else:
#             retrieval_logits = retrieval_logits.view(-1)
#             outputs = (retrieval_logits, retrieval_probs)

#         return outputs

##################xbli
class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, feature_vectors, labels):
        # Normalize feature vectors
        feature_vectors_normalized = F.normalize(feature_vectors, p=2, dim=1)
        # Compute logits
        logits = torch.div(
            torch.matmul(
                feature_vectors_normalized, torch.transpose(feature_vectors_normalized, 0, 1)
            ),
            self.temperature,
        )
        return losses.NTXentLoss(temperature=0.07)(logits, torch.squeeze(labels))
class MarginContrastiveLoss(nn.Module):
    def __init__(self, margin):
        super().__init__()
        self.margin = margin

        ######### train和evaluate的gap
    def forward(self, output1, output2, targets): #[batch,sample_num,hidden_size] #targets应该是起始位置和结束位置
        ########### 问题：位置信息没有加入
        #########   使用一个线性层，分别过ground truth和fake case
        d = (output1 - output2).pow(2).sum(1).sqrt() # distance
        # pdb.set_trace()
        loss = torch.mean(0.5 * targets.float() * d.pow(2) + \
                          0.5 * (1 - targets.float()) * F.relu(self.margin - d).pow(2))
        return loss# we could also return distance; it is needed to evaluate the model
def gaussi_like_distribute(x,means,sigma=2,T=256):
    gaussi = np.exp(-(((x-means)/T)**2)/(2*sigma**2))/(sigma*np.sqrt(2*np.pi))
    return gaussi