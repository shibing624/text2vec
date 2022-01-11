"""
@file   : model.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-08-02
"""

import torch
from torch import nn
from transformers import BertConfig, BertModel


class SentenceBert(nn.Module):
    def __init__(self, model_name='bert-base-chinese'):
        super(SentenceBert, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)

    def get_embedding_vec(self, output, mask):
        token_embedding = output[0]
        input_mask_expanded = mask.unsqueeze(-1).expand(token_embedding.size()).float()
        sum_embedding = torch.sum(token_embedding * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embedding / sum_mask

    def forward(self, s1_input_ids, s2_input_ids, encoder_type='pooler'):
        """
        :param encoder_type: encoder_type:  "first-last-avg", "last-avg", "cls", "pooler(cls + dense)"
        """
        s1_mask = torch.ne(s1_input_ids, 0)
        s2_mask = torch.ne(s2_input_ids, 0)

        s1_output = self.bert(input_ids=s1_input_ids, attention_mask=s1_mask, output_hidden_states=True)
        s2_output = self.bert(input_ids=s2_input_ids, attention_mask=s2_mask, output_hidden_states=True)

        s1_vec = self.get_output_layer(s1_output, encoder_type)
        s2_vec = self.get_output_layer(s2_output, encoder_type)
        return s1_vec, s2_vec

    def get_output_layer(self, output, encoder_type):
        """
        :param output:
        :param encoder_type: encoder_type:  "first-last-avg", "last-avg", "cls", "pooler(cls + dense)"
        :return:
        """
        if encoder_type == 'fist-last-avg':
            # 第一层和最后一层的隐层取出  然后经过平均池化
            first = output.hidden_states[1]   # hidden_states列表有13个hidden_state，第一个其实是embeddings，第二个元素才是第一层的hidden_state
            last = output.hidden_states[-1]
            seq_length = first.size(1)   # 序列长度

            first_avg = torch.avg_pool1d(first.transpose(1, 2), kernel_size=seq_length).squeeze(-1)  # batch, hid_size
            last_avg = torch.avg_pool1d(last.transpose(1, 2), kernel_size=seq_length).squeeze(-1)  # batch, hid_size
            final_encoding = torch.avg_pool1d(torch.cat([first_avg.unsqueeze(1), last_avg.unsqueeze(1)], dim=1).transpose(1, 2), kernel_size=2).squeeze(-1)
            return final_encoding

        if encoder_type == 'last-avg':
            sequence_output = output.last_hidden_state  # (batch_size, max_len, hidden_size)
            seq_length = sequence_output.size(1)
            final_encoding = torch.avg_pool1d(sequence_output.transpose(1, 2), kernel_size=seq_length).squeeze(-1)
            return final_encoding

        if encoder_type == "cls":
            sequence_output = output.last_hidden_state
            cls = sequence_output[:, 0]  # [b,d]
            return cls

        if encoder_type == "pooler":
            pooler_output = output.pooler_output  # [b,d]
            return pooler_output

