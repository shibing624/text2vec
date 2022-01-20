# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""

import torch
from torch import nn
from transformers import BertModel


class Model(nn.Module):
    def __init__(self, model_name='bert-base-chinese', encoder_type='first-last-avg'):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.encoder_type = encoder_type

    def forward(self, input_ids, attention_mask, token_type_ids):
        """
        :param input_ids:
        :param attention_mask:
        :param encoder_type: encoder_type:  "first-last-avg", "last-avg", "cls", "pooler(cls + dense)"
        :return:
        """
        output = self.bert(input_ids, attention_mask, token_type_ids, output_hidden_states=True)

        if self.encoder_type == 'first-last-avg':
            # 第一层和最后一层的隐层取出  然后经过平均池化
            # hidden_states列表有13个hidden_state，第一个其实是embeddings，第二个元素才是第一层的hidden_state
            first = output.hidden_states[1]
            last = output.hidden_states[-1]
            seq_length = first.size(1)  # 序列长度

            first_avg = torch.avg_pool1d(first.transpose(1, 2), kernel_size=seq_length).squeeze(-1)  # batch, hid_size
            last_avg = torch.avg_pool1d(last.transpose(1, 2), kernel_size=seq_length).squeeze(-1)  # batch, hid_size
            final_encoding = torch.avg_pool1d(
                torch.cat([first_avg.unsqueeze(1), last_avg.unsqueeze(1)], dim=1).transpose(1, 2),
                kernel_size=2).squeeze(-1)
            return final_encoding

        if self.encoder_type == 'last-avg':
            sequence_output = output.last_hidden_state  # (batch_size, max_len, hidden_size)
            seq_length = sequence_output.size(1)
            final_encoding = torch.avg_pool1d(sequence_output.transpose(1, 2), kernel_size=seq_length).squeeze(-1)
            return final_encoding

        if self.encoder_type == "cls":
            sequence_output = output.last_hidden_state
            cls = sequence_output[:, 0]  # [batch_size, 768]
            return cls

        if self.encoder_type == "pooler":
            pooler_output = output.pooler_output  # [batch_size, 768]
            return pooler_output
