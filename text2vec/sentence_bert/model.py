# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""

import torch
from torch import nn
from transformers import BertModel


class Model(nn.Module):
    def __init__(self, model_name='bert-base-chinese', encoder_type='first-last-avg', num_classes=2):
        """
        Init model
        :param model_name:
        :param encoder_type: encoder_type:  "first-last-avg", "last-avg", "cls", "pooler(cls + dense)"
        """
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.encoder_type = encoder_type
        self.fc = nn.Linear(self.bert.config.hidden_size * 3, num_classes)

    def get_output_layer(self, output):
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

        elif self.encoder_type == 'last-avg':
            sequence_output = output.last_hidden_state  # (batch_size, max_len, hidden_size)
            seq_length = sequence_output.size(1)
            final_encoding = torch.avg_pool1d(sequence_output.transpose(1, 2), kernel_size=seq_length).squeeze(-1)
            return final_encoding

        elif self.encoder_type == "cls":
            sequence_output = output.last_hidden_state
            return sequence_output[:, 0]  # [batch_size, 768]

        elif self.encoder_type == "pooler":
            return output.pooler_output  # [batch_size, 768]
        else:
            return output.pooler_output

    def forward(self, source_input_ids, source_attention_mask, source_token_type_ids,
                target_input_ids, target_attention_mask, target_token_type_ids, is_train=True):
        """
        Output the bert sentence embeddings, pass to classifier module. Applies different
        concats and finally the linear layer to produce class scores
        :param source_input_ids:
        :param source_attention_mask:
        :param source_token_type_ids:
        :param target_input_ids:
        :param target_attention_mask:
        :param target_token_type_ids:
        :return: embeddings
        """
        source_output = self.bert(source_input_ids, source_attention_mask, source_token_type_ids,
                                  output_hidden_states=True)
        target_output = self.bert(target_input_ids, target_attention_mask, target_token_type_ids,
                                  output_hidden_states=True)
        source_emb = self.get_output_layer(source_output)
        target_emb = self.get_output_layer(target_output)
        if is_train:
            # (u, v, |u - v|)
            embs = [source_emb, target_emb, torch.abs(source_emb - target_emb)]
            input_embs = torch.cat(embs, 1)
            # softmax
            outputs = self.fc(input_embs)
        else:
            outputs = source_emb, target_emb
        return outputs
