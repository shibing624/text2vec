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
    def __init__(self):
        super(SentenceBert, self).__init__()

        self.config = BertConfig.from_pretrained('../mengzi_pretrain/config.json')
        self.model = BertModel.from_pretrained('../mengzi_pretrain/pytorch_model.bin', config=self.config)

        # self.model = AutoModel.from_pretrained('./pretrain_weight')

    def get_embedding_vec(self, output, mask):
        token_embedding = output[0]
        input_mask_expanded = mask.unsqueeze(-1).expand(token_embedding.size()).float()
        sum_embedding = torch.sum(token_embedding * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embedding / sum_mask

    def forward(self, s1_input_ids, s2_input_ids):
        s1_mask = torch.ne(s1_input_ids, 0)
        s2_mask = torch.ne(s2_input_ids, 0)

        s1_output = self.model(input_ids=s1_input_ids, attention_mask=s1_mask)
        s2_output = self.model(input_ids=s2_input_ids, attention_mask=s2_mask)

        # s1_vec = self.get_embedding_vec(s1_output, s1_mask)
        # s2_vec = self.get_embedding_vec(s2_output, s2_mask)
        s1_vec = s1_output[1]
        s2_vec = s2_output[1]
        return s1_vec, s2_vec

