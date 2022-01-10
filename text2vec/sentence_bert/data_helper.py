"""
@file   : data_helper.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-08-02
"""
import torch
import gzip
import pickle
import pandas as pd
import random
from tqdm import tqdm
from transformers.models.bert import BertTokenizer


class Features:
    def __init__(self, s1_input_ids=None, s2_input_ids=None, label=None):
        self.s1_input_ids = s1_input_ids
        self.s2_input_ids = s2_input_ids
        self.label = label

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "s1_input_ids: %s" % (self.s1_input_ids)
        s += ", s2_input_ids: %s" % (self.s2_input_ids)
        s += ", label: %d" % (self.label)
        return s


def convert_token_to_id(path):
    '''
    将句子转为id序列
    :return:
    '''
    with open(path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        features = []
        for line in tqdm(lines):
            s, t, lab = line.strip().split('\t')
            s_input = tokenizer.encode(s)
            t_input = tokenizer.encode(t)
            
            if len(s_input) > max_len:
                s_input = s_input[:max_len]
            else:
                s_input = s_input + (max_len - len(s_input)) * [0]
            

            if len(t_input) > max_len:
                t_input = t_input[:max_len]
            else:
                t_input = t_input + (max_len - len(t_input)) * [0]
            lab = int(lab)
            feature = Features(s1_input_ids=s_input, s2_input_ids=t_input, label=lab)
            features.append(feature)
        return features


if __name__ == '__main__':
    train_data_path = '../data/ATEC/ATEC.train.data'
    test_data_path = '../data/ATEC/ATEC.test.data'

    tokenizer = BertTokenizer.from_pretrained('../mengzi_pretrain')

    max_len = 64
    train_features = convert_token_to_id(train_data_path)
    test_features = convert_token_to_id(test_data_path)
    with gzip.open('./data/train_features.pkl.gz', 'wb') as fout:
        pickle.dump(train_features, fout)
    with gzip.open('./data/test_features.pkl.gz', 'wb') as fout:
        pickle.dump(test_features, fout)



