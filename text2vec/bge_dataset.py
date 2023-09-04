# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import math
import os
import random

import datasets
from loguru import logger
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


def load_bge_train_data(train_file):
    """
    Load train dataset from file.
        args: file path
        return: list of (text_a, text_b, score)
    """
    if os.path.isdir(train_file):
        train_datas = []
        for file in os.listdir(train_file):
            temp_data = datasets.load_dataset(
                'json',
                data_files=os.path.join(train_file, file),
                split='train'
            )
            train_datas.append(temp_data)
        dataset = datasets.concatenate_datasets(train_datas)
    elif os.path.isfile(train_file):
        dataset = datasets.load_dataset('json', data_files=train_file, split='train')
    else:
        try:
            dataset = datasets.load_dataset(train_file, split="train")
        except Exception as e:
            logger.error(f'load_dataset error: {e}')
            dataset = []
    return dataset


class BgeTrainDataset(Dataset):
    """Bge文本匹配训练数据集, 重写__getitem__和__len__方法"""

    def __init__(
            self,
            tokenizer: PreTrainedTokenizer,
            data_file_or_name: str,
            query_max_len: int = 32,
            passage_max_len: int = 128,
            train_group_size: int = 8
    ):
        self.tokenizer = tokenizer
        self.dataset = load_bge_train_data(data_file_or_name)
        self.query_max_len = query_max_len
        self.passage_max_len = passage_max_len
        self.train_group_size = train_group_size

    def __len__(self):
        return len(self.dataset)

    def text_2_id(self, text, max_len: int):
        return self.tokenizer(
            text,
            max_length=max_len,
            truncation=True,
            padding=True,
            return_tensors='pt'
        )

    def __getitem__(self, index: int):
        query = self.dataset[index]['query']
        passage = []
        pos = random.choice(self.dataset[index]['pos'])
        passage.append(pos)

        if len(self.dataset[index]['neg']) < self.train_group_size - 1:
            num = math.ceil((self.train_group_size - 1) / len(self.dataset[index]['neg']))
            negs = random.sample(self.dataset[index]['neg'] * num, self.train_group_size - 1)
        else:
            negs = random.sample(self.dataset[index]['neg'], self.train_group_size - 1)
        passage.extend(negs)
        return query, passage


