# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import os
from torch.utils.data import Dataset
from loguru import logger
from transformers import PreTrainedTokenizer
from datasets import load_dataset


def load_cosent_train_data(path):
    data = []
    if not os.path.isfile(path):
        return data
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip().split('\t')
            if len(line) != 3:
                logger.warning(f'line size not match, pass: {line}')
                continue
            data.append((line[0], int(line[2])))
            data.append((line[1], int(line[2])))
    return data


class CosentTrainDataset(Dataset):
    """Cosent文本匹配训练数据集, 重写__getitem__和__len__方法"""

    def __init__(self, tokenizer: PreTrainedTokenizer, data: list, max_len: int = 64):
        self.tokenizer = tokenizer
        self.data = data
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def text_2_id(self, text: str):
        return self.tokenizer(text, max_length=self.max_len, truncation=True,
                              padding='max_length', return_tensors='pt')

    def __getitem__(self, index: int):
        line = self.data[index]
        return self.text_2_id(line[0]), line[1]


class HFCosentTrainDataset(Dataset):
    """Load HuggingFace datasets to Cosent format

    Args:
        tokenizer (PreTrainedTokenizer): tokenizer
        name (str): dataset name
        max_len (int): max length of sentence
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, name="STS-B", max_len: int = 64):
        self.tokenizer = tokenizer
        dataset = load_dataset("shibing624/nli_zh", name.upper(), split="train")
        self.data = self.convert_to_rank(dataset)
        self.max_len = max_len

    def convert_to_rank(self, dataset):
        """
        Flatten the dataset to a list of tuples
        """
        data = []
        for line in dataset:
            data.append((line['sentence1'], line['label']))
            data.append((line['sentence2'], line['label']))
        return data

    def __len__(self):
        return len(self.data)

    def text_2_id(self, text: str):
        return self.tokenizer(text, max_length=self.max_len, truncation=True,
                              padding='max_length', return_tensors='pt')

    def __getitem__(self, index: int):
        line = self.data[index]
        return self.text_2_id(line[0]), line[1]
