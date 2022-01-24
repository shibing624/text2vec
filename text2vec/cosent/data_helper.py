# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""
from torch.utils.data import Dataset
from loguru import logger


def load_train_data(path):
    data = []
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip().split('\t')
            if len(line) != 3:
                logger.warning(f'line size not match, pass: {line}')
                continue
            data.append((line[0], int(line[2])))
            data.append((line[1], int(line[2])))
    return data


def load_test_data(path):
    data = []
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip().split('\t')
            if len(line) != 3:
                logger.warning(f'line size not match, pass: {line}')
                continue
            data.append((line[0], line[1], int(line[2])))
    return data


class TrainDataset(Dataset):
    """训练数据集, 重写__getitem__和__len__方法"""

    def __init__(self, data, tokenizer, max_len=64):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def text_2_id(self, text: str):
        return self.tokenizer(text, max_length=self.max_len, truncation=True, padding='max_length', return_tensors='pt')

    def __getitem__(self, index: int):
        line = self.data[index]
        return self.text_2_id([line[0]]), line[1]


class TestDataset(Dataset):
    """测试数据集, 重写__getitem__和__len__方法"""

    def __init__(self, data, tokenizer, max_len=64):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def text_2_id(self, text: str):
        return self.tokenizer(text, max_length=self.max_len, truncation=True, padding='max_length', return_tensors='pt')

    def __getitem__(self, index: int):
        line = self.data[index]
        return self.text_2_id([line[0]]), self.text_2_id([line[1]]), line[2]
