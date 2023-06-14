# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import os

from datasets import load_dataset
from loguru import logger
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from text2vec.utils.io_util import load_jsonl


def load_text_matching_train_data(path):
    """
    Load train data from file.
        args: file path
        return: list of (text_a, text_b, score)
    """
    data = []
    if not os.path.isfile(path):
        logger.warning(f'file not exist: {path}')
        return data

    def get_field_names(data_item):
        if "text1" in data_item and "text2" in data_item:
            return "text1", "text2"
        elif "sentence1" in data_item and "sentence2" in data_item:
            return "sentence1", "sentence2"
        else:
            return None, None

    if path.endswith('.jsonl'):
        data_list = load_jsonl(path)
        for entry in data_list:
            field1, field2 = get_field_names(entry)
            if not field1 or not field2:
                continue

            text_a, text_b, score = entry[field1], entry[field2], int(entry["label"])
            if 'STS' in path.upper():
                score = int(score > 2.5)
            data.append((text_a, text_b, score))
    else:
        with open(path, 'r', encoding='utf8') as f:
            for line in f:
                line = line.strip().split('\t')
                if len(line) != 3:
                    logger.warning(f'line size not match, pass: {line}')
                    continue
                score = int(line[2])
                if 'STS' in path.upper():
                    score = int(score > 2.5)
                data.append((line[0], line[1], score))
    return data


def load_text_matching_test_data(path):
    """
    Load test data from file.
        args: file path
        return: list of (text_a, text_b, score)
    """
    data = []
    if not os.path.isfile(path):
        logger.warning(f'file not exist: {path}')
        return data

    def get_field_names(data_item):
        if "text1" in data_item and "text2" in data_item:
            return "text1", "text2"
        elif "sentence1" in data_item and "sentence2" in data_item:
            return "sentence1", "sentence2"
        else:
            return None, None

    if path.endswith('.jsonl'):
        data_list = load_jsonl(path)
        for entry in data_list:
            field1, field2 = get_field_names(entry)
            if not field1 or not field2:
                continue

            text_a, text_b, score = entry[field1], entry[field2], int(entry["label"])
            data.append((text_a, text_b, score))
    else:
        with open(path, 'r', encoding='utf8') as f:
            for line in f:
                line = line.strip().split('\t')
                if len(line) != 3:
                    logger.warning(f'line size not match, pass: {line}')
                    continue
                score = int(line[2])
                data.append((line[0], line[1], score))
    return data


class TextMatchingTrainDataset(Dataset):
    """文本匹配训练数据集, 重写__getitem__和__len__方法"""

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
        return self.text_2_id(line[0]), self.text_2_id(line[1]), line[2]


class TextMatchingTestDataset(Dataset):
    """文本匹配测试数据集, 重写__getitem__和__len__方法"""

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
        return self.text_2_id(line[0]), self.text_2_id(line[1]), line[2]


class HFTextMatchingTrainDataset(Dataset):
    """Load HuggingFace datasets to SBERT format

    Args:
        tokenizer (PreTrainedTokenizer): tokenizer
        name (str): dataset name
        max_len (int): max length of sentence
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, name="STS-B", max_len: int = 64):
        self.tokenizer = tokenizer
        self.data = load_dataset("shibing624/nli_zh", name.upper(), split="train")
        self.max_len = max_len
        self.name = name.upper()

    def __len__(self):
        return len(self.data)

    def text_2_id(self, text: str):
        return self.tokenizer(text, max_length=self.max_len, truncation=True,
                              padding='max_length', return_tensors='pt')

    def __getitem__(self, index: int):
        line = self.data[index]
        # STS-B convert to 0/1 label
        return self.text_2_id(line['sentence1']), self.text_2_id(line['sentence2']), int(
            line['label'] > 2.5) if 'STS' in self.name else line['label']


class HFTextMatchingTestDataset(Dataset):
    """Load HuggingFace datasets to SBERT format

    Args:
        tokenizer (PreTrainedTokenizer): tokenizer
        name (str): dataset name
        max_len (int): max length of sentence
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, name="STS-B", max_len: int = 64, split="validation"):
        self.tokenizer = tokenizer
        self.data = load_dataset("shibing624/nli_zh", name.upper(), split=split)
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def text_2_id(self, text: str):
        return self.tokenizer(text, max_length=self.max_len, truncation=True,
                              padding='max_length', return_tensors='pt')

    def __getitem__(self, index: int):
        line = self.data[index]
        return self.text_2_id(line['sentence1']), self.text_2_id(line['sentence2']), line['label']
