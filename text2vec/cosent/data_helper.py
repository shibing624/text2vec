# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com), xiaolu(luxiaonlp@163.com)
@description:
"""
import torch
from torch.utils.data import Dataset
from loguru import logger


def load_data(path):
    sents, labels = [], []
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip().split('\t')
            if len(line) != 3:
                logger.warning(f'line size not match, pass: {line}')
                continue
            sents.extend([line[0], line[1]])
            score = int(line[2])
            # if 'STS' in path:
            #     score = float(score) / 5.0  # Normalize score to range 0 ... 1
            labels.extend([score, score])
    return sents, labels


def load_test_data(path):
    sents1, sents2, labels = [], [], []
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip().split('\t')
            if len(line) != 3:
                logger.warning(f'line size not match, pass: {line}')
                continue
            sents1.append(line[0])
            sents2.append(line[1])
            score = int(line[2])
            # if 'STS' in path:
            #     score = float(score) / 5.0  # Normalize score to range 0 ... 1
            labels.append(score)
    return sents1, sents2, labels


class CustomDataset(Dataset):
    def __init__(self, sentences, labels, tokenizer):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        inputs = self.tokenizer.encode_plus(
            text=self.sentences[index],
            text_pair=None,
            add_special_tokens=True,
            return_token_type_ids=True
        )
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'labels': self.labels[index]
        }


def pad_to_maxlen(input_ids, max_len=64, pad_value=0):
    """
    Padding text
    """
    if len(input_ids) >= max_len:
        input_ids = input_ids[:max_len]
    else:
        input_ids = input_ids + [pad_value] * (max_len - len(input_ids))
    return input_ids


def collate_fn(batch):
    # 按batch进行padding获取当前batch中最大长度
    max_len = max([len(d['input_ids']) for d in batch])
    input_ids, attention_mask, token_type_ids, labels = [], [], [], []

    for item in batch:
        input_ids.append(pad_to_maxlen(item['input_ids'], max_len=max_len))
        attention_mask.append(pad_to_maxlen(item['attention_mask'], max_len=max_len))
        token_type_ids.append(pad_to_maxlen(item['token_type_ids'], max_len=max_len))
        labels.append(item['labels'])

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    attention_mask = torch.tensor(attention_mask, dtype=torch.long)
    token_type_ids = torch.tensor(token_type_ids, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.float)
    return input_ids, attention_mask, token_type_ids, labels
