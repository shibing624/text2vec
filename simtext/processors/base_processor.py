# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: base_processor.py
# time: 2019-05-21 11:27

import collections
import operator
from typing import List, Optional, Union, Dict, Any

import numpy as np


def get_list_subset(target: List, index_list: List[int]) -> List:
    return [target[i] for i in index_list if i < len(target)]


class BaseProcessor(object):
    """
    Corpus Pre Processor class
    """

    def __init__(self, **kwargs):
        self.token2idx: Dict[str, int] = kwargs.get('token2idx', {})
        self.idx2token: Dict[int, str] = dict([(v, k) for (k, v) in self.token2idx.items()])

        self.token2count: Dict = {}

        self.label2idx: Dict[str, int] = kwargs.get('label2idx', {})
        self.idx2label: Dict[int, str] = dict([(v, k) for (k, v) in self.label2idx.items()])

        self.token_pad: str = kwargs.get('token_pad', '<PAD>')
        self.token_unk: str = kwargs.get('token_unk', '<UNK>')
        self.token_bos: str = kwargs.get('token_bos', '<BOS>')
        self.token_eos: str = kwargs.get('token_eos', '<EOS>')

        self.dataset_info: Dict[str, Any] = kwargs.get('dataset_info', {})

        self.add_bos_eos: bool = kwargs.get('add_bos_eos', False)

        self.sequence_length = kwargs.get('sequence_length', None)

    def info(self):
        return {
            'class_name': self.__class__.__name__,
            'config': {
                'label2idx': self.label2idx,
                'token2idx': self.token2idx,
                'token_pad': self.token_pad,
                'token_unk': self.token_unk,
                'token_bos': self.token_bos,
                'token_eos': self.token_eos,
                'dataset_info': self.dataset_info,
                'add_bos_eos': self.add_bos_eos,
                'sequence_length': self.sequence_length
            },
            'module': self.__class__.__module__,
        }

    def analyze_corpus(self,
                       corpus: Union[List[List[str]]],
                       labels: Union[List[List[str]], List[str]],
                       min_count: int = 3,
                       force: bool = False):
        rec_len = sorted([len(seq) for seq in corpus])[int(0.95 * len(corpus))]
        self.dataset_info['RECOMMEND_LEN'] = rec_len

        if len(self.token2idx) == 0 or force:
            self._build_token_dict(corpus, min_count)
        if len(self.label2idx) == 0 or force:
            self._build_label_dict(labels)

    def _build_token_dict(self, corpus: List[List[str]], min_count: int = 3):
        """
        Build token index dictionary using corpus

        Args:
            corpus: List of tokenized sentences, like ``[['I', 'love', 'tf'], ...]``
            min_count:
        """
        token2idx = {
            self.token_pad: 0,
            self.token_unk: 1,
            self.token_bos: 2,
            self.token_eos: 3
        }

        token2count = {}
        for sentence in corpus:
            for token in sentence:
                count = token2count.get(token, 0)
                token2count[token] = count + 1

        # 按照词频降序排序
        sorted_token2count = sorted(token2count.items(),
                                    key=operator.itemgetter(1),
                                    reverse=True)
        token2count = collections.OrderedDict(sorted_token2count)

        for token, token_count in token2count.items():
            if token not in token2idx and token_count > min_count:
                token2idx[token] = len(token2idx)

        self.token2idx = token2idx
        self.idx2token = dict([(value, key)
                               for key, value in self.token2idx.items()])
        self.dataset_info['token_count'] = len(self.token2idx)

    def _build_label_dict(self, corpus: Union[List[List[str]], List[str]]):
        raise NotImplementedError

    def process_x_dataset(self,
                          data: List[List[str]],
                          max_len: Optional[int] = None,
                          subset: Optional[List[int]] = None) -> np.ndarray:
        from tensorflow.python.keras.preprocessing.sequence import pad_sequences
        if max_len is None:
            max_len = self.sequence_length
        if subset is not None:
            target = get_list_subset(data, subset)
        else:
            target = data
        numerized_samples = self.numerize_token_sequences(target)

        return pad_sequences(numerized_samples, max_len, padding='post', truncating='post')

    def process_y_dataset(self,
                          data: Union[List[List[str]], List[str]],
                          max_len: Optional[int],
                          subset: Optional[List[int]] = None) -> np.ndarray:
        raise NotImplementedError

    def numerize_token_sequences(self,
                                 sequences: List[List[str]]):
        raise NotImplementedError

    def numerize_label_sequences(self,
                                 sequences: List[List[str]]) -> List[List[int]]:
        raise NotImplementedError

    def reverse_numerize_label_sequences(self, sequence, **kwargs):
        raise NotImplementedError

    def __repr__(self):
        return f"<{self.__class__}>"

    def __str__(self):
        return self.__repr__()
