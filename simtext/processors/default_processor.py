# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description:
"""

from typing import List, Optional

import numpy as np

from .base_processor import BaseProcessor, get_list_subset


class DefaultProcessor(BaseProcessor):
    """
    Corpus Pre Processor class
    """

    def __init__(self, multi_label=False, **kwargs):
        super(DefaultProcessor, self).__init__(**kwargs)
        self.multi_label = multi_label
        self.multi_label_binarizer = None

    def info(self):
        info = super(DefaultProcessor, self).info()
        info['multi_label'] = self.multi_label
        return info

    def _build_label_dict(self,
                          labels: List[str]):
        from sklearn.preprocessing import MultiLabelBinarizer
        if self.multi_label:
            label_set = set()
            for i in labels:
                label_set = label_set.union(list(i))
        else:
            label_set = set(labels)
        self.label2idx = {}
        for idx, label in enumerate(sorted(label_set)):
            self.label2idx[label] = len(self.label2idx)

        self.idx2label = dict([(value, key) for key, value in self.label2idx.items()])
        self.dataset_info['label_count'] = len(self.label2idx)
        self.multi_label_binarizer = MultiLabelBinarizer(classes=list(self.label2idx.keys()))

    def process_y_dataset(self,
                          data: List[str],
                          max_len: Optional[int] = None,
                          subset: Optional[List[int]] = None) -> np.ndarray:
        from tensorflow.python.keras.utils import to_categorical
        if subset is not None:
            target = get_list_subset(data, subset)
        else:
            target = data
        if self.multi_label:
            return self.multi_label_binarizer.fit_transform(target)
        else:
            numerized_samples = self.numerize_label_sequences(target)
            return to_categorical(numerized_samples, len(self.label2idx))

    def numerize_token_sequences(self,
                                 sequences: List[List[str]]):
        result = []
        for seq in sequences:
            if self.add_bos_eos:
                seq = [self.token_bos] + seq + [self.token_eos]
            unk_index = self.token2idx[self.token_unk]
            result.append([self.token2idx.get(token, unk_index) for token in seq])
        return result

    def numerize_label_sequences(self,
                                 sequences: List[str]) -> List[int]:
        """
        Convert label sequence to label-index sequence
        ``['O', 'O', 'B-ORG'] -> [0, 0, 2]``

        Args:
            sequences: label sequence, list of str

        Returns:
            label-index sequence, list of int
        """
        return [self.label2idx[label] for label in sequences]

    def reverse_numerize_label_sequences(self, sequences, **kwargs):
        if self.multi_label:
            return self.multi_label_binarizer.inverse_transform(sequences)
        else:
            return [self.idx2label[label] for label in sequences]
