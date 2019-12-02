# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""
import json
import os
import pydoc
from typing import Union, List, Optional, Dict

import numpy as np

from text2vec.processors.base_processor import BaseProcessor
from text2vec.processors.default_processor import DefaultProcessor
from text2vec.utils.logger import get_logger

logger = get_logger(__name__)


class Embedding(object):
    """Base class for Embedding Model"""

    def info(self) -> Dict:
        return {
            'processor': self.processor.info(),
            'class_name': self.__class__.__name__,
            'module': self.__class__.__module__,
            'config': {
                'sequence_length': self.sequence_length,
                'embedding_size': self.embedding_size,
            },
            'embed_model': json.loads(self.embed_model.to_json()) if self.embed_model else None,
        }

    def __init__(self,
                 sequence_length: Union[int, str] = 128,
                 embedding_size: int = 100,
                 processor: Optional[BaseProcessor] = None):
        self.embedding_size = embedding_size

        if processor is None:
            self.processor = DefaultProcessor()
        else:
            self.processor = processor

        self.sequence_length = sequence_length
        self.embed_model = None
        self.tokenizer = None

    @property
    def token_count(self) -> int:
        """
        corpus token count
        """
        return len(self.processor.token2idx)

    @property
    def sequence_length(self) -> Union[int, str]:
        """
        model sequence length
        """
        return self.processor.sequence_length

    @property
    def label2idx(self) -> Dict[str, int]:
        """
        label to index dict
        """
        return self.processor.label2idx

    @property
    def token2idx(self) -> Dict[str, int]:
        """
        token to index dict
        """
        return self.processor.token2idx

    @sequence_length.setter
    def sequence_length(self, val: Union[int, str]):
        if isinstance(val, str):
            if val == 'auto':
                logger.debug("Sequence length will auto set at 95% of sequence length")
            elif val == 'variable':
                val = None
            else:
                raise ValueError("sequence_length must be an int or 'auto' or 'variable'")
        self.processor.sequence_length = val

    def _build_model(self, **kwargs):
        raise NotImplementedError

    def analyze_corpus(self,
                       x: List[List[str]],
                       y: Union[List[List[str]], List[str]]):
        """
        Prepare embedding layer and pre-processor for labeling task

        Args:
            x:
            y:

        Returns:

        """
        self.processor.analyze_corpus(x, y)
        self._build_model()

    def embed_one(self, sentence: Union[List[str], List[int]]) -> np.array:
        """
        Convert one sentence to vector

        Args:
            sentence: target sentence, list of str

        Returns:
            vectorized sentence
        """
        return self.embed([sentence])[0]

    def embed(self,
              sentence_list: Union[List[List[str]], List[List[int]]],
              debug: bool = False) -> np.ndarray:
        raise NotImplementedError

    def process_x_dataset(self,
                          data: List[List[str]],
                          subset: Optional[List[int]] = None) -> np.ndarray:
        """
        batch process feature data while training

        Args:
            data: target dataset
            subset: subset index list

        Returns:
            vectorized feature tensor
        """
        return self.processor.process_x_dataset(data, self.sequence_length, subset)

    def process_y_dataset(self,
                          data: List[List[str]],
                          subset: Optional[List[int]] = None) -> np.ndarray:
        """
        batch process labels data while training

        Args:
            data: target dataset
            subset: subset index list

        Returns:
            vectorized feature tensor
        """
        return self.processor.process_y_dataset(data, self.sequence_length, subset)

    def reverse_numerize_label_sequences(self,
                                         sequences,
                                         lengths=None):
        return self.processor.reverse_numerize_label_sequences(sequences, lengths=lengths)

    def __repr__(self):
        return f"<{self.__class__} seq_len: {self.sequence_length}>"

    def __str__(self):
        return self.__repr__()
