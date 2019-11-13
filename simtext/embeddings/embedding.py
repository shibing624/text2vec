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

from simtext.processors.base_processor import BaseProcessor
from simtext.processors.default_processor import DefaultProcessor
from simtext.utils.logger import get_logger

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
            'embed_model': json.loads(self.embed_model.to_json()),
        }

    @classmethod
    def _load_saved_instance(cls,
                             config_dict: Dict,
                             model_path: str,
                             tf_model):
        os.environ['TF_KERAS'] = '1'
        from tensorflow import keras
        import keras_bert
        L = keras.layers
        custom_objects = keras_bert.get_custom_objects()
        processor_info = config_dict['processor']
        processor_class = pydoc.locate(f"{processor_info['module']}.{processor_info['class_name']}")
        processor = processor_class(**processor_info['config'])

        instance = cls(processor=processor,
                       from_saved_model=True, **config_dict['config'])

        embed_model_json_str = json.dumps(config_dict['embed_model'])
        instance.embed_model = keras.models.model_from_json(embed_model_json_str,
                                                            custom_objects=custom_objects)

        # Load Weights from model
        for layer in instance.embed_model.layers:
            layer.set_weights(tf_model.get_layer(layer.name).get_weights())
        return instance

    def __init__(self,
                 sequence_length: Union[int, str] = 'auto',
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
        if self.sequence_length == 'auto':
            self.sequence_length = self.processor.dataset_info['RECOMMEND_LEN']
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
        """
        batch embed sentences

        Args:
            sentence_list: Sentence list to embed
            debug: show debug info
        Returns:
            vectorized sentence list
        """
        tensor_x = self.process_x_dataset(sentence_list)

        if debug:
            logger.debug(f'sentence tensor: {tensor_x}')
        embed_results = self.embed_model.predict(tensor_x)
        return embed_results

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


if __name__ == "__main__":
    print("Hello world")
