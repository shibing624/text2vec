# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description:
"""

import codecs
import os
from typing import Union, Optional, Any, List, Tuple

import numpy as np
import tensorflow as tf

import text2vec
from text2vec.embeddings.embedding import Embedding
from text2vec.processors.base_processor import BaseProcessor
from text2vec.utils import get_file
from text2vec.utils.logger import get_logger
from text2vec.utils.non_masking_layer import NonMaskingLayer

os.environ['TF_KERAS'] = '1'
import keras_bert

logger = get_logger(__name__)


class BERTEmbedding(Embedding):
    """Pre-trained BERT embedding"""
    model_key_map = {
        'bert-base-uncased': 'uncased_L-12_H-768_A-12',
        'bert-large-uncased': 'uncased_L-24_H-1024_A-16',
        'bert-base-cased': 'cased_L-12_H-768_A-12',
        'bert-large-cased': 'cased_L-24_H-1024_A-16',
        'bert-base-multilingual-cased': 'multi_cased_L-12_H-768_A-12',
        'bert-base-chinese': 'chinese_L-12_H-768_A-12'
    }

    pre_trained_models = {
        # BERT-Base, Uncased: 12-layer, 768-hidden, 12-heads, 110M parameters
        'uncased_L-12_H-768_A-12': 'https://storage.googleapis.com/bert_models/2018_10_18/'
                                   'uncased_L-12_H-768_A-12.zip',
        # BERT-Large, Uncased
        # 24-layer, 1024-hidden, 16-heads, 340M parameters
        'uncased_L-24_H-1024_A-16': 'https://storage.googleapis.com/bert_models/2018_10_18/'
                                    'uncased_L-24_H-1024_A-16.zip',
        # BERT-Base, Cased
        # 12-layer, 768-hidden, 12-heads , 110M parameters
        'cased_L-12_H-768_A-12': 'https://storage.googleapis.com/bert_models/2018_10_18/'
                                 'cased_L-12_H-768_A-12.zip',
        # BERT-Large, Cased
        # 24-layer, 1024-hidden, 16-heads, 340M parameters
        'cased_L-24_H-1024_A-16': 'https://storage.googleapis.com/bert_models/2018_10_18/'
                                  'cased_L-24_H-1024_A-16.zip',
        # BERT-Base, Multilingual Cased (New, recommended)
        # 104 languages, 12-layer, 768-hidden, 12-heads, 110M parameters
        'multi_cased_L-12_H-768_A-12': 'https://storage.googleapis.com/bert_models/2018_11_23/'
                                       'multi_cased_L-12_H-768_A-12.zip',
        # BERT-Base, Multilingual Uncased (Orig, not recommended)
        # 12-layer, 768-hidden, 12-heads, 110M parameters
        'multilingual_L-12_H-768_A-12': 'https://storage.googleapis.com/bert_models/2018_11_03/'
                                        'multilingual_L-12_H-768_A-12.zip',
        # BERT-Base, Chinese
        # Chinese Simplified and Traditional, 12-layer, 768-hidden, 12-heads, 110M
        'chinese_L-12_H-768_A-12': 'https://storage.googleapis.com/bert_models/2018_11_03/'
                                   'chinese_L-12_H-768_A-12.zip',

        # https://github.com/ymcui/Chinese-BERT-wwm 提供优化后的RoBERTa-wwm-ext-large中文模型，可供下载，
        # 但由于提升有限（LCQMC句对语义任务识别任务中，BERT的测试集准确率86.9，RoBERTa-wwm-ext-large的测试集准确率87.0），
        # 本项目暂不使用。
    }

    def info(self):
        info = super(BERTEmbedding, self).info()
        info['config'] = {
            'model_folder': self.model_folder,
            'sequence_length': self.sequence_length
        }
        return info

    def __init__(self,
                 model_folder: str = '',
                 layer_nums: int = 4,
                 trainable: bool = False,
                 sequence_length: Union[str, int] = 128,
                 processor: Optional[BaseProcessor] = None):
        """

        Args:
            model_folder:
            layer_nums: number of layers whose outputs will be concatenated into a single tensor,
                           default `4`, output the last 4 hidden layers as the thesis suggested
            trainable: whether if the model is trainable, default `False` and set it to `True`
                        for fine-tune this embedding layer during your training
            sequence_length:
            processor:
        """
        self.trainable = trainable
        # Do not need to train the whole bert model if just to use its feature output
        self.training = False
        self.layer_nums = layer_nums
        if isinstance(sequence_length, tuple):
            raise ValueError('BERT embedding only accept `int` type `sequence_length`')

        if sequence_length == 'variable':
            raise ValueError('BERT embedding only accept sequences in equal length')

        super(BERTEmbedding, self).__init__(sequence_length=sequence_length,
                                            embedding_size=100,
                                            processor=processor)

        self.processor.token_pad = '[PAD]'
        self.processor.token_unk = '[UNK]'
        self.processor.token_bos = '[CLS]'
        self.processor.token_eos = '[SEP]'

        self.processor.add_bos_eos = False  # bert_tokenizer added

        self.model_folder = model_folder
        self._build_token2idx_from_bert()
        self._build_model()

    def _build_token2idx_from_bert(self):
        dict_path = os.path.join(self.model_folder, 'vocab.txt')
        if not os.path.exists(dict_path):
            model_name = self.model_key_map.get(self.model_folder, 'chinese_L-12_H-768_A-12')
            url = self.pre_trained_models.get(model_name)
            get_file(
                model_name + ".zip", url, extract=True,
                cache_dir=text2vec.USER_DIR,
                cache_subdir=text2vec.USER_DATA_DIR,
                verbose=1
            )
            self.model_folder = os.path.join(text2vec.USER_DATA_DIR, model_name)
            dict_path = os.path.join(self.model_folder, 'vocab.txt')
        logger.debug(f'load vocab.txt from {dict_path}')
        token2idx = {}
        with codecs.open(dict_path, 'r', encoding='utf-8') as f:
            for line in f:
                token = line.strip()
                token2idx[token] = len(token2idx)

        self.bert_token2idx = token2idx
        self.tokenizer = keras_bert.Tokenizer(token2idx)
        self.processor.token2idx = self.bert_token2idx
        self.processor.idx2token = dict([(value, key) for key, value in token2idx.items()])

    def _build_model(self, **kwargs):
        if self.embed_model is None:
            seq_len = self.sequence_length
            if isinstance(seq_len, tuple):
                seq_len = seq_len[0]
            config_path = os.path.join(self.model_folder, 'bert_config.json')
            check_point_path = os.path.join(self.model_folder, 'bert_model.ckpt')
            logger.debug('load bert model from %s' % check_point_path)
            bert_model = keras_bert.load_trained_model_from_checkpoint(config_path,
                                                                       check_point_path,
                                                                       seq_len=seq_len,
                                                                       output_layer_num=self.layer_nums,
                                                                       training=self.training,
                                                                       trainable=self.trainable)

            self._model = tf.keras.Model(bert_model.inputs, bert_model.output)
            bert_seq_len = int(bert_model.output.shape[1])
            if bert_seq_len < seq_len:
                logger.warning(f"Sequence length limit set to {bert_seq_len} by pre-trained model")
                self.sequence_length = bert_seq_len
            self.embedding_size = int(bert_model.output.shape[-1])
            output_features = NonMaskingLayer()(bert_model.output)
            self.embed_model = tf.keras.Model(bert_model.inputs, output_features)
            logger.debug(f'seq_len: {self.sequence_length}')

    def analyze_corpus(self,
                       x: Union[Tuple[List[List[str]], ...], List[List[str]]],
                       y: Union[List[List[Any]], List[Any]]):
        """
        Prepare embedding layer and pre-processor for labeling task

        Args:
            x:
            y:

        Returns:

        """
        if len(self.processor.token2idx) == 0:
            self._build_token2idx_from_bert()
        super(BERTEmbedding, self).analyze_corpus(x, y)

    def embed(self,
              sentence_list: Union[Tuple[List[List[str]], ...], List[List[str]]],
              debug: bool = False) -> np.ndarray:
        """
        batch embed sentences

        Args:
            sentence_list: Sentence list to embed
            debug: show debug log
        Returns:
            vectorized sentence list

            print(token, predicts[i].tolist()[:4])
            [CLS] [0.24250675737857819, 0.04605229198932648, ...]
            from [0.2858668565750122, 0.12927496433258057,  ...]
            that [-0.7514970302581787, 0.14548861980438232, ...]
            day [0.32245880365371704, -0.043174318969249725, ...]
            ...
        """
        if self.embed_model is None:
            raise ValueError('need to build model for embed sentence')

        tensor_x = self.process_x_dataset(sentence_list)
        if debug:
            logger.debug(f'sentence tensor: {tensor_x}')
        embed_results = self.embed_model.predict(tensor_x)
        return embed_results

    def process_x_dataset(self,
                          data: Union[Tuple[List[List[str]], ...], List[List[str]]],
                          subset: Optional[List[int]] = None) -> Tuple[np.ndarray, ...]:
        """
        batch process feature data while training

        Args:
            data: target dataset
            subset: subset index list

        Returns:
            vectorized feature tensor
        """
        x1 = None
        if isinstance(data, tuple):
            if len(data) == 2:
                x0 = self.processor.process_x_dataset(data[0], self.sequence_length, subset)
                x1 = self.processor.process_x_dataset(data[1], self.sequence_length, subset)
            else:
                x0 = self.processor.process_x_dataset(data[0], self.sequence_length, subset)
        else:
            x0 = self.processor.process_x_dataset(data, self.sequence_length, subset)
        if x1 is None:
            x1 = np.zeros(x0.shape, dtype=np.int32)
        return x0, x1
