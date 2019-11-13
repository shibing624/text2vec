# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""

from typing import Union, Optional, Dict, Any, List, Tuple

import numpy as np
from gensim.models import KeyedVectors

from simtext.embeddings.embedding import Embedding
from simtext.processors.base_processor import BaseProcessor
from simtext.utils.logger import get_logger
from simtext.utils.tokenizer import Tokenizer

logger = get_logger(__name__)


class WordEmbedding(Embedding):
    """Pre-trained word2vec embedding"""

    def info(self):
        info = super(WordEmbedding, self).info()
        info['config'] = {
            'w2v_path': self.w2v_path,
            'w2v_kwargs': self.w2v_kwargs,
            'sequence_length': self.sequence_length
        }
        return info

    def __init__(self,
                 w2v_path: str,
                 w2v_kwargs: Dict[str, Any] = None,
                 sequence_length: Union[Tuple[int, ...], str, int] = 'auto',
                 processor: Optional[BaseProcessor] = None):
        """

        Args:
            w2v_path: word2vec file path
            w2v_kwargs: params pass to the ``load_word2vec_format()`` function of ``gensim.models.KeyedVectors`` -
                https://radimrehurek.com/gensim/models/keyedvectors.html#module-gensim.models.keyedvectors
            sequence_length: ``'auto'``, ``'variable'`` or integer. When using ``'auto'``, use the 95% of corpus length
                as sequence length. When using ``'variable'``, model input shape will set to None, which can handle
                various length of input, it will use the length of max sequence in every batch for sequence length.
                If using an integer, let's say ``50``, the input output sequence length will set to 50.
            processor:
        """
        if w2v_kwargs is None:
            w2v_kwargs = {}
        self.w2v_path = w2v_path
        self.w2v_kwargs = w2v_kwargs
        self.w2v = None
        self.w2v_model_loaded = False
        logger.debug('load w2v embedding')
        super(WordEmbedding, self).__init__(sequence_length=sequence_length,
                                            embedding_size=0,
                                            processor=processor)
        self._build_token2idx_from_w2v()

    def _build_token2idx_from_w2v(self):
        logger.debug('load w2v from %s' % self.w2v_path)
        w2v = KeyedVectors.load_word2vec_format(self.w2v_path, **self.w2v_kwargs)

        token2idx = {
            self.processor.token_pad: 0,
            self.processor.token_unk: 1,
            self.processor.token_bos: 2,
            self.processor.token_eos: 3
        }

        for token in w2v.index2word:
            token2idx[token] = len(token2idx)

        vector_matrix = np.zeros((len(token2idx), w2v.vector_size))
        vector_matrix[1] = np.random.rand(w2v.vector_size)
        vector_matrix[4:] = w2v.vectors

        self.embedding_size = w2v.vector_size
        self.w2v_vector_matrix = vector_matrix
        self.w2v_token2idx = token2idx
        self.w2v_top_words = w2v.index2entity[:50]
        self.w2v_model_loaded = True
        self.w2v = w2v

        self.processor.token2idx = self.w2v_token2idx
        self.processor.idx2token = dict([(value, key) for key, value in self.w2v_token2idx.items()])
        logger.debug('word count   : {}'.format(len(self.w2v_vector_matrix)))
        logger.debug('emb size     : {}'.format(self.embedding_size))
        logger.debug('Top 50 word  : {}'.format(self.w2v_top_words))
        self.tokenizer = Tokenizer()

    def _build_model(self, **kwargs):
        logger.debug('no need build model for w2v')

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
        """
        if self.w2v is None:
            raise ValueError('need to build model for embed sentence')

        embeds = []
        for sentence in sentence_list:
            emb = []
            for word in sentence:
                if word in self.w2v.vocab:
                    e = self.w2v[word]
                else:
                    e = 0
                emb.append(e)
            tensor_x = np.array(emb).sum(axis=0)
            embeds.append(tensor_x)
        embeds = np.array(embeds)
        if debug:
            logger.debug(f'sentence tensor shape: {embeds.shape}')
        return embeds
