# -*- coding: utf-8 -*-
"""
@author:XuMing<xuming624@qq.com>
@description: 
"""

import numpy as np

from text2vec.algorithm.distance import cosine_distance
from text2vec.algorithm.rank_bm25 import BM25Okapi
from text2vec.utils.logger import get_logger
from text2vec.utils.tokenizer import Tokenizer
from text2vec.vector import Vector, EmbType

logger = get_logger(__name__)


class SimType(object):
    COSINE = 'cosine'
    WMD = 'wmd'


class Similarity(Vector):
    def __init__(self, similarity_type=SimType.COSINE,
                 embedding_type=EmbType.W2V,
                 sequence_length=128,
                 w2v_path='',
                 w2v_kwargs=None,
                 bert_model_folder='',
                 bert_layer_nums=4):
        """
        Cal text similarity
        :param similarity_type:
        :param embedding_type:
        :param sequence_length:
        :param w2v_path:
        :param w2v_kwargs:
        :param bert_model_folder:
        :param bert_layer_nums:
        """
        super(Similarity, self).__init__(embedding_type=embedding_type,
                                         w2v_path=w2v_path,
                                         w2v_kwargs=w2v_kwargs,
                                         sequence_length=sequence_length,
                                         bert_model_folder=bert_model_folder,
                                         bert_layer_nums=bert_layer_nums)
        self.similarity_type = similarity_type
        self.sequence_length = sequence_length

    def get_score(self, text1, text2):
        """
        Get score between text1 and text2
        :param text1: str
        :param text2: str
        :return: float, score
        """
        ret = 0.0
        if not text1.strip() or not text2.strip():
            return ret
        token_1 = self.tokenize(text1)
        token_2 = self.tokenize(text2)
        if self.similarity_type == SimType.COSINE:
            emb_1 = self.encode(token_1)
            emb_2 = self.encode(token_2)
            ret = cosine_distance(emb_1, emb_2)
        elif self.similarity_type == SimType.WMD:
            ret = 1. / (1. + self.model.w2v.wmdistance(token_1, token_2))
        return ret


class SearchSimilarity(object):
    def __init__(self, corpus):
        """
        Search sim doc with rank bm25
        :param corpus: list of str.
            A list of doc.(no need segment, do it in init)
        """
        self.corpus = corpus
        self.corpus_seg = None
        self.bm25_instance = None
        self.tokenizer = Tokenizer()

    def init(self):
        if not self.bm25_instance:
            if not self.corpus:
                logger.error('corpus is none, set corpus with docs.')
                raise ValueError("must set corpus, which is documents, list of str")

            if isinstance(self.corpus, str):
                self.corpus = [self.corpus]

            self.corpus_seg = {k: self.tokenizer.tokenize(k) for k in self.corpus}
            self.bm25_instance = BM25Okapi(corpus=list(self.corpus_seg.values()))

    def get_similarities(self, query, n=5):
        """
        Get similarity between `query` and this docs.
        :param query: str
        :param n: int, num_best
        :return: result, dict, float scores, docs rank
        """
        scores = self.get_scores(query)
        rank_n = np.argsort(scores)[::-1]
        if n > 0:
            rank_n = rank_n[:n]
        return [self.corpus[i] for i in rank_n]

    def get_scores(self, query):
        """
        Get scores between query and docs
        :param query: input str
        :return: numpy array, scores for query between docs
        """
        self.init()
        tokens = self.tokenizer.tokenize(query)
        return self.bm25_instance.get_scores(query=tokens)
