# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import numpy as np

from text2vec.utils.rank_bm25 import BM25Okapi
from text2vec.utils.tokenizer import Tokenizer
from text2vec.utils.log import logger


class BM25(object):
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
                logger.error('corpus is None, set corpus with documents.')
                raise ValueError("must set corpus, which is documents, list of str")

            if isinstance(self.corpus, str) or not hasattr(self.corpus, '__len__'):
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
