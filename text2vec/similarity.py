# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""

from text2vec.utils.bm25 import BM25
from text2vec.utils.distance import cosine_distance
from text2vec.utils.logger import get_logger
from text2vec.vector import Vector, EmbType

logger = get_logger(__name__)


class SimType(object):
    COSINE = 'cosine'
    WMD = 'wmd'


class Similarity(Vector):
    def __init__(self, similarity_type=SimType.COSINE,
                 sequence_length=128):
        """
        corpus: list of token list.
            A list of query in segment tokens.
        num_best: int, optional
            Number of results to retrieve.
        :param similarity_type:
        :param corpus:
        """
        super(Similarity, self).__init__()
        self.similarity_type = similarity_type
        self.sequence_length = sequence_length

    def score(self, text1, text2):
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


class SearchSimilarity(Similarity):
    def __init__(self, similarity_type=SimType.COSINE, corpus=None, num_best=10, search_type='bm25'):
        """
        corpus: list of token list.
            A list of query in segment tokens.
        num_best: int, optional
            Number of results to retrieve.
        :param similarity_type:
        :param corpus:
        """
        super(SearchSimilarity, self).__init__(similarity_type=similarity_type)
        self.similarity_type = similarity_type
        self.corpus = corpus
        self.num_best = num_best
        self.search_type = search_type
        self.bm25_instance = None

    def init(self):
        if not self.bm25_instance:
            self.bm25_instance = BM25(corpus=self.corpus)

    def get_similarities(self, query):
        """Get similarity between `query` and this docs.

        Parameters
        ----------
        query : str.

        Return
        ------
        :float scores.

        """
        self.init()
        tokens = self.tokenize(query)
        sim_items = self.bm25_instance.similarity(tokens, self.num_best)
        docs = [(self.corpus[id], score) for id, score in sim_items]
        return docs
