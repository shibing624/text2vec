# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""
import numpy

from text2vec.utils import cosine_distance
from text2vec.utils.bm25 import BM25
from text2vec.vector import Vector, EmbType


class Similarity(Vector):
    def __init__(self, embedding_type='w2v', similarity_type='cosine', corpus=None, num_best=10, search_type='bm25'):
        """
        corpus: list of token list.
            A list of query in segment tokens.
        num_best: int, optional
            Number of results to retrieve.
        :param embedding_type:
        :param similarity_type:
        :param corpus:
        """
        super(Similarity, self).__init__(embedding_type=embedding_type)
        self.similarity_type = similarity_type
        self.corpus = corpus
        self.num_best = num_best
        self.search_type = search_type
        self.bm25_instance = None
        self.wmd_instance = None

    def init(self):
        if not self.bm25_instance:
            self.bm25_instance = BM25(corpus=self.corpus)
        if not self.wmd_instance:
            from gensim.similarities import WmdSimilarity
            self.embedding_type = EmbType.W2V
            self.load_model()
            self.wmd_instance = WmdSimilarity(corpus=self.corpus, w2v_model=self.model.w2v, num_best=self.num_best)

    def score(self, text1, text2):
        ret = 0.0
        if not text1.strip() or not text2.strip():
            return ret
        token_1 = self.tokenize(text1)
        token_2 = self.tokenize(text2)
        emb_1 = self.encode(token_1)
        emb_2 = self.encode(token_2)
        if self.similarity_type == 'cosine':
            ret = cosine_distance(emb_1, emb_2)
        elif self.similarity_type == 'wmd' and self.embedding_type == 'w2v':
            ret = 1. / (1. + self.model.w2v.wmdistance(token_1, token_2))
        return ret

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
