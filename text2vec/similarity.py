# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""

from text2vec.utils import cosine_distance
from text2vec.vector import Vector


class Similarity(Vector):
    def __init__(self, embedding_type='w2v', similarity_type='cosine', corpus = None):
        super(Similarity, self).__init__(embedding_type=embedding_type)
        self.similarity_type = similarity_type
        self.corpus = corpus

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
            ret = 1 - self.model.w2v.wmdistance(token_1, token_2)
        return ret
