# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

from typing import List, Union, Tuple, Optional
from text2vec.utils.rank_bm25 import BM25Okapi
from text2vec.utils.tokenizer import JiebaTokenizer


class BM25:
    def __init__(self, corpus: Union[List[str], str]):
        """
        Search sim doc with rank bm25
        :param corpus: list of str.
            A list of doc.(no need segment, do it in init)
        """
        self.corpus = corpus
        self.corpus_seg = None
        self.bm25_instance = None
        self.jieba_tokenizer = JiebaTokenizer()

    def init(self):
        if not self.bm25_instance:
            if not self.corpus:
                raise ValueError("Must set corpus, which is documents, list of str")

            if isinstance(self.corpus, str) or not hasattr(self.corpus, '__len__'):
                self.corpus = [self.corpus]

            self.corpus_seg = {k: self.jieba_tokenizer.tokenize(k, HMM=False) for k in self.corpus}
            self.bm25_instance = BM25Okapi(corpus=list(self.corpus_seg.values()))

    def get_scores(self, query: str, top_k: Optional[int] = None) -> List[Tuple[str, float]]:
        """
        Get scores between query and docs
        :param query: str, input str, no need segment, auto segment by jieba
        :param top_k: int, top k results, default is None, means return all results
        :return: list, List[corpus_str, score] for query between docs
        """
        self.init()
        tokens = self.jieba_tokenizer.tokenize(query, HMM=False)
        scores = self.bm25_instance.get_scores(query=tokens)

        corpus_scores = list(zip(self.corpus, scores))
        corpus_scores_sort = sorted(corpus_scores, key=lambda x: x[1], reverse=True)
        return corpus_scores_sort[:top_k]
