# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import numpy as np
from loguru import logger
from text2vec.utils.rank_bm25 import BM25Okapi
from text2vec.utils.tokenizer import JiebaTokenizer
from text2vec.word2vec import Word2Vec
from text2vec.sbert import SBert, cos_sim
from text2vec.utils.distance import cosine_distance


class EmbType(object):
    W2V = 'w2v'
    SBERT = 'sbert'


class SimType(object):
    COSINE = 'cosine'
    WMD = 'wmd'


class Similarity(object):
    def __init__(self, model_name_or_path="", similarity_type=SimType.COSINE, embedding_type=EmbType.SBERT):
        """
        Cal text similarity
        :param similarity_type:
        :param embedding_type:
        """
        if embedding_type not in [EmbType.SBERT, EmbType.W2V]:
            logger.warning('embedding_type set error, use default sbert')
            embedding_type = EmbType.SBERT
        if similarity_type not in [SimType.COSINE, SimType.WMD]:
            logger.warning('similarity_type set error, use default cosine')
            similarity_type = SimType.COSINE
        if similarity_type == SimType.WMD and embedding_type != EmbType.W2V:
            logger.warning('wmd sim type, emb type must be w2v')
            embedding_type = EmbType.W2V
        self.similarity_type = similarity_type
        self.embedding_type = embedding_type
        self.model_name_or_path = model_name_or_path
        self.jieba_tokenizer = JiebaTokenizer()
        self.model = None

    def load_model(self):
        if self.model is None:
            if self.embedding_type == EmbType.W2V:
                if self.model_name_or_path:
                    self.model = Word2Vec(self.model_name_or_path)
                else:
                    self.model = Word2Vec()
            if self.embedding_type == EmbType.SBERT:
                if self.model_name_or_path:
                    self.model = SBert(self.model_name_or_path)
                else:
                    self.model = SBert()

    def get_score(self, sentence1, sentence2):
        """
        Get score between text1 and text2
        :param sentence1: str
        :param sentence2: str
        :return: float, score
        """
        res = 0.0
        sentence1 = sentence1.strip()
        sentence2 = sentence2.strip()
        if not sentence1 or not sentence2:
            return res
        self.load_model()
        if self.similarity_type == SimType.COSINE:
            emb1 = self.model.encode(sentence1)
            emb2 = self.model.encode(sentence2)
            res = cos_sim(emb1, emb2)[0] if self.embedding_type == EmbType.SBERT else cosine_distance(emb1, emb2)
            res = float(res)
        elif self.similarity_type == SimType.WMD:
            token1 = self.jieba_tokenizer.tokenize(sentence1)
            token2 = self.jieba_tokenizer.tokenize(sentence2)
            res = 1. / (1. + self.model.w2v.wmdistance(token1, token2))
        return res

    def get_scores(self, sentences1, sentences2, only_aligned=False):
        """
        Get similarity scores between texts1 and texts2
        :param sentences1: list
        :param sentences2: list
        :param only_aligned: bool, default False，如果sentences1和sentences2为size对齐的数据，是否仅计算scores[i][i]的结果
        :return: return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
        """
        if not sentences1 or not sentences2:
            return None
        if only_aligned and len(sentences1) != len(sentences2):
            logger.warning('sentences size not equal, auto set is_aligned=False')
            only_aligned = False
        self.load_model()
        embs1 = self.model.encode(sentences1)
        embs2 = self.model.encode(sentences2)
        if self.embedding_type == EmbType.SBERT:
            scores = cos_sim(embs1, embs2).numpy()
        else:
            scores = np.zeros((len(sentences1), len(sentences2)), dtype=np.float32)
            if only_aligned:
                for i, e in enumerate(zip(embs1, embs2)):
                    scores[i][i] = cosine_distance(e[0], e[1])
            else:
                for i, e1 in enumerate(embs1):
                    for j, e2 in enumerate(embs2):
                        scores[i][j] = cosine_distance(e1, e2)
        return scores


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
        self.jieba_tokenizer = JiebaTokenizer()

    def init(self):
        if not self.bm25_instance:
            if not self.corpus:
                raise ValueError("must set corpus, which is documents, list of str")

            if isinstance(self.corpus, str):
                self.corpus = [self.corpus]

            self.corpus_seg = {k: self.jieba_tokenizer.tokenize(k) for k in self.corpus}
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
        tokens = self.jieba_tokenizer.tokenize(query)
        return self.bm25_instance.get_scores(query=tokens)
