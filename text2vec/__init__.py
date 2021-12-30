# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

from text2vec.version import __version__
from text2vec.word2vec import Word2Vec
from text2vec.sbert import SBert
from text2vec.bm25 import BM25

from sentence_transformers.util import semantic_search, cos_sim, paraphrase_mining, community_detection, \
    normalize_embeddings
from text2vec.similarity import Similarity, SearchSimilarity, SimType