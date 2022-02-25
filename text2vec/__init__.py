# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

from text2vec.version import __version__
from text2vec.word2vec import Word2Vec
from text2vec.sentence_model import SentenceModel, EncoderType
from text2vec.bm25 import BM25
from text2vec.similarity import Similarity, SimilarityType, EmbeddingType, semantic_search, cos_sim
from text2vec.ngram import NGram
from text2vec.utils.stats_util import compute_spearmanr, compute_pearsonr