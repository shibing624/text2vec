# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

from text2vec.version import __version__

from text2vec.word2vec import Word2Vec, load_stopwords
from text2vec.sentence_model import SentenceModel, EncoderType
from text2vec.sentence_model import SentenceModel as SBert
from text2vec.bm25 import BM25
from text2vec.similarity import Similarity, SimilarityType, EmbeddingType, semantic_search, cos_sim
from text2vec.ngram import NGram
from text2vec.cosent_model import CosentModel
from text2vec.sentencebert_model import SentenceBertModel
from text2vec.bertmatching_model import BertMatchModel
from text2vec.text_matching_dataset import TextMatchingTrainDataset, TextMatchingTestDataset, load_test_data, \
    load_train_data, HFTextMatchingTrainDataset, HFTextMatchingTestDataset
from text2vec.cosent_dataset import CosentTrainDataset, HFCosentTrainDataset
from text2vec.utils.tokenizer import JiebaTokenizer
from text2vec.utils.get_file import http_get, get_file
from text2vec.utils.stats_util import compute_spearmanr, compute_pearsonr
