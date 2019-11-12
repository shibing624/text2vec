# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""

from pathlib import Path

import jieba

from simtext.embeddings.bert_embedding import BERTEmbedding
from simtext.embeddings.word_embedding import WordEmbedding
from simtext.utils.distance import cos_dist

jieba.setLogLevel('ERROR')
USER_DIR = Path.expanduser(Path('~')).joinpath('.simtext')
if not USER_DIR.exists():
    USER_DIR.mkdir()
USER_DATA_DIR = USER_DIR.joinpath('datasets')
if not USER_DATA_DIR.exists():
    USER_DATA_DIR.mkdir()


class EmbType(object):
    BERT = 'bert'
    W2V = 'w2v'


class Similarity(object):
    def __init__(self, embedding_type='bert'):
        self.embedding_type = embedding_type
        self.model = None

    def load_model(self):
        if not self.model:
            if self.embedding_type == EmbType.BERT:
                self.model = BERTEmbedding(sequence_length=128)
            elif self.embedding_type == EmbType.W2V:
                self.model = WordEmbedding(w2v_path='/Users/xuming06/Codes/ai-server-xuming/data/sentence_w2v.bin',
                                           w2v_kwargs={'binary': True})
            else:
                raise ValueError('set error embedding_type.')

    def score(self, text1, text2):
        ret = 0.0
        if not text1.strip() or not text2.strip():
            return ret
        self.load_model()
        if self.embedding_type == EmbType.BERT:
            tokens_1 = self.model.tokenizer.tokenize(text1)
            tokens_2 = self.model.tokenizer.tokenize(text2)
        else:
            tokens_1 = jieba.lcut(text1)
            tokens_2 = jieba.lcut(text2)
        emb_1 = self.model.embed([tokens_1])[0]
        emb_2 = self.model.embed([tokens_2])[0]
        ret = cos_dist(emb_1, emb_2)
        return ret
