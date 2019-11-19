# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""

from pathlib import Path

from simtext.utils import cosine_distance

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
    def __init__(self, embedding_type='w2v'):
        self.embedding_type = embedding_type
        self.model = None

    def load_model(self):
        if not self.model:
            if self.embedding_type == EmbType.BERT:
                from simtext.embeddings.bert_embedding import BERTEmbedding
                self.model = BERTEmbedding(sequence_length=128)
            elif self.embedding_type == EmbType.W2V:
                from simtext.embeddings.word_embedding import WordEmbedding
                self.model = WordEmbedding()
            else:
                raise ValueError('set error embedding_type.')

    def encode(self, text):
        ret = 0.0
        if not text.strip():
            return ret
        self.load_model()
        tokens = self.model.tokenizer.tokenize(text)
        return self.model.embed_one(tokens)

    def score(self, text1, text2):
        ret = 0.0
        if not text1.strip() or not text2.strip():
            return ret
        emb_1 = self.encode(text1)
        emb_2 = self.encode(text2)
        ret = cosine_distance(emb_1, emb_2)
        return ret


SIM = Similarity()
score = SIM.score
encode = SIM.encode
