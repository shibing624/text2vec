# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""


class EmbType(object):
    BERT = 'bert'
    W2V = 'w2v'


class Vector(object):
    def __init__(self, embedding_type='w2v'):
        self.embedding_type = embedding_type
        self.model = None

    def load_model(self):
        if not self.model:
            if self.embedding_type == EmbType.BERT:
                from text2vec.embeddings.bert_embedding import BERTEmbedding
                self.model = BERTEmbedding(sequence_length=128)
            elif self.embedding_type == EmbType.W2V:
                from text2vec.embeddings.word_embedding import WordEmbedding
                self.model = WordEmbedding()
            else:
                raise ValueError('set error embedding type.')

    def tokenize(self, text):
        if not text.strip():
            return []
        self.load_model()
        return self.model.tokenizer.tokenize(text)

    def encode(self, tokens):
        ret = 0.0
        if not tokens:
            return ret
        self.load_model()
        if isinstance(tokens, str):
            tokens = self.tokenize(tokens)
        return self.model.embed_one(tokens)
