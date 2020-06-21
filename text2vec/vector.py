# -*- coding: utf-8 -*-
"""
@author:XuMing<xuming624@qq.com>
@description: 
"""


class EmbType(object):
    BERT = 'bert'
    W2V = 'w2v'


class Vector(object):
    def __init__(self, embedding_type=EmbType.W2V,
                 w2v_path='',
                 w2v_kwargs=None,
                 sequence_length=128,
                 processor=None,
                 trainable=False,
                 bert_model_folder='',
                 bert_layer_nums=4):
        self.embedding_type = embedding_type
        self.w2v_path = w2v_path
        self.w2v_kwargs = w2v_kwargs  # default: {binary:False}
        self.sequence_length = sequence_length
        self.processor = processor
        self.trainable = trainable
        self.bert_model_folder = bert_model_folder
        self.bert_layer_nums = bert_layer_nums
        self.model = None

    def load_model(self):
        if not self.model:
            if self.embedding_type == EmbType.BERT:
                from text2vec.embeddings.bert_embedding import BERTEmbedding
                self.model = BERTEmbedding(model_folder=self.bert_model_folder,
                                           layer_nums=self.bert_layer_nums,
                                           trainable=self.trainable,
                                           sequence_length=self.sequence_length,
                                           processor=self.processor)
            elif self.embedding_type == EmbType.W2V:
                from text2vec.embeddings.word_embedding import WordEmbedding
                self.model = WordEmbedding(w2v_path=self.w2v_path,
                                           w2v_kwargs=self.w2v_kwargs,
                                           sequence_length=self.sequence_length,
                                           processor=self.processor,
                                           trainable=self.trainable)
            else:
                raise ValueError('set error embedding type.')

    def tokenize(self, text):
        if not text.strip():
            return []
        self.load_model()
        return self.model.tokenizer.tokenize(text.lower().strip())

    def encode(self, tokens):
        ret = 0.0
        if not tokens:
            return ret
        self.load_model()
        if isinstance(tokens, str):
            tokens = self.tokenize(tokens)
        return self.model.embed_one(tokens)
