# -*- coding: utf-8 -*-
"""
@author:XuMing<xuming624@qq.com>
@description:
"""

import logging
import unittest

SEQUENCE_LENGTH = 30


class TestWordEmbeddings(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from text2vec.embeddings.word_embedding import WordEmbedding
        cls.embedding = WordEmbedding(sequence_length=SEQUENCE_LENGTH)

    def test_tokenize(self):
        sentence = ['我', '想', '看', '电影', '%%##!$#%']
        tokens = self.embedding.embed_one(sentence)

        logging.info('tokenize test: {} -> {}'.format(sentence, tokens))
        assert len(tokens) > 0

        token_list = self.embedding.embed([sentence])
        assert len(token_list[0]) > 0

    def test_embed(self):
        sentence = ['我', '想', '看', '电影', '%%##!$#%']
        embedded_sentence = self.embedding.embed_one(sentence)
        embedded_sentences = self.embedding.embed([sentence])
        logging.info('embed test: {} -> {}'.format(sentence, embedded_sentence))
        assert len(embedded_sentence.shape) > 0
        assert len(embedded_sentences.shape) > 0


class TestBERTEmbedding(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from text2vec.embeddings.bert_embedding import BERTEmbedding
        cls.embedding = BERTEmbedding(sequence_length=SEQUENCE_LENGTH)

    def test_build(self):
        assert self.embedding.embedding_size > 0

    def test_embed(self):
        sentence = ['我', '想', '看', '电影', '%%##!$#%']
        embedded_sentence = self.embedding.embed_one(sentence)
        embedded_sentences = self.embedding.embed([sentence])
        logging.info('embed test: {} -> {}'.format(sentence, embedded_sentence))

        assert len(embedded_sentence.shape) > 0
        assert len(embedded_sentences.shape) > 0


if __name__ == "__main__":
    unittest.main()
