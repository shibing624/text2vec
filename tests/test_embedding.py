# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""
import sys
import unittest

sys.path.append('..')

SEQUENCE_LENGTH = 30


class TestWordEmbeddings(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from text2vec.embeddings.word_embedding import WordEmbedding
        cls.embedding = WordEmbedding(sequence_length=SEQUENCE_LENGTH)

    def test_embed(self):
        sentence = ['我', '想', '看', '电影', '%%##!$#%']
        embedded_sentence = self.embedding.embed_one(sentence)
        embedded_sentences = self.embedding.embed([sentence])
        print('embed test: {} -> {}'.format(sentence, embedded_sentence))

        print(embedded_sentence.shape)
        assert embedded_sentence.shape == (200,)

        print([round(i, 3) for i in embedded_sentence[:3]])
        self.assertTrue(["{:.3f}".format(i) for i in embedded_sentence[:3]] == ["0.015", "-0.094", "-0.020"])

        print(embedded_sentences.shape)
        assert embedded_sentences.shape == (1, 200)


class TestBERTEmbedding(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from text2vec.embeddings.bert_embedding import BERTEmbedding
        cls.embedding = BERTEmbedding(sequence_length=SEQUENCE_LENGTH)

    def test_build(self):
        print(self.embedding.embedding_size)
        self.assertTrue(self.embedding.embedding_size == 768 * 4)

    def test_embed(self):
        sentence = ['我', '想', '看', '电影', '%%##!$#%']
        embedded_sentence = self.embedding.embed_one(sentence)
        embedded_sentences = self.embedding.embed([sentence])
        print('embed test: {} -> {}'.format(sentence, embedded_sentence))

        print(embedded_sentence.shape)
        assert embedded_sentence.shape == (30, 768 * 4)

        print(embedded_sentences.shape)
        self.assertTrue(embedded_sentences.shape == (1, 30, 3072))

        print(embedded_sentences)

    def test_token(self):
        tokens = self.embedding.process_x_dataset([['语', '言', '模', '型']])[0]
        target_index = [6427, 6241, 3563, 1798]
        print(list(tokens[0])[:4])
        assert list(tokens[0])[:4] == target_index


if __name__ == "__main__":
    unittest.main()
