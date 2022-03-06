# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import sys
import unittest
from time import time
import os

sys.path.append('..')
from text2vec import Similarity, SimilarityType, EmbeddingType, compute_spearmanr

pwd_path = os.path.abspath(os.path.dirname(__file__))
sts_test_path = os.path.join(pwd_path, '../examples/data/STS-B/STS-B.test.data')


def load_test_data(path):
    sents1, sents2, labels = [], [], []
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip().split('\t')
            if len(line) != 3:
                continue
            sents1.append(line[0])
            sents2.append(line[1])
            labels.append(int(line[2]))
            if len(sents1) > 10:
                break
    return sents1, sents2, labels


class SimTestCase(unittest.TestCase):
    def test_w2v_sim_each(self):
        """测试w2v_sim_each"""
        m = Similarity(similarity_type=SimilarityType.COSINE, embedding_type=EmbeddingType.WORD2VEC)
        print(m)
        sents1, sents2, labels = load_test_data(sts_test_path)
        t1 = time()
        scores = []
        for s1, s2 in zip(sents1, sents2):
            s = m.get_score(s1, s2)
            scores.append(s)
        spend_time = time() - t1
        corr = compute_spearmanr(scores, labels)
        print('scores:', scores[:10])
        print('labels:', labels[:10])
        print('w2v_each_sim spearman corr:', corr)
        print('spend time:', spend_time, ' seconds count:', len(sents1) * 2, 'qps:', len(sents1) * 2 / spend_time)

    def test_w2v_sim_batch(self):
        """测试w2v_sim_batch"""
        m = Similarity(similarity_type=SimilarityType.COSINE, embedding_type=EmbeddingType.WORD2VEC)
        sents1, sents2, labels = load_test_data(sts_test_path)
        t1 = time()
        scores = m.get_scores(sents1, sents2)
        sims = []
        for i in range(len(sents1)):
            sims.append(scores[i][i])
        spend_time = time() - t1
        corr = compute_spearmanr(sims, labels)
        print('scores:', sims[:10])
        print('labels:', labels[:10])
        print('w2v_batch_sim spearman corr:', corr)
        print('spend time:', spend_time, ' seconds count:', len(sents1) * 2, 'qps:', len(sents1) * 2 / spend_time)

    def test_sbert_sim_each(self):
        """测试sbert_sim_each"""
        m = Similarity(similarity_type=SimilarityType.COSINE, embedding_type=EmbeddingType.BERT)
        sents1, sents2, labels = load_test_data(sts_test_path)
        t1 = time()
        scores = []
        for s1, s2 in zip(sents1, sents2):
            s = m.get_score(s1, s2)
            scores.append(s)
        spend_time = time() - t1
        corr = compute_spearmanr(scores, labels)
        print('scores:', scores[:10])
        print('labels:', labels[:10])
        print('sbert_each_sim spearman corr:', corr)
        print('spend time:', spend_time, ' seconds count:', len(sents1) * 2, 'qps:', len(sents1) * 2 / spend_time)

    def test_sbert_sim_batch(self):
        """测试sbert_sim_each_batch"""
        m = Similarity(similarity_type=SimilarityType.COSINE, embedding_type=EmbeddingType.BERT)
        sents1, sents2, labels = load_test_data(sts_test_path)
        t1 = time()
        scores = m.get_scores(sents1, sents2)
        sims = []
        for i in range(len(sents1)):
            for j in range(len(sents2)):
                print(scores[i][j], sents1[i], sents2[j])
        print()
        for i in range(len(sents1)):
            sims.append(scores[i][i])
            print(scores[i][i], sents1[i], sents2[i])

        spend_time = time() - t1
        corr = compute_spearmanr(sims, labels)
        print('scores:', sims[:10])
        print('labels:', labels[:10])
        print('sbert_batch_sim spearman corr:', corr)
        print('spend time:', spend_time, ' seconds count:', len(sents1) * 2, 'qps:', len(sents1) * 2 / spend_time)


if __name__ == '__main__':
    unittest.main()
