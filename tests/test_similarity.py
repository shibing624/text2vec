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
from text2vec import Similarity, SimType, EmbType
from text2vec.cosent.train import compute_corrcoef
from text2vec.cosent.data_helper import load_test_data

pwd_path = os.path.abspath(os.path.dirname(__file__))
sts_test_path = os.path.join(pwd_path, '../text2vec/data/STS-B/STS-B.test.data')


class SimTestCase(unittest.TestCase):
    def test_w2v_sim_each(self):
        """测试w2v_sim_each"""
        m = Similarity(similarity_type=SimType.COSINE, embedding_type=EmbType.W2V)
        sents1, sents2, labels = load_test_data(sts_test_path)
        t1 = time()
        scores = []
        for s1, s2 in zip(sents1, sents2):
            s = m.get_score(s1, s2)
            scores.append(s)
        spend_time = time() - t1
        corr = compute_corrcoef(scores, labels)
        print('scores:', scores[:10])
        print('labels:', labels[:10])
        print('w2v_each_sim spearman corr:', corr)
        print('spend time:', spend_time, ' seconds count:', len(sents1) * 2, 'qps:', len(sents1) * 2 / spend_time)

    def test_w2v_sim_batch(self):
        """测试w2v_sim_batch"""
        m = Similarity(similarity_type=SimType.COSINE, embedding_type=EmbType.W2V)
        sents1, sents2, labels = load_test_data(sts_test_path)
        t1 = time()
        scores = m.get_scores(sents1, sents2)
        sims = []
        for i in range(len(sents1)):
            sims.append(scores[i][i])
        spend_time = time() - t1
        corr = compute_corrcoef(sims, labels)
        print('scores:', sims[:10])
        print('labels:', labels[:10])
        print('w2v_batch_sim spearman corr:', corr)
        print('spend time:', spend_time, ' seconds count:', len(sents1) * 2, 'qps:', len(sents1) * 2 / spend_time)

    def test_sbert_sim_each(self):
        """测试sbert_sim_each"""
        m = Similarity(similarity_type=SimType.COSINE, embedding_type=EmbType.SBERT)
        sents1, sents2, labels = load_test_data(sts_test_path)
        t1 = time()
        scores = []
        for s1, s2 in zip(sents1, sents2):
            s = m.get_score(s1, s2)
            scores.append(s)
        spend_time = time() - t1
        corr = compute_corrcoef(scores, labels)
        print('scores:', scores[:10])
        print('labels:', labels[:10])
        print('sbert_each_sim spearman corr:', corr)
        print('spend time:', spend_time, ' seconds count:', len(sents1) * 2, 'qps:', len(sents1) * 2 / spend_time)

    def test_sbert_sim_batch(self):
        """测试sbert_sim_each_batch"""
        m = Similarity(similarity_type=SimType.COSINE, embedding_type=EmbType.SBERT)
        sents1, sents2, labels = load_test_data(sts_test_path)
        t1 = time()
        scores = m.get_scores(sents1, sents2)
        sims = []
        for i in range(len(sents1)):
            sims.append(scores[i][i])
        spend_time = time() - t1
        corr = compute_corrcoef(sims, labels)
        print('scores:', sims[:10])
        print('labels:', labels[:10])
        print('sbert_batch_sim spearman corr:', corr)
        print('spend time:', spend_time, ' seconds count:', len(sents1) * 2, 'qps:', len(sents1) * 2 / spend_time)

if __name__ == '__main__':
    unittest.main()
