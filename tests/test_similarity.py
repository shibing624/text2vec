# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import sys
import unittest
import os

sys.path.append('..')
from text2vec import Similarity, SimType, EmbType
from text2vec.cosent.train import compute_corrcoef

pwd_path = os.path.abspath(os.path.dirname(__file__))
sts_test_path = os.path.join(pwd_path, '../text2vec/data/STS-B/STS-B.test.data')


def load_test_data(path):
    sents1, sents2, labels = [], [], []
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip().split('\t')
            if len(line) != 3:
                continue
            sents1.append(line[0])
            sents2.append(line[1])
            score = int(line[2])
            labels.append(score)
    return sents1, sents2, labels


class SimTestCase(unittest.TestCase):
    def test_w2v_sim_each(self):
        """测试w2v_sim_each"""
        m = Similarity(similarity_type=SimType.COSINE, embedding_type=EmbType.W2V)
        sents1, sents2, labels = load_test_data(sts_test_path)
        scores = []
        for s1, s2 in zip(sents1, sents2):
            s = m.get_score(s1, s2)
            scores.append(s)
        corr = compute_corrcoef(scores, labels)
        print('scores:', scores[:10])
        print('labels:', labels[:10])
        print('w2v_each_sim spearman corr:', corr)

    def test_w2v_sim_batch(self):
        """测试w2v_sim_batch"""
        m = Similarity(similarity_type=SimType.COSINE, embedding_type=EmbType.W2V)
        sents1, sents2, labels = load_test_data(sts_test_path)
        scores = m.batch_sim_score(sents1, sents2)
        corr = compute_corrcoef(scores, labels)
        print('scores:', scores[:10])
        print('labels:', labels[:10])
        print('w2v_batch_sim spearman corr:', corr)

    def test_sbert_sim_each(self):
        """测试sbert_sim_each"""
        m = Similarity(similarity_type=SimType.COSINE, embedding_type=EmbType.SBERT)
        sents1, sents2, labels = load_test_data(sts_test_path)
        scores = []
        for s1, s2 in zip(sents1, sents2):
            s = m.get_score(s1, s2)
            scores.append(s)
        corr = compute_corrcoef(scores, labels)
        print('scores:', scores[:10])
        print('labels:', labels[:10])
        print('sbert_each_sim spearman corr:', corr)

    def test_sbert_sim_batch(self):
        """测试sbert_sim_each_batch"""
        m = Similarity(similarity_type=SimType.COSINE, embedding_type=EmbType.SBERT)
        sents1, sents2, labels = load_test_data(sts_test_path)
        scores = m.batch_sim_score(sents1, sents2)
        sims = []
        for i in range(len(sents1)):
            sims.append(scores[i][i])
        corr = compute_corrcoef(sims, labels)
        print('scores:', sims[:10])
        print('labels:', labels[:10])
        print('sbert_batch_sim spearman corr:', corr)


if __name__ == '__main__':
    unittest.main()
