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


class SimModelTestCase(unittest.TestCase):
    def test_sbert_sim_stsb_batch(self):
        """测试sbert_sim_each_batch"""
        model_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
        m = Similarity(model_name, similarity_type=SimType.COSINE, embedding_type=EmbType.SBERT)
        test_path = os.path.join(pwd_path, '../text2vec/data/STS-B/STS-B.test.data')
        sents1, sents2, labels = load_test_data(test_path)
        t1 = time()
        scores = m.get_scores(sents1, sents2)
        sims = []
        for i in range(len(sents1)):
            sims.append(scores[i][i])
        spend_time = time() - t1
        corr = compute_corrcoef(sims, labels)
        print('scores:', sims[:10])
        print('labels:', labels[:10])
        print('sbert_batch_sim STS-B spearman corr:', corr)
        print('spend time:', spend_time, ' seconds count:', len(sents1) * 2, 'qps:', len(sents1) * 2 / spend_time)

        # ATEC
        test_path = os.path.join(pwd_path, '../text2vec/data/ATEC/ATEC.test.data')
        sents1, sents2, labels = load_test_data(test_path)
        t1 = time()
        scores = m.get_scores(sents1, sents2)
        sims = []
        for i in range(len(sents1)):
            sims.append(scores[i][i])
        spend_time = time() - t1
        corr = compute_corrcoef(sims, labels)
        print('scores:', sims[:10])
        print('labels:', labels[:10])
        print('sbert_batch_sim ATEC spearman corr:', corr)
        print('spend time:', spend_time, ' seconds count:', len(sents1) * 2, 'qps:', len(sents1) * 2 / spend_time)

        # BQ
        test_path = os.path.join(pwd_path, '../text2vec/data/BQ/BQ.test.data')
        sents1, sents2, labels = load_test_data(test_path)
        t1 = time()
        scores = m.get_scores(sents1, sents2)
        sims = []
        for i in range(len(sents1)):
            sims.append(scores[i][i])
        spend_time = time() - t1
        corr = compute_corrcoef(sims, labels)
        print('scores:', sims[:10])
        print('labels:', labels[:10])
        print('sbert_batch_sim BQ spearman corr:', corr)
        print('spend time:', spend_time, ' seconds count:', len(sents1) * 2, 'qps:', len(sents1) * 2 / spend_time)

        # LCQMC
        test_path = os.path.join(pwd_path, '../text2vec/data/LCQMC/LCQMC.test.data')
        sents1, sents2, labels = load_test_data(test_path)
        t1 = time()
        scores = m.get_scores(sents1, sents2)
        sims = []
        for i in range(len(sents1)):
            sims.append(scores[i][i])
        spend_time = time() - t1
        corr = compute_corrcoef(sims, labels)
        print('scores:', sims[:10])
        print('labels:', labels[:10])
        print('sbert_batch_sim LCQMC spearman corr:', corr)
        print('spend time:', spend_time, ' seconds count:', len(sents1) * 2, 'qps:', len(sents1) * 2 / spend_time)

        # PAWSX
        test_path = os.path.join(pwd_path, '../text2vec/data/PAWSX/PAWSX.test.data')
        sents1, sents2, labels = load_test_data(test_path)
        t1 = time()
        scores = m.get_scores(sents1, sents2)
        sims = []
        for i in range(len(sents1)):
            sims.append(scores[i][i])
        spend_time = time() - t1
        corr = compute_corrcoef(sims, labels)
        print('scores:', sims[:10])
        print('labels:', labels[:10])
        print('sbert_batch_sim PAWSX spearman corr:', corr)
        print('spend time:', spend_time, ' seconds count:', len(sents1) * 2, 'qps:', len(sents1) * 2 / spend_time)

    def test_set_sim_model_batch(self):
        """测试test_set_sim_model_batch"""
        model_path = os.path.join(pwd_path, '../text2vec/cosent/text2vec-base-chinese-stsb')
        if not os.path.exists(os.path.join(model_path, 'vocab.txt')):
            return
        m = Similarity(model_path, similarity_type=SimType.COSINE, embedding_type=EmbType.SBERT)
        test_path = os.path.join(pwd_path, '../text2vec/data/STS-B/STS-B.test.data')
        sents1, sents2, labels = load_test_data(test_path)
        t1 = time()
        scores = m.get_scores(sents1, sents2)
        sims = []
        for i in range(len(sents1)):
            sims.append(scores[i][i])
        spend_time = time() - t1
        corr = compute_corrcoef(sims, labels)
        print('scores:', sims[:10])
        print('labels:', labels[:10])
        print('sbert_batch_sim STS-B spearman corr:', corr)
        print('spend time:', spend_time, ' seconds count:', len(sents1) * 2, 'qps:', len(sents1) * 2 / spend_time)

        # ATEC
        test_path = os.path.join(pwd_path, '../text2vec/data/ATEC/ATEC.test.data')
        sents1, sents2, labels = load_test_data(test_path)
        t1 = time()
        scores = m.get_scores(sents1, sents2)
        sims = []
        for i in range(len(sents1)):
            sims.append(scores[i][i])
        spend_time = time() - t1
        corr = compute_corrcoef(sims, labels)
        print('scores:', sims[:10])
        print('labels:', labels[:10])
        print('sbert_batch_sim ATEC spearman corr:', corr)
        print('spend time:', spend_time, ' seconds count:', len(sents1) * 2, 'qps:', len(sents1) * 2 / spend_time)

        # BQ
        test_path = os.path.join(pwd_path, '../text2vec/data/BQ/BQ.test.data')
        sents1, sents2, labels = load_test_data(test_path)
        t1 = time()
        scores = m.get_scores(sents1, sents2)
        sims = []
        for i in range(len(sents1)):
            sims.append(scores[i][i])
        spend_time = time() - t1
        corr = compute_corrcoef(sims, labels)
        print('scores:', sims[:10])
        print('labels:', labels[:10])
        print('sbert_batch_sim BQ spearman corr:', corr)
        print('spend time:', spend_time, ' seconds count:', len(sents1) * 2, 'qps:', len(sents1) * 2 / spend_time)

        # LCQMC
        test_path = os.path.join(pwd_path, '../text2vec/data/LCQMC/LCQMC.test.data')
        sents1, sents2, labels = load_test_data(test_path)
        t1 = time()
        scores = m.get_scores(sents1, sents2)
        sims = []
        for i in range(len(sents1)):
            sims.append(scores[i][i])
        spend_time = time() - t1
        corr = compute_corrcoef(sims, labels)
        print('scores:', sims[:10])
        print('labels:', labels[:10])
        print('sbert_batch_sim LCQMC spearman corr:', corr)
        print('spend time:', spend_time, ' seconds count:', len(sents1) * 2, 'qps:', len(sents1) * 2 / spend_time)

        # PAWSX
        test_path = os.path.join(pwd_path, '../text2vec/data/PAWSX/PAWSX.test.data')
        sents1, sents2, labels = load_test_data(test_path)
        t1 = time()
        scores = m.get_scores(sents1, sents2)
        sims = []
        for i in range(len(sents1)):
            sims.append(scores[i][i])
        spend_time = time() - t1
        corr = compute_corrcoef(sims, labels)
        print('scores:', sims[:10])
        print('labels:', labels[:10])
        print('sbert_batch_sim PAWSX spearman corr:', corr)
        print('spend time:', spend_time, ' seconds count:', len(sents1) * 2, 'qps:', len(sents1) * 2 / spend_time)


if __name__ == '__main__':
    unittest.main()
