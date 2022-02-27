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

def load_test_data(path):
    sents1, sents2, labels = [], [], []
    if not os.path.isfile(path):
        return sents1, sents2, labels
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



class SimModelTestCase(unittest.TestCase):
    def test_w2v_sim_batch(self):
        """测试test_w2v_sim_batch"""
        model_name = 'w2v-light-tencent-chinese'
        print(model_name)
        m = Similarity(model_name, similarity_type=SimilarityType.COSINE, embedding_type=EmbeddingType.WORD2VEC)
        test_path = os.path.join(pwd_path, '../examples/data/STS-B/STS-B.test.data')
        sents1, sents2, labels = load_test_data(test_path)
        t1 = time()
        scores = m.get_scores(sents1, sents2, only_aligned=True)
        sims = []
        for i in range(len(sents1)):
            sims.append(scores[i][i])
        spend_time = min(time() - t1, 1e-9)
        corr = compute_spearmanr(sims, labels)
        print('scores:', sims[:10])
        print('labels:', labels[:10])
        print('sbert_batch_sim STS-B spearman corr:', corr)
        print('spend time:', spend_time, ' seconds count:', len(sents1) * 2, 'qps:', len(sents1) * 2 / spend_time)

        # ATEC
        test_path = os.path.join(pwd_path, '../examples/data/ATEC/ATEC.test.data')
        sents1, sents2, labels = load_test_data(test_path)
        t1 = time()
        scores = m.get_scores(sents1, sents2, only_aligned=True)
        sims = []
        for i in range(len(sents1)):
            sims.append(scores[i][i])
        spend_time = min(time() - t1, 1e-9)
        corr = compute_spearmanr(sims, labels)
        print('scores:', sims[:10])
        print('labels:', labels[:10])
        print('sbert_batch_sim ATEC spearman corr:', corr)
        print('spend time:', spend_time, ' seconds count:', len(sents1) * 2, 'qps:', len(sents1) * 2 / spend_time)

        # BQ
        test_path = os.path.join(pwd_path, '../examples/data/BQ/BQ.test.data')
        sents1, sents2, labels = load_test_data(test_path)
        t1 = time()
        scores = m.get_scores(sents1, sents2, only_aligned=True)
        sims = []
        for i in range(len(sents1)):
            sims.append(scores[i][i])
        spend_time = min(time() - t1, 1e-9)
        corr = compute_spearmanr(sims, labels)
        print('scores:', sims[:10])
        print('labels:', labels[:10])
        print('sbert_batch_sim BQ spearman corr:', corr)
        print('spend time:', spend_time, ' seconds count:', len(sents1) * 2, 'qps:', len(sents1) * 2 / spend_time)

        # LCQMC
        test_path = os.path.join(pwd_path, '../examples/data/LCQMC/LCQMC.test.data')
        sents1, sents2, labels = load_test_data(test_path)
        t1 = time()
        scores = m.get_scores(sents1, sents2, only_aligned=True)
        sims = []
        for i in range(len(sents1)):
            sims.append(scores[i][i])
        spend_time = min(time() - t1, 1e-9)
        corr = compute_spearmanr(sims, labels)
        print('scores:', sims[:10])
        print('labels:', labels[:10])
        print('sbert_batch_sim LCQMC spearman corr:', corr)
        print('spend time:', spend_time, ' seconds count:', len(sents1) * 2, 'qps:', len(sents1) * 2 / spend_time)

        # PAWSX
        test_path = os.path.join(pwd_path, '../examples/data/PAWSX/PAWSX.test.data')
        sents1, sents2, labels = load_test_data(test_path)
        t1 = time()
        scores = m.get_scores(sents1, sents2, only_aligned=True)
        sims = []
        for i in range(len(sents1)):
            sims.append(scores[i][i])
        spend_time = min(time() - t1, 1e-9)
        corr = compute_spearmanr(sims, labels)
        print('scores:', sims[:10])
        print('labels:', labels[:10])
        print('sbert_batch_sim PAWSX spearman corr:', corr)
        print('spend time:', spend_time, ' seconds count:', len(sents1) * 2, 'qps:', len(sents1) * 2 / spend_time)

    def test_sbert_sim_stsb_batch(self):
        """测试sbert_sim_each_batch"""
        model_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
        print(model_name)
        m = Similarity(model_name, similarity_type=SimilarityType.COSINE, embedding_type=EmbeddingType.BERT)
        test_path = os.path.join(pwd_path, '../examples/data/STS-B/STS-B.test.data')
        sents1, sents2, labels = load_test_data(test_path)
        t1 = time()
        scores = m.get_scores(sents1, sents2)
        sims = []
        for i in range(len(sents1)):
            sims.append(scores[i][i])
        spend_time = min(time() - t1, 1e-9)
        corr = compute_spearmanr(sims, labels)
        print('scores:', sims[:10])
        print('labels:', labels[:10])
        print('sbert_batch_sim STS-B spearman corr:', corr)
        print('spend time:', spend_time, ' seconds count:', len(sents1) * 2, 'qps:', len(sents1) * 2 / spend_time)

        # ATEC
        test_path = os.path.join(pwd_path, '../examples/data/ATEC/ATEC.test.data')
        sents1, sents2, labels = load_test_data(test_path)
        t1 = time()
        scores = m.get_scores(sents1, sents2)
        sims = []
        for i in range(len(sents1)):
            sims.append(scores[i][i])
        spend_time = min(time() - t1, 1e-9)
        corr = compute_spearmanr(sims, labels)
        print('scores:', sims[:10])
        print('labels:', labels[:10])
        print('sbert_batch_sim ATEC spearman corr:', corr)
        print('spend time:', spend_time, ' seconds count:', len(sents1) * 2, 'qps:', len(sents1) * 2 / spend_time)

        # BQ
        test_path = os.path.join(pwd_path, '../examples/data/BQ/BQ.test.data')
        sents1, sents2, labels = load_test_data(test_path)
        t1 = time()
        scores = m.get_scores(sents1, sents2)
        sims = []
        for i in range(len(sents1)):
            sims.append(scores[i][i])
        spend_time = min(time() - t1, 1e-9)
        corr = compute_spearmanr(sims, labels)
        print('scores:', sims[:10])
        print('labels:', labels[:10])
        print('sbert_batch_sim BQ spearman corr:', corr)
        print('spend time:', spend_time, ' seconds count:', len(sents1) * 2, 'qps:', len(sents1) * 2 / spend_time)

        # LCQMC
        test_path = os.path.join(pwd_path, '../examples/data/LCQMC/LCQMC.test.data')
        sents1, sents2, labels = load_test_data(test_path)
        t1 = time()
        scores = m.get_scores(sents1, sents2)
        sims = []
        for i in range(len(sents1)):
            sims.append(scores[i][i])
        spend_time = min(time() - t1, 1e-9)
        corr = compute_spearmanr(sims, labels)
        print('scores:', sims[:10])
        print('labels:', labels[:10])
        print('sbert_batch_sim LCQMC spearman corr:', corr)
        print('spend time:', spend_time, ' seconds count:', len(sents1) * 2, 'qps:', len(sents1) * 2 / spend_time)

        # PAWSX
        test_path = os.path.join(pwd_path, '../examples/data/PAWSX/PAWSX.test.data')
        sents1, sents2, labels = load_test_data(test_path)
        t1 = time()
        scores = m.get_scores(sents1, sents2)
        sims = []
        for i in range(len(sents1)):
            sims.append(scores[i][i])
        spend_time = min(time() - t1, 1e-9)
        corr = compute_spearmanr(sims, labels)
        print('scores:', sims[:10])
        print('labels:', labels[:10])
        print('sbert_batch_sim PAWSX spearman corr:', corr)
        print('spend time:', spend_time, ' seconds count:', len(sents1) * 2, 'qps:', len(sents1) * 2 / spend_time)

    def test_set_sim_model_batch(self):
        """测试test_set_sim_model_batch"""
        m = Similarity('shibing624/text2vec-base-chinese', similarity_type=SimilarityType.COSINE, embedding_type=EmbeddingType.BERT)
        test_path = os.path.join(pwd_path, '../examples/data/STS-B/STS-B.test.data')
        sents1, sents2, labels = load_test_data(test_path)
        t1 = time()
        scores = m.get_scores(sents1, sents2)
        sims = []
        for i in range(len(sents1)):
            sims.append(scores[i][i])
        spend_time = min(time() - t1, 1e-9)
        corr = compute_spearmanr(sims, labels)
        print('scores:', sims[:10])
        print('labels:', labels[:10])
        print('sbert_batch_sim STS-B spearman corr:', corr)
        print('spend time:', spend_time, ' seconds count:', len(sents1) * 2, 'qps:', len(sents1) * 2 / spend_time)

        # ATEC
        test_path = os.path.join(pwd_path, '../examples/data/ATEC/ATEC.test.data')
        sents1, sents2, labels = load_test_data(test_path)
        t1 = time()
        scores = m.get_scores(sents1, sents2)
        sims = []
        for i in range(len(sents1)):
            sims.append(scores[i][i])
        spend_time = min(time() - t1, 1e-9)
        corr = compute_spearmanr(sims, labels)
        print('scores:', sims[:10])
        print('labels:', labels[:10])
        print('sbert_batch_sim ATEC spearman corr:', corr)
        print('spend time:', spend_time, ' seconds count:', len(sents1) * 2, 'qps:', len(sents1) * 2 / spend_time)

        # BQ
        test_path = os.path.join(pwd_path, '../examples/data/BQ/BQ.test.data')
        sents1, sents2, labels = load_test_data(test_path)
        t1 = time()
        scores = m.get_scores(sents1, sents2)
        sims = []
        for i in range(len(sents1)):
            sims.append(scores[i][i])
        spend_time = min(time() - t1, 1e-9)
        corr = compute_spearmanr(sims, labels)
        print('scores:', sims[:10])
        print('labels:', labels[:10])
        print('sbert_batch_sim BQ spearman corr:', corr)
        print('spend time:', spend_time, ' seconds count:', len(sents1) * 2, 'qps:', len(sents1) * 2 / spend_time)

        # LCQMC
        test_path = os.path.join(pwd_path, '../examples/data/LCQMC/LCQMC.test.data')
        sents1, sents2, labels = load_test_data(test_path)
        t1 = time()
        scores = m.get_scores(sents1, sents2)
        sims = []
        for i in range(len(sents1)):
            sims.append(scores[i][i])
        spend_time = min(time() - t1, 1e-9)
        corr = compute_spearmanr(sims, labels)
        print('scores:', sims[:10])
        print('labels:', labels[:10])
        print('sbert_batch_sim LCQMC spearman corr:', corr)
        print('spend time:', spend_time, ' seconds count:', len(sents1) * 2, 'qps:', len(sents1) * 2 / spend_time)

        # PAWSX
        test_path = os.path.join(pwd_path, '../examples/data/PAWSX/PAWSX.test.data')
        sents1, sents2, labels = load_test_data(test_path)
        t1 = time()
        scores = m.get_scores(sents1, sents2)
        sims = []
        for i in range(len(sents1)):
            sims.append(scores[i][i])
        spend_time = min(time() - t1, 1e-9)
        corr = compute_spearmanr(sims, labels)
        print('scores:', sims[:10])
        print('labels:', labels[:10])
        print('sbert_batch_sim PAWSX spearman corr:', corr)
        print('spend time:', spend_time, ' seconds count:', len(sents1) * 2, 'qps:', len(sents1) * 2 / spend_time)


if __name__ == '__main__':
    unittest.main()
