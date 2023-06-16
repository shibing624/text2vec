# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import os
import sys
import unittest
from time import time

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
        spend_time = max(time() - t1, 1e-9)
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
        spend_time = max(time() - t1, 1e-9)
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
        spend_time = max(time() - t1, 1e-9)
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
        spend_time = max(time() - t1, 1e-9)
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
        spend_time = max(time() - t1, 1e-9)
        corr = compute_spearmanr(sims, labels)
        print('scores:', sims[:10])
        print('labels:', labels[:10])
        print('sbert_batch_sim PAWSX spearman corr:', corr)
        print('spend time:', spend_time, ' seconds count:', len(sents1) * 2, 'qps:', len(sents1) * 2 / spend_time)

    def test_sbert_sim_stsb_batch(self):
        """测试sbert_sim_each_batch"""
        model_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
        print(model_name)
        m = Similarity(
            model_name,
            similarity_type=SimilarityType.COSINE,
            embedding_type=EmbeddingType.BERT,
            encoder_type="FIRST_LAST_AVG"
        )
        test_path = os.path.join(pwd_path, '../examples/data/STS-B/STS-B.test.data')
        sents1, sents2, labels = load_test_data(test_path)
        t1 = time()
        scores = m.get_scores(sents1, sents2)
        sims = []
        for i in range(len(sents1)):
            sims.append(scores[i][i])
        spend_time = max(time() - t1, 1e-9)
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
        spend_time = max(time() - t1, 1e-9)
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
        spend_time = max(time() - t1, 1e-9)
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
        spend_time = max(time() - t1, 1e-9)
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
        spend_time = max(time() - t1, 1e-9)
        corr = compute_spearmanr(sims, labels)
        print('scores:', sims[:10])
        print('labels:', labels[:10])
        print('sbert_batch_sim PAWSX spearman corr:', corr)
        print('spend time:', spend_time, ' seconds count:', len(sents1) * 2, 'qps:', len(sents1) * 2 / spend_time)

    def test_set_sim_model_batch(self):
        """测试test_set_sim_model_batch"""
        m = Similarity(
            'shibing624/text2vec-base-chinese',
            similarity_type=SimilarityType.COSINE,
            embedding_type=EmbeddingType.BERT,
            encoder_type="FIRST_LAST_AVG"
        )
        test_path = os.path.join(pwd_path, '../examples/data/STS-B/STS-B.test.data')
        sents1, sents2, labels = load_test_data(test_path)
        t1 = time()
        scores = m.get_scores(sents1, sents2)
        sims = []
        for i in range(len(sents1)):
            sims.append(scores[i][i])
        spend_time = max(time() - t1, 1e-9)
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
        spend_time = max(time() - t1, 1e-9)
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
        spend_time = max(time() - t1, 1e-9)
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
        spend_time = max(time() - t1, 1e-9)
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
        spend_time = max(time() - t1, 1e-9)
        corr = compute_spearmanr(sims, labels)
        print('scores:', sims[:10])
        print('labels:', labels[:10])
        print('sbert_batch_sim PAWSX spearman corr:', corr)
        print('spend time:', spend_time, ' seconds count:', len(sents1) * 2, 'qps:', len(sents1) * 2 / spend_time)

    def test_uer_sbert_nli_model(self):
        # uer/sbert-base-chinese-nli
        # STS-B spearman corr: 0.7179
        # ATEC spearman corr: 0.2953
        # BQ spearman corr: 0.4332
        # LCQMC spearman corr: 0.6239
        # PAWSX spearman corr: 0.1345
        # avg: 0.44096
        pass

    def test_ernie3_0_nano_model(self):
        # nghuyong/ernie-3.0-nano-zh
        # STS-B spearman corr: 0.6677
        # ATEC spearman corr: 0.2331
        # BQ spearman corr: 0.3716
        # LCQMC spearman corr: 0.6007
        # PAWSX spearman corr: 0.0970
        # avg: 0.3918
        # V100 QPS: 2858
        pass

    def test_ernie3_0_base_model(self):
        # nghuyong/ernie-3.0-base-zh
        # training with first_last_avg pooling and inference with mean pooling
        # STS-B spearman corr: 0.7981
        # ATEC spearman corr: 0.2965
        # BQ spearman corr: 0.3535
        # LCQMC spearman corr: 0.7184
        # PAWSX spearman corr: 0.1453
        # avg: 0.4619
        # V100 QPS: 1547

        # training with first_last_avg pooling and inference with first_last_avg pooling
        # STS-B spearman corr: 0.7931
        # ATEC spearman corr: 0.2997
        # BQ spearman corr: 0.3749
        # LCQMC spearman corr: 0.7110
        # PAWSX spearman corr: 0.1326
        # avg: 0.4421
        # V100 QPS: 1613

        # training with mean pooling and inference with mean pooling
        # STS-B spearman corr: 0.8153
        # ATEC spearman corr: 0.3319
        # BQ spearman corr: 0.4284
        # LCQMC spearman corr: 0.7293
        # PAWSX spearman corr: 0.1499
        # avg: 0.4909   (best)
        # V100 QPS: 1588

        # training with mean pooling and inference with mean pooling
        # training data: STS-B + ATEC + BQ + LCQMC + PAWSX
        # STS-B spearman corr: 0.80700
        # ATEC spearman corr: 0.5126
        # BQ spearman corr: 0.6872
        # LCQMC spearman corr: 0.7913
        # PAWSX spearman corr: 0.3428
        # avg: 0.6281
        # V100 QPS: 1526
        pass

    def test_ernie3_0_xbase_model(self):
        # nghuyong/ernie-3.0-xbase-zh
        # STS-B spearman corr: 0.7827
        # ATEC spearman corr: 0.3463
        # BQ spearman corr: 0.4267
        # LCQMC spearman corr: 0.7181
        # PAWSX spearman corr: 0.1318
        # avg: 0.4777
        # V100 QPS: 468
        pass

    def test_hfl_chinese_bert_wwm_ext_model(self):
        # hfl/chinese-bert-wwm-ext
        # STS-B spearman corr: 0.7635
        # ATEC spearman corr: 0.2708
        # BQ spearman corr: 0.3480
        # LCQMC spearman corr: 0.7056
        # PAWSX spearman corr: 0.1699
        # avg: 0.4515
        # V100 QPS: 1507
        pass

    def test_hfl_chinese_roberta_wwm_ext_model(self):
        # hfl/chinese-roberta-wwm-ext
        # training with first_last_avg pooling and inference with mean pooling
        # STS-B spearman corr: 0.7894
        # ATEC spearman corr: 0.3241
        # BQ spearman corr: 0.4362
        # LCQMC spearman corr: 0.7107
        # PAWSX spearman corr: 0.1446
        # avg: 0.4808
        # V100 QPS: 1472

        # hfl/chinese-roberta-wwm-ext
        # training with first_last_avg pooling and inference with first_last_avg pooling
        # STS-B spearman corr: 0.7854
        # ATEC spearman corr: 0.3234
        # BQ spearman corr: 0.4402
        # LCQMC spearman corr: 0.7029
        # PAWSX spearman corr: 0.1295
        # avg: 0.4739
        # V100 QPS: 1581

        # hfl/chinese-roberta-wwm-ext
        # training with mean pooling and inference with mean pooling
        # STS-B spearman corr: 0.7996
        # ATEC spearman corr: 0.3315
        # BQ spearman corr: 0.4364
        # LCQMC spearman corr: 0.7175
        # PAWSX spearman corr: 0.1472
        # avg: 0.4864
        # V100 QPS: 1487
        pass

    def test_hfl_chinese_macbert_large_model(self):
        # hfl/chinese-macbert-large
        # STS-B spearman corr: 0.7495
        # ATEC spearman corr: 0.3222
        # BQ spearman corr: 0.4608
        # LCQMC spearman corr: 0.6784
        # PAWSX spearman corr: 0.1081
        # avg: 0.4634
        # V100 QPS: 672
        pass



if __name__ == '__main__':
    unittest.main()
