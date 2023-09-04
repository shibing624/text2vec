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
from text2vec import load_jsonl

pwd_path = os.path.abspath(os.path.dirname(__file__))

is_debug = True


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
            if is_debug and len(sents1) > 10:
                break
    return sents1, sents2, labels


def get_corr(model, test_path):
    sents1, sents2, labels = load_test_data(test_path)
    t1 = time()
    scores = model.get_scores(sents1, sents2, only_aligned=True)
    sims = []
    for i in range(len(sents1)):
        sims.append(scores[i][i])
    spend_time = max(time() - t1, 1e-9)
    corr = compute_spearmanr(sims, labels)
    print('scores:', sims[:10])
    print('labels:', labels[:10])
    print(f'{test_path} spearman corr:', corr)
    print('spend time:', spend_time, ' seconds count:', len(sents1) * 2, 'qps:', len(sents1) * 2 / spend_time)
    return corr


class SimModelTestCase(unittest.TestCase):
    def test_w2v_sim_batch(self):
        """测试test_w2v_sim_batch"""
        model_name = 'w2v-light-tencent-chinese'
        print(model_name)
        m = Similarity(model_name, similarity_type=SimilarityType.COSINE, embedding_type=EmbeddingType.WORD2VEC)
        test_path = os.path.join(pwd_path, '../examples/data/STS-B/STS-B.test.data')
        get_corr(m, test_path)

        # ATEC
        test_path = os.path.join(pwd_path, '../examples/data/ATEC/ATEC.test.data')
        get_corr(m, test_path)

        # BQ
        test_path = os.path.join(pwd_path, '../examples/data/BQ/BQ.test.data')
        get_corr(m, test_path)

        # LCQMC
        test_path = os.path.join(pwd_path, '../examples/data/LCQMC/LCQMC.test.data')
        get_corr(m, test_path)

        # PAWSX
        test_path = os.path.join(pwd_path, '../examples/data/PAWSX/PAWSX.test.data')
        get_corr(m, test_path)

    def test_sbert_sim_stsb_batch(self):
        """测试sbert_sim_each_batch"""
        model_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
        print(model_name)
        m = Similarity(
            model_name,
            similarity_type=SimilarityType.COSINE,
            embedding_type=EmbeddingType.BERT,
            encoder_type="MEAN"
        )
        test_path = os.path.join(pwd_path, '../examples/data/STS-B/STS-B.test.data')
        get_corr(m, test_path)

        # ATEC
        test_path = os.path.join(pwd_path, '../examples/data/ATEC/ATEC.test.data')
        get_corr(m, test_path)

        # BQ
        test_path = os.path.join(pwd_path, '../examples/data/BQ/BQ.test.data')
        get_corr(m, test_path)

        # LCQMC
        test_path = os.path.join(pwd_path, '../examples/data/LCQMC/LCQMC.test.data')
        get_corr(m, test_path)

        # PAWSX
        test_path = os.path.join(pwd_path, '../examples/data/PAWSX/PAWSX.test.data')
        get_corr(m, test_path)

    def test_set_sim_model_batch(self):
        """测试test_set_sim_model_batch"""
        m = Similarity(
            'shibing624/text2vec-base-chinese',
            similarity_type=SimilarityType.COSINE,
            embedding_type=EmbeddingType.BERT,
            encoder_type="MEAN"
        )
        print(m)
        test_path = os.path.join(pwd_path, '../examples/data/STS-B/STS-B.test.data')
        c1 = get_corr(m, test_path)

        # ATEC
        test_path = os.path.join(pwd_path, '../examples/data/ATEC/ATEC.test.data')
        c2 = get_corr(m, test_path)

        # BQ
        test_path = os.path.join(pwd_path, '../examples/data/BQ/BQ.test.data')
        c3 = get_corr(m, test_path)

        # LCQMC
        test_path = os.path.join(pwd_path, '../examples/data/LCQMC/LCQMC.test.data')
        c4 = get_corr(m, test_path)

        # PAWSX
        test_path = os.path.join(pwd_path, '../examples/data/PAWSX/PAWSX.test.data')
        c5 = get_corr(m, test_path)

        # SOHU-dd
        test_path = os.path.join(pwd_path, '../examples/data/SOHU/dd-test.jsonl')
        data = load_jsonl(test_path)
        sents1, sents2, labels = [], [], []
        for item in data:
            sents1.append(item['sentence1'])
            sents2.append(item['sentence2'])
            labels.append(item['label'])
        t1 = time()
        scores = m.get_scores(sents1, sents2)
        sims = []
        for i in range(len(sents1)):
            sims.append(scores[i][i])
        spend_time = max(time() - t1, 1e-9)
        corr = compute_spearmanr(sims, labels)
        print('scores:', sims[:10])
        print('labels:', labels[:10])
        print(f'{test_path} spearman corr:', corr)
        print('spend time:', spend_time, ' seconds count:', len(sents1) * 2, 'qps:', len(sents1) * 2 / spend_time)
        c6 = corr

        # SOHU-dc
        test_path = os.path.join(pwd_path, '../examples/data/SOHU/dc-test.jsonl')
        data = load_jsonl(test_path)
        sents1, sents2, labels = [], [], []
        for item in data:
            sents1.append(item['sentence1'])
            sents2.append(item['sentence2'])
            labels.append(item['label'])
        t1 = time()
        scores = m.get_scores(sents1, sents2)
        sims = []
        for i in range(len(sents1)):
            sims.append(scores[i][i])
        spend_time = max(time() - t1, 1e-9)
        corr = compute_spearmanr(sims, labels)
        print('scores:', sims[:10])
        print('labels:', labels[:10])
        print(f'{test_path} spearman corr:', corr)
        print('spend time:', spend_time, ' seconds count:', len(sents1) * 2, 'qps:', len(sents1) * 2 / spend_time)
        c7 = corr
        print('average spearman corr:', (c1 + c2 + c3 + c4 + c5 + c6 + c7) / 7)

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
        # sohu-dd spearman corr: 0.7032
        # sohu-dc spearman corr: 0.5723
        # add sohu-dd and sohu-dc avg: 0.5329
        # V100 QPS: 1588

        # training with mean pooling and inference with mean pooling
        # retrain with 512 length
        # STS-B spearman corr: 0.7962
        # ATEC spearman corr: 0.2852
        # BQ spearman corr:  0.34746
        # LCQMC spearman corr: 0.7073
        # PAWSX spearman corr: 0.16109
        # avg:
        # V100 QPS: 1552

        # training with mean pooling and inference with mean pooling
        # retrain with 256 length
        # STS-B spearman corr: 0.8080
        # ATEC spearman corr: 0.3006
        # BQ spearman corr: 0.3927
        # LCQMC spearman corr: 0.71993
        # PAWSX spearman corr: 0.1371
        # avg:
        # V100 QPS:

        # training with mean pooling and inference with mean pooling
        # training data: STS-B + ATEC + BQ + LCQMC + PAWSX
        # STS-B spearman corr: 0.8070
        # ATEC spearman corr: 0.5126
        # BQ spearman corr: 0.6872
        # LCQMC spearman corr: 0.7913
        # PAWSX spearman corr: 0.3428
        # avg: 0.6281
        # sohu-dd spearman corr: 0.6939
        # sohu-dc spearman corr: 0.5544
        # add sohu-dd and sohu-dc avg: 0.6270
        # V100 QPS: 1526

        # training with mean pooling and inference with mean pooling
        # training data: all nli-zh-all random sampled data
        # STS-B spearman corr: 0.7742
        # ATEC spearman corr: 0.4394
        # BQ spearman corr: 0.6436
        # LCQMC spearman corr: 0.7345
        # PAWSX spearman corr: 0.3914
        # avg: 0.5966
        # sohu-dd spearman corr: 0.49124
        # sohu-dc spearman corr: 0.3653
        # V100 QPS: 1603

        # training with mean pooling and inference with mean pooling
        # training data: all nli-zh-all human sampled data (v2)
        # STS-B spearman corr: 0.7825
        # ATEC spearman corr: 0.4337
        # BQ spearman corr: 0.6143
        # LCQMC spearman corr: 0.7348
        # PAWSX spearman corr: 0.3890
        # avg: 0.5908
        # sohu-dd spearman corr:  0.7034
        # sohu-dc spearman corr: 0.5491
        # add sohu avg: 0.6009
        # V100 QPS:  1560

        # training with mean pooling and inference with mean pooling
        # training data: all sts-zh sampled data only stsb + sts22 (v3) (drop)
        # STS-B spearman corr: 0.8163
        # ATEC spearman corr: 0.3236
        # BQ spearman corr: 0.3905
        # LCQMC spearman corr: 0.7279
        # PAWSX spearman corr: 0.1377
        # avg: 0.4811
        # sohu-dd spearman corr: 0.7162
        # sohu-dc spearman corr: 0.5721
        # V100 QPS:  1557

        # training with mean pooling and inference with mean pooling
        # training data: all sts-zh human sampled data only stsb + nli (v4)
        # STS-B spearman corr: 0.7893
        # ATEC spearman corr: 0.4489
        # BQ spearman corr: 0.6358
        # LCQMC spearman corr: 0.7424
        # PAWSX spearman corr: 0.4090
        # avg: 0.6072
        # sohu-ddb spearman corr: 0.7670
        # sohu-dcb spearman corr: 0.6330
        # add sohu avg: 0.6308
        # V100 QPS: 1601
        pass

    def test_ernie3_0_xbase_model(self):
        # nghuyong/ernie-3.0-xbase-zh
        # STS-B spearman corr: 0.7827
        # ATEC spearman corr: 0.3463
        # BQ spearman corr: 0.4267
        # LCQMC spearman corr: 0.7181
        # PAWSX spearman corr: 0.1318
        # avg: 0.4811
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

    def test_m3e_base_model(self):
        # moka-ai/m3e-base
        # STS-B spearman corr: 0.7696
        # ATEC spearman corr: 0.4127
        # BQ spearman corr: 0.6381
        # LCQMC spearman corr: 0.7487
        # PAWSX spearman corr: 0.1220
        # avg: 0.5378
        # V100 QPS: 1490
        # sohu-dd spearman corr: 0.7583
        # sohu-dc spearman corr: 0.6055
        # add sohu avg: 0.5793
        pass

    def test_bge_large_zh_noinstruct_model(self):
        # BAAI/bge-large-zh-noinstruct
        # STS-B spearman corr: 0.7292
        # ATEC spearman corr: 0.4466
        # BQ spearman corr: 0.54995
        # LCQMC spearman corr: 0.69834
        # PAWSX spearman corr: 0.15612
        # avg: 0.51606
        # V100 QPS: 470
        # sohu-dd spearman corr: 0.53378
        # sohu-dc spearman corr: 0.198637
        # add sohu avg: 0.4732
        pass

    def test_bge_large_zh_noinstruct_cosent_model(self):
        # BAAI/bge-large-zh-noinstruct with sts-b cosent finetuned
        # STS-B spearman corr: 0.8059
        # ATEC spearman corr: 0.4234
        # BQ spearman corr: 0.515842
        # LCQMC spearman corr: 0.7291
        # PAWSX spearman corr: 0.1249
        # avg: 0.5198
        # V100 QPS: 498
        # sohu-dd spearman corr: 0.7243
        # sohu-dc spearman corr: 0.58399
        # add sohu avg: 0.5582
        pass

    def test_bge_large_zh_noinstruct_cosent_passage_model(self):
        # BAAI/bge-large-zh-noinstruct with sts-b cosent finetuned v2
        # STS-B spearman corr: 0.7644
        # ATEC spearman corr: 0.38411
        # BQ spearman corr: 0.61348
        # LCQMC spearman corr: 0.717220
        # PAWSX spearman corr: 0.351538
        # avg: 0.5661
        # V100 QPS:427
        # sohu-dd spearman corr: 0.7181
        # sohu-dc spearman corr: 0.631528
        # add sohu avg: 0.5972
        pass

    def test_bge_large_zh_noinstruct_bge_model(self):
        # BAAI/bge-large-zh-noinstruct with bge finetuned v2
        # STS-B spearman corr: 0.8093
        # ATEC spearman corr: 0.45839
        # BQ spearman corr: 0.56505
        # LCQMC spearman corr: 0.742664
        # PAWSX spearman corr: 0.11136
        # avg: 0.53736
        # V100 QPS: 605
        # sohu-dd spearman corr: 0.566741
        # sohu-dc spearman corr: 0.2098
        # add sohu avg: 0.4947
        pass


if __name__ == '__main__':
    unittest.main()
