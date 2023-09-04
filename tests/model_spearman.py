# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import os
import sys
from time import time

sys.path.append('..')
from text2vec import Similarity, SimilarityType, EmbeddingType, compute_spearmanr
from text2vec import load_jsonl

pwd_path = os.path.abspath(os.path.dirname(__file__))

is_debug = False


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


def get_sim_model_batch(model_name):
    """测试test_set_sim_model_batch"""
    m = Similarity(
        model_name,
        similarity_type=SimilarityType.COSINE,
        embedding_type=EmbeddingType.BERT,
        encoder_type="MEAN"
    )
    print(m)
    # STS-B
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
    print('\naverage 5 dataset spearman corr:', (c1 + c2 + c3 + c4 + c5) / 5)

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


if __name__ == '__main__':
    if (len(sys.argv)) >= 2:
        model_name = sys.argv[1]
    else:
        model_name = 'shibing624/text2vec-base-chinese'
    get_sim_model_batch(model_name)
