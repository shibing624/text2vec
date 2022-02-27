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
from text2vec import Word2Vec, SentenceModel

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


class QPSEncoderTestCase(unittest.TestCase):
    def test_cosent_speed(self):
        """测试cosent_speed"""
        sents1, sents2, labels = load_test_data(sts_test_path)
        m = SentenceModel('shibing624/text2vec-base-chinese')
        sents = sents1 + sents2
        print('sente size:', len(sents))
        t1 = time()
        m.encode(sents)
        spend_time = time() - t1
        print('spend time:', spend_time, ' seconds')
        print('cosent_sbert qps:', len(sents) / spend_time)

    def test_sbert_speed(self):
        """测试sbert_speed"""
        sents1, sents2, labels = load_test_data(sts_test_path)
        m = SentenceModel()
        sents = sents1 + sents2
        print('sente size:', len(sents))
        t1 = time()
        m.encode(sents)
        spend_time = time() - t1
        print('spend time:', spend_time, ' seconds')
        print('sbert qps:', len(sents) / spend_time)

    def test_w2v_speed(self):
        """测试w2v_speed"""
        sents1, sents2, labels = load_test_data(sts_test_path)
        m = Word2Vec()
        sents = sents1 + sents2
        print('sente size:', len(sents))
        t1 = time()
        m.encode(sents)
        spend_time = time() - t1
        print('spend time:', spend_time, ' seconds')
        print('w2v qps:', len(sents) / spend_time)


if __name__ == '__main__':
    unittest.main()
