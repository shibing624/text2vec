# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import os
import sys
import unittest
from loguru import logger
import time

sys.path.append('..')
from text2vec import Word2Vec, SentenceModel

pwd_path = os.path.abspath(os.path.dirname(__file__))
logger.add('test.log')

data = ['如何更换花呗绑定银行卡',
        '花呗更改绑定银行卡']
print("data:", data)
num_tokens = sum([len(i) for i in data])


class QPSEncoderTestCase(unittest.TestCase):
    def test_cosent_speed(self):
        """测试cosent_speed"""
        model = SentenceModel('shibing624/text2vec-base-chinese')
        for j in range(10):
            tmp = data * (2 ** j)
            c_num_tokens = num_tokens * (2 ** j)
            start_t = time.time()
            r = model.encode(tmp)
            assert r is not None
            print('result shape', r.shape)
            time_t = time.time() - start_t
            logger.info("----\ncosent:")
            logger.info('encoding %d sentences, spend %.2fs, %4d samples/s, %6d tokens/s' %
                        (len(tmp), time_t, int(len(tmp) / time_t), int(c_num_tokens / time_t)))

    def test_sbert_speed(self):
        """测试sbert_speed"""
        model = SentenceModel('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        for j in range(10):
            tmp = data * (2 ** j)
            c_num_tokens = num_tokens * (2 ** j)
            start_t = time.time()
            r = model.encode(tmp)
            assert r is not None
            print('result shape', r.shape)
            time_t = time.time() - start_t
            logger.info("----\nsbert:")
            logger.info('encoding %d sentences, spend %.2fs, %4d samples/s, %6d tokens/s' %
                        (len(tmp), time_t, int(len(tmp) / time_t), int(c_num_tokens / time_t)))

    def test_w2v_speed(self):
        """测试w2v_speed"""
        model = Word2Vec()
        for j in range(10):
            tmp = data * (2 ** j)
            c_num_tokens = num_tokens * (2 ** j)
            start_t = time.time()
            r = model.encode(tmp)
            assert r is not None
            print('result shape', r.shape)
            time_t = time.time() - start_t
            logger.info("----\nword2vec:")
            logger.info('encoding %d sentences, spend %.2fs, %4d samples/s, %6d tokens/s' %
                        (len(tmp), time_t, int(len(tmp) / time_t), int(c_num_tokens / time_t)))


if __name__ == '__main__':
    unittest.main()
