# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import sys
import unittest

sys.path.append('..')
import text2vec
import numpy as np


class BaseTestCase(unittest.TestCase):
    def test_encode_char(self):
        """测试文本 char encode结果"""
        char = '卡'
        emb = text2vec.encode(char)
        t = type(emb)
        print(t)
        self.assertTrue(t == np.ndarray)

        print(char, emb, emb.shape)
        self.assertEqual(emb.shape, (200,))

        print(' '.join(["{:.3f}".format(i) for i in emb[:3]]))
        self.assertTrue(' '.join(["{:.3f}".format(i) for i in emb[:3]]) == "0.068 -0.110 -0.048")

    def test_encode_word(self):
        """测试文本 word encode结果"""
        word = '银行卡'
        emb = text2vec.encode(word)
        print(word, emb)
        self.assertEqual(emb.shape, (200,))
        self.assertTrue(' '.join(["{:.3f}".format(i) for i in emb[:3]]) == "0.002 -0.126 0.053")

    def test_encode_text(self):
        """测试文本 text encode结果"""
        a = '如何更换花呗绑定银行卡'
        emb = text2vec.encode(a)
        print(a, emb)
        self.assertEqual(emb.shape, (200,))
        self.assertTrue(' '.join(["{:.3f}".format(i) for i in emb[:3]]) == "0.041 -0.126 0.019")

    def test_oov_emb(self):
        """测试 OOV word embedding"""
        w = '，'
        comma_res = text2vec.encode(w)
        print(w, comma_res)
        self.assertEqual(comma_res, 0.0)

        w = '特价机票'
        r = text2vec.encode(w)
        print(w, r)

        w = '特价'
        r1 = text2vec.encode(w)
        print(w, r1)

        w = '机票'
        r2 = text2vec.encode(w)
        print(w, r2)

        emb = [r1, r2]
        r_average = np.array(emb).sum(axis=0) / 2.0
        print('r_average:', r_average)

        if str(r) == str(r_average):
            print('same')
        self.assertTrue(str(r) == str(r_average))


if __name__ == '__main__':
    unittest.main()
