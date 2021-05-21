# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import sys
import unittest

sys.path.append('..')
from text2vec import Similarity, EmbType, SimType


class WmdTestCase(unittest.TestCase):
    def test_encode(self):
        """测试文本wmd encode结果"""
        sim = Similarity(embedding_type=EmbType.W2V, similarity_type=SimType.WMD)
        a = '如何更换花呗绑定银行卡'
        b = '花呗更改绑定银行卡'
        c = '我什么时候开通了花呗'

        emb = sim.encode(a)
        print(a, emb)
        print(emb.shape)
        self.assertEqual(emb.shape, (200,))

        s = sim.get_score(a, b)
        print(a, b, s)
        self.assertTrue("{:.3f}".format(s) == "0.746")

        s = sim.get_score(a, c)
        print(a, c, s)
        self.assertTrue("{:.3f}".format(s) == "0.573")

        s = sim.get_score(b, c)
        print(b, c, s)
        self.assertTrue("{:.3f}".format(s) == "0.563")


if __name__ == '__main__':
    unittest.main()
