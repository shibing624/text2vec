# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import sys
import unittest

sys.path.append('..')
from text2vec import Similarity

a = '什么是智能手环'
b = '智能手环是个啥'
c = '智能手环有什么用'
d = '智能手环能干什么'


class SimTestCase(unittest.TestCase):
    def test_sents_score(self):
        """测试句子之间相似度值-word2vec"""
        sim = Similarity()

        print(sim.get_score(a, b), a, b)
        print(sim.get_score(a, c), a, c)
        print(sim.get_score(a, d), a, d)
        print(sim.get_score(b, c), b, c)
        print(sim.get_score(b, d), b, d)
        print(sim.get_score(c, d), c, d)

        print("{:.3f}".format(sim.get_score(a, b)))
        self.assertTrue("{:.3f}".format(sim.get_score(a, b)) == "1.000")
        self.assertTrue("{:.3f}".format(sim.get_score(a, c)) == "1.000")
        self.assertTrue("{:.3f}".format(sim.get_score(a, d)) == "0.903")
        self.assertTrue("{:.3f}".format(sim.get_score(b, c)) == "1.000")
        self.assertTrue("{:.3f}".format(sim.get_score(b, d)) == "0.903")
        self.assertTrue("{:.3f}".format(sim.get_score(c, d)) == "0.903")

    def test_sents_score_bert(self):
        """测试句子之间相似度值-bert"""
        sim = Similarity(embedding_type='bert')
        print("{:.3f}".format(sim.get_score(a, b)))
        self.assertTrue("{:.3f}".format(sim.get_score(a, b)) == "0.915")

    def test_search_score(self):
        """测试计算句子与文档集之间的相似度值"""
        from text2vec import SearchSimilarity

        corpus = [a, b, c]
        print(corpus)
        search_sim = SearchSimilarity(corpus=corpus)

        scores = search_sim.get_scores(query=d)
        print(d, 'scores:', scores)
        self.assertTrue(' '.join(["{:.3f}".format(i) for i in scores]) == "-0.192 -0.174 -0.174")
        print(d, 'rank similarities:', search_sim.get_similarities(query=d))
        self.assertTrue(search_sim.get_similarities(query=d) == ['智能手环有什么用', '智能手环是个啥', '什么是智能手环'])

    def test_oov_sim(self):
        """测试OOV word 相似度"""
        sim = Similarity()
        a = '，'
        b = '花'
        s = sim.get_score(a, b)
        print(a, b, s)
        self.assertTrue(s == 0.0)

        a = '，画画'
        b = '花画画'
        s = sim.get_score(a, b)
        print(a, b, s)
        self.assertTrue(round(s,3) == 0.822)

        a = '，'
        b = '花画画'
        s = sim.get_score(a, b)
        print(a, b, s)
        self.assertTrue(round(s,3) == 0.000)

        a = '，机票'
        b = '特价机票'
        s = sim.get_score(a, b)
        print(a, b, s)
        self.assertTrue(round(s,3) == 0.884)

        a = '机票'
        b = '特价机票'
        s = sim.get_score(a, b)
        print(a, b, s)
        self.assertTrue(round(s,3) == 0.884)

        a = '特价机票'
        b = '特价的机票'
        s = sim.get_score(a, b)
        print(a, b, s)
        self.assertTrue(round(s,3) == 1.000)


if __name__ == '__main__':
    unittest.main()
