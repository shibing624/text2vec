# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import sys
import unittest

sys.path.append('..')

from text2vec import SentenceModel, cos_sim
from text2vec import BM25

sbert_model = SentenceModel()


def sbert_sim_score(str_a, str_b):
    a_emb = sbert_model.encode(str_a)
    b_emb = sbert_model.encode(str_b)
    return cos_sim(a_emb, b_emb)


class IssueTestCase(unittest.TestCase):

    def test_sim_diff(self):
        a = '研究团队面向国家重大战略需求追踪国际前沿发展借鉴国际人工智能研究领域的科研模式有效整合创新资源解决复'
        b = '英汉互译比较语言学'
        r = sbert_sim_score(a, b)
        print(a, b, r)
        self.assertTrue(abs(float(r) - 0.4098) < 0.001)

    def test_sim_same(self):
        a = '汉英翻译比较语言学'
        b = '英汉互译比较语言学'
        r = sbert_sim_score(a, b)
        print(a, b, r)
        self.assertTrue(abs(float(r) - 0.8905) < 0.001)

    def test_search_sim(self):
        sentences = [
            '原称《车骑出行》。', '画面从左至右为：四导骑，', '两两并排前行，', '骑手面相对，', '似交谈；', '三导骑，',
            '并排前行；', '二马轺车，', '轮有辐，',
            '车上一驭者执鞭，', '一尊者坐，', '回首；', '二导从，', '骑手面相对，', '似交谈；', '二马轺车，', '轮有辐，',
            '车上一驭者执鞭，', '一尊者坐；', '四导从，', '两两相对并排前行；', '两骑手，', '反身张弓射虎；', '虎，',
            '跃起前扑。', '上下右三边有框，', '上沿双边框内填刻三角纹，', '下沿双边框内填刻斜条纹。']
        self.assertEqual(len(sentences), 28)
        uniq_sentences = list(set(sentences))
        print(uniq_sentences)
        print(len(uniq_sentences))
        self.assertEqual(len(uniq_sentences), 23)

        search_sim = BM25(corpus=uniq_sentences)
        print(len(search_sim.corpus))
        query = '上沿双边框内填刻三角形纹'
        scores = search_sim.get_scores(query=query, top_k=None)
        print(scores)
        print(len(scores))
        self.assertEqual(len(scores), 23)


if __name__ == '__main__':
    unittest.main()
