# -*- coding: utf-8 -*-
"""
@author:XuMing<xuming624@qq.com>
@description: 
"""
import text2vec
from text2vec import Similarity

sim = Similarity()


def test_sim_diff():
    a = '研究团队面向国家重大战略需求追踪国际前沿发展借鉴国际人工智能研究领域的科研模式有效整合创新资源解决复'
    b = '英汉互译比较语言学'

    print(a, b, sim.get_score(a, b))


def test_sim_same():
    a = '汉英翻译比较语言学'
    b = '英汉互译比较语言学'

    print(a, b, sim.get_score(a, b))


def test_diff_len():
    sentences = ['原称《车骑出行》。', '画面从左至右为：四导骑，', '两两并排前行，', '骑手面相对，', '似交谈；', '三导骑，', '并排前行；', '二马轺车，', '轮有辐，', '车上一驭者执鞭，',
                 '一尊者坐，', '回首；', '二导从，', '骑手面相对，', '似交谈；', '二马轺车，', '轮有辐，', '车上一驭者执鞭，', '一尊者坐；', '四导从，', '两两相对并排前行；',
                 '两骑手，', '反身张弓射虎；', '虎，', '跃起前扑。', '上下右三边有框，', '上沿双边框内填刻三角纹，', '下沿双边框内填刻斜条纹。']
    print(len(sentences))
    uniq_sentences = set(sentences)
    print(len(uniq_sentences))
    search_sim = text2vec.SearchSimilarity(corpus=uniq_sentences)
    print(len(search_sim.corpus))
    query = '上沿双边框内填刻三角形纹'
    scores = search_sim.get_scores(query=query)
    print(scores)
    print(len(scores))