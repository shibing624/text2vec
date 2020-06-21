# -*- coding: utf-8 -*-
"""
@author:XuMing<xuming624@qq.com>
@description: 
"""

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
