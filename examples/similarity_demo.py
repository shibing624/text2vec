# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description:
"""

from text2vec import Similarity

a = '如何更换花呗绑定银行卡'
b = '花呗更改绑定银行卡'
c = '我什么时候开通了花呗'

sim = Similarity()
s = sim.get_score(a, b)
print(a, b, s)

s = sim.get_score(a, c)
print(a, c, s)

s = sim.get_score(b, c)
print(b, c, s)

from text2vec import SearchSimilarity

corpus = [a, b, c]
search_sim = SearchSimilarity(corpus=corpus)

print(search_sim.get_scores(query=a))
print(search_sim.get_similarities(query=a))
