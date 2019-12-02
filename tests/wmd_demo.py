# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""

from text2vec import Similarity
sim = Similarity(embedding_type='w2v', similarity_type='wmd')
a = '如何更换花呗绑定银行卡'
b = '花呗更改绑定银行卡'
c = '我什么时候开通了花呗'
emb = sim.encode(a)
print(emb)

s = sim.score(a, b)
print(a, b, s)


s = sim.score(a, c)
print(a, c, s)

s = sim.score(b, c)
print(b, c, s)
