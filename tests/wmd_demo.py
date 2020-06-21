# -*- coding: utf-8 -*-
"""
@author:XuMing<xuming624@qq.com>
@description: 
"""

from text2vec import Similarity, EmbType, SimType

sim = Similarity(embedding_type=EmbType.W2V, similarity_type=SimType.WMD)
a = '如何更换花呗绑定银行卡'
b = '花呗更改绑定银行卡'
c = '我什么时候开通了花呗'
emb = sim.encode(a)
print(a, emb)
print(emb.shape)
s = sim.get_score(a, b)
print(a, b, s)

s = sim.get_score(a, c)
print(a, c, s)

s = sim.get_score(b, c)
print(b, c, s)
