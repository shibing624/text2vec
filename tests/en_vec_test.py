# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""

from text2vec.similarity import Similarity

a = ['如何更换花呗绑定银行卡', '花呗更改绑定银行卡', '我什么时候开通了花呗']
v = Similarity(a)
emb = v.get_tfidf()
print(emb)
