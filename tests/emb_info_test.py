# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""

a = '湖北人爱吃鱼'
b = '甘肃人不爱吃鱼'
from text2vec import Similarity

ss = Similarity(embedding_type='w2v')
ss.score(a, b)
print(ss.model.info())

ss = Similarity(embedding_type='bert')
ss.score(a, b)
print(ss.model.info())
