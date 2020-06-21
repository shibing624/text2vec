# -*- coding: utf-8 -*-
"""
@author:XuMing<xuming624@qq.com>
@description: 
"""
from text2vec import Similarity

a = '湖北人爱吃鱼'
b = '甘肃人不爱吃鱼'

ss = Similarity(embedding_type='w2v')
ss.get_score(a, b)
print(ss.model.info())

ss = Similarity(embedding_type='bert')
ss.get_score(a, b)
print(ss.model.info())
