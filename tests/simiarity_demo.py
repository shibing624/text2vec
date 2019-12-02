# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""

from text2vec.similarity import Similarity
from text2vec.utils.tokenizer import segment

a = ['如何更换花呗绑定银行卡', '花呗更改绑定银行卡', '我什么时候开通了花呗']
corpus_data = [segment(i) for i in a]
sim = Similarity(corpus=corpus_data)
print(sim.get_similarities(query='花呗更改绑定银行卡'))
