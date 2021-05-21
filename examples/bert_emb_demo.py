# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import sys

sys.path.append('..')
from text2vec import Vector
from text2vec import Similarity

if __name__ == '__main__':
    vec = Vector(embedding_type='bert')
    char = '卡'
    emb = vec.encode(char)
    # <class 'numpy.ndarray'> (128, 3072) 128=seq_len, 3072=768*4
    print(type(emb), emb.shape)
    print(char, emb)

    word = '银行卡'
    print(word, vec.encode(word))

    a = '如何更换花呗绑定银行卡'
    emb = vec.encode(a)
    print(a, emb)
    print(emb.shape)

    sim = Similarity(embedding_type='bert')
    b = '花呗更改绑定银行卡'
    print(sim.get_score(a, b))
