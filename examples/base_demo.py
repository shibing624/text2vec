# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import sys

import numpy as np

sys.path.append('..')
import text2vec

text2vec.set_log_level('INFO')

if __name__ == '__main__':
    char = '卡'
    emb = text2vec.encode(char)
    print(type(emb), emb.shape)
    print(char, emb)

    word = '银行卡'
    print(word, text2vec.encode(word))

    a = '如何更换花呗绑定银行卡'
    emb = text2vec.encode(a)
    print(a, emb)

    b = ['卡',
         '银行卡',
         '如何更换花呗绑定银行卡',
         '如何更换花呗绑定银行卡,如何更换花呗绑定银行卡。如何更换花呗绑定银行卡？。。。这个，如何更换花呗绑定银行卡！']
    res = []
    for i in b:
        emb = text2vec.encode(i)
        print(i, emb)
        res.append(emb)
    print(b, res)

    print(np.array(res).shape)
