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
    result = text2vec.encode(char)
    print(type(result))
    print(char, result)
    s = np.sum(result)
    print(s)

    word = '银行卡'
    print(word, text2vec.encode(word))

    a = '如何更换花呗绑定银行卡'
    emb = text2vec.encode(a)
    print(a, emb)

    b = ['卡',
         '银行卡',
         '如何更换花呗绑定银行卡',
         '如何更换花呗绑定银行卡,如何更换花呗绑定银行卡。如何更换花呗绑定银行卡？。。。这个，如何更换花呗绑定银行卡！']
    result = []
    for i in b:
        r = text2vec.encode(i)
        result.append(r)
    print(b, result)

    print(np.array(result).shape)
