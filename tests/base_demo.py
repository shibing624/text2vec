# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""

import text2vec

a = '如何更换花呗绑定银行卡'
b = '花呗更改绑定银行卡'
c = '我什么时候开通了花呗'

char = '我'
print(char, text2vec.encode(char))

word = '如何'
print(word, text2vec.encode(word))

emb = text2vec.encode(a)
print(a, emb)

s = text2vec.score(a, b)
print(a, b, s)

s = text2vec.score(a, c)
print(a, c, s)

s = text2vec.score(b, c)
print(b, c, s)
