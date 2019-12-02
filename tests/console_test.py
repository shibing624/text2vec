# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""

import text2vec

if __name__ == '__main__':
    a = '什么是智能手环'
    b = '智能手环是个啥'
    c = '智能手环有什么用'
    d = '智能手环能干什么'
    print(text2vec.score(a, b), a, b)
    print(text2vec.score(a, c), a, c)
    print(text2vec.score(a, d), a, d)
    print(text2vec.score(b, c), b, c)
    print(text2vec.score(b, d), b, d)
    print(text2vec.score(c, d), c, d)

    print("example:")
    while True:
        a = input('input1:')
        b = input('input2:')
        print(text2vec.score(a, b), a, b)
