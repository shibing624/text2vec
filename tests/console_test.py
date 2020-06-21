# -*- coding: utf-8 -*-
"""
@author:XuMing<xuming624@qq.com>
@description: 
"""

from text2vec import Similarity
sim = Similarity()

if __name__ == '__main__':
    a = '什么是智能手环'
    b = '智能手环是个啥'
    c = '智能手环有什么用'
    d = '智能手环能干什么'
    print(sim.get_score(a, b), a, b)
    print(sim.get_score(a, c), a, c)
    print(sim.get_score(a, d), a, d)
    print(sim.get_score(b, c), b, c)
    print(sim.get_score(b, d), b, d)
    print(sim.get_score(c, d), c, d)

    print("example:")
    while True:
        a = input('input1:')
        b = input('input2:')
        print(sim.get_score(a, b), a, b)
