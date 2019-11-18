# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""

import simtext

if __name__ == '__main__':
    a = '如何更换花呗绑定银行卡'
    b = '花呗更改绑定银行卡'
    s = simtext.score(a, b)
    print("example:")
    print(s, a, b)
    while True:
        a = input('input1:')
        b = input('input2:')
        print(simtext.score(a, b), a, b)
