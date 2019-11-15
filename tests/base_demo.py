# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""

import simtext

a = '如何更换花呗绑定银行卡'
b = '花呗更改绑定银行卡'
c = '我什么时候开通了花呗'
s = simtext.score(a, b)
print(a, b, s)

s = simtext.score(a, c)
print(a, c, s)

s = simtext.score(b, c)
print(b, c, s)
