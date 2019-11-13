# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""

import simtext

a = '湖北人爱吃鱼'
b = '甘肃人不爱吃鱼'
c = '武汉人'
s = simtext.score(a, b)
print(a, b, s)

s = simtext.score(a, c)
print(a, c, s)

s = simtext.score(b, c)
print(b, c, s)
